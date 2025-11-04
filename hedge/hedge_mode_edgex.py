import asyncio
import json
import signal
import logging
import os
import sys
import time
import requests
import argparse
import traceback
import csv
from decimal import Decimal
from typing import Dict, Any, Tuple

from lighter.signer_client import SignerClient
from edgex_sdk import Client, OrderSide, WebSocketManager, CancelOrderParams
import websockets
from datetime import datetime
import pytz
import dotenv

dotenv.load_dotenv()


class HedgeBot:
    """Trading bot that places post-only orders on edgeX and hedges with market orders on Lighter."""

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, iterations: int = 20, sleep_time: int = 0):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.edgex_position = Decimal('0')
        self.lighter_position = Decimal('0')
        self.edgex_client_order_id = ''

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/{ticker}_hedge_mode_trades.csv"
        self.original_stdout = sys.stdout

        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv_file()

        # Setup logger
        self.logger = logging.getLogger(f"hedge_bot_{ticker}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Disable verbose logging from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create different formatters for file and console
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
        
        # State management
        self.stop_flag = False
        self.order_counter = 0
        
        # edgeX state
        self.edgex_client = None
        self.edgex_ws_manager = None
        self.edgex_contract_id = None
        self.edgex_tick_size = None
        self.edgex_order_status = None
        
        # edgeX websocket order book state
        self.edgex_order_book = {"bids": {}, "asks": {}}
        self.edgex_best_bid = None
        self.edgex_best_ask = None
        self.edgex_order_book_ready = False
        
        # Lighter order book state
        self.lighter_client = None
        self.lighter_order_book = {"bids": {}, "asks": {}}
        self.lighter_best_bid = None
        self.lighter_best_ask = None
        self.lighter_order_book_ready = False
        self.lighter_order_book_offset = 0
        self.lighter_order_book_sequence_gap = False
        self.lighter_snapshot_loaded = False
        self.lighter_order_book_lock = asyncio.Lock()
        
        # Lighter WebSocket state
        self.lighter_ws_task = None
        
        # Lighter order management
        self.lighter_order_price = None
        self.lighter_order_side = None
        self.lighter_order_size = None
        self.lighter_order_filled = False
        
        # Strategy state
        self.waiting_for_lighter_fill = False
        
        # Order execution tracking
        self.order_execution_complete = False
        
        
        # Lighter API configuration
        self.lighter_base_url = "https://mainnet.zklighter.elliot.ai"
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX'))
        
        # edgeX configuration
        self.edgex_account_id = os.getenv('EDGEX_ACCOUNT_ID')
        self.edgex_stark_private_key = os.getenv('EDGEX_STARK_PRIVATE_KEY')
        self.edgex_base_url = os.getenv('EDGEX_BASE_URL', 'https://pro.edgex.exchange')
        self.edgex_ws_url = os.getenv('EDGEX_WS_URL', 'wss://quote.edgex.exchange')

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Close WebSocket connections
        if self.edgex_ws_manager:
            try:
                self.edgex_ws_manager.disconnect_all()
                self.logger.info("🔌 edgeX WebSocket connections disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting edgeX WebSocket: {e}")

        # Cancel Lighter WebSocket task
        if self.lighter_ws_task and not self.lighter_ws_task.done():
            try:
                self.lighter_ws_task.cancel()
                self.logger.info("🔌 Lighter WebSocket task cancelled")
            except Exception as e:
                self.logger.error(f"Error cancelling Lighter WebSocket task: {e}")

        # Close logging handlers properly
        for handler in self.logger.handlers[:]:
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass

    def _initialize_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['exchange', 'timestamp', 'side', 'price', 'quantity'])
    
    def log_trade_to_csv(self, exchange: str, side: str, price: str, quantity: str):
        """Log trade details to CSV file."""
        timestamp = datetime.now(pytz.UTC).isoformat()
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                exchange,
                timestamp,
                side,
                price,
                quantity
            ])
        
        self.logger.info(f"📊 Trade logged to CSV: {exchange} {side} {quantity} @ {price}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    async def initialize_lighter_client(self):
        """Initialize the Lighter client."""
        if self.lighter_client is None:
            api_key_private_key = os.getenv('API_KEY_PRIVATE_KEY')
            if not api_key_private_key:
                raise Exception("API_KEY_PRIVATE_KEY environment variable not set")

            self.lighter_client = SignerClient(
                url=self.lighter_base_url,
                private_key=api_key_private_key,
                api_key_index=self.api_key_index,
                account_index=self.account_index,
            )

            # Check client
            err = self.lighter_client.check_client()
            if err is not None:
                raise Exception(f"CheckClient error: {err}")

            self.logger.info("✅ Lighter client initialized successfully")
        return self.lighter_client

    def initialize_edgex_client(self):
        """Initialize the edgeX client."""
        if not self.edgex_account_id or not self.edgex_stark_private_key:
            raise ValueError("EDGEX_ACCOUNT_ID and EDGEX_STARK_PRIVATE_KEY must be set in environment variables")

        # Initialize edgeX client using official SDK
        self.edgex_client = Client(
            base_url=self.edgex_base_url,
            account_id=int(self.edgex_account_id),
            stark_private_key=self.edgex_stark_private_key
        )

        # Initialize WebSocket manager using official SDK
        self.edgex_ws_manager = WebSocketManager(
            base_url=self.edgex_ws_url,
            account_id=int(self.edgex_account_id),
            stark_pri_key=self.edgex_stark_private_key
        )

        self.logger.info("✅ edgeX client initialized successfully")
        return self.edgex_client

    async def get_lighter_market_config(self) -> Tuple[int, int, int, Decimal]:
        """Get Lighter market configuration."""
        url = f"{self.lighter_base_url}/api/v1/orderBooks"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            if not response.text.strip():
                raise Exception("Empty response from Lighter API")

            data = response.json()

            if "order_books" not in data:
                raise Exception("Unexpected response format")

            for market in data["order_books"]:
                if market["symbol"] == self.ticker:
                    price_multiplier = pow(10, market["supported_price_decimals"])
                    return (market["market_id"], 
                           pow(10, market["supported_size_decimals"]), 
                           price_multiplier,
                           Decimal("1") / (Decimal("10") ** market["supported_price_decimals"])
                           )
            raise Exception(f"Ticker {self.ticker} not found")

        except Exception as e:
            self.logger.error(f"⚠️ Error getting market config: {e}")
            raise

    async def get_edgex_contract_info(self) -> Tuple[str, Decimal]:
        """Get edgeX contract ID and tick size."""
        if not self.edgex_client:
            raise Exception("edgeX client not initialized")
            
        response = await self.edgex_client.get_metadata()
        data = response.get('data', {})
        if not data:
            raise ValueError("Failed to get edgeX metadata")

        contract_list = data.get('contractList', [])
        if not contract_list:
            raise ValueError("Failed to get edgeX contract list")

        current_contract = None
        for c in contract_list:
            if c.get('contractName') == self.ticker + 'USD':
                current_contract = c
                break

        if not current_contract:
            raise ValueError(f"Failed to get contract ID for ticker {self.ticker}")

        contract_id = current_contract.get('contractId')
        min_quantity = Decimal(current_contract.get('minOrderSize'))
        tick_size = Decimal(current_contract.get('tickSize'))
        
        if self.order_quantity < min_quantity:
            raise ValueError(f"Order quantity is less than min quantity: {self.order_quantity} < {min_quantity}")

        return contract_id, tick_size

    async def fetch_edgex_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from edgeX websocket data."""
        if not self.edgex_order_book_ready:
            # Fallback to REST API if websocket data is not ready
            self.logger.warning("WebSocket order book not ready, falling back to REST API")
            return await self.fetch_edgex_bbo_prices_rest()
        
        # Use websocket data
        best_bid = self.edgex_best_bid if self.edgex_best_bid is not None else Decimal('0')
        best_ask = self.edgex_best_ask if self.edgex_best_ask is not None else Decimal('0')
        
        # If we have invalid data, fallback to REST API
        if best_bid == Decimal('0') or best_ask == Decimal('0'):
            self.logger.warning("Invalid WebSocket data (zero prices), falling back to REST API")
            return await self.fetch_edgex_bbo_prices_rest()
        
        return best_bid, best_ask

    async def fetch_edgex_bbo_prices_rest(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from edgeX REST API (fallback)."""
        if not self.edgex_client:
            raise Exception("edgeX client not initialized")
            
        from edgex_sdk import GetOrderBookDepthParams
        
        depth_params = GetOrderBookDepthParams(contract_id=self.edgex_contract_id, limit=15)
        order_book = await self.edgex_client.quote.get_order_book_depth(depth_params)
        order_book_data = order_book['data']

        # Get the first order book entry
        order_book_entry = order_book_data[0]
        bids = sorted(order_book_entry.get('bids', []), key=lambda x: Decimal(x['price']), reverse=True)  # Sort bids high to low
        asks = sorted(order_book_entry.get('asks', []), key=lambda x: Decimal(x['price']))  # Sort asks low to high

        # Best bid is the highest price someone is willing to buy at
        best_bid = Decimal(bids[0]['price']) if bids and len(bids) > 0 else Decimal('0')
        # Best ask is the lowest price someone is willing to sell at
        best_ask = Decimal(asks[0]['price']) if asks and len(asks) > 0 else Decimal('0')
        
        return best_bid, best_ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        if self.edgex_tick_size is None:
            return price
        return (price / self.edgex_tick_size).quantize(Decimal('1')) * self.edgex_tick_size

    async def place_bbo_order(self, side: str, quantity: Decimal):
        # Get best bid/ask prices
        best_bid, best_ask = await self.fetch_edgex_bbo_prices()
        data_source = "WebSocket" if self.edgex_order_book_ready else "REST API"
        self.logger.info(f"[{data_source}] Best bid: {best_bid}, Best ask: {best_ask}, Tick size: {self.edgex_tick_size}")
        
        if side.lower() == 'buy':
            # For buy orders, place slightly below best ask to ensure execution
            order_price = best_ask - self.edgex_tick_size
            # order_price = best_ask - Decimal('8')
            order_side = OrderSide.BUY
        else:
            # For sell orders, place slightly above best bid to ensure execution
            order_price = best_bid + self.edgex_tick_size
            # order_price = best_bid + Decimal('8')
            order_side = OrderSide.SELL

        self.edgex_client_order_id = str(int(time.time() * 1000))
        # Place the order using official SDK (post-only to ensure maker order)
        order_result = await self.edgex_client.create_limit_order(
            contract_id=self.edgex_contract_id,
            size=str(quantity),
            price=str(self.round_to_tick(order_price)),
            side=order_side,
            post_only=True,
            client_order_id=self.edgex_client_order_id
        )
        order_id = order_result['data'].get('orderId')
        return order_id

    async def place_edgex_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on edgeX."""
        if not self.edgex_client:
            raise Exception("edgeX client not initialized")
        
        if side == 'buy':
            order_type = "OPEN"
        else:
            order_type = "CLOSE"

        self.edgex_order_status = None
        self.logger.info(f"[{order_type}] [edgeX] [{side}] Placing edgeX POST-ONLY order")
        order_id = await self.place_bbo_order(side, quantity)

        start_time = time.time()
        while not self.stop_flag:
            if self.edgex_order_status == 'CANCELED':
                self.edgex_order_status = 'NEW'
                order_id = await self.place_bbo_order(side, quantity)
                start_time = time.time()
                await asyncio.sleep(1)
            elif self.edgex_order_status in ['NEW', 'OPEN', 'PENDING', 'CANCELING']:
                await asyncio.sleep(0.1)
                if time.time() - start_time > self.fill_timeout:
                    try:
                        cancel_params = CancelOrderParams(order_id=order_id)
                        # Cancel the order using official SDK
                        self.logger.info(f"[{order_id}] [OPEN] [edgeX] [{side}] Time out - Canceling edgeX order.")
                        cancel_result = await self.edgex_client.cancel_order(cancel_params)
                        await asyncio.sleep(1)
                    except Exception as e:
                        self.logger.error(f"❌ Error canceling edgeX order: {e}")
            elif self.edgex_order_status == 'FILLED':
                break
            else:
                if self.edgex_order_status is not None:
                    self.logger.error(f"❌ Unknown edgeX order status: {self.edgex_order_status}")
                    break
                else:
                    await asyncio.sleep(0.5)

    def handle_edgex_order_book_update(self, message):
        """Handle edgeX order book updates from WebSocket."""
        try:
            if isinstance(message, str):
                message = json.loads(message)
            
            self.logger.debug(f"Received depth message: {message}")
            
            # Check if this is a quote-event message with depth data
            if message.get("type") == "quote-event":
                content = message.get("content", {})
                channel = message.get("channel", "")
                
                self.logger.debug(f"Quote event message - channel: {channel}")
                
                if channel.startswith("depth."):
                    data = content.get('data', [])
                    if data and len(data) > 0:
                        order_book_data = data[0]
                        depth_type = order_book_data.get('depthType', '')
                        
                        self.logger.debug(f"Order book data (type: {depth_type})")
                        
                        # Handle SNAPSHOT (full data) or CHANGED (incremental updates)
                        if depth_type in ['SNAPSHOT', 'CHANGED']:
                            # Update bids - format is [{"price": "121699.0", "size": "5.128"}, ...]
                            bids = order_book_data.get('bids', [])
                            for bid in bids:
                                price = Decimal(bid['price'])
                                size = Decimal(bid['size'])
                                if size > 0:
                                    self.edgex_order_book['bids'][price] = size
                                else:
                                    # Remove zero size orders
                                    self.edgex_order_book['bids'].pop(price, None)
                            
                            # Update asks - format is [{"price": "121699.0", "size": "5.128"}, ...]
                            asks = order_book_data.get('asks', [])
                            for ask in asks:
                                price = Decimal(ask['price'])
                                size = Decimal(ask['size'])
                                if size > 0:
                                    self.edgex_order_book['asks'][price] = size
                                else:
                                    # Remove zero size orders
                                    self.edgex_order_book['asks'].pop(price, None)
                            
                            # Update best bid and ask
                            if self.edgex_order_book['bids']:
                                self.edgex_best_bid = max(self.edgex_order_book['bids'].keys())
                            if self.edgex_order_book['asks']:
                                self.edgex_best_ask = min(self.edgex_order_book['asks'].keys())
                            
                            if not self.edgex_order_book_ready:
                                self.edgex_order_book_ready = True
                                self.logger.info(f"📊 edgeX order book ready - Best bid: {self.edgex_best_bid}, Best ask: {self.edgex_best_ask}")
                            else:
                                self.logger.debug(f"📊 Order book updated - Best bid: {self.edgex_best_bid}, Best ask: {self.edgex_best_ask}")
                        
        except Exception as e:
            self.logger.error(f"Error handling edgeX order book update: {e}")
            self.logger.error(f"Message content: {message}")

    def handle_lighter_order_result(self, order_data):
        """Handle Lighter order result from WebSocket."""
        try:
            order_data["avg_filled_price"] = (Decimal(order_data["filled_quote_amount"]) /
                                              Decimal(order_data["filled_base_amount"]))
            if order_data["is_ask"]:
                order_data["side"] = "SHORT"
                self.lighter_position -= Decimal(order_data["filled_base_amount"])
            else:
                order_data["side"] = "LONG"
                self.lighter_position += Decimal(order_data["filled_base_amount"])

            self.logger.info(f"📊 Lighter order filled: {order_data['side']} {order_data['filled_base_amount']} @ {order_data['avg_filled_price']}")
            
            # Log Lighter trade to CSV
            self.log_trade_to_csv(
                exchange='Lighter',
                side=order_data['side'],
                price=str(order_data['avg_filled_price']),
                quantity=str(order_data['filled_base_amount'])
            )
            
            # Mark execution as complete
            self.lighter_order_filled = True  # Mark order as filled
            self.order_execution_complete = True
            
        except Exception as e:
            self.logger.error(f"Error handling Lighter order result: {e}")

    async def reset_lighter_order_book(self):
        """Reset Lighter order book state."""
        async with self.lighter_order_book_lock:
            self.lighter_order_book["bids"].clear()
            self.lighter_order_book["asks"].clear()
            self.lighter_order_book_offset = 0
            self.lighter_order_book_sequence_gap = False
            self.lighter_snapshot_loaded = False
            self.lighter_best_bid = None
            self.lighter_best_ask = None

    def update_lighter_order_book(self, side: str, levels: list):
        """Update Lighter order book with new levels."""
        for level in levels:
            # Handle different data structures - could be list [price, size] or dict {"price": ..., "size": ...}
            if isinstance(level, list) and len(level) >= 2:
                price = Decimal(level[0])
                size = Decimal(level[1])
            elif isinstance(level, dict):
                price = Decimal(level.get("price", 0))
                size = Decimal(level.get("size", 0))
            else:
                self.logger.warning(f"⚠️ Unexpected level format: {level}")
                continue
                
            if size > 0:
                self.lighter_order_book[side][price] = size
            else:
                # Remove zero size orders
                self.lighter_order_book[side].pop(price, None)

    def validate_order_book_offset(self, new_offset: int) -> bool:
        """Validate order book offset sequence."""
        if new_offset <= self.lighter_order_book_offset:
            self.logger.warning(f"⚠️ Out-of-order update: new_offset={new_offset}, current_offset={self.lighter_order_book_offset}")
            return False
        return True

    def validate_order_book_integrity(self) -> bool:
        """Validate order book integrity."""
        # Check for negative prices or sizes
        for side in ["bids", "asks"]:
            for price, size in self.lighter_order_book[side].items():
                if price <= 0 or size <= 0:
                    self.logger.error(f"❌ Invalid order book data: {side} price={price}, size={size}")
                    return False
        return True

    def get_lighter_best_levels(self) -> Tuple[Tuple[Decimal, Decimal], Tuple[Decimal, Decimal]]:
        """Get best bid and ask levels from Lighter order book."""
        best_bid = None
        best_ask = None
        
        if self.lighter_order_book["bids"]:
            best_bid_price = max(self.lighter_order_book["bids"].keys())
            best_bid_size = self.lighter_order_book["bids"][best_bid_price]
            best_bid = (best_bid_price, best_bid_size)
        
        if self.lighter_order_book["asks"]:
            best_ask_price = min(self.lighter_order_book["asks"].keys())
            best_ask_size = self.lighter_order_book["asks"][best_ask_price]
            best_ask = (best_ask_price, best_ask_size)
        
        return best_bid, best_ask

    def get_lighter_order_price(self, is_ask: bool) -> Decimal:
        """Get mid price from Lighter order book."""
        best_bid, best_ask = self.get_lighter_best_levels()
        
        if best_bid is None or best_ask is None:
            raise Exception("Cannot calculate mid price - missing order book data")
        
        if is_ask:
            order_price = best_bid[0]+self.tick_size
        else:
            order_price = best_ask[0]-self.tick_size

        return order_price

    def calculate_adjusted_price(self, original_price: Decimal, side: str, adjustment_percent: Decimal) -> Decimal:
        """Calculate adjusted price for order modification."""
        adjustment = original_price * adjustment_percent
        
        if side.lower() == 'buy':
            # For buy orders, increase price to improve fill probability
            return original_price + adjustment
        else:
            # For sell orders, decrease price to improve fill probability
            return original_price - adjustment

    async def request_fresh_snapshot(self, ws):
        """Request fresh order book snapshot."""
        await ws.send(json.dumps({"type": "subscribe", "channel": f"order_book/{self.lighter_market_index}"}))

    async def handle_lighter_ws(self):
        """Handle Lighter WebSocket connection and messages."""
        url = "wss://mainnet.zklighter.elliot.ai/stream"
        cleanup_counter = 0
        
        while not self.stop_flag:
            timeout_count = 0
            try:
                # Reset order book state before connecting
                await self.reset_lighter_order_book()
                
                async with websockets.connect(url) as ws:
                    # Subscribe to order book updates
                    await ws.send(json.dumps({"type": "subscribe", "channel": f"order_book/{self.lighter_market_index}"}))

                    # Subscribe to account orders updates
                    account_orders_channel = f"account_orders/{self.lighter_market_index}/{self.account_index}"

                    # Get auth token for the subscription
                    try:
                        # Set auth token to expire in 10 minutes
                        ten_minutes_deadline = int(time.time() + 10 * 60)
                        auth_token, err = self.lighter_client.create_auth_token_with_expiry(ten_minutes_deadline)
                        if err is not None:
                            self.logger.warning(f"⚠️ Failed to create auth token for account orders subscription: {err}")
                        else:
                            auth_message = {
                                "type": "subscribe",
                                "channel": account_orders_channel,
                                "auth": auth_token
                            }
                            await ws.send(json.dumps(auth_message))
                            self.logger.info("✅ Subscribed to account orders with auth token (expires in 10 minutes)")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error creating auth token for account orders subscription: {e}")

                    while not self.stop_flag:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1)

                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"⚠️ JSON parsing error in Lighter websocket: {e}")
                                continue

                            # Reset timeout counter on successful message
                            timeout_count = 0

                            async with self.lighter_order_book_lock:
                                if data.get("type") == "subscribed/order_book":
                                    # Initial snapshot - clear and populate the order book
                                    self.lighter_order_book["bids"].clear()
                                    self.lighter_order_book["asks"].clear()

                                    # Handle the initial snapshot
                                    order_book = data.get("order_book", {})
                                    if order_book and "offset" in order_book:
                                        self.lighter_order_book_offset = order_book["offset"]
                                        self.logger.info(f"✅ Initial order book offset set to: {self.lighter_order_book_offset}")

                                    # Debug: Log the structure of bids and asks
                                    bids = order_book.get("bids", [])
                                    asks = order_book.get("asks", [])
                                    if bids:
                                        self.logger.debug(f"📊 Sample bid structure: {bids[0] if bids else 'None'}")
                                    if asks:
                                        self.logger.debug(f"📊 Sample ask structure: {asks[0] if asks else 'None'}")

                                    self.update_lighter_order_book("bids", bids)
                                    self.update_lighter_order_book("asks", asks)
                                    self.lighter_snapshot_loaded = True
                                    self.lighter_order_book_ready = True

                                    self.logger.info(f"✅ Lighter order book snapshot loaded with "
                                          f"{len(self.lighter_order_book['bids'])} bids and "
                                          f"{len(self.lighter_order_book['asks'])} asks")

                                elif data.get("type") == "update/order_book" and self.lighter_snapshot_loaded:
                                    # Extract offset from the message
                                    order_book = data.get("order_book", {})
                                    if not order_book or "offset" not in order_book:
                                        self.logger.warning("⚠️ Order book update missing offset, skipping")
                                        continue

                                    new_offset = order_book["offset"]

                                    # Validate offset sequence
                                    if not self.validate_order_book_offset(new_offset):
                                        self.lighter_order_book_sequence_gap = True
                                        break

                                    # Update the order book with new data
                                    self.update_lighter_order_book("bids", order_book.get("bids", []))
                                    self.update_lighter_order_book("asks", order_book.get("asks", []))

                                    # Validate order book integrity after update
                                    if not self.validate_order_book_integrity():
                                        self.logger.warning("🔄 Order book integrity check failed, requesting fresh snapshot...")
                                        break

                                    # Get the best bid and ask levels
                                    best_bid, best_ask = self.get_lighter_best_levels()

                                    # Update global variables
                                    if best_bid is not None:
                                        self.lighter_best_bid = best_bid[0]
                                    if best_ask is not None:
                                        self.lighter_best_ask = best_ask[0]

                                elif data.get("type") == "ping":
                                    # Respond to ping with pong
                                    await ws.send(json.dumps({"type": "pong"}))
                                elif data.get("type") == "update/account_orders":
                                    # Handle account orders updates
                                    orders = data.get("orders", {}).get(str(self.lighter_market_index), [])
                                    if len(orders) == 1:
                                        order_data = orders[0]
                                        if order_data.get("status") == "filled":
                                            self.handle_lighter_order_result(order_data)
                                elif data.get("type") == "update/order_book" and not self.lighter_snapshot_loaded:
                                    # Ignore updates until we have the initial snapshot
                                    continue

                            # Periodic cleanup outside the lock
                            cleanup_counter += 1
                            if cleanup_counter >= 1000:
                                cleanup_counter = 0

                            # Handle sequence gap and integrity issues outside the lock
                            if self.lighter_order_book_sequence_gap:
                                try:
                                    await self.request_fresh_snapshot(ws)
                                    self.lighter_order_book_sequence_gap = False
                                except Exception as e:
                                    self.logger.error(f"⚠️ Failed to request fresh snapshot: {e}")
                                    break

                        except asyncio.TimeoutError:
                            timeout_count += 1
                            if timeout_count % 3 == 0:
                                self.logger.warning(f"⏰ No message from Lighter websocket for {timeout_count} seconds")
                            continue
                        except websockets.exceptions.ConnectionClosed as e:
                            self.logger.warning(f"⚠️ Lighter websocket connection closed: {e}")
                            break
                        except websockets.exceptions.WebSocketException as e:
                            self.logger.warning(f"⚠️ Lighter websocket error: {e}")
                            break
                        except Exception as e:
                            self.logger.error(f"⚠️ Error in Lighter websocket: {e}")
                            self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                            break
            except Exception as e:
                self.logger.error(f"⚠️ Failed to connect to Lighter websocket: {e}")
            
            # Wait a bit before reconnecting
            await asyncio.sleep(2)

    def handle_edgex_order_update(self, order_data):
        """Handle edgeX order updates from WebSocket."""
        order_id = order_data.get('order_id')
        status = order_data.get('status')
        side = order_data.get('side', '').lower()
        filled_size = Decimal(order_data.get('filled_size', '0'))
        price = Decimal(order_data.get('price', '0'))

        if side == 'buy':
            self.edgex_position += filled_size
            lighter_side = 'sell'
        else:
            self.edgex_position -= filled_size
            lighter_side = 'buy'
        
        # Store order details for immediate execution
        self.current_lighter_side = lighter_side
        self.current_lighter_quantity = filled_size
        self.current_lighter_price = price

        self.lighter_order_info = {
            'lighter_side': lighter_side,
            'quantity': filled_size,
            'price': price
        }

        self.waiting_for_lighter_fill = True
        
        self.logger.info(f"📋 Ready to place Lighter order: {lighter_side} {filled_size} @ {price}")

    async def place_lighter_limit_order(self, lighter_side: str, quantity: Decimal, price: Decimal):
        """Place a limit order on Lighter with mid price strategy."""
        if not self.lighter_client:
            await self.initialize_lighter_client()

        # Determine order parameters
        if lighter_side.lower() == 'buy':
            is_ask = False
            order_type = "CLOSE"
        else:
            order_type = "OPEN"
            is_ask = True

        self.logger.info(f"Placing Lighter limit order: {lighter_side} {quantity} @ {price} | is_ask: {is_ask}")

        # Calculate mid price from Lighter order book
        try:
            order_price = self.get_lighter_order_price(is_ask)
            # Convert to Lighter price format (multiply by price multiplier)
            lighter_price = int(order_price * self.price_multiplier)
        except Exception as e:
            self.logger.error(f"❌ Cannot calculate mid price: {e}")
            return None

        if lighter_price == 0:
            self.logger.error("❌ Cannot place Lighter order - invalid mid price")
            return None

        self.logger.info(f"[{order_type}] [Lighter] [{lighter_side}] Placing limit order at mid price: {order_price}")
        
        # Reset order state
        self.lighter_order_filled = False
        self.lighter_order_price = order_price
        self.lighter_order_side = lighter_side
        self.lighter_order_size = quantity
        
        # Sign the order transaction
        client_order_index = int(time.time() * 1000) % 1000000  # Simple unique ID
        
        try:
            # Use the native SignerClient's create_order method
            created_order, tx_hash, error = await self.lighter_client.create_order(
                market_index=self.lighter_market_index,
                client_order_index=client_order_index,
                base_amount=int(quantity * self.base_amount_multiplier),
                price=lighter_price,
                is_ask=is_ask,
                order_type=self.lighter_client.ORDER_TYPE_LIMIT,
                time_in_force=self.lighter_client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                reduce_only=False,
                trigger_price=0,
            )
            
            if error is not None:
                self.logger.error(f"❌ Lighter order error: {error}")
                return None

            self.logger.info(f"🚀 Lighter limit order sent: {lighter_side} {quantity} @ {order_price}")
            
            # Start monitoring in a separate task
            await self.monitor_lighter_order(client_order_index)
            
            return tx_hash
            
        except Exception as e:
            self.logger.error(f"❌ Error placing Lighter order: {e}")
            return None

    async def monitor_lighter_order(self, client_order_index: int):
        """Monitor Lighter order and adjust price if needed."""
        self.logger.info(f"🔍 Starting to monitor Lighter order - Order ID: {client_order_index}")

        start_time = time.time()
        order_place_time = time.time()
        while not self.lighter_order_filled and not self.stop_flag:
            # Check if 1 second has passed and we haven't adjusted price yet
            if time.time() - start_time > 1:
                try:
                    # Calculate adjusted price
                    adjusted_price = self.calculate_adjusted_price(
                        self.lighter_order_price, 
                        self.lighter_order_side, 
                        Decimal('0.0006')
                    )
                    
                    self.logger.info(f"🔄 Adjusting Lighter order price from {self.lighter_order_price} to {adjusted_price}")
                    
                    # Modify current order with adjusted price using client_order_index
                    await self.modify_lighter_order(client_order_index, adjusted_price)
                    self.lighter_order_price = adjusted_price
                    start_time = time.time()  # Reset timer after adjustment
                    
                except Exception as e:
                    self.logger.error(f"❌ Error adjusting Lighter order price: {e}")
                    break
            
            # Check for timeout (30 seconds total)
            if time.time() - order_place_time > 30:
                self.logger.error(f"❌ Timeout waiting for Lighter order fill after {time.time() - order_place_time:.1f}s")
                self.logger.error(f"❌ Order state - Filled: {self.lighter_order_filled}")
                
                # Fallback: Mark as filled to continue trading
                self.logger.warning("⚠️ Using fallback - marking order as filled to continue trading")
                self.lighter_order_filled = True
                self.waiting_for_lighter_fill = False
                self.order_execution_complete = True
                break
                
            await asyncio.sleep(0.1)  # Check every 100ms

    async def modify_lighter_order(self, client_order_index: int, new_price: Decimal):
        """Modify current Lighter order with new price using client_order_index."""
        try:
            if client_order_index is None:
                self.logger.error("❌ Cannot modify order - no order ID available")
                return

            # Calculate new Lighter price
            lighter_price = int(new_price * self.price_multiplier)
            
            self.logger.info(f"🔧 Attempting to modify order - Market: {self.lighter_market_index}, Client Order Index: {client_order_index}, New Price: {lighter_price}")
            
            # Use the native SignerClient's modify_order method
            tx_info, tx_hash, error = await self.lighter_client.modify_order(
                market_index=self.lighter_market_index,
                order_index=client_order_index,  # Use client_order_index directly
                base_amount=int(self.lighter_order_size * self.base_amount_multiplier),
                price=lighter_price,
                trigger_price=0
            )
            
            if error is not None:
                self.logger.error(f"❌ Lighter order modification error: {error}")
                return

            self.lighter_order_price = new_price
            self.logger.info(f"🔄 Lighter order modified successfully: {self.lighter_order_side} {self.lighter_order_size} @ {new_price}")
            
        except Exception as e:
            self.logger.error(f"❌ Error modifying Lighter order: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")

    async def setup_edgex_websocket(self):
        """Setup edgeX websocket for order updates and market data."""
        if not self.edgex_ws_manager:
            raise Exception("edgeX WebSocket manager not initialized")
            
        def order_update_handler(message):
            """Handle order updates from edgeX WebSocket."""
            # Parse the message structure
            if isinstance(message, str):
                message = json.loads(message)

            # Check if this is a trade-event with ORDER_UPDATE
            content = message.get("content", {})
            event = content.get("event", "")
            try:
                if event == "ORDER_UPDATE":
                    # Extract order data from the nested structure
                    data = content.get('data', {})
                    orders = data.get('order', [])

                    if orders and len(orders) > 0:
                        for order in orders:
                            if order.get('contractId') != self.edgex_contract_id:
                                continue
                            
                            if order.get('clientOrderId') != self.edgex_client_order_id:
                                continue
                            order_id = order.get('id')
                            status = order.get('status')
                            side = order.get('side', '').lower()
                            filled_size = Decimal(order.get('cumMatchSize'))
                            size = Decimal(order.get('size'))

                            if status == 'CANCELED':
                                if filled_size > 0:
                                    status = 'FILLED'
                                else:
                                    status = 'CANCELED'

                            if side == 'buy':
                                order_type = "OPEN"
                            else:
                                order_type = "CLOSE"

                            # Handle the order update
                            if status == 'FILLED' and self.edgex_order_status != 'FILLED':
                                self.logger.info(f"[{order_id}] [{order_type}] [edgeX] [{status}]: {filled_size} @ {order.get('price')}")
                                self.edgex_order_status = status
                                
                                # Log edgeX trade to CSV
                                self.log_trade_to_csv(
                                    exchange='edgeX',
                                    side=side,
                                    price=str(order.get('price', '0')),
                                    quantity=str(filled_size)
                                )
                                
                                # Call handle_edgex_order_update directly to avoid delay
                                order_data = {
                                    'order_id': order_id,
                                    'side': side,
                                    'status': status,
                                    'size': order.get('size'),
                                    'price': order.get('price'),
                                    'contract_id': order.get('contractId'),
                                    'filled_size': filled_size
                                }
                                
                                # Call handle_edgex_order_update directly (now sync!)
                                self.handle_edgex_order_update(order_data)
                            elif status == 'FILLED' and self.edgex_order_status == 'FILLED':
                                # Duplicate FILLED message - ignore to prevent double processing
                                self.logger.debug(f"[{order_id}] [{order_type}] [edgeX] Duplicate FILLED message ignored")
                            else:
                                self.logger.info(f"[{order_id}] [{order_type}] [edgeX] [{status}]: {size} @ {order.get('price')}.")
                                self.edgex_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling edgeX order update: {e}")

        try:
            # Setup private client for order updates
            private_client = self.edgex_ws_manager.get_private_client()
            private_client.on_message("trade-event", order_update_handler)
            self.logger.info("✅ edgeX WebSocket order update handler set up")
            
            # Setup public client for market data
            public_client = self.edgex_ws_manager.get_public_client()
            
            # Register handler for depth messages
            public_client.on_message("depth", self.handle_edgex_order_book_update)
            self.logger.info("✅ edgeX WebSocket depth handler registered")
            
        except Exception as e:
            self.logger.error(f"Could not setup edgeX WebSocket handlers: {e}")

    async def trading_loop(self):
        """Main trading loop implementing the new strategy."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")
        
        # Initialize clients
        try:
            await self.initialize_lighter_client()
            self.initialize_edgex_client()
            
            # Get contract info
            self.edgex_contract_id, self.edgex_tick_size = await self.get_edgex_contract_info()
            self.lighter_market_index, self.base_amount_multiplier, self.price_multiplier, self.tick_size = await self.get_lighter_market_config()
            
            self.logger.info(f"Contract info loaded - edgeX: {self.edgex_contract_id}, Lighter: {self.lighter_market_index}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return

        # Setup edgeX websocket
        try:
            await self.setup_edgex_websocket()
            # Connect both public (for market data) and private (for order updates) websockets
            self.edgex_ws_manager.connect_public()
            self.edgex_ws_manager.connect_private()
            self.logger.info("✅ edgeX WebSocket connections established")
            
            # Subscribe to depth channel after connection is established
            public_client = self.edgex_ws_manager.get_public_client()
            public_client.subscribe(f"depth.{self.edgex_contract_id}.15")
            self.logger.info(f"✅ Subscribed to depth channel: depth.{self.edgex_contract_id}.15")
            
            # Wait for initial order book data with timeout
            self.logger.info("⏳ Waiting for initial order book data...")
            timeout = 10  # seconds
            start_time = time.time()
            while not self.edgex_order_book_ready and not self.stop_flag:
                if time.time() - start_time > timeout:
                    self.logger.warning(f"⚠️ Timeout waiting for WebSocket order book data after {timeout}s")
                    break
                await asyncio.sleep(0.5)
            
            if self.edgex_order_book_ready:
                self.logger.info("✅ WebSocket order book data received")
            else:
                self.logger.warning("⚠️ WebSocket order book not ready, will use REST API fallback")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup edgeX websocket: {e}")
            return

        # Setup Lighter websocket
        try:
            self.lighter_ws_task = asyncio.create_task(self.handle_lighter_ws())
            self.logger.info("✅ Lighter WebSocket task started")
            
            # Wait for initial Lighter order book data with timeout
            self.logger.info("⏳ Waiting for initial Lighter order book data...")
            timeout = 10  # seconds
            start_time = time.time()
            while not self.lighter_order_book_ready and not self.stop_flag:
                if time.time() - start_time > timeout:
                    self.logger.warning(f"⚠️ Timeout waiting for Lighter WebSocket order book data after {timeout}s")
                    break
                await asyncio.sleep(0.5)
            
            if self.lighter_order_book_ready:
                self.logger.info("✅ Lighter WebSocket order book data received")
            else:
                self.logger.warning("⚠️ Lighter WebSocket order book not ready")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Lighter websocket: {e}")
            return

        await asyncio.sleep(5)

        iterations = 0
        while iterations < self.iterations and not self.stop_flag:
            iterations += 1
            self.logger.info("-----------------------------------------------")
            self.logger.info(f"🔄 Trading loop iteration {iterations} / {self.iterations}")
            self.logger.info("-----------------------------------------------")

            self.logger.info(f"[STEP 1] EdgeX position: {self.edgex_position} | Lighter position: {self.lighter_position}")

            if abs(self.edgex_position + self.lighter_position) > self.order_quantity*2:
                self.logger.error(f"❌ Position diff is too large: {self.edgex_position + self.lighter_position}")
                break

            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            try:
                # Determine side based on some logic (for now, alternate)
                side = 'buy'
                await self.place_edgex_post_only_order(side, self.order_quantity)
            except ValueError as e:
                if "DEADLINE_EXCEEDED" in str(e):
                    self.logger.warning(f"⏰ EdgeX order deadline exceeded, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue  # Retry the order instead of breaking
                else:
                    self.logger.error(f"⚠️ ValueError in trading loop: {e}")
                    self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                    break
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            # Wait for edgeX order to fill and then place Lighter order
            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if edgeX order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_limit_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break
                
                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error(f"❌ Timeout waiting for trade completion")
                    break

            if self.stop_flag:
                break

            # Sleep after step 1
            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds after STEP 1...")
                await asyncio.sleep(self.sleep_time)

            # Close position
            self.logger.info(f"[STEP 2] EdgeX position: {self.edgex_position} | Lighter position: {self.lighter_position}")
            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            try:
                # Determine side based on some logic (for now, alternate)
                side = 'sell'
                await self.place_edgex_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            # Wait for edgeX order to fill and then place Lighter order
            start_time2 = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if edgeX order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_limit_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break
                await asyncio.sleep(0.01)
                if time.time() - start_time2 > 180:
                    self.logger.error(f"❌ Timeout waiting for trade completion")
                    break

            # Sleep after step 2
            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds after STEP 2...")
                await asyncio.sleep(self.sleep_time)
            
            # Close remaining position
            self.logger.info(f"[STEP 3] EdgeX position: {self.edgex_position} | Lighter position: {self.lighter_position}")
            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            if self.edgex_position == 0:
                continue
            elif self.edgex_position > 0:
                side = 'sell'
            else:
                side = 'buy'

            try:
                # Determine side based on some logic (for now, alternate)
                await self.place_edgex_post_only_order(side, abs(self.edgex_position))
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time3 = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if edgeX order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_limit_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break
                await asyncio.sleep(0.01)
                if time.time() - start_time3 > 180:
                    self.logger.error(f"❌ Timeout waiting for trade completion")
                    break

    async def run(self):
        """Run the hedge bot."""
        self.setup_signal_handlers()
        
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Received interrupt signal...")
        finally:
            self.logger.info("🔄 Cleaning up...")
            self.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading bot for edgeX and Lighter')
    parser.add_argument('--ticker', type=str, default='BTC',
                        help='Ticker symbol (default: BTC)')
    parser.add_argument('--size', type=str, default='0.001',
                        help='Number of tokens to buy/sell per order (default: 0.001)')
    parser.add_argument('--iter', type=int, default=20,
                        help='Number of iterations to run (default: 20)')
    parser.add_argument('--fill-timeout', type=int, default=5,
                        help='Timeout in seconds for maker order fills (default: 5)')
    parser.add_argument('--sleep', type=int, default=0,
                        help='Sleep time in seconds after each step (default: 0)')

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Create and run the hedge bot
    bot = HedgeBot(
        ticker=args.ticker.upper(),
        order_quantity=Decimal(args.size),
        fill_timeout=args.fill_timeout,
        iterations=args.iter,
        sleep_time=args.sleep
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
