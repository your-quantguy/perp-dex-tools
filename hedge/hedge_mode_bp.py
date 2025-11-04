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
from typing import Tuple

from lighter.signer_client import SignerClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.backpack import BackpackClient
import websockets
from datetime import datetime
import pytz

class Config:
    """Simple config class to wrap dictionary for Backpack client."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on Backpack and hedges with market orders on Lighter."""

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, iterations: int = 20, sleep_time: int = 0):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.lighter_order_filled = False
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.backpack_position = Decimal('0')
        self.lighter_position = Decimal('0')
        self.current_order = {}

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/backpack_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/backpack_{ticker}_hedge_mode_trades.csv"
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

        # Backpack state
        self.backpack_client = None
        self.backpack_contract_id = None
        self.backpack_tick_size = None
        self.backpack_order_status = None

        # Backpack order book state for websocket-based BBO
        self.backpack_order_book = {'bids': {}, 'asks': {}}
        self.backpack_best_bid = None
        self.backpack_best_ask = None
        self.backpack_order_book_ready = False

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
        self.lighter_order_result = None

        # Lighter order management
        self.lighter_order_status = None
        self.lighter_order_price = None
        self.lighter_order_side = None
        self.lighter_order_size = None
        self.lighter_order_start_time = None

        # Strategy state
        self.waiting_for_lighter_fill = False
        self.wait_start_time = None

        # Order execution tracking
        self.order_execution_complete = False

        # Current order details for immediate execution
        self.current_lighter_side = None
        self.current_lighter_quantity = None
        self.current_lighter_price = None
        self.lighter_order_info = None

        # Lighter API configuration
        self.lighter_base_url = "https://mainnet.zklighter.elliot.ai"
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX'))

        # Backpack configuration
        self.backpack_public_key = os.getenv('BACKPACK_PUBLIC_KEY')
        self.backpack_secret_key = os.getenv('BACKPACK_SECRET_KEY')

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Close WebSocket connections
        if self.backpack_client:
            try:
                # Note: disconnect() is async, but shutdown() is sync
                # We'll let the cleanup happen naturally
                self.logger.info("🔌 Backpack WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting Backpack WebSocket: {e}")

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

    def handle_lighter_order_result(self, order_data):
        """Handle Lighter order result from WebSocket."""
        try:
            order_data["avg_filled_price"] = (Decimal(order_data["filled_quote_amount"]) /
                                              Decimal(order_data["filled_base_amount"]))
            if order_data["is_ask"]:
                order_data["side"] = "SHORT"
                order_type = "OPEN"
                self.lighter_position -= Decimal(order_data["filled_base_amount"])
            else:
                order_data["side"] = "LONG"
                order_type = "CLOSE"
                self.lighter_position += Decimal(order_data["filled_base_amount"])
            
            client_order_index = order_data["client_order_id"]

            self.logger.info(f"[{client_order_index}] [{order_type}] [Lighter] [FILLED]: "
                             f"{order_data['filled_base_amount']} @ {order_data['avg_filled_price']}")

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
            self.logger.warning(
                f"⚠️ Out-of-order update: new_offset={new_offset}, current_offset={self.lighter_order_book_offset}")
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

    def get_lighter_mid_price(self) -> Decimal:
        """Get mid price from Lighter order book."""
        best_bid, best_ask = self.get_lighter_best_levels()

        if best_bid is None or best_ask is None:
            raise Exception("Cannot calculate mid price - missing order book data")

        mid_price = (best_bid[0] + best_ask[0]) / Decimal('2')
        return mid_price

    def get_lighter_order_price(self, is_ask: bool) -> Decimal:
        """Get order price from Lighter order book."""
        best_bid, best_ask = self.get_lighter_best_levels()

        if best_bid is None or best_ask is None:
            raise Exception("Cannot calculate order price - missing order book data")

        if is_ask:
            order_price = best_bid[0] + self.tick_size
        else:
            order_price = best_ask[0] - self.tick_size

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
                                    for order in orders:
                                        if order.get("status") == "filled":
                                            self.handle_lighter_order_result(order)
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

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def initialize_lighter_client(self):
        """Initialize the Lighter client."""
        if self.lighter_client is None:
            api_key_private_key = os.getenv('API_KEY_PRIVATE_KEY')
            if not api_key_private_key:
                raise Exception("API_KEY_PRIVATE_KEY environment variable not set")

            self.lighter_client = SignerClient(
                url=self.lighter_base_url,
                private_key=api_key_private_key,
                account_index=self.account_index,
                api_key_index=self.api_key_index,
            )

            # Check client
            err = self.lighter_client.check_client()
            if err is not None:
                raise Exception(f"CheckClient error: {err}")

            self.logger.info("✅ Lighter client initialized successfully")
        return self.lighter_client

    def initialize_backpack_client(self):
        """Initialize the Backpack client."""
        if not self.backpack_public_key or not self.backpack_secret_key:
            raise ValueError("BACKPACK_PUBLIC_KEY and BACKPACK_SECRET_KEY must be set in environment variables")

        # Create config for Backpack client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',  # Will be set when we get contract info
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),  # Will be updated when we get contract info
            'close_order_side': 'sell'  # Default, will be updated based on strategy
        }

        # Wrap in Config class for Backpack client
        config = Config(config_dict)

        # Initialize Backpack client
        self.backpack_client = BackpackClient(config)

        self.logger.info("✅ Backpack client initialized successfully")
        return self.backpack_client

    def get_lighter_market_config(self) -> Tuple[int, int, int, Decimal]:
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

    async def get_backpack_contract_info(self) -> Tuple[str, Decimal]:
        """Get Backpack contract ID and tick size."""
        if not self.backpack_client:
            raise Exception("Backpack client not initialized")

        contract_id, tick_size = await self.backpack_client.get_contract_attributes()

        if self.order_quantity < self.backpack_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.backpack_client.config.quantity}")

        return contract_id, tick_size

    async def fetch_backpack_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from Backpack using websocket data."""
        # Use WebSocket data if available
        if self.backpack_order_book_ready and self.backpack_best_bid and self.backpack_best_ask:
            if self.backpack_best_bid > 0 and self.backpack_best_ask > 0 and self.backpack_best_bid < self.backpack_best_ask:
                return self.backpack_best_bid, self.backpack_best_ask

        # Fallback to REST API if websocket data is not available
        self.logger.warning("WebSocket BBO data not available, falling back to REST API")
        if not self.backpack_client:
            raise Exception("Backpack client not initialized")

        best_bid, best_ask = await self.backpack_client.fetch_bbo_prices(self.backpack_contract_id)

        return best_bid, best_ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        if self.backpack_tick_size is None:
            return price
        return (price / self.backpack_tick_size).quantize(Decimal('1')) * self.backpack_tick_size

    async def place_bbo_order(self, side: str, quantity: Decimal):
        # Get best bid/ask prices
        best_bid, best_ask = await self.fetch_backpack_bbo_prices()

        # Place the order using Backpack client
        order_result = await self.backpack_client.place_open_order(
            contract_id=self.backpack_contract_id,
            quantity=quantity,
            direction=side.lower()
        )

        if order_result.success:
            return order_result.order_id
        else:
            raise Exception(f"Failed to place order: {order_result.error_message}")

    async def place_backpack_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on Backpack."""
        if not self.backpack_client:
            raise Exception("Backpack client not initialized")

        self.backpack_order_status = None
        self.logger.info(f"[OPEN] [Backpack] [{side}] Placing Backpack POST-ONLY order")
        order_id = await self.place_bbo_order(side, quantity)

        start_time = time.time()
        while not self.stop_flag:
            if self.backpack_order_status == 'CANCELED':
                self.backpack_order_status = 'NEW'
                order_id = await self.place_bbo_order(side, quantity)
                start_time = time.time()
                await asyncio.sleep(0.5)
            elif self.backpack_order_status in ['NEW', 'OPEN', 'PENDING', 'CANCELING', 'PARTIALLY_FILLED']:
                await asyncio.sleep(0.5)
                if time.time() - start_time > 10:
                    try:
                        # Cancel the order using Backpack client
                        cancel_result = await self.backpack_client.cancel_order(order_id)
                        if not cancel_result.success:
                            self.logger.error(f"❌ Error canceling Backpack order: {cancel_result.error_message}")
                    except Exception as e:
                        self.logger.error(f"❌ Error canceling Backpack order: {e}")
            elif self.backpack_order_status == 'FILLED':
                break
            else:
                if self.backpack_order_status is not None:
                    self.logger.error(f"❌ Unknown Backpack order status: {self.backpack_order_status}")
                    break
                else:
                    await asyncio.sleep(0.5)

    def handle_backpack_order_book_update(self, message):
        """Handle Backpack order book updates from WebSocket."""
        try:
            if isinstance(message, str):
                message = json.loads(message)

            self.logger.debug(f"Received Backpack depth message: {message}")

            # Check if this is a depth update message
            if message.get("stream") and "depth" in message.get("stream", ""):
                data = message.get("data", {})

                if data:
                    # Update bids - format is [["price", "size"], ...]
                    # Backpack API uses 'b' for bids
                    bids = data.get('b', [])
                    for bid in bids:
                        price = Decimal(bid[0])
                        size = Decimal(bid[1])
                        if size > 0:
                            self.backpack_order_book['bids'][price] = size
                        else:
                            # Remove zero size orders
                            self.backpack_order_book['bids'].pop(price, None)

                    # Update asks - format is [["price", "size"], ...]
                    # Backpack API uses 'a' for asks
                    asks = data.get('a', [])
                    for ask in asks:
                        price = Decimal(ask[0])
                        size = Decimal(ask[1])
                        if size > 0:
                            self.backpack_order_book['asks'][price] = size
                        else:
                            # Remove zero size orders
                            self.backpack_order_book['asks'].pop(price, None)

                    # Update best bid and ask
                    if self.backpack_order_book['bids']:
                        self.backpack_best_bid = max(self.backpack_order_book['bids'].keys())
                    if self.backpack_order_book['asks']:
                        self.backpack_best_ask = min(self.backpack_order_book['asks'].keys())

                    if not self.backpack_order_book_ready:
                        self.backpack_order_book_ready = True
                        self.logger.info(f"📊 Backpack order book ready - Best bid: {self.backpack_best_bid}, "
                                         f"Best ask: {self.backpack_best_ask}")
                    else:
                        self.logger.debug(f"📊 Order book updated - Best bid: {self.backpack_best_bid}, "
                                          f"Best ask: {self.backpack_best_ask}")

        except Exception as e:
            self.logger.error(f"Error handling Backpack order book update: {e}")
            self.logger.error(f"Message content: {message}")

    def handle_backpack_order_update(self, order_data):
        """Handle Backpack order updates from WebSocket."""
        side = order_data.get('side', '').lower()
        filled_size = Decimal(order_data.get('filled_size', '0'))
        price = Decimal(order_data.get('price', '0'))

        if side == 'buy':
            lighter_side = 'sell'
        else:
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


    async def place_lighter_market_order(self, lighter_side: str, quantity: Decimal, price: Decimal):
        if not self.lighter_client:
            await self.initialize_lighter_client()

        best_bid, best_ask = self.get_lighter_best_levels()

        # Determine order parameters
        if lighter_side.lower() == 'buy':
            order_type = "CLOSE"
            is_ask = False
            price = best_ask[0] * Decimal('1.002')
        else:
            order_type = "OPEN"
            is_ask = True
            price = best_bid[0] * Decimal('0.998')


        # Reset order state
        self.lighter_order_filled = False
        self.lighter_order_price = price
        self.lighter_order_side = lighter_side
        self.lighter_order_size = quantity

        try:
            client_order_index = int(time.time() * 1000)
            # Sign the order transaction
            tx_info, error = self.lighter_client.sign_create_order(
                market_index=self.lighter_market_index,
                client_order_index=client_order_index,
                base_amount=int(quantity * self.base_amount_multiplier),
                price=int(price * self.price_multiplier),
                is_ask=is_ask,
                order_type=self.lighter_client.ORDER_TYPE_LIMIT,
                time_in_force=self.lighter_client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                reduce_only=False,
                trigger_price=0,
            )
            if error is not None:
                raise Exception(f"Sign error: {error}")

            # Prepare the form data
            tx_hash = await self.lighter_client.send_tx(
                tx_type=self.lighter_client.TX_TYPE_CREATE_ORDER,
                tx_info=tx_info
            )

            self.logger.info(f"[{client_order_index}] [{order_type}] [Lighter] [OPEN]: {quantity}")

            await self.monitor_lighter_order(client_order_index)

            return tx_hash
        except Exception as e:
            self.logger.error(f"❌ Error placing Lighter order: {e}")
            return None

    async def monitor_lighter_order(self, client_order_index: int):
        """Monitor Lighter order and adjust price if needed."""

        start_time = time.time()
        while not self.lighter_order_filled and not self.stop_flag:
            # Check for timeout (30 seconds total)
            if time.time() - start_time > 30:
                self.logger.error(f"❌ Timeout waiting for Lighter order fill after {time.time() - start_time:.1f}s")
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

            self.logger.info(f"🔧 Attempting to modify order - Market: {self.lighter_market_index}, "
                             f"Client Order Index: {client_order_index}, New Price: {lighter_price}")

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
            self.logger.info(f"🔄 Lighter order modified successfully: {self.lighter_order_side} "
                             f"{self.lighter_order_size} @ {new_price}")

        except Exception as e:
            self.logger.error(f"❌ Error modifying Lighter order: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")

    async def setup_backpack_websocket(self):
        """Setup Backpack websocket for order updates and order book data."""
        if not self.backpack_client:
            raise Exception("Backpack client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from Backpack WebSocket."""
            if order_data.get('contract_id') != self.backpack_contract_id:
                return
            try:
                order_id = order_data.get('order_id')
                status = order_data.get('status')
                side = order_data.get('side', '').lower()
                filled_size = Decimal(order_data.get('filled_size', '0'))
                size = Decimal(order_data.get('size', '0'))
                price = order_data.get('price', '0')

                if side == 'buy':
                    order_type = "OPEN"
                else:
                    order_type = "CLOSE"
                
                if status == 'CANCELED' and filled_size > 0:
                    status = 'FILLED'

                # Handle the order update
                if status == 'FILLED' and self.backpack_order_status != 'FILLED':
                    if side == 'buy':
                        self.backpack_position += filled_size
                    else:
                        self.backpack_position -= filled_size
                    self.logger.info(f"[{order_id}] [{order_type}] [Backpack] [{status}]: {filled_size} @ {price}")
                    self.backpack_order_status = status

                    # Log Backpack trade to CSV
                    self.log_trade_to_csv(
                        exchange='Backpack',
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

                    self.handle_backpack_order_update({
                        'order_id': order_id,
                        'side': side,
                        'status': status,
                        'size': size,
                        'price': price,
                        'contract_id': self.backpack_contract_id,
                        'filled_size': filled_size
                    })
                elif self.backpack_order_status != 'FILLED':
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [{order_type}] [Backpack] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [Backpack] [{status}]: {filled_size} @ {price}")
                    self.backpack_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling Backpack order update: {e}")

        try:
            # Setup order update handler
            self.backpack_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ Backpack WebSocket order update handler set up")

            # Connect to Backpack WebSocket
            await self.backpack_client.connect()
            self.logger.info("✅ Backpack WebSocket connection established")

            # Setup separate WebSocket connection for depth updates
            await self.setup_backpack_depth_websocket()

        except Exception as e:
            self.logger.error(f"Could not setup Backpack WebSocket handlers: {e}")

    async def setup_backpack_depth_websocket(self):
        """Setup separate WebSocket connection for Backpack depth updates."""
        try:
            import websockets

            async def handle_depth_websocket():
                """Handle depth WebSocket connection."""
                url = "wss://ws.backpack.exchange"

                while not self.stop_flag:
                    try:
                        async with websockets.connect(url) as ws:
                            # Subscribe to depth updates
                            subscribe_message = {
                                "method": "SUBSCRIBE",
                                "params": [f"depth.{self.backpack_contract_id}"]
                            }
                            await ws.send(json.dumps(subscribe_message))
                            self.logger.info(f"✅ Subscribed to depth updates for {self.backpack_contract_id}")

                            # Listen for messages
                            async for message in ws:
                                if self.stop_flag:
                                    break

                                try:
                                    # Handle ping frames
                                    if isinstance(message, bytes) and message == b'\x09':
                                        await ws.pong()
                                        continue

                                    data = json.loads(message)

                                    # Handle depth updates
                                    if data.get('stream') and 'depth' in data.get('stream', ''):
                                        self.handle_backpack_order_book_update(data)

                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Failed to parse depth WebSocket message: {e}")
                                except Exception as e:
                                    self.logger.error(f"Error handling depth WebSocket message: {e}")

                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning("Depth WebSocket connection closed, reconnecting...")
                    except Exception as e:
                        self.logger.error(f"Depth WebSocket error: {e}")

                    # Wait before reconnecting
                    if not self.stop_flag:
                        await asyncio.sleep(2)

            # Start depth WebSocket in background
            asyncio.create_task(handle_depth_websocket())
            self.logger.info("✅ Backpack depth WebSocket task started")

        except Exception as e:
            self.logger.error(f"Could not setup Backpack depth WebSocket: {e}")

    async def trading_loop(self):
        """Main trading loop implementing the new strategy."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")

        # Initialize clients
        try:
            self.initialize_lighter_client()
            self.initialize_backpack_client()

            # Get contract info
            self.backpack_contract_id, self.backpack_tick_size = await self.get_backpack_contract_info()
            self.lighter_market_index, self.base_amount_multiplier, self.price_multiplier, self.tick_size = self.get_lighter_market_config()

            self.logger.info(f"Contract info loaded - Backpack: {self.backpack_contract_id}, "
                             f"Lighter: {self.lighter_market_index}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return

        # Setup Backpack websocket
        try:
            await self.setup_backpack_websocket()
            self.logger.info("✅ Backpack WebSocket connection established")

            # Wait for initial order book data with timeout
            self.logger.info("⏳ Waiting for initial order book data...")
            timeout = 10  # seconds
            start_time = time.time()
            while not self.backpack_order_book_ready and not self.stop_flag:
                if time.time() - start_time > timeout:
                    self.logger.warning(f"⚠️ Timeout waiting for WebSocket order book data after {timeout}s")
                    break
                await asyncio.sleep(0.5)

            if self.backpack_order_book_ready:
                self.logger.info("✅ WebSocket order book data received")
            else:
                self.logger.warning("⚠️ WebSocket order book not ready, will use REST API fallback")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup Backpack websocket: {e}")
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
            self.logger.info(f"🔄 Trading loop iteration {iterations}")
            self.logger.info("-----------------------------------------------")

            self.logger.info(f"[STEP 1] Backpack position: {self.backpack_position} | Lighter position: {self.lighter_position}")

            if abs(self.backpack_position + self.lighter_position) > self.order_quantity*2:
                self.logger.error(f"❌ Position diff is too large: {self.backpack_position + self.lighter_position}")
                break

            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            try:
                # Determine side based on some logic (for now, alternate)
                side = 'buy'
                await self.place_backpack_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if Backpack order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_market_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

            if self.stop_flag:
                break

            # Sleep after step 1
            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds after STEP 1...")
                await asyncio.sleep(self.sleep_time)

            # Close position
            self.logger.info(f"[STEP 2] Backpack position: {self.backpack_position} | Lighter position: {self.lighter_position}")
            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            try:
                # Determine side based on some logic (for now, alternate)
                side = 'sell'
                await self.place_backpack_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            while not self.order_execution_complete and not self.stop_flag:
                # Check if Backpack order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_market_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

            # Sleep after step 2
            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds after STEP 2...")
                await asyncio.sleep(self.sleep_time)

            # Close remaining position
            self.logger.info(f"[STEP 3] Backpack position: {self.backpack_position} | Lighter position: {self.lighter_position}")
            self.order_execution_complete = False
            self.waiting_for_lighter_fill = False
            if self.backpack_position == 0:
                continue
            elif self.backpack_position > 0:
                side = 'sell'
            else:
                side = 'buy'

            try:
                # Determine side based on some logic (for now, alternate)
                await self.place_backpack_post_only_order(side, abs(self.backpack_position))
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            # Wait for order to be filled via WebSocket
            while not self.order_execution_complete and not self.stop_flag:
                # Check if Backpack order filled and we need to place Lighter order
                if self.waiting_for_lighter_fill:
                    await self.place_lighter_market_order(
                        self.current_lighter_side,
                        self.current_lighter_quantity,
                        self.current_lighter_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
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
    parser = argparse.ArgumentParser(description='Trading bot for Backpack and Lighter')
    parser.add_argument('--exchange', type=str,
                        help='Exchange')
    parser.add_argument('--ticker', type=str, default='BTC',
                        help='Ticker symbol (default: BTC)')
    parser.add_argument('--size', type=str,
                        help='Number of tokens to buy/sell per order')
    parser.add_argument('--iter', type=int,
                        help='Number of iterations to run')
    parser.add_argument('--fill-timeout', type=int, default=5,
                        help='Timeout in seconds for maker order fills (default: 5)')
    parser.add_argument('--sleep', type=int, default=0,
                        help='Sleep time in seconds after each step (default: 0)')

    return parser.parse_args()
