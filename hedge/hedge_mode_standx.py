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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.standx import StandXClient
import websockets
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Simple config class to wrap dictionary for StandX client."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on StandX and hedges with market orders on Lighter."""

    # StandX offset: 9 basis points (bps) from BBO
    STANDX_OFFSET_BPS = Decimal('0.0009')  # 9 bps = 0.09%

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, 
                 iterations: int = 20, sleep_time: int = 0, max_position: Decimal = Decimal('0')):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time
        if max_position == Decimal('0'):
            self.max_position = order_quantity
        else:
            self.max_position = max_position

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/standx_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/standx_{ticker}_hedge_mode_trades.csv"

        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv_file()

        # Setup logger
        self.logger = logging.getLogger(f"hedge_bot_standx_{ticker}")
        self.logger.setLevel(logging.INFO)
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

        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        # State management
        self.stop_flag = False
        self.order_counter = 0

        # StandX state
        self.standx_client = None
        self.standx_contract_id = None
        self.standx_tick_size = None
        self.standx_order_status = None
        self.standx_position = Decimal(0)

        # StandX order book state
        self.standx_best_bid = None
        self.standx_best_ask = None
        self.standx_order_book_ready = False

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
        self.lighter_order_filled = False

        # Lighter order management
        self.lighter_order_status = None
        self.lighter_order_price = None
        self.lighter_order_side = None
        self.lighter_order_size = None

        # Lighter position
        self.lighter_position = Decimal(0)

        # Strategy state
        self.order_execution_complete = False

        # Lighter API configuration
        self.lighter_base_url = "https://mainnet.zklighter.elliot.ai"
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX', '0'))

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        if self.standx_client:
            try:
                self.logger.info("🔌 StandX client will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting StandX client: {e}")

        if self.lighter_ws_task and not self.lighter_ws_task.done():
            try:
                self.lighter_ws_task.cancel()
                self.logger.info("🔌 Lighter WebSocket task cancelled")
            except Exception as e:
                self.logger.error(f"Error cancelling Lighter WebSocket task: {e}")

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
            writer.writerow([exchange, timestamp, side, price, quantity])

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

            self.log_trade_to_csv(
                exchange='Lighter',
                side=order_data['side'],
                price=str(order_data['avg_filled_price']),
                quantity=str(order_data['filled_base_amount'])
            )

            self.lighter_order_filled = True
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
            if isinstance(level, list) and len(level) >= 2:
                price = Decimal(level[0])
                size = Decimal(level[1])
            elif isinstance(level, dict):
                price = Decimal(level.get("price", 0))
                size = Decimal(level.get("size", 0))
            else:
                continue

            if size > 0:
                self.lighter_order_book[side][price] = size
            else:
                self.lighter_order_book[side].pop(price, None)

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

    async def handle_lighter_ws(self):
        """Handle Lighter WebSocket connection and messages."""
        url = "wss://mainnet.zklighter.elliot.ai/stream"

        while not self.stop_flag:
            timeout_count = 0
            try:
                await self.reset_lighter_order_book()

                async with websockets.connect(url) as ws:
                    # Subscribe to order book updates
                    await ws.send(json.dumps({"type": "subscribe", "channel": f"order_book/{self.lighter_market_index}"}))

                    # Subscribe to account orders updates
                    account_orders_channel = f"account_orders/{self.lighter_market_index}/{self.account_index}"

                    try:
                        ten_minutes_deadline = int(time.time() + 10 * 60)
                        auth_token, err = self.lighter_client.create_auth_token_with_expiry(ten_minutes_deadline)
                        if err is None:
                            auth_message = {
                                "type": "subscribe",
                                "channel": account_orders_channel,
                                "auth": auth_token
                            }
                            await ws.send(json.dumps(auth_message))
                            self.logger.info("✅ Subscribed to account orders with auth token")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error creating auth token: {e}")

                    while not self.stop_flag:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1)

                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError:
                                continue

                            timeout_count = 0

                            async with self.lighter_order_book_lock:
                                if data.get("type") == "subscribed/order_book":
                                    self.lighter_order_book["bids"].clear()
                                    self.lighter_order_book["asks"].clear()

                                    order_book = data.get("order_book", {})
                                    if order_book and "offset" in order_book:
                                        self.lighter_order_book_offset = order_book["offset"]

                                    self.update_lighter_order_book("bids", order_book.get("bids", []))
                                    self.update_lighter_order_book("asks", order_book.get("asks", []))
                                    self.lighter_snapshot_loaded = True
                                    self.lighter_order_book_ready = True

                                    self.logger.info(f"✅ Lighter order book snapshot loaded")

                                elif data.get("type") == "update/order_book" and self.lighter_snapshot_loaded:
                                    order_book = data.get("order_book", {})
                                    if not order_book or "offset" not in order_book:
                                        continue

                                    self.update_lighter_order_book("bids", order_book.get("bids", []))
                                    self.update_lighter_order_book("asks", order_book.get("asks", []))

                                    best_bid, best_ask = self.get_lighter_best_levels()
                                    if best_bid is not None:
                                        self.lighter_best_bid = best_bid[0]
                                    if best_ask is not None:
                                        self.lighter_best_ask = best_ask[0]

                                elif data.get("type") == "ping":
                                    await ws.send(json.dumps({"type": "pong"}))

                                elif data.get("type") == "update/account_orders":
                                    orders = data.get("orders", {}).get(str(self.lighter_market_index), [])
                                    for order in orders:
                                        if order.get("status") == "filled":
                                            self.handle_lighter_order_result(order)

                        except asyncio.TimeoutError:
                            timeout_count += 1
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            break
                        except Exception as e:
                            self.logger.error(f"⚠️ Error in Lighter websocket: {e}")
                            break

            except Exception as e:
                self.logger.error(f"⚠️ Failed to connect to Lighter websocket: {e}")

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

            err = self.lighter_client.check_client()
            if err is not None:
                raise Exception(f"CheckClient error: {err}")

            self.logger.info("✅ Lighter client initialized successfully")
        return self.lighter_client

    def initialize_standx_client(self):
        """Initialize the StandX client."""
        # Create config for StandX client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'close_order_side': 'sell'
        }

        config = Config(config_dict)
        self.standx_client = StandXClient(config)
        self.logger.info("✅ StandX client initialized successfully")
        return self.standx_client

    def get_lighter_market_config(self) -> Tuple[int, int, int, Decimal]:
        """Get Lighter market configuration."""
        url = f"{self.lighter_base_url}/api/v1/orderBooks"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            for market in data["order_books"]:
                if market["symbol"] == self.ticker:
                    price_multiplier = pow(10, market["supported_price_decimals"])
                    return (market["market_id"],
                            pow(10, market["supported_size_decimals"]),
                            price_multiplier,
                            Decimal("1") / (Decimal("10") ** market["supported_price_decimals"]))
            raise Exception(f"Ticker {self.ticker} not found")

        except Exception as e:
            self.logger.error(f"⚠️ Error getting market config: {e}")
            raise

    async def get_standx_contract_info(self) -> Tuple[str, Decimal]:
        """Get StandX contract ID and tick size."""
        if not self.standx_client:
            raise Exception("StandX client not initialized")

        contract_id, tick_size = await self.standx_client.get_contract_attributes()
        return contract_id, tick_size

    async def get_standx_position(self) -> Decimal:
        """Get StandX position."""
        if not self.standx_client:
            raise Exception("StandX client not initialized")

        return await self.standx_client.get_account_positions()

    async def fetch_standx_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from StandX."""
        if not self.standx_client:
            raise Exception("StandX client not initialized")

        best_bid, best_ask = await self.standx_client.fetch_bbo_prices(self.standx_contract_id)
        return best_bid, best_ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        if self.standx_tick_size is None:
            return price
        return (price / self.standx_tick_size).quantize(Decimal('1')) * self.standx_tick_size

    async def place_standx_post_only_order(self, side: str, quantity: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Place a post-only order on StandX with 9 USD offset from BBO.
        Uses query_open_orders to track order status since new_order only returns request_id.
        """
        if not self.standx_client:
            raise Exception("StandX client not initialized")

        self.standx_order_status = None
        self.logger.info(f"[StandX] [{side}] Placing POST-ONLY order with 9 bps offset")

        # Get BBO prices
        best_bid, best_ask = await self.fetch_standx_bbo_prices()

        # Calculate price with 9 bps offset
        if side.lower() == 'buy':
            order_price = best_ask * (Decimal('1') - self.STANDX_OFFSET_BPS)
        else:
            order_price = best_bid * (Decimal('1') + self.STANDX_OFFSET_BPS)

        order_price = self.round_to_tick(order_price)

        self.logger.info(f"[StandX] [{side}] BBO: {best_bid}/{best_ask}, Order price: {order_price}")

        # Get open orders before placing new order
        open_orders_before = await self.standx_client.get_active_orders(self.standx_contract_id)
        order_ids_before = {o.order_id for o in open_orders_before}

        # Place the order
        payload = {
            "symbol": self.standx_contract_id,
            "side": side.lower(),
            "order_type": "limit",
            "qty": str(quantity),
            "price": str(order_price),
            "time_in_force": "alo",
            "reduce_only": False
        }

        result = await self.standx_client._make_request("POST", "/api/new_order",
                                                         data=payload, signed=True)

        if result.get('code', -1) != 0:
            raise Exception(f"Order failed: {result.get('message', 'Unknown error')}")

        self.logger.info(f"[StandX] [{side}] Order submitted, waiting for confirmation...")

        # Wait a moment for order to be processed
        await asyncio.sleep(1)

        # Find the new order by comparing open orders
        our_order = None
        for _ in range(10):  # Try up to 10 times
            open_orders_after = await self.standx_client.get_active_orders(self.standx_contract_id)
            for order in open_orders_after:
                if order.order_id not in order_ids_before and order.side == side.lower():
                    our_order = order
                    break
            if our_order:
                break
            await asyncio.sleep(0.5)

        if our_order is None:
            # Order might have been filled immediately or rejected
            self.logger.info(f"[StandX] [{side}] Order not found in open orders - may have filled or rejected")
            # Check position to determine if filled
            new_position = await self.standx_client.get_account_positions()
            if side.lower() == 'buy' and new_position > self.standx_position:
                filled_size = new_position - self.standx_position
                return filled_size, order_price
            elif side.lower() == 'sell' and new_position < self.standx_position:
                filled_size = self.standx_position - new_position
                return filled_size, order_price
            return Decimal(0), Decimal(0)

        order_id = our_order.order_id
        self.logger.info(f"[StandX] [{side}] Order placed: {order_id} @ {order_price}")

        # Poll for order completion (disappears from open orders = filled or canceled)
        start_time = time.time()
        while not self.stop_flag:
            open_orders = await self.standx_client.get_active_orders(self.standx_contract_id)
            order_still_open = any(o.order_id == order_id for o in open_orders)

            if not order_still_open:
                # Order completed - check position to determine if filled
                new_position = await self.standx_client.get_account_positions()
                if side.lower() == 'buy':
                    filled_size = new_position - self.standx_position
                else:
                    filled_size = self.standx_position - new_position

                if filled_size > 0:
                    self.standx_order_status = 'FILLED'
                    return abs(filled_size), order_price
                else:
                    self.standx_order_status = 'CANCELED'
                    return Decimal(0), Decimal(0)

            elif time.time() - start_time > self.fill_timeout:
                # Check if price has moved
                new_best_bid, new_best_ask = await self.fetch_standx_bbo_prices()
                if side.lower() == 'buy':
                    current_price = new_best_ask * (Decimal('1') - self.STANDX_OFFSET_BPS)
                else:
                    current_price = new_best_bid * (Decimal('1') + self.STANDX_OFFSET_BPS)

                current_price = self.round_to_tick(current_price)

                if order_price != current_price:
                    self.logger.info(f"[StandX] [{side}] [CANCELING] Price changed, canceling order")
                    try:
                        await self.standx_client.cancel_order(order_id)
                        await asyncio.sleep(0.5)
                        # Check if any fill happened
                        new_position = await self.standx_client.get_account_positions()
                        if side.lower() == 'buy':
                            filled_size = new_position - self.standx_position
                        else:
                            filled_size = self.standx_position - new_position
                        return abs(filled_size), order_price
                    except Exception as e:
                        self.logger.error(f"❌ Error canceling StandX order: {e}")
                        return Decimal(0), Decimal(0)
                else:
                    # Price hasn't changed, keep waiting
                    start_time = time.time()  # Reset timeout

            await asyncio.sleep(0.5)

        return Decimal(0), Decimal(0)

    async def place_lighter_market_order(self, lighter_side: str, quantity: Decimal):
        """Place a market order on Lighter for hedging."""
        if not self.lighter_client:
            self.initialize_lighter_client()

        best_bid, best_ask = self.get_lighter_best_levels()

        if lighter_side.lower() == 'buy':
            order_type = "CLOSE"
            is_ask = False
            price = best_ask[0] * Decimal('1.002')
        else:
            order_type = "OPEN"
            is_ask = True
            price = best_bid[0] * Decimal('0.998')

        self.lighter_order_filled = False
        self.lighter_order_price = price
        self.lighter_order_side = lighter_side
        self.lighter_order_size = quantity

        try:
            client_order_index = int(time.time() * 1000)
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

            tx_hash = await self.lighter_client.send_tx(
                tx_type=self.lighter_client.TX_TYPE_CREATE_ORDER,
                tx_info=tx_info
            )

            self.logger.info(f"[Lighter] [{order_type}]: {quantity}")

            await self.monitor_lighter_order(client_order_index)

            return tx_hash
        except Exception as e:
            self.logger.error(f"❌ Error placing Lighter order: {e}")
            return None

    async def monitor_lighter_order(self, client_order_index: int):
        """Monitor Lighter order for fill."""
        start_time = time.time()
        while not self.lighter_order_filled and not self.stop_flag:
            if time.time() - start_time > 30:
                self.logger.warning("⚠️ Timeout waiting for Lighter order fill")
                self.lighter_order_filled = True
                self.order_execution_complete = True
                break

            await asyncio.sleep(0.1)

    def get_lighter_position(self) -> Decimal:
        """Get Lighter position via REST API."""
        url = "https://mainnet.zklighter.elliot.ai/api/v1/account"
        headers = {"accept": "application/json"}
        parameters = {"by": "index", "value": self.account_index}

        try:
            response = requests.get(url, headers=headers, params=parameters, timeout=10)
            response.raise_for_status()

            data = response.json()

            positions = data['accounts'][0].get('positions', [])
            for position in positions:
                if position.get('symbol') == self.ticker:
                    return Decimal(position['position']) * position['sign']

            return Decimal(0)

        except Exception as e:
            self.logger.error(f"⚠️ Error getting Lighter position: {e}")
            return self.lighter_position

    async def trading_loop(self):
        """Main trading loop implementing the hedge strategy."""
        self.logger.info(f"🚀 Starting StandX hedge bot for {self.ticker}")

        # Initialize clients
        try:
            self.initialize_lighter_client()
            self.initialize_standx_client()

            # Get contract info
            self.standx_contract_id, self.standx_tick_size = await self.get_standx_contract_info()
            self.lighter_market_index, self.base_amount_multiplier, self.price_multiplier, self.tick_size = self.get_lighter_market_config()

            self.logger.info(f"Contract info loaded - StandX: {self.standx_contract_id}, "
                             f"Lighter: {self.lighter_market_index}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return

        # Connect to StandX
        try:
            await self.standx_client.connect()
            self.logger.info("✅ StandX client connected")

            # Fetch initial BBO prices
            best_bid, best_ask = await self.fetch_standx_bbo_prices()
            if best_bid > 0 and best_ask > 0:
                self.standx_best_bid = best_bid
                self.standx_best_ask = best_ask
                self.standx_order_book_ready = True
                self.logger.info(f"✅ StandX order book ready - Best bid: {best_bid}, Best ask: {best_ask}")

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to StandX: {e}")
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return

        # Setup Lighter websocket
        try:
            self.lighter_ws_task = asyncio.create_task(self.handle_lighter_ws())
            self.logger.info("✅ Lighter WebSocket task started")

            # Wait for initial Lighter order book data
            timeout = 10
            start_time = time.time()
            while not self.lighter_order_book_ready and not self.stop_flag:
                if time.time() - start_time > timeout:
                    self.logger.warning(f"⚠️ Timeout waiting for Lighter WebSocket")
                    break
                await asyncio.sleep(0.5)

        except Exception as e:
            self.logger.error(f"❌ Failed to setup Lighter websocket: {e}")
            return

        await asyncio.sleep(5)

        iterations = 0
        self.lighter_position = self.get_lighter_position()
        self.standx_position = await self.get_standx_position()

        while iterations < self.iterations and not self.stop_flag:
            iterations += 1
            self.logger.info("-----------------------------------------------")
            self.logger.info(f"🔄 Trading loop iteration {iterations}")
            self.logger.info("-----------------------------------------------")

            # BUY phase: Build long position on StandX, hedge short on Lighter
            while self.standx_position < self.max_position and not self.stop_flag:
                self.lighter_position = self.get_lighter_position()
                self.standx_position = await self.get_standx_position()
                self.logger.info(f"Buying up to {self.max_position} | StandX pos: {self.standx_position} | Lighter pos: {self.lighter_position}")

                if abs(self.standx_position + self.lighter_position) > self.order_quantity * 2:
                    self.logger.error(f"❌ Position diff too large: {self.standx_position + self.lighter_position}")
                    sys.exit(1)

                self.order_execution_complete = False

                try:
                    filled_size, filled_price = await self.place_standx_post_only_order('buy', self.order_quantity)
                    if filled_size == 0:
                        self.logger.info(f"[StandX] [buy] [CANCELLED]")
                        continue
                except Exception as e:
                    self.logger.error(f"⚠️ Error placing StandX post only order: {e}")
                    self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                    continue

                self.logger.info(f"[StandX] [buy] [FILLED]: {filled_size} @ {filled_price}")
                self.log_trade_to_csv(
                    exchange='StandX',
                    side='buy',
                    price=str(filled_price),
                    quantity=str(filled_size)
                )

                self.standx_position += filled_size

                # Hedge on Lighter
                await self.place_lighter_market_order('sell', abs(filled_size))

            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds ...")
                await asyncio.sleep(self.sleep_time)

            # SELL phase: Build short position on StandX, hedge long on Lighter
            exit_after_next_trade = False
            while self.standx_position > -1 * self.max_position and not self.stop_flag:
                self.lighter_position = self.get_lighter_position()
                self.standx_position = await self.get_standx_position()
                self.logger.info(f"Selling up to -{self.max_position} | StandX pos: {self.standx_position} | Lighter pos: {self.lighter_position}")

                if abs(self.standx_position + self.lighter_position) > self.order_quantity * 2:
                    self.logger.error(f"❌ Position diff too large: {self.standx_position + self.lighter_position}")
                    sys.exit(1)

                if iterations == self.iterations:
                    if self.standx_position > 0 and self.standx_position <= self.order_quantity:
                        exit_after_next_trade = True

                try:
                    if exit_after_next_trade:
                        filled_size, filled_price = await self.place_standx_post_only_order('sell', abs(self.standx_position))
                    else:
                        filled_size, filled_price = await self.place_standx_post_only_order('sell', self.order_quantity)
                    if filled_size == 0:
                        self.logger.info(f"[StandX] [sell] [CANCELLED]")
                        continue
                except Exception as e:
                    self.logger.error(f"⚠️ Error placing StandX post only order: {e}")
                    self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                    continue

                self.logger.info(f"[StandX] [sell] [FILLED]: {filled_size} @ {filled_price}")
                self.log_trade_to_csv(
                    exchange='StandX',
                    side='sell',
                    price=str(filled_price),
                    quantity=str(filled_size)
                )

                self.standx_position -= abs(filled_size)

                # Hedge on Lighter
                await self.place_lighter_market_order('buy', abs(filled_size))

                if exit_after_next_trade:
                    self.logger.info("Position back to zero. Done! Exiting...")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StandX Hedge Mode Bot')
    parser.add_argument('--ticker', type=str, required=True, help='Trading pair (e.g. BTC)')
    parser.add_argument('--quantity', type=str, required=True, help='Order quantity')
    parser.add_argument('--fill-timeout', type=int, default=5, help='Fill timeout in seconds')
    parser.add_argument('--iterations', type=int, default=20, help='Number of trading iterations')
    parser.add_argument('--sleep-time', type=int, default=0, help='Sleep time between iterations')
    parser.add_argument('--max-position', type=str, default='0', help='Maximum position size')

    args = parser.parse_args()

    bot = HedgeBot(
        ticker=args.ticker,
        order_quantity=Decimal(args.quantity),
        fill_timeout=args.fill_timeout,
        iterations=args.iterations,
        sleep_time=args.sleep_time,
        max_position=Decimal(args.max_position)
    )

    asyncio.run(bot.run())
