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
import statistics
from decimal import Decimal
from typing import Tuple
from collections import deque

from lighter.signer_client import SignerClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.grvt import GrvtClient
import websockets
from datetime import datetime
import pytz

class Config:
    """Simple config class to wrap dictionary for GRVT client."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on GRVT and hedges with market orders on Lighter."""

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, max_position: Decimal = Decimal('0')):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.lighter_order_filled = False
        self.current_order = {}
        self.max_position = max_position
        self.spread_history = deque(maxlen=2000)

        self.exp_grvt_price = 0
        self.exp_lighter_price = 0

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/grvt_{ticker}_hedge_mode_trades.csv"
        self.bbo_csv_filename = f"logs/grvt_{ticker}_bbo_data.csv"
        self.thresholds_json_filename = f"logs/grvt_{ticker}_thresholds.json"
        self.original_stdout = sys.stdout

        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv_file()
        self._initialize_bbo_csv_file()

        # Setup logger
        self.logger = logging.getLogger(f"hedge_bot_{ticker}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Disable verbose logging from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)
        
        # Completely disable all pysdk logging (set to level higher than CRITICAL)
        pysdk_loggers = [
            'pysdk',
            'pysdk.grvt_ccxt_logging_selector',
            'pysdk.grvt_ccxt_base',
            'pysdk.grvt_ccxt_pro',
            'pysdk.grvt_ccxt',
            'pysdk.grvt_ccxt_ws'
        ]
        for logger_name in pysdk_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL to silence everything
            logger.propagate = False
            logger.handlers = []  # Remove all handlers
        
        # Disable aiohttp and asyncio logging
        for logger_name in ['aiohttp', 'asyncio']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False
            logger.handlers = []
        
        # Disable root logger to prevent INFO:root: messages (like get_signable_message)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL + 1)
        root_logger.handlers = []  # Remove default handlers
        root_logger.propagate = False

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

        # GRVT state
        self.grvt_client = None
        self.grvt_contract_id = None
        self.grvt_tick_size = None
        self.grvt_order_status = None

        # GRVT order book state (using WebSocket)
        self.grvt_order_book = {"bids": {}, "asks": {}}
        self.grvt_best_bid = None
        self.grvt_best_ask = None
        self.grvt_order_book_ready = False
        self.grvt_order_book_lock = asyncio.Lock()
        self.grvt_snapshot_received = False
        self.grvt_order_book_ws_task = None

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
        self.current_lighter_price = None
        self.lighter_order_info = None

        # Position tracking
        self.grvt_position = Decimal('0')
        self.lighter_position = Decimal('0')

        # CSV file handles for efficient writing (kept open)
        self.bbo_csv_file = None
        self.bbo_csv_writer = None
        self.bbo_write_counter = 0
        self.bbo_flush_interval = 10  # Flush every N writes

        # Lighter API configuration
        self.lighter_base_url = "https://mainnet.zklighter.elliot.ai"
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX'))

        # GRVT configuration
        self.grvt_trading_account_id = os.getenv('GRVT_TRADING_ACCOUNT_ID')
        self.grvt_private_key = os.getenv('GRVT_PRIVATE_KEY')
        self.grvt_api_key = os.getenv('GRVT_API_KEY')
        self.grvt_environment = os.getenv('GRVT_ENVIRONMENT', 'prod')

    def shutdown(self, signum=None, frame=None):
        """Synchronous shutdown handler (called by signal handler)."""
        # Just set the stop flag - actual cleanup happens in async_shutdown()
        self.stop_flag = True

    async def async_shutdown(self):
        """Async shutdown handler for proper cleanup."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Cancel Lighter WebSocket task
        if self.lighter_ws_task and not self.lighter_ws_task.done():
            try:
                self.lighter_ws_task.cancel()
                await asyncio.sleep(0.1)  # Give task time to cancel
                self.logger.info("🔌 Lighter WebSocket task cancelled")
            except Exception as e:
                self.logger.error(f"Error cancelling Lighter WebSocket task: {e}")

        # Cancel GRVT order book WebSocket task
        if self.grvt_order_book_ws_task and not self.grvt_order_book_ws_task.done():
            try:
                self.grvt_order_book_ws_task.cancel()
                await asyncio.sleep(0.1)  # Give task time to cancel
                self.logger.info("🔌 GRVT order book WebSocket task cancelled")
            except Exception as e:
                self.logger.error(f"Error cancelling GRVT order book WebSocket task: {e}")

        # Disconnect GRVT WebSocket properly
        if self.grvt_client and hasattr(self.grvt_client, '_ws_client') and self.grvt_client._ws_client:
            try:
                # Use asyncio.wait_for with timeout to prevent hanging
                await asyncio.wait_for(self.grvt_client.disconnect(), timeout=2.0)
                self.logger.info("🔌 GRVT WebSocket disconnected")
            except (asyncio.TimeoutError, RuntimeError, Exception) as e:
                # Ignore errors during shutdown (event loop may be closing or already closed)
                # RuntimeError: no running event loop can occur during cleanup
                pass

        # Close CSV file handles
        if self.bbo_csv_file:
            try:
                self.bbo_csv_file.flush()
                self.bbo_csv_file.close()
                self.logger.info("📊 BBO CSV file closed")
            except Exception as e:
                self.logger.error(f"Error closing BBO CSV file: {e}")

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
                writer.writerow(['exchange', 'timestamp', 'side', 'price', 'quantity', 'expected_price'])

    def _initialize_bbo_csv_file(self):
        """Initialize BBO CSV file with headers if it doesn't exist."""
        file_exists = os.path.exists(self.bbo_csv_filename)
        
        # Open file in append mode (will create if doesn't exist)
        self.bbo_csv_file = open(self.bbo_csv_filename, 'a', newline='', buffering=8192)  # 8KB buffer
        self.bbo_csv_writer = csv.writer(self.bbo_csv_file)
        
        # Write header only if file is new
        if not file_exists:
            self.bbo_csv_writer.writerow([
                'timestamp',
                'grvt_bid',
                'grvt_ask',
                'lighter_bid',
                'lighter_ask',
                'long_grvt_spread',
                'short_grvt_spread',
                'long_grvt',
                'short_grvt'
            ])
            self.bbo_csv_file.flush()  # Ensure header is written immediately

    def log_trade_to_csv(self, exchange: str, side: str, price: str, quantity: str, expected_price: str):
        """Log trade details to CSV file."""
        timestamp = datetime.now(pytz.UTC).isoformat()

        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                exchange,
                timestamp,
                side,
                price,
                quantity,
                expected_price
            ])

        self.logger.info(f"📊 Trade logged to CSV: {exchange} {side} {quantity} @ {price}")

    def log_bbo_to_csv(self, grvt_bid: Decimal, grvt_ask: Decimal, lighter_bid: Decimal, lighter_ask: Decimal, long_grvt: bool, short_grvt: bool):
        """Log BBO data to CSV file using buffered writes."""
        if not self.bbo_csv_file or not self.bbo_csv_writer:
            # Fallback: reinitialize if file handle is lost
            self._initialize_bbo_csv_file()
        
        timestamp = datetime.now(pytz.UTC).isoformat()
        
        # Calculate spreads
        long_grvt_spread = lighter_bid - grvt_bid if lighter_bid and lighter_bid > 0 and grvt_bid > 0 else Decimal('0')
        short_grvt_spread = grvt_ask - lighter_ask if grvt_ask > 0 and lighter_ask and lighter_ask > 0 else Decimal('0')
        
        try:
            self.bbo_csv_writer.writerow([
                timestamp,
                float(grvt_bid),
                float(grvt_ask),
                float(lighter_bid) if lighter_bid and lighter_bid > 0 else 0.0,
                float(lighter_ask) if lighter_ask and lighter_ask > 0 else 0.0,
                float(long_grvt_spread),
                float(short_grvt_spread),
                long_grvt,
                short_grvt
            ])
            
            # Increment counter and flush periodically
            self.bbo_write_counter += 1
            if self.bbo_write_counter >= self.bbo_flush_interval:
                self.bbo_csv_file.flush()
                self.bbo_write_counter = 0
        except Exception as e:
            self.logger.error(f"Error writing to BBO CSV: {e}")
            # Try to reinitialize on error
            try:
                if self.bbo_csv_file:
                    self.bbo_csv_file.close()
            except Exception:
                pass
            self._initialize_bbo_csv_file()

    def log_thresholds_to_json(self, long_grvt_threshold: Decimal, short_grvt_threshold: Decimal):
        """Log threshold values to JSON file."""
        try:
            timestamp = datetime.now(pytz.UTC).isoformat()
            thresholds_data = {
                "timestamp": timestamp,
                "long_grvt_threshold": float(long_grvt_threshold),
                "short_grvt_threshold": float(short_grvt_threshold)
            }
            with open(self.thresholds_json_filename, 'w') as json_file:
                json.dump(thresholds_data, json_file, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing thresholds to JSON: {e}")

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
                quantity=str(order_data['filled_base_amount']),
                expected_price=str(self.exp_lighter_price)
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
            bid_levels = [(price, size) for price, size in self.lighter_order_book["bids"].items()
                if size * price >= 4000]
            best_bid = max(bid_levels) if bid_levels else (None, None)

        if self.lighter_order_book["asks"]:
            ask_levels = [(price, size) for price, size in self.lighter_order_book["asks"].items() 
                if size * price >= 4000]
            best_ask = min(ask_levels) if ask_levels else (None, None)

        return best_bid, best_ask

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
        def signal_handler(signum, frame):
            """Handle shutdown signals by setting stop flag."""
            self.stop_flag = True
            self.logger.info("\n🛑 Received interrupt signal (Ctrl+C)...")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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

    def initialize_grvt_client(self):
        """Initialize the GRVT client."""
        if not all([self.grvt_trading_account_id, self.grvt_private_key, self.grvt_api_key]):
            raise ValueError("GRVT_TRADING_ACCOUNT_ID, GRVT_PRIVATE_KEY, and GRVT_API_KEY must be set in environment variables")

        # Create config for GRVT client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',  # Will be set when we get contract info
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),  # Will be updated when we get contract info
            'close_order_side': 'sell'  # Default, will be updated based on strategy
        }

        # Wrap in Config class for GRVT client
        config = Config(config_dict)

        # Initialize GRVT client
        self.grvt_client = GrvtClient(config)

        self.logger.info("✅ GRVT client initialized successfully")
        return self.grvt_client

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

    async def get_grvt_contract_info(self) -> Tuple[str, Decimal]:
        """Get GRVT contract ID and tick size."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        contract_id, tick_size = await self.grvt_client.get_contract_attributes()

        if self.order_quantity < self.grvt_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.grvt_client.config.quantity}")

        return contract_id, tick_size

    async def get_grvt_position(self) -> Decimal:
        """Get GRVT position."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        return await self.grvt_client.get_account_positions()

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        if self.grvt_tick_size is None:
            return price
        return (price / self.grvt_tick_size).quantize(Decimal('1')) * self.grvt_tick_size

    async def place_grvt_market_order(self, side: str, quantity: Decimal):
        """Place a market order on GRVT."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")
        self.grvt_order_status = None

        return await self.grvt_client.place_market_order(self.grvt_contract_id, quantity, side.lower())

    async def place_grvt_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on GRVT at best bid/ask."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        # Determine order price
        if side.lower() == 'buy':
            order_price = self.grvt_best_ask - self.grvt_tick_size
        else:  # sell
            order_price = self.grvt_best_bid + self.grvt_tick_size

        order_price = self.round_to_tick(order_price)

        self.grvt_order_status = None
        self.logger.info(f"[OPEN] [GRVT] [{side}] Placing GRVT POST-ONLY order: {quantity} @ {order_price}")

        # Place post-only order using GRVT client
        order_result = await self.grvt_client.place_open_order(
            contract_id=self.grvt_contract_id,
            quantity=quantity,
            direction=side.lower()
        )

        if not order_result.success:
            raise Exception(f"Failed to place order: {order_result.error_message}")

        return order_result.order_id

    async def place_lighter_market_order(self, lighter_side: str, quantity: Decimal):
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

    async def setup_grvt_websocket(self):
        """Setup GRVT websocket for order updates."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from GRVT WebSocket."""
            if order_data.get('contract_id') != self.grvt_contract_id:
                return
            try:
                order_id = order_data.get('order_id')
                status = order_data.get('status')
                side = order_data.get('side', '').lower()
                filled_size = Decimal(order_data.get('filled_size', '0'))
                size = Decimal(order_data.get('size', '0'))
                price = order_data.get('price', '0')
                
                if status == 'CANCELED' and filled_size > 0:
                    status = 'FILLED'

                # Handle the order update
                if status == 'FILLED' and self.grvt_order_status != 'FILLED':
                    if side == 'buy':
                        self.grvt_position += filled_size
                    else:
                        self.grvt_position -= filled_size
                    self.logger.info(f"[{order_id}] [GRVT] [{status}]: {filled_size} @ {price}")
                    self.grvt_order_status = status
                    if filled_size > 0.0001:
                        # Log GRVT trade to CSV
                        self.log_trade_to_csv(
                            exchange='GRVT',
                            side=side,
                            price=str(price),
                            quantity=str(filled_size),
                            expected_price=str(self.exp_grvt_price)
                        )
                elif self.grvt_order_status != 'FILLED':
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [GRVT] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [GRVT] [{status}]: {filled_size} @ {price}")
                    self.grvt_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling GRVT order update: {e}")

        try:
            # Setup order update handler
            self.grvt_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ GRVT WebSocket order update handler set up")

            # Connect to GRVT WebSocket
            await self.grvt_client.connect()
            self.logger.info("✅ GRVT WebSocket connection established")

        except Exception as e:
            self.logger.error(f"Could not setup GRVT WebSocket handlers: {e}")

    def _parse_grvt_level(self, level):
        """Parse a level which can be dict or list format."""
        if isinstance(level, dict):
            return level.get('price', '0'), level.get('size', '0')
        elif isinstance(level, list) and len(level) >= 2:
            return str(level[0]), str(level[1])
        return None, None

    def _update_grvt_orderbook_side(self, orderbook_side, levels):
        """Update order book side with new levels."""
        for level in levels:
            price, size = self._parse_grvt_level(level)
            if price is None:
                continue
            
            # Convert size to float for comparison
            try:
                size_float = float(size)
            except (ValueError, TypeError):
                size_float = 0.0
            
            # If size is 0 or '0' or '0.0', remove the level
            if size_float <= 0:
                orderbook_side.pop(price, None)
            else:
                orderbook_side[price] = size

    def _get_grvt_best_levels(self) -> Tuple[Tuple[Decimal, Decimal], Tuple[Decimal, Decimal]]:
        """Get best bid and ask levels from GRVT order book."""
        best_bid = None
        best_ask = None

        if self.grvt_order_book["bids"]:
            valid_bids = {price: size for price, size in self.grvt_order_book["bids"].items() 
                         if float(size) > 0}
            if valid_bids:
                best_bid_price = max(valid_bids.keys(), key=lambda x: float(x))
                best_bid = (Decimal(best_bid_price), Decimal(valid_bids[best_bid_price]))

        if self.grvt_order_book["asks"]:
            valid_asks = {price: size for price, size in self.grvt_order_book["asks"].items() 
                         if float(size) > 0}
            if valid_asks:
                best_ask_price = min(valid_asks.keys(), key=lambda x: float(x))
                best_ask = (Decimal(best_ask_price), Decimal(valid_asks[best_ask_price]))

        return best_bid, best_ask

    def _get_grvt_instrument_name(self) -> str:
        """Convert ticker to GRVT instrument format (e.g., BTC -> BTC_USDT_Perp)."""
        return f"{self.ticker}_USDT_Perp"

    async def handle_grvt_orderbook_ws(self):
        """Handle GRVT order book WebSocket connection and messages."""
        ws_url = "wss://market-data.grvt.io/ws/full"
        instrument = self._get_grvt_instrument_name()
        rate = 100  # milliseconds
        
        # Subscription message (JSONRPC 2.0 format)
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {
                "stream": "v1.book.d",
                "selectors": [f"{instrument}@{rate}"]
            },
            "id": 1
        }
        
        last_sequence = None
        
        while not self.stop_flag:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Reset order book state before connecting
                    async with self.grvt_order_book_lock:
                        self.grvt_order_book["bids"].clear()
                        self.grvt_order_book["asks"].clear()
                        self.grvt_snapshot_received = False
                        self.grvt_order_book_ready = False
                    
                    # Send subscription message
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.info(f"📡 Subscribed to GRVT order book: {instrument}@{rate}ms")
                    
                    async for message in ws:
                        if self.stop_flag:
                            break
                            
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"⚠️ JSON parsing error in GRVT orderbook websocket: {e}")
                            continue
                        
                        # Handle subscription confirmation
                        if 'result' in data or ('stream' in data and 'subs' in data):
                            result = data.get('result', data)
                            subs = result.get('subs', [])
                            if subs:
                                self.logger.info(f"✅ GRVT order book subscription confirmed: {subs}")
                            continue
                        
                        # Handle feed data stream
                        if 'stream' in data and 'feed' in data:
                            stream = data.get('stream')
                            selector = data.get('selector', '')
                            sequence = data.get('sequence_number', '0')
                            feed_data = data.get('feed', {})
                            
                            # Validate sequence number
                            if last_sequence is not None:
                                try:
                                    seq_int = int(sequence)
                                    last_seq_int = int(last_sequence)
                                    if seq_int < last_seq_int:
                                        self.logger.warning(f"⚠️ GRVT sequence number decreased! {last_sequence} -> {sequence}")
                                    elif seq_int > last_seq_int + 1 and last_sequence != 0:
                                        self.logger.warning(f"⚠️ GRVT sequence number gap! {last_sequence} -> {sequence}")
                                except ValueError:
                                    pass
                            last_sequence = sequence
                            
                            # Check if this is an order book update
                            if 'bids' in feed_data or 'asks' in feed_data:
                                bids = feed_data.get('bids', [])
                                asks = feed_data.get('asks', [])
                                
                                async with self.grvt_order_book_lock:
                                    # Handle initial snapshot (sequence_number = 0)
                                    if sequence == '0' or sequence == 0:
                                        # Clear order book for snapshot
                                        self.grvt_order_book["bids"].clear()
                                        self.grvt_order_book["asks"].clear()
                                        self.grvt_snapshot_received = True
                                        self.logger.info(f"📸 GRVT order book snapshot received (seq: {sequence})")
                                    
                                    # Update order book
                                    if bids:
                                        self._update_grvt_orderbook_side(self.grvt_order_book['bids'], bids)
                                    if asks:
                                        self._update_grvt_orderbook_side(self.grvt_order_book['asks'], asks)
                                    
                                    # Update best bid/ask
                                    best_bid, best_ask = self._get_grvt_best_levels()
                                    
                                    if best_bid is not None:
                                        self.grvt_best_bid = best_bid[0]
                                        self.grvt_best_bid_size = best_bid[1]
                                    if best_ask is not None:
                                        self.grvt_best_ask = best_ask[0]
                                        self.grvt_best_ask_size = best_ask[1]
                                    
                                    # Mark as ready after snapshot
                                    if self.grvt_snapshot_received:
                                        self.grvt_order_book_ready = True
                            
                            # Handle errors
                            if 'error' in data or ('code' in data and 'message' in data):
                                error = data.get('error', data)
                                code = error.get('code', 'N/A')
                                message = error.get('message', str(error))
                                self.logger.error(f"❌ GRVT orderbook error [{code}]: {message}")
                                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("⚠️ GRVT orderbook websocket connection closed")
            except websockets.exceptions.WebSocketException as e:
                self.logger.warning(f"⚠️ GRVT orderbook websocket error: {e}")
            except Exception as e:
                self.logger.error(f"⚠️ Error in GRVT orderbook websocket: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
            
            # Wait before reconnecting
            if not self.stop_flag:
                await asyncio.sleep(2)

    async def get_lighter_position(self):
        url = "https://mainnet.zklighter.elliot.ai/api/v1/account"
        headers = {"accept": "application/json"}

        current_position = None
        parameters = {"by": "index", "value": self.account_index}
        attempts = 0
        while current_position is None and attempts < 10:
            try:
                response = requests.get(url, headers=headers, params=parameters, timeout=10)
                response.raise_for_status()

                # Check if response has content
                if not response.text.strip():
                    print("⚠️ Empty response from Lighter API for position check")
                    return self.lighter_position

                data = response.json()

                if 'accounts' not in data or not data['accounts']:
                    print(f"⚠️ Unexpected response format from Lighter API: {data}")
                    return self.lighter_position

                positions = data['accounts'][0].get('positions', [])
                for position in positions:
                    if position.get('symbol') == self.ticker:
                        current_position = Decimal(position['position']) * position['sign']
                        break
                if current_position is None:
                    current_position = 0

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error getting position: {e}")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error in position response: {e}")
                print(f"Response text: {response.text[:200]}...")  # Show first 200 chars
            except Exception as e:
                print(f"⚠️ Unexpected error getting position: {e}")
            finally:
                attempts += 1
                await asyncio.sleep(1)

        if current_position is None:
            self.logger.error(f"❌ Failed to get Lighter position after {attempts} attempts")
            sys.exit(1)

        return current_position
    
    async def check_position_balance(self, log_position: bool = True) -> bool:
        attempts = 0
        position_is_balanced = False
        while attempts < 4:
            attempts += 1
            self.lighter_position = await self.get_lighter_position()
            self.grvt_position = await self.get_grvt_position()
            if log_position:
                self.logger.info(f"GRVT position: {self.grvt_position} | Lighter position: {self.lighter_position}")

            if abs(self.grvt_position + self.lighter_position) > self.order_quantity:
                self.logger.error(f"❌ Attempt {attempts} | Position imbalance: {self.grvt_position + self.lighter_position}")
                await asyncio.sleep(5)
            else:
                position_is_balanced = True
                break
        return position_is_balanced


    async def trading_loop(self):
        """Main trading loop implementing the new strategy."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")

        # Initialize clients
        try:
            self.initialize_lighter_client()
            self.initialize_grvt_client()

            # Get contract info
            self.grvt_contract_id, self.grvt_tick_size = await self.get_grvt_contract_info()
            self.lighter_market_index, self.base_amount_multiplier, self.price_multiplier, self.tick_size = self.get_lighter_market_config()

            self.logger.info(f"Contract info loaded - GRVT: {self.grvt_contract_id}, "
                             f"Lighter: {self.lighter_market_index}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return

        # Setup GRVT websocket for order updates
        try:
            await self.setup_grvt_websocket()
            self.logger.info("✅ GRVT WebSocket connection established for order updates")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup GRVT websocket: {e}")
            return

        # Setup GRVT order book websocket
        try:
            self.grvt_order_book_ws_task = asyncio.create_task(self.handle_grvt_orderbook_ws())
            self.logger.info("✅ GRVT order book WebSocket task started")

            # Wait for initial GRVT order book data with timeout
            self.logger.info("⏳ Waiting for initial GRVT order book data...")
            timeout = 10  # seconds
            start_time = time.time()
            while not self.grvt_order_book_ready and not self.stop_flag:
                if time.time() - start_time > timeout:
                    self.logger.warning(f"⚠️ Timeout waiting for GRVT WebSocket order book data after {timeout}s")
                    break
                await asyncio.sleep(0.5)

            if self.grvt_order_book_ready:
                self.logger.info("✅ GRVT WebSocket order book data received")
            else:
                self.logger.warning("⚠️ GRVT WebSocket order book not ready")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup GRVT order book websocket: {e}")
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

        last_position_log = time.time()
        while not self.stop_flag:
            if time.time() - last_position_log > 10:
                log_position = True
                last_position_log = time.time()
            else:
                log_position = False

            position_is_balanced = await self.check_position_balance(log_position)
            if not position_is_balanced:
                self.stop_flag = True
                break

            if None in [self.lighter_best_bid, self.lighter_best_ask, self.grvt_best_bid, self.grvt_best_ask]:
                await asyncio.sleep(1)
                continue

            self.spread_history.append(self.lighter_best_bid - self.grvt_best_bid)

            if len(self.spread_history) > 1000:
                data = list(self.spread_history)
                median_val = statistics.median(data)
                long_grvt_threshold = median_val + self.grvt_best_ask * Decimal("0.0002")
                short_grvt_threshold = -(median_val - self.grvt_best_ask * Decimal("0.0002"))
                # Log thresholds to JSON file
                self.log_thresholds_to_json(long_grvt_threshold, short_grvt_threshold)
            else:
                if log_position:
                    self.logger.info(f"logging spread history. {len(self.spread_history)}/1000")
                    self.logger.info(f"best bid: {self.lighter_best_bid} | best ask: {self.lighter_best_ask}")
                await asyncio.sleep(1)
                continue            
            long_grvt = False
            short_grvt = False
            if self.lighter_best_bid and self.grvt_best_ask and self.lighter_best_bid - self.grvt_best_ask > long_grvt_threshold and self.grvt_position <= self.max_position:
                self.exp_grvt_price = self.grvt_best_ask
                self.exp_lighter_price = self.lighter_best_bid
                long_grvt = True
            elif self.grvt_best_bid and self.lighter_best_ask and self.grvt_best_bid - self.lighter_best_ask > short_grvt_threshold and self.grvt_position >= -1*self.max_position:
                self.exp_grvt_price = self.grvt_best_bid
                self.exp_lighter_price = self.lighter_best_ask
                short_grvt = True

            if long_grvt:
                order_quantity = min(self.order_quantity, self.grvt_best_ask_size)

                try:
                    # Place both trades concurrently
                    await asyncio.gather(
                        self.place_grvt_market_order('buy', order_quantity),
                        self.place_lighter_market_order('sell', order_quantity)
                    )
                except Exception as e:
                    self.logger.error(f"⚠️ Error in trading loop: {e}")
                    self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")

            elif short_grvt:
                order_quantity = min(self.order_quantity, self.grvt_best_bid_size)

                try:
                    # Place both trades concurrently
                    await asyncio.gather(
                        self.place_grvt_market_order('sell', order_quantity),
                        self.place_lighter_market_order('buy', order_quantity)
                    )
                except Exception as e:
                    self.logger.error(f"⚠️ Error in trading loop: {e}")
                    self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")

            else:
                await asyncio.sleep(1)

    async def run(self):
        """Run the hedge bot."""
        self.setup_signal_handlers()

        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Received interrupt signal...")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            self.logger.info("🔄 Cleaning up...")
            try:
                await self.async_shutdown()
            except Exception as e:
                # Ignore errors during final cleanup
                pass
