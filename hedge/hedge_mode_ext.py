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
import random
from collections import defaultdict
from decimal import Decimal
from typing import Tuple

from lighter.signer_client import SignerClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.extended import ExtendedClient
import websockets
from datetime import datetime
import pytz


class Config:
    """Simple config class to wrap dictionary for Extended client."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on Extended and hedges with market orders on Lighter."""

    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        fill_timeout: int = 5,
        iterations: int = 20,
        sleep_time: int = 0,
        entry_bps: float = 2.0,
        exit_good_bps: float = 0.0,
        exit_ok_bps: float = -0.5,
        exit_bad_bps: float = -1.0,
        soft_unhedged_pos: float = 0.02,
        max_unhedged_pos: float = 0.03,
        max_unhedged_ms: int = 1000,
        unwind_trigger_bps: float = -0.3,
        unwind_confirm_count: int = 3,
        unwind_cooldown_ms: int = 5000,
        hedge_ioc: bool = False,
        ioc_tick_offset: int = 2,
        ioc_max_retries: int = 3,
    ):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.lighter_order_filled = False
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.extended_position = Decimal('0')
        self.lighter_position = Decimal('0')
        self.current_order = {}

        # Strategy parameters
        self.entry_bps = Decimal(str(entry_bps))
        self.exit_good_bps = Decimal(str(exit_good_bps))
        self.exit_ok_bps = Decimal(str(exit_ok_bps))
        self.exit_bad_bps = Decimal(str(exit_bad_bps))
        self.soft_unhedged_pos = Decimal(str(soft_unhedged_pos))
        self.max_unhedged_pos = Decimal(str(max_unhedged_pos))
        self.max_unhedged_ms = max_unhedged_ms
        self.unwind_trigger_bps = Decimal(str(unwind_trigger_bps))
        self.unwind_confirm_count = unwind_confirm_count
        self.unwind_cooldown_ms = unwind_cooldown_ms
        self.hedge_ioc = hedge_ioc
        self.ioc_tick_offset = Decimal(str(ioc_tick_offset))
        self.ioc_max_retries = ioc_max_retries

        # Position state
        self.unhedged_pos = Decimal("0")
        self.hedged_pos = Decimal("0")
        self.unhedged_since_ms = None
        self.unwind_cooldown_until_ms = 0
        self.unwind_edge_bad_count = 0
        self.hedged_qty_by_order = defaultdict(Decimal)

        self.hedge_lock = asyncio.Lock()
        self.unwind_lock = asyncio.Lock()

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/extended_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/extended_{ticker}_hedge_mode_trades.csv"
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

        # Event logging
        self.event_csv_filename = f"logs/extended_{ticker}_hedge_events.csv"
        if not os.path.exists(self.event_csv_filename):
            with open(self.event_csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'event', 'reason', 'edge_bps', 'qty', 'unhedged_before',
                    'unhedged_after', 'hedged_pos', 'age_ms'
                ])

        # State management
        self.stop_flag = False
        self.order_counter = 0

        # Extended state
        self.extended_client = None
        self.extended_contract_id = None
        self.extended_tick_size = None
        self.extended_order_status = None

        # Extended order book state for websocket-based BBO
        self.extended_order_book = {'bids': {}, 'asks': {}}
        self.extended_best_bid = None
        self.extended_best_ask = None
        self.extended_order_book_ready = False

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

        # Extended configuration
        self.extended_vault = os.getenv('EXTENDED_VAULT')
        self.extended_stark_key_private = os.getenv('EXTENDED_STARK_KEY_PRIVATE')
        self.extended_stark_key_public = os.getenv('EXTENDED_STARK_KEY_PUBLIC')
        self.extended_api_key = os.getenv('EXTENDED_API_KEY')

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Close WebSocket connections
        if self.extended_client:
            try:
                # Note: disconnect() is async, but shutdown() is sync
                # We'll let the cleanup happen naturally
                self.logger.info("🔌 Extended WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting Extended WebSocket: {e}")

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

    def log_event(self, event: str, reason: str, edge_bps: Decimal, qty: Decimal,
                  unhedged_before: Decimal, unhedged_after: Decimal, age_ms: int):
        """Log decision events to CSV for audit."""
        timestamp = datetime.now(pytz.UTC).isoformat()
        with open(self.event_csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                event,
                reason,
                f"{edge_bps}",
                f"{qty}",
                f"{unhedged_before}",
                f"{unhedged_after}",
                f"{self.hedged_pos}",
                age_ms,
            ])

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

            self.logger.info(f"📊 Lighter order filled: {order_data['side']} "
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

    def choose_mid_price(self) -> Decimal:
        """Choose a mid price from available books."""
        candidates = []
        try:
            candidates.append(self.get_lighter_mid_price())
        except Exception:
            pass

        if self.extended_best_bid and self.extended_best_ask and self.extended_best_bid < self.extended_best_ask:
            candidates.append((self.extended_best_bid + self.extended_best_ask) / Decimal('2'))

        if not candidates:
            raise ValueError("No mid price available")

        return sum(candidates) / Decimal(len(candidates))

    def edge_entry_bps(self, side: str, extended_post_price: Decimal) -> Decimal:
        mid = self.choose_mid_price()
        if side.lower() == "buy":
            edge = (self.lighter_best_bid - extended_post_price) / mid * Decimal("1e4")
        else:
            edge = (extended_post_price - self.lighter_best_ask) / mid * Decimal("1e4")
        return edge

    def edge_exit_bps_for_unhedged(self, unhedged_pos: Decimal) -> Decimal:
        mid = self.choose_mid_price()

        if unhedged_pos > 0:
            if self.extended_best_bid is None:
                raise ValueError("Extended best bid missing")
            edge = (self.extended_best_bid - self.lighter_best_ask) / mid * Decimal("1e4")
        else:
            if self.extended_best_ask is None:
                raise ValueError("Extended best ask missing")
            edge = (self.lighter_best_bid - self.extended_best_ask) / mid * Decimal("1e4")
        return edge

    def exit_edge_bps_for_unwind(self) -> Decimal:
        mid = self.choose_mid_price()
        if self.hedged_pos > 0:
            edge = (self.extended_best_bid - self.lighter_best_ask) / mid * Decimal("1e4")
        else:
            edge = (self.lighter_best_bid - self.extended_best_ask) / mid * Decimal("1e4")
        return edge

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

                                    asyncio.create_task(self.maybe_hedge_unhedged_pos(reason="L2_UPDATE"))
                                    asyncio.create_task(self.maybe_unwind_hedged_pos(source="L2_UPDATE"))

                                elif data.get("type") == "ping":
                                    # Respond to ping with pong
                                    await ws.send(json.dumps({"type": "pong"}))
                                elif data.get("type") == "update/account_orders":
                                    # Handle account orders updates
                                    orders = data.get("orders", {}).get(str(self.lighter_market_index), [])
                                    for order_data in orders:
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

    def initialize_extended_client(self):
        """Initialize the Extended client."""
        if not all([self.extended_vault, self.extended_stark_key_private, self.extended_stark_key_public, self.extended_api_key]):
            raise ValueError("EXTENDED_VAULT, EXTENDED_STARK_KEY_PRIVATE, EXTENDED_STARK_KEY_PUBLIC, and EXTENDED_API_KEY must be set in environment variables")

        # Create config for Extended client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',  # Will be set when we get contract info
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),  # Will be updated when we get contract info
            'close_order_side': 'sell'  # Default, will be updated based on strategy
        }

        # Wrap in Config class for Extended client
        config = Config(config_dict)

        # Initialize Extended client
        self.extended_client = ExtendedClient(config)

        self.logger.info("✅ Extended client initialized successfully")
        return self.extended_client

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

    async def get_extended_contract_info(self) -> Tuple[str, Decimal]:
        """Get Extended contract ID and tick size."""
        if not self.extended_client:
            raise Exception("Extended client not initialized")

        contract_id, tick_size = await self.extended_client.get_contract_attributes()

        if self.order_quantity < self.extended_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.extended_client.config.quantity}")

        return contract_id, tick_size

    async def fetch_extended_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from Extended using websocket data."""
        # Use WebSocket data if available
        if self.extended_order_book_ready and self.extended_best_bid and self.extended_best_ask:
            if self.extended_best_bid > 0 and self.extended_best_ask > 0 and self.extended_best_bid < self.extended_best_ask:
                return self.extended_best_bid, self.extended_best_ask

        # Fallback to REST API if websocket data is not available
        self.logger.warning("WebSocket BBO data not available, falling back to REST API")
        if not self.extended_client:
            raise Exception("Extended client not initialized")

        best_bid, best_ask = await self.extended_client.fetch_bbo_prices(self.extended_contract_id)

        return best_bid, best_ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round price to tick size."""
        if self.extended_tick_size is None:
            return price
        return (price / self.extended_tick_size).quantize(Decimal('1')) * self.extended_tick_size

    async def place_bbo_order(self, side: str, quantity: Decimal):
        # Get best bid/ask prices
        best_bid, best_ask = await self.fetch_extended_bbo_prices()

        # Place the order using Extended client
        order_result = await self.extended_client.place_open_order(
            contract_id=self.extended_contract_id,
            quantity=quantity,
            direction=side.lower()
        )

        if order_result.success:
            return order_result.order_id, order_result.price
        else:
            raise Exception(f"Failed to place order: {order_result.error_message}")

    async def place_extended_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on Extended."""
        if not self.extended_client:
            raise Exception("Extended client not initialized")

        self.extended_order_status = None
        now_ms = int(time.time() * 1000)
        if now_ms < self.unwind_cooldown_until_ms:
            self.logger.info("COOLDOWN_SKIP: entry blocked due to unwind cooldown")
            self.log_event("ENTRY_SKIP", "COOLDOWN_SKIP", Decimal("0"), Decimal("0"), self.unhedged_pos, self.unhedged_pos, 0)
            return

        # Determine tentative post price using current best quotes
        best_bid, best_ask = await self.fetch_extended_bbo_prices()
        post_price = best_bid if side == "buy" else best_ask
        edge_bps = self.edge_entry_bps(side, post_price)
        if edge_bps < self.entry_bps:
            self.logger.info(f"ENTRY_GATE_SKIP: edge {edge_bps:.4f} < entry_bps {self.entry_bps}")
            self.log_event("ENTRY_SKIP", "ENTRY_GATE_SKIP", edge_bps, Decimal("0"), self.unhedged_pos, self.unhedged_pos, 0)
            return

        self.logger.info(f"[OPEN] [Extended] [{side}] Placing Extended POST-ONLY order with edge {edge_bps:.4f} bps")
        order_id, order_price = await self.place_bbo_order(side, quantity)

        start_time = time.time()
        last_cancel_time = 0
        
        while not self.stop_flag:
            if self.extended_order_status in ['CANCELED', 'CANCELLED']:
                self.logger.info(f"Order {order_id} was canceled, placing new order")
                self.extended_order_status = None  # Reset to None to trigger new order
                order_id, order_price = await self.place_bbo_order(side, quantity)
                start_time = time.time()
                last_cancel_time = 0  # Reset cancel timer
                await asyncio.sleep(0.5)
            elif self.extended_order_status in ['NEW', 'OPEN', 'PENDING', 'CANCELING', 'PARTIALLY_FILLED']:
                await asyncio.sleep(0.5)
                
                # Check if we need to cancel and replace the order
                should_cancel = False
                if side == 'buy':
                    if order_price < self.extended_best_bid:
                        should_cancel = True
                else:
                    if order_price > self.extended_best_ask:
                        should_cancel = True

                # Cancel order if it's been too long or price is off
                current_time = time.time()
                if current_time - start_time > 10:
                    if should_cancel and current_time - last_cancel_time > 5:  # Prevent rapid cancellations
                        try:
                            self.logger.info(f"Canceling order {order_id} due to timeout/price mismatch")
                            cancel_result = await self.extended_client.cancel_order(order_id)
                            self.logger.info(f"cancel_result: {cancel_result}")
                            if cancel_result.success:
                                last_cancel_time = current_time
                                # Don't reset start_time here, let the cancellation trigger new order
                            else:
                                self.logger.error(f"❌ Error canceling Extended order: {cancel_result.error_message}")
                        except Exception as e:
                            self.logger.error(f"❌ Error canceling Extended order: {e}")
                    elif not should_cancel:
                        self.logger.info(f"Waiting for Extended order to be filled (order price is at best bid/ask)")
            elif self.extended_order_status == 'FILLED':
                self.logger.info(f"Order {order_id} filled successfully")
                break
            else:
                if self.extended_order_status is not None:
                    self.logger.error(f"❌ Unknown Extended order status: {self.extended_order_status}")
                    break
                else:
                    await asyncio.sleep(0.5)

        return True

    def handle_extended_order_book_update(self, message):
        """Handle Extended order book updates from WebSocket."""
        try:
            if isinstance(message, str):
                message = json.loads(message)

            self.logger.debug(f"Received Extended order book message: {message}")

            # Check if this is an order book update message
            if message.get("type") in ["SNAPSHOT", "DELTA"]:
                data = message.get("data", {})

                if data:
                    # Handle SNAPSHOT - replace entire order book
                    if message.get("type") == "SNAPSHOT":
                        self.extended_order_book['bids'].clear()
                        self.extended_order_book['asks'].clear()

                    # Update bids - Extended format is [{"p": "price", "q": "size"}, ...]
                    bids = data.get('b', [])
                    for bid in bids:
                        if isinstance(bid, dict):
                            price = Decimal(bid.get('p', '0'))
                            size = Decimal(bid.get('q', '0'))
                        else:
                            # Fallback for array format [price, size]
                            price = Decimal(bid[0])
                            size = Decimal(bid[1])
                        
                        if size > 0:
                            self.extended_order_book['bids'][price] = size
                        else:
                            # Remove zero size orders
                            self.extended_order_book['bids'].pop(price, None)

                    # Update asks - Extended format is [{"p": "price", "q": "size"}, ...]
                    asks = data.get('a', [])
                    for ask in asks:
                        if isinstance(ask, dict):
                            price = Decimal(ask.get('p', '0'))
                            size = Decimal(ask.get('q', '0'))
                        else:
                            # Fallback for array format [price, size]
                            price = Decimal(ask[0])
                            size = Decimal(ask[1])
                        
                        if size > 0:
                            self.extended_order_book['asks'][price] = size
                        else:
                            # Remove zero size orders
                            self.extended_order_book['asks'].pop(price, None)

                    # Update best bid and ask
                    if self.extended_order_book['bids']:
                        self.extended_best_bid = max(self.extended_order_book['bids'].keys())
                    if self.extended_order_book['asks']:
                        self.extended_best_ask = min(self.extended_order_book['asks'].keys())

                    if not self.extended_order_book_ready:
                        self.extended_order_book_ready = True
                        self.logger.info(f"📊 Extended order book ready - Best bid: {self.extended_best_bid}, "
                                         f"Best ask: {self.extended_best_ask}")
                    else:
                        self.logger.debug(f"📊 Order book updated - Best bid: {self.extended_best_bid}, "
                                          f"Best ask: {self.extended_best_ask}")

        except Exception as e:
            self.logger.error(f"Error handling Extended order book update: {e}")
            self.logger.error(f"Message content: {message}")

    def handle_extended_order_update(self, order_data):
        """Handle Extended order updates from WebSocket."""
        side = order_data.get('side', '').lower()
        filled_size = Decimal(order_data.get('filled_size', '0'))
        oid = order_data.get('order_id')
        now_ms = int(time.time() * 1000)

        delta = filled_size - self.hedged_qty_by_order[oid]
        if delta <= 0:
            return

        self.hedged_qty_by_order[oid] = filled_size

        if side == 'buy':
            self.unhedged_pos += delta
        else:
            self.unhedged_pos -= delta

        if self.unhedged_since_ms is None and self.unhedged_pos != 0:
            self.unhedged_since_ms = now_ms

        asyncio.create_task(self.maybe_hedge_unhedged_pos(reason="EXT_FILL"))

    async def maybe_hedge_unhedged_pos(self, reason: str):
        async with self.hedge_lock:
            if self.unhedged_pos == 0:
                return

            now_ms = int(time.time() * 1000)
            abs_pos = abs(self.unhedged_pos)
            age_ms = now_ms - (self.unhedged_since_ms or now_ms)
            try:
                edge_bps = self.edge_exit_bps_for_unhedged(self.unhedged_pos)
            except Exception as exc:
                self.logger.error(f"Hedge edge calc failed: {exc}")
                return

            qty = Decimal("0")
            reason_code = reason
            if edge_bps >= self.exit_good_bps:
                qty = abs_pos
                reason_code = "GOOD_EDGE"
            elif edge_bps >= self.exit_ok_bps:
                qty = max(Decimal("0"), abs_pos - self.soft_unhedged_pos)
                reason_code = "OK_EDGE"
            else:
                if abs_pos > self.max_unhedged_pos:
                    qty = abs_pos - self.max_unhedged_pos
                    reason_code = "HARD_LIMIT"
                elif age_ms > self.max_unhedged_ms:
                    qty = max(Decimal("0"), abs_pos - self.soft_unhedged_pos)
                    reason_code = "TIMEOUT"
                else:
                    qty = Decimal("0")
                    reason_code = "WAIT"

            unhedged_before = self.unhedged_pos
            if qty > 0:
                lighter_side = "sell" if self.unhedged_pos > 0 else "buy"
                if self.hedge_ioc:
                    success = await self.place_lighter_ioc_progressive(lighter_side, qty)
                    if not success:
                        self.log_event("HEDGE_SKIP", "IOC_FAIL", edge_bps, qty, unhedged_before, self.unhedged_pos, age_ms)
                        return
                else:
                    await self.place_lighter_market_order(lighter_side, qty, Decimal("0"))

                if lighter_side == "sell":
                    self.unhedged_pos -= qty
                    if self.unhedged_pos == 0:
                        self.hedged_pos += qty
                else:
                    self.unhedged_pos += qty
                    if self.unhedged_pos == 0:
                        self.hedged_pos -= qty

                if self.unhedged_pos == 0:
                    self.unhedged_since_ms = None

                self.log_event("HEDGE_EXEC", reason_code, edge_bps, qty, unhedged_before, self.unhedged_pos, age_ms)
            else:
                self.log_event("HEDGE_SKIP", reason_code, edge_bps, qty, unhedged_before, self.unhedged_pos, age_ms)

    async def maybe_unwind_hedged_pos(self, source: str):
        async with self.unwind_lock:
            if self.hedged_pos == 0:
                return
            now_ms = int(time.time() * 1000)
            if now_ms < self.unwind_cooldown_until_ms:
                return

            try:
                edge_bps = self.exit_edge_bps_for_unwind()
            except Exception as exc:
                self.logger.error(f"Unwind edge calc failed: {exc}")
                return

            if edge_bps <= self.unwind_trigger_bps:
                self.unwind_edge_bad_count += 1
            else:
                self.unwind_edge_bad_count = 0
                return

            if self.unwind_edge_bad_count < self.unwind_confirm_count:
                return

            unwind_qty = abs(self.hedged_pos)
            extended_side = "sell" if self.hedged_pos > 0 else "buy"
            lighter_side = "buy" if self.hedged_pos > 0 else "sell"

            try:
                await self.place_bbo_order(extended_side, unwind_qty)
                if self.hedge_ioc:
                    await self.place_lighter_ioc_progressive(lighter_side, unwind_qty)
                else:
                    await self.place_lighter_market_order(lighter_side, unwind_qty, Decimal("0"))
            finally:
                self.hedged_pos = Decimal("0")
                self.unwind_cooldown_until_ms = now_ms + self.unwind_cooldown_ms
                self.unwind_edge_bad_count = 0
                self.log_event("UNWIND", "UNWIND_EDGE_BAD", edge_bps, unwind_qty, self.unhedged_pos, self.unhedged_pos, 0)

    async def place_lighter_market_order(self, lighter_side: str, quantity: Decimal, price: Decimal):
        if not self.lighter_client:
            await self.initialize_lighter_client()

        best_bid, best_ask = self.get_lighter_best_levels()

        # Determine order parameters
        if lighter_side.lower() == 'buy':
            is_ask = False
            price = best_ask[0] * Decimal('1.002')
        else:
            is_ask = True
            price = best_bid[0] * Decimal('0.998')

        self.logger.info(f"Placing Lighter market order: {lighter_side} {quantity} | is_ask: {is_ask}")

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
            self.logger.info(f"🚀 Lighter limit order sent: {lighter_side} {quantity}")
            await self.monitor_lighter_order(client_order_index)

            return tx_hash
        except Exception as e:
            self.logger.error(f"❌ Error placing Lighter order: {e}")
            return None

    async def place_lighter_ioc_progressive(self, lighter_side: str, quantity: Decimal) -> bool:
        """Attempt IOC style execution with progressive tick offsets."""
        best_bid, best_ask = self.get_lighter_best_levels()
        offset = self.ioc_tick_offset
        for attempt in range(self.ioc_max_retries):
            if lighter_side.lower() == "buy":
                price = best_ask[0] + offset * self.tick_size
            else:
                price = best_bid[0] - offset * self.tick_size

            self.logger.info(f"IOC attempt {attempt+1}/{self.ioc_max_retries}: {lighter_side} {quantity} @ {price}")
            try:
                client_order_index = int(time.time() * 1000)
                time_in_force = getattr(self.lighter_client, "ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL", None)
                if time_in_force is None:
                    time_in_force = self.lighter_client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME

                tx_info, error = self.lighter_client.sign_create_order(
                    market_index=self.lighter_market_index,
                    client_order_index=client_order_index,
                    base_amount=int(quantity * self.base_amount_multiplier),
                    price=int(price * self.price_multiplier),
                    is_ask=lighter_side.lower() == "sell",
                    order_type=self.lighter_client.ORDER_TYPE_LIMIT,
                    time_in_force=time_in_force,
                    reduce_only=False,
                    trigger_price=0,
                )
                if error is not None:
                    raise Exception(error)

                await self.lighter_client.send_tx(
                    tx_type=self.lighter_client.TX_TYPE_CREATE_ORDER,
                    tx_info=tx_info,
                )
                return True
            except Exception as exc:
                self.logger.error(f"IOC attempt failed: {exc}")
                offset += self.ioc_tick_offset
                await asyncio.sleep(0.05)
        return False

    async def monitor_lighter_order(self, client_order_index: int):
        """Monitor Lighter order and adjust price if needed."""
        self.logger.info(f"🔍 Starting to monitor Lighter order - Order ID: {client_order_index}")

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

    async def setup_extended_websocket(self):
        """Setup Extended websocket for order updates and order book data."""
        if not self.extended_client:
            raise Exception("Extended client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from Extended WebSocket."""
            if order_data.get('contract_id') != self.extended_contract_id:
                self.logger.info(f"Ignoring order update from {order_data.get('contract_id')}")
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

                # Handle the order update
                if status == 'FILLED':
                    if side == 'buy':
                        self.extended_position += filled_size
                    else:
                        self.extended_position -= filled_size
                    self.logger.info(f"[{order_id}] [{order_type}] [Extended] [{status}]: {filled_size} @ {price}")
                    self.extended_order_status = status

                    # Log Extended trade to CSV
                    self.log_trade_to_csv(
                        exchange='Extended',
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

                    self.handle_extended_order_update({
                        'order_id': order_id,
                        'side': side,
                        'status': status,
                        'size': size,
                        'price': price,
                        'contract_id': self.extended_contract_id,
                        'filled_size': filled_size
                    })
                else:
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [{order_type}] [Extended] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [Extended] [{status}]: {filled_size} @ {price}")
                    # Update order status for all non-filled statuses
                    if status == 'PARTIALLY_FILLED':
                        self.extended_order_status = "OPEN"
                    elif status in ['CANCELED', 'CANCELLED']:
                        self.extended_order_status = status
                    elif status in ['NEW', 'OPEN', 'PENDING', 'CANCELING']:
                        self.extended_order_status = status
                    else:
                        self.logger.warning(f"Unknown order status: {status}")
                        self.extended_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling Extended order update: {e}")

        try:
            # Setup order update handler
            self.extended_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ Extended WebSocket order update handler set up")

            # Connect to Extended WebSocket
            await self.extended_client.connect()
            self.logger.info("✅ Extended WebSocket connection established")

            # Setup separate WebSocket connection for depth updates
            await self.setup_extended_depth_websocket()

        except Exception as e:
            self.logger.error(f"Could not setup Extended WebSocket handlers: {e}")

    async def setup_extended_depth_websocket(self):
        """Setup separate WebSocket connection for Extended depth updates."""
        try:
            import websockets

            async def handle_depth_websocket():
                """Handle depth WebSocket connection."""
                # Use the correct Extended WebSocket URL for order book stream
                market_name = f"{self.ticker}-USD"  # Extended uses format like BTC-USD
                url = f"wss://api.starknet.extended.exchange/stream.extended.exchange/v1/orderbooks/{market_name}?depth=1"

                while not self.stop_flag:
                    try:
                        async with websockets.connect(url) as ws:
                            self.logger.info(f"✅ Connected to Extended order book stream for {market_name}")

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
                                    self.logger.debug(f"Received Extended order book message: {data}")

                                    # Handle order book updates
                                    if data.get("type") in ["SNAPSHOT", "DELTA"]:
                                        self.handle_extended_order_book_update(data)

                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Failed to parse Extended order book message: {e}")
                                except Exception as e:
                                    self.logger.error(f"Error handling Extended order book message: {e}")

                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning("Extended order book WebSocket connection closed, reconnecting...")
                    except Exception as e:
                        self.logger.error(f"Extended order book WebSocket error: {e}")

                    # Wait before reconnecting
                    if not self.stop_flag:
                        await asyncio.sleep(2)

            # Start depth WebSocket in background
            asyncio.create_task(handle_depth_websocket())
            self.logger.info("✅ Extended order book WebSocket task started")

        except Exception as e:
            self.logger.error(f"Could not setup Extended order book WebSocket: {e}")

    async def trading_loop(self):
        """Main trading loop implementing risk-gated Extended entries only."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")

        try:
            self.initialize_lighter_client()
            self.initialize_extended_client()
            self.extended_contract_id, self.extended_tick_size = await self.get_extended_contract_info()
            self.lighter_market_index, self.base_amount_multiplier, self.price_multiplier, self.tick_size = self.get_lighter_market_config()
            self.logger.info(f"Contract info loaded - Extended: {self.extended_contract_id}, Lighter: {self.lighter_market_index}")
            await self.setup_extended_websocket()
            self.lighter_ws_task = asyncio.create_task(self.handle_lighter_ws())
        except Exception as exc:
            self.logger.error(f"❌ Initialization failure: {exc}")
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return

        iteration = 0
        while not self.stop_flag and iteration < self.iterations:
            iteration += 1
            self.logger.info(
                f"HEARTBEAT iter={iteration}/{self.iterations} ext_pos={self.extended_position} "
                f"unhedged={self.unhedged_pos} hedged={self.hedged_pos} skip={self.entry_skip_count}"
            )

            if abs(self.extended_position) >= self.max_extended_position:
                self.logger.info("RISK_GUARD: extended position at limit, skipping entry")
                await asyncio.sleep(self.sleep_time or 1)
                continue

            side = 'buy' if iteration % 2 == 1 else 'sell'
            try:
                placed = await self.place_extended_post_only_order(side, self.order_quantity)
            except Exception as exc:
                self.logger.error(f"⚠️ Error placing Extended order: {exc}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            if not placed:
                continue

            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time)

        self.logger.info("✅ Trading loop finished")

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
    parser = argparse.ArgumentParser(description='Trading bot for Extended and Lighter')
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
