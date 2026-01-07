"""
Custom Lighter WebSocket implementation without using the official SDK.
Based on the sample code provided by the user.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
import websockets


class LighterCustomWebSocketManager:
    """Custom WebSocket manager for Lighter order updates and order book without SDK."""

    def __init__(self, config: Dict[str, Any], order_update_callback: Optional[Callable] = None):
        self.config = config
        self.order_update_callback = order_update_callback
        self.logger = None
        self.running = False
        self.ws = None

        # Order book state
        self.order_book = {"bids": {}, "asks": {}}
        self.best_bid = None
        self.best_ask = None
        self.snapshot_loaded = False
        self.order_book_offset = None
        self.order_book_sequence_gap = False
        self.order_book_lock = asyncio.Lock()

        # WebSocket URL
        self.ws_url = "wss://mainnet.zklighter.elliot.ai/stream"
        self.market_index = config.contract_id
        self.account_index = config.account_index
        self.lighter_client = config.lighter_client

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    def _log(self, message: str, level: str = "INFO"):
        """Log message using the logger if available."""
        if self.logger:
            self.logger.log(message, level)

    def update_order_book(self, side: str, updates: List[Dict[str, Any]]):
        """Update the order book with new price/size information."""
        if side not in ["bids", "asks"]:
            self._log(f"Invalid side parameter: {side}. Must be 'bids' or 'asks'", "ERROR")
            return

        ob = self.order_book[side]

        if not isinstance(updates, list):
            self._log(f"Invalid updates format for {side}: expected list, got {type(updates)}", "ERROR")
            return

        for update in updates:
            try:
                if not isinstance(update, dict):
                    self._log(f"Invalid update format: expected dict, got {type(update)}", "ERROR")
                    continue

                if "price" not in update or "size" not in update:
                    self._log(f"Missing required fields in update: {update}", "ERROR")
                    continue

                price = float(update["price"])
                size = float(update["size"])

                # Validate price and size are reasonable
                if price <= 0:
                    self._log(f"Invalid price in update: {price}", "ERROR")
                    continue

                if size < 0:
                    self._log(f"Invalid size in update: {size}", "ERROR")
                    continue

                if size == 0:
                    ob.pop(price, None)
                else:
                    ob[price] = size
            except (KeyError, ValueError, TypeError) as e:
                self._log(f"Error processing order book update: {e}, update: {update}", "ERROR")
                continue

    def validate_order_book_offset(self, new_offset: int) -> bool:
        """Validate that the new offset is sequential and handle gaps."""
        if self.order_book_offset is None:
            # First offset, always valid
            self.order_book_offset = new_offset
            return True

        # Check if the new offset is sequential (should be +1)
        expected_offset = self.order_book_offset + 1
        if new_offset >= expected_offset:
            # Sequential update, update our offset
            self.order_book_offset = new_offset
            self.order_book_sequence_gap = False
            return True
        elif new_offset < expected_offset:
            # Gap detected - we missed some updates
            self._log(f"Order book sequence gap detected! Expected offset {expected_offset}, got {new_offset}", "WARNING")
            self.order_book_sequence_gap = True
            return False

    def handle_order_book_cutoff(self, data: Dict[str, Any]) -> bool:
        """Handle cases where order book updates might be cutoff or incomplete."""
        order_book = data.get("order_book", {})

        # Validate required fields
        if not order_book or "code" not in order_book or "offset" not in order_book:
            self._log("Incomplete order book update - missing required fields", "WARNING")
            return False

        # Check if the order book has the expected structure
        if "asks" not in order_book or "bids" not in order_book:
            self._log("Incomplete order book update - missing bids/asks", "WARNING")
            return False

        # Validate that asks and bids are lists
        if not isinstance(order_book["asks"], list) or not isinstance(order_book["bids"], list):
            self._log("Invalid order book structure - asks/bids should be lists", "WARNING")
            return False

        return True

    def validate_order_book_integrity(self) -> bool:
        """Validate that the order book is internally consistent."""
        try:
            if not self.order_book["bids"] or not self.order_book["asks"]:
                # Empty order book is valid
                return True

            # Get best bid and best ask
            best_bid = max(self.order_book["bids"].keys())
            best_ask = min(self.order_book["asks"].keys())

            # Check if best bid is higher than best ask (inconsistent)
            if best_bid >= best_ask:
                self._log(f"Order book inconsistency detected! Best bid: {best_bid}, Best ask: {best_ask}", "WARNING")
                return False

            return True
        except (ValueError, KeyError) as e:
            self._log(f"Error validating order book integrity: {e}", "ERROR")
            return False

    async def request_fresh_snapshot(self):
        """Request a fresh order book snapshot when we detect inconsistencies."""
        try:
            if not self.ws:
                return

            # Unsubscribe and resubscribe to get a fresh snapshot
            unsubscribe_msg = json.dumps({"type": "unsubscribe", "channel": f"order_book/{self.market_index}"})
            await self.ws.send(unsubscribe_msg)

            # Wait a moment for the unsubscribe to process
            await asyncio.sleep(1)

            # Resubscribe to get a fresh snapshot
            subscribe_msg = json.dumps({"type": "subscribe", "channel": f"order_book/{self.market_index}"})
            await self.ws.send(subscribe_msg)

            self._log("Requested fresh order book snapshot", "INFO")
        except Exception as e:
            self._log(f"Error requesting fresh snapshot: {e}", "ERROR")
            raise

    def get_best_levels(self) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
        """Get the best bid and ask levels with sufficient size for our order (~$5000)."""
        try:
            # Get all bid levels with sufficient size
            bid_levels = [(price, size) for price, size in self.order_book["bids"].items()
                          if size * price >= 40000]

            # Get all ask levels with sufficient size
            ask_levels = [(price, size) for price, size in self.order_book["asks"].items()
                          if size * price >= 40000]

            # Get best bid (highest price) and best ask (lowest price)
            best_bid = max(bid_levels) if bid_levels else (None, None)
            best_ask = min(ask_levels) if ask_levels else (None, None)

            return best_bid, best_ask
        except (ValueError, KeyError) as e:
            self._log(f"Error getting best levels: {e}", "ERROR")
            return (None, None), (None, None)

    def cleanup_old_order_book_levels(self):
        """Clean up old order book levels to prevent memory leaks."""
        try:
            # Keep only the top 100 levels on each side to prevent memory bloat
            max_levels = 100

            # Clean up bids (keep highest prices)
            if len(self.order_book["bids"]) > max_levels:
                sorted_bids = sorted(self.order_book["bids"].items(), reverse=True)
                self.order_book["bids"].clear()
                for price, size in sorted_bids[:max_levels]:
                    self.order_book["bids"][price] = size

            # Clean up asks (keep lowest prices)
            if len(self.order_book["asks"]) > max_levels:
                sorted_asks = sorted(self.order_book["asks"].items())
                self.order_book["asks"].clear()
                for price, size in sorted_asks[:max_levels]:
                    self.order_book["asks"][price] = size

        except Exception as e:
            self._log(f"Error cleaning up order book levels: {e}", "ERROR")

    async def reset_order_book(self):
        """Reset the order book state when reconnecting."""
        async with self.order_book_lock:
            self.order_book["bids"].clear()
            self.order_book["asks"].clear()
            self.snapshot_loaded = False
            self.best_bid = None
            self.best_ask = None
            self.order_book_offset = None
            self.order_book_sequence_gap = False

    def handle_order_update(self, order_data_list: List[Dict[str, Any]]):
        """Handle order update from WebSocket."""
        try:
            # Call the order update callback if it exists
            if self.order_update_callback:
                self.order_update_callback(order_data_list)

        except Exception as e:
            self._log(f"Error handling order update: {e}", "ERROR")

    async def connect(self):
        """Connect to Lighter WebSocket using custom implementation."""
        cleanup_counter = 0
        timeout_count = 0
        reconnect_delay = 1  # Start with 1 second delay
        max_reconnect_delay = 30  # Maximum delay of 30 seconds

        while True:
            try:
                # Reset order book state before connecting
                await self.reset_order_book()

                async with websockets.connect(self.ws_url) as self.ws:
                    # Subscribe to order book updates
                    await self.ws.send(json.dumps({
                        "type": "subscribe",
                        "channel": f"order_book/{self.market_index}"
                    }))

                    # Subscribe to account orders updates
                    account_orders_channel = f"account_orders/{self.market_index}/{self.account_index}"

                    # Get auth token for the subscription
                    try:
                        if self.lighter_client:
                            # Set auth token to expire in 10 minutes
                            ten_minutes_deadline = int(time.time() + 10 * 60)
                            auth_token, err = self.lighter_client.create_auth_token_with_expiry(ten_minutes_deadline)
                            if err is not None:
                                self._log(f"Failed to create auth token for account orders subscription: {err}", "WARNING")
                            else:
                                auth_message = {
                                    "type": "subscribe",
                                    "channel": account_orders_channel,
                                    "auth": auth_token
                                }
                                await self.ws.send(json.dumps(auth_message))
                                self._log("Subscribed to account orders with auth token (expires in 10 minutes)", "INFO")
                    except Exception as e:
                        self._log(f"Error creating auth token for account orders subscription: {e}", "WARNING")

                    self.running = True
                    # Reset reconnect delay on successful connection
                    reconnect_delay = 1
                    self._log("WebSocket connected using custom implementation", "INFO")

                    # Main message processing loop
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(self.ws.recv(), timeout=1)

                            try:
                                data = json.loads(msg)
                            except json.JSONDecodeError as e:
                                self._log(f"JSON parsing error in Lighter websocket: {e}", "ERROR")
                                continue

                            # Reset timeout counter on successful message
                            timeout_count = 0

                            async with self.order_book_lock:
                                if data.get("type") == "subscribed/order_book":
                                    # Initial snapshot - clear and populate the order book
                                    self.order_book["bids"].clear()
                                    self.order_book["asks"].clear()

                                    # Handle the initial snapshot
                                    order_book = data.get("order_book", {})
                                    if order_book and "offset" in order_book:
                                        # Set the initial offset from the snapshot
                                        self.order_book_offset = order_book["offset"]
                                        self._log(f"Initial order book offset set to: {self.order_book_offset}", "INFO")

                                    self.update_order_book("bids", order_book.get("bids", []))
                                    self.update_order_book("asks", order_book.get("asks", []))
                                    self.snapshot_loaded = True

                                    self._log(f"Lighter order book snapshot loaded with "
                                              f"{len(self.order_book['bids'])} bids and "
                                              f"{len(self.order_book['asks'])} asks", "INFO")

                                elif data.get("type") == "update/order_book" and self.snapshot_loaded:
                                    # Check for cutoff/incomplete updates first
                                    if not self.handle_order_book_cutoff(data):
                                        self._log("Skipping incomplete order book update", "WARNING")
                                        continue

                                    # Extract offset from the message
                                    order_book = data.get("order_book", {})
                                    if not order_book or "offset" not in order_book:
                                        self._log("Order book update missing offset, skipping", "WARNING")
                                        continue

                                    new_offset = order_book["offset"]

                                    # Validate offset sequence
                                    if not self.validate_order_book_offset(new_offset):
                                        # Sequence gap detected, try to request fresh snapshot first
                                        if self.order_book_sequence_gap:
                                            self._log("Sequence gap detected, requesting fresh snapshot...", "WARNING")
                                            # Release lock before network I/O
                                            break
                                        else:
                                            # For out-of-order updates, just continue
                                            continue

                                    # Update the order book with new data
                                    self.update_order_book("bids", order_book.get("bids", []))
                                    self.update_order_book("asks", order_book.get("asks", []))

                                    # Validate order book integrity after update
                                    if not self.validate_order_book_integrity():
                                        self._log("Order book integrity check failed, requesting fresh snapshot...", "WARNING")
                                        # Release lock before network I/O
                                        break

                                    # Get the best bid and ask levels
                                    (best_bid_price, best_bid_size), (best_ask_price, best_ask_size) = self.get_best_levels()

                                    # Update global variables
                                    if best_bid_price is not None:
                                        self.best_bid = best_bid_price
                                    if best_ask_price is not None:
                                        self.best_ask = best_ask_price

                                elif data.get("type") == "ping":
                                    # Respond to ping with pong
                                    await self.ws.send(json.dumps({"type": "pong"}))
                                elif data.get("type") == "update/account_orders":
                                    # Handle account orders updates
                                    orders = data.get("orders", {}).get(str(self.market_index), [])
                                    self.handle_order_update(orders)
                                elif data.get("type") == "update/order_book" and not self.snapshot_loaded:
                                    # Ignore updates until we have the initial snapshot
                                    continue
                                else:
                                    self._log(f"Unknown message type: {data.get('type', 'unknown')}", "DEBUG")

                            # Periodic cleanup outside the lock
                            cleanup_counter += 1
                            if cleanup_counter >= 1000:  # Clean up every 1000 messages
                                self.cleanup_old_order_book_levels()
                                cleanup_counter = 0

                            # Handle sequence gap and integrity issues outside the lock
                            if self.order_book_sequence_gap:
                                try:
                                    await self.request_fresh_snapshot()
                                    self.order_book_sequence_gap = False
                                except Exception as e:
                                    self._log(f"Failed to request fresh snapshot: {e}", "ERROR")
                                    self._log("Reconnecting due to sequence gap...", "WARNING")
                                    break

                        except asyncio.TimeoutError:
                            timeout_count += 1
                            if timeout_count % 30 == 0:
                                self._log(f"No message from Lighter websocket for {timeout_count} seconds "
                                          f"(abnormal behavior)", "WARNING")
                            continue
                        except websockets.exceptions.ConnectionClosed as e:
                            self._log(f"Lighter websocket connection closed: {e}", "WARNING")
                            self._log("Connection lost, will attempt to reconnect...", "INFO")
                            break  # Break inner loop to reconnect
                        except websockets.exceptions.WebSocketException as e:
                            self._log(f"Lighter websocket error: {e}", "ERROR")
                            break  # Break inner loop to reconnect
                        except Exception as e:
                            self._log(f"Error in Lighter websocket: {e}", "ERROR")
                            import traceback
                            self._log(f"Full traceback: {traceback.format_exc()}", "ERROR")
                            break  # Break inner loop to reconnect

            except Exception as e:
                self._log(f"Failed to connect to Lighter websocket: {e}", "ERROR")

            # Wait before reconnecting with exponential backoff
            if self.running:
                self._log(f"Waiting {reconnect_delay} seconds before reconnecting...", "INFO")
                await asyncio.sleep(reconnect_delay)
                # Exponential backoff: double the delay, but cap at max_reconnect_delay
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                self._log(f"Error closing websocket: {e}", "ERROR")
        self._log("WebSocket disconnected", "INFO")
