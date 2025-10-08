"""
Simplified Paradex exchange client implementation - L2 credentials only.
"""

import os
import asyncio
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .base import BaseExchangeClient, OrderResult, OrderInfo
from helpers.logger import TradingLogger


def patch_paradex_http_client():
    """Patch Paradex SDK HttpClient to suppress unwanted print statements."""
    try:
        from paradex_py.api.http_client import HttpClient

        def patched_request(self, url, http_method, params=None, payload=None, headers=None):
            res = self.client.request(
                method=http_method.value,
                url=url,
                params=params,
                json=payload,
                headers=headers,
            )
            if res.status_code >= 300:
                from paradex_py.api.models import ApiErrorSchema
                error = ApiErrorSchema().loads(res.text)
                raise Exception(error)
            try:
                return res.json()
            except ValueError:
                # Suppress the "No response request" print statement
                # This is expected for DELETE requests that don't return JSON
                # The original code would print: f"HttpClient: No response request({url}, {http_method.value})"
                pass

        # Replace the request method
        HttpClient.request = patched_request

    except ImportError:
        # Paradex SDK not available, skip patching
        pass


class ParadexClient(BaseExchangeClient):
    """Simplified Paradex exchange client - L2 credentials only."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Paradex client with L2 credentials only."""
        # Import paradex_py modules only when this class is instantiated
        from paradex_py import Paradex
        from paradex_py.environment import Environment, TESTNET, PROD
        from paradex_py.common.order import Order, OrderType, OrderSide, OrderStatus
        from paradex_py.api.ws_client import ParadexWebsocketChannel
        # from paradex_py.common.console_logging import console_logger  # Disabled to turn off native logging

        # Apply the patch when this class is instantiated
        patch_paradex_http_client()

        # Set config first
        self.config = config

        # Paradex credentials from environment - L1 address + L2 private key
        self.l1_address = os.getenv('PARADEX_L1_ADDRESS')
        self.l2_private_key_hex = os.getenv('PARADEX_L2_PRIVATE_KEY')
        self.l2_address = os.getenv('PARADEX_L2_ADDRESS')
        self.environment = os.getenv('PARADEX_ENVIRONMENT', 'prod')

        # Validate that required credentials are provided
        if not self.l1_address:
            raise ValueError(
                "PARADEX_L1_ADDRESS must be set in environment variables.\n"
                "This is your Ethereum L1 address."
            )

        if not self.l2_private_key_hex:
            raise ValueError(
                "PARADEX_L2_PRIVATE_KEY must be set in environment variables.\n"
                "Run 'python get_paradex_api_key.py' to generate L2 credentials from L1 credentials."
            )

        # Convert L2 private key from hex to int
        try:
            from starknet_py.common import int_from_hex
            self.l2_private_key = int_from_hex(self.l2_private_key_hex)
        except Exception as e:
            raise ValueError(f"Invalid L2 private key format: {e}")

        # Convert environment string to proper enum
        env_map = {
            'prod': PROD,
            'testnet': TESTNET,
            'nightly': TESTNET  # Use testnet for nightly
        }
        self.env = env_map.get(self.environment.lower(), TESTNET)

        # Initialize logger
        self.logger = TradingLogger(exchange="paradex", ticker=self.config.ticker, log_to_console=False)

        # Initialize Paradex client with L2 credentials only
        self._initialize_paradex_client()

        self._order_update_handler = None
        self.order_size_increment = ''

    def _initialize_paradex_client(self) -> None:
        """Initialize the Paradex client with L2 credentials only."""
        try:
            # Import paradex_py modules locally
            from paradex_py import Paradex

            # Initialize Paradex client without credentials first
            self.paradex = Paradex(
                env=self.env,
                logger=None  # Disabled native logging
            )

            # Initialize account with L2 private key
            self.paradex.init_account(
                l1_address=self.l1_address,
                l2_private_key=self.l2_private_key
            )

            # Log the L2 address being used
            if self.l2_address:
                self.logger.log(f"Using L2 address: {self.l2_address}", "INFO")

        except Exception as e:
            raise ValueError(f"Failed to initialize Paradex client: {e}")

    def _validate_config(self) -> None:
        """Validate Paradex configuration."""
        if not self.l2_private_key_hex:
            raise ValueError("L2 private key is required for trading operations")

    async def connect(self) -> None:
        """Connect to Paradex WebSocket."""
        is_connected = False
        while not is_connected:
            is_connected = await self.paradex.ws_client.connect()
            if not is_connected:
                self.logger.log("Connection failed, retrying in 1 second...", "WARN")
                await asyncio.sleep(1)
        # Wait a moment for connection to establish
        await asyncio.sleep(2)
        self._ws_connected = True

        # Setup WebSocket subscription for order updates if handler is set
        await self._setup_websocket_subscription()

    async def disconnect(self) -> None:
        """Disconnect from Paradex."""
        try:
            if hasattr(self, 'paradex') and self.paradex:
                # Close WebSocket connection properly
                if hasattr(self.paradex, 'ws_client') and self.paradex.ws_client:
                    await self.paradex.ws_client._close_connection()
                
                # Close HTTP client session if it exists
                if hasattr(self.paradex, 'http_client') and hasattr(self.paradex.http_client, 'client'):
                    if hasattr(self.paradex.http_client.client, 'close'):
                        await self.paradex.http_client.client.close()
                
                self._ws_connected = False
                self.logger.log("Paradex connection closed properly", "INFO")
        except Exception as e:
            self.logger.log(f"Error during Paradex disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "paradex"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

        async def order_update_handler(ws_channel, message):
            """Handle order updates from WebSocket."""
            from paradex_py.api.ws_client import ParadexWebsocketChannel

            params = message.get("params", {})
            data = params.get("data", {})

            if ws_channel == ParadexWebsocketChannel.ORDERS:
                # Extract order data
                order_id = data.get("id")
                status = data.get("status")
                side = data.get("side", "").lower()
                remaining_size = data.get("remaining_size")
                size = data.get("size")
                price = data.get("price")
                contract_id = data.get("market")
                filled_size = str(Decimal(size) - Decimal(remaining_size))
                if contract_id != self.config.contract_id:
                    return

                if order_id and status:
                    # Determine order type based on side
                    if side == self.config.close_order_side:
                        order_type = "CLOSE"
                    else:
                        order_type = "OPEN"

                    # Map Paradex status to our status
                    status_map = {
                        'NEW': 'OPEN',
                        'OPEN': 'OPEN',
                        'CLOSED': 'CANCELED' if data.get("cancel_reason") else 'FILLED'
                    }
                    mapped_status = status_map.get(status, status)

                    # Handle partially filled orders
                    if status == 'OPEN' and Decimal(filled_size) > 0:
                        mapped_status = "PARTIALLY_FILLED"

                    if mapped_status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED']:
                        if self._order_update_handler:
                            self._order_update_handler({
                                'order_id': order_id,
                                'side': side,
                                'order_type': order_type,
                                'status': mapped_status,
                                'size': size,
                                'price': price,
                                'contract_id': contract_id,
                                'filled_size': filled_size
                            })

        # Store the handler for later use
        self._ws_order_update_handler = order_update_handler

    async def _setup_websocket_subscription(self) -> None:
        """Setup WebSocket subscription for order updates."""
        if not hasattr(self, '_ws_order_update_handler'):
            return

        # Ensure WebSocket is connected
        if not hasattr(self, '_ws_connected') or not self._ws_connected:
            is_connected = False
            while not is_connected:
                is_connected = await self.paradex.ws_client.connect()
                if not is_connected:
                    self.logger.log("WebSocket connection failed, retrying in 1 second...", "WARN")
                    await asyncio.sleep(1)
            self._ws_connected = True
            self.logger.log("WebSocket connected for order monitoring", "INFO")

        # Subscribe to orders channel for the specific market
        from paradex_py.api.ws_client import ParadexWebsocketChannel

        contract_id = self.config.contract_id
        try:
            await self.paradex.ws_client.subscribe(
                ParadexWebsocketChannel.ORDERS,
                callback=self._ws_order_update_handler,
                params={"market": contract_id}
            )
            self.logger.log(f"Subscribed to order updates for {contract_id}", "INFO")
        except Exception as e:
            self.logger.log(f"Failed to subscribe to order updates: {e}", "ERROR")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def fetch_bbo_prices(self, contract_id: str) -> Dict[str, Any]:
        """Get orderbook using official SDK."""
        orderbook_data = self.paradex.api_client.fetch_orderbook(contract_id, {"depth": 1})
        if not orderbook_data:
            self.logger.log("Failed to get orderbook", "ERROR")
            raise ValueError("Failed to get orderbook")

        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        if not bids or not asks:
            self.logger.log("Failed to get bid/ask data", "ERROR")
            raise ValueError("Failed to get bid/ask data")

        # Get best bid and ask prices
        best_bid = Decimal(bids[0][0])
        best_ask = Decimal(asks[0][0])

        if best_bid <= 0 or best_ask <= 0:
            self.logger.log("Invalid bid/ask prices", "ERROR")
            raise ValueError("Invalid bid/ask prices")

        return best_bid, best_ask

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price of an order with Paradex using official SDK."""
        # Get current market prices
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)

        # Determine order side and price
        from paradex_py.common.order import OrderSide

        if direction == 'buy':
            # For buy orders, place slightly below best ask to ensure execution
            order_price = best_ask - self.config.tick_size
            order_side = OrderSide.Buy
        elif direction == 'sell':
            # For sell orders, place slightly above best bid to ensure execution
            order_price = best_bid + self.config.tick_size
            order_side = OrderSide.Sell
        else:
            raise Exception(f"[OPEN] Invalid direction: {direction}")

        order_price = self.round_to_tick(order_price)
        return order_price


    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _submit_order_with_retry(self, order) -> OrderResult:
        """Submit an order with Paradex using official SDK."""
        # Submit order using official SDK
        order_result = self.paradex.api_client.submit_order(order)

        # Extract order ID from response
        order_id = order_result.get('id')
        if not order_id:
            return OrderResult(success=False, error_message='No order ID in response')
        return order_result

    async def place_post_only_order(self, contract_id: str, quantity: Decimal, price: Decimal,
                                    side: str) -> OrderResult:
        """Place a post only order with Paradex using official SDK."""
        from paradex_py.common.order import Order, OrderType, OrderSide, OrderStatus

        # Create order using Paradex SDK
        order = Order(
            market=contract_id,
            order_type=OrderType.Limit,
            order_side=side,
            size=quantity.quantize(self.order_size_increment, rounding=ROUND_HALF_UP),
            limit_price=price,
            instruction="POST_ONLY"
        )

        order_result = self._submit_order_with_retry(order)

        order_id = order_result.get('id')
        order_status = order_result.get('status')
        order_status_start_time = time.time()
        order_info = await self.get_order_info(order_id)
        if order_info is not None:
            order_status = order_info.status
        while order_status in ['NEW'] and time.time() - order_status_start_time < 10:
            # Check order status after a short delay
            await asyncio.sleep(0.01)
            order_info = await self.get_order_info(order_id)
            if order_info is not None:
                order_status = order_info.status

        if order_status == 'NEW':
            raise Exception('Paradex Server Error: Order not processed after 10 seconds')
        else:
            return order_info

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with Paradex using official SDK."""
        attempt = 0
        from paradex_py.common.order import OrderSide
        while True:
            attempt += 1
            if attempt % 5 == 0:
                self.logger.log(f"[OPEN] Attempt {attempt} to place order", "INFO")
                active_orders = await self.get_active_orders(contract_id)
                active_open_orders = 0
                for order in active_orders:
                    if order.side == self.config.direction:
                        active_open_orders += 1
                if active_open_orders > 1:
                    self.logger.log(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}", "ERROR")
                    raise Exception(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}")

            if direction == 'buy':
                order_side = OrderSide.Buy
            elif direction == 'sell':
                order_side = OrderSide.Sell
            else:
                raise Exception(f"[OPEN] Invalid direction: {direction}")

            order_price = await self.get_order_price(direction)
            order_result = await self.place_post_only_order(contract_id, quantity, order_price, order_side)
            order_status = order_result.status
            order_id = order_result.order_id

            if order_status == 'CLOSED':
                remaining_size = order_result.remaining_size
                cancel_reason = order_result.cancel_reason
                if remaining_size == 0:
                    break
                elif cancel_reason == 'POST_ONLY_WOULD_CROSS':
                    continue
                else:
                    raise Exception(f"[OPEN] [{order_id}] Error placing order: {cancel_reason}")
            else:
                break

        if order_status in ['OPEN']:
            # Order successfully placed
            return OrderResult(
                success=True,
                order_id=order_id,
                side=direction,
                size=quantity,
                price=order_price,
                status=order_status
            )
        elif order_status == 'CLOSED' and remaining_size == 0:
            return OrderResult(
                success=True,
                order_id=order_id,
                side=direction,
                size=quantity,
                price=order_price,
                status=order_status
            )
        else:
            raise Exception(f"[OPEN] [{order_id}] Unexpected order status: {order_status}")

    async def place_market_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place a market order on Paradex for immediate execution."""
        from paradex_py.common.order import Order, OrderType, OrderSide
        if direction == 'buy':
            order_side = OrderSide.Buy
        elif direction == 'sell':
            order_side = OrderSide.Sell
        else:
            raise Exception(f"[MARKET] Invalid direction: {direction}")

        order = Order(
            market=contract_id,
            order_type=OrderType.Market,
            order_side=order_side,
            size=quantity.quantize(self.order_size_increment, rounding=ROUND_HALF_UP)
        )

        order_result = self._submit_order_with_retry(order)
        order_id = order_result.get('id')
        start_time = time.time()
        status = order_result.get('status', 'NEW')
        info = await self.get_order_info(order_id)
        if info is not None:
            status = info.status
        while status in ['NEW', 'OPEN'] and time.time() - start_time < 10:
            await asyncio.sleep(0.05)
            info = await self.get_order_info(order_id)
            if info is not None:
                status = info.status

        if info is None:
            raise Exception(f"[MARKET] Failed to fetch order info: {order_id}")

        return OrderResult(
            success=True,
            order_id=info.order_id,
            side=direction,
            size=info.size,
            price=info.price,
            status=info.status
        )

    async def place_aggressive_limit_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Fallback: aggressive limit crossing the spread to ensure execution (non post-only)."""
        from paradex_py.common.order import Order, OrderType, OrderSide
        best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

        if direction == 'buy':
            order_side = OrderSide.Buy
            limit_price = self.round_to_tick(best_ask + self.config.tick_size)
        elif direction == 'sell':
            order_side = OrderSide.Sell
            limit_price = self.round_to_tick(best_bid - self.config.tick_size)
        else:
            raise Exception(f"[LIMIT] Invalid direction: {direction}")

        order = Order(
            market=contract_id,
            order_type=OrderType.Limit,
            order_side=order_side,
            size=quantity.quantize(self.order_size_increment, rounding=ROUND_HALF_UP),
            limit_price=limit_price
        )

        order_result = self._submit_order_with_retry(order)
        order_id = order_result.get('id')
        start_time = time.time()
        status = order_result.get('status', 'NEW')
        info = await self.get_order_info(order_id)
        if info is not None:
            status = info.status
        while status in ['NEW', 'OPEN'] and time.time() - start_time < 10:
            await asyncio.sleep(0.05)
            info = await self.get_order_info(order_id)
            if info is not None:
                status = info.status

        if info is None:
            raise Exception(f"[LIMIT] Failed to fetch order info: {order_id}")

        return OrderResult(
            success=True,
            order_id=info.order_id,
            side=direction,
            size=info.size,
            price=info.price,
            status=info.status
        )

    async def _get_active_close_orders(self, contract_id: str) -> int:
        """Get active close orders for a contract using official SDK."""
        active_orders = await self.get_active_orders(contract_id)
        active_close_orders = 0
        for order in active_orders:
            if order.side == self.config.close_order_side:
                active_close_orders += 1
        return active_close_orders

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with Paradex using official SDK."""
        # Get current market prices
        attempt = 0
        active_close_orders = await self._get_active_close_orders(contract_id)
        while True:
            attempt += 1
            if attempt % 5 == 0:
                self.logger.log(f"[CLOSE] Attempt {attempt} to place order", "INFO")
                current_close_orders = await self._get_active_close_orders(contract_id)

                if current_close_orders - active_close_orders > 1:
                    self.logger.log(f"[CLOSE] ERROR: Active close orders abnormal: {active_close_orders}, {current_close_orders}", "ERROR")
                    raise Exception(f"[CLOSE] ERROR: Active close orders abnormal: {active_close_orders}, {current_close_orders}")
                else:
                    active_close_orders = current_close_orders
            # Get current market prices
            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

            # Convert side string to OrderSide enum
            from paradex_py.common.order import OrderSide
            order_side = OrderSide.Buy if side.lower() == 'buy' else OrderSide.Sell

            # Adjust order price based on market conditions and side
            if side.lower() == 'sell':
                # For sell orders, ensure price is above best bid to be a maker order
                if price <= best_bid:
                    adjusted_price = best_bid + self.config.tick_size
                else:
                    adjusted_price = price
            elif side.lower() == 'buy':
                # For buy orders, ensure price is below best ask to be a maker order
                if price >= best_ask:
                    adjusted_price = best_ask - self.config.tick_size
                else:
                    adjusted_price = price

            adjusted_price = self.round_to_tick(adjusted_price)
            order_result = await self.place_post_only_order(contract_id, quantity, adjusted_price, order_side)
            order_status = order_result.status
            order_id = order_result.order_id

            if order_status == 'CLOSED':
                remaining_size = order_result.remaining_size
                cancel_reason = order_result.cancel_reason
                if remaining_size == 0:
                    break
                elif cancel_reason == 'POST_ONLY_WOULD_CROSS':
                    continue
                else:
                    raise Exception(f"[CLOSE] [{order_id}] Error placing order: {cancel_reason}")
            else:
                break

        return OrderResult(
            success=True,
            order_id=order_id,
            side=side,
            size=quantity,
            price=adjusted_price,
            status=order_status
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with Paradex using official SDK."""
        try:
            # Cancel the order using official SDK
            self.paradex.api_client.cancel_order(order_id)
            return OrderResult(success=True)

        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from Paradex using official SDK."""
        try:
            # Get order by ID using official SDK
            order_data = self.paradex.api_client.fetch_order(order_id)
            size = Decimal(order_data.get('size', 0)).quantize(self.order_size_increment, rounding=ROUND_HALF_UP)
            remaining_size = Decimal(order_data.get('remaining_size', 0))
            return OrderInfo(
                order_id=order_data.get('id', ''),
                side=order_data.get('side', '').lower(),
                size=size,
                price=Decimal(order_data.get('price', 0)),
                status=order_data.get('status', ''),
                filled_size=size - remaining_size,
                remaining_size=remaining_size,
                cancel_reason=order_data.get('cancel_reason', '')
            )

        except Exception as e:
            self.logger.log(f"Error getting order info: {e}", "ERROR")
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_orders_with_retry(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get orders using official SDK."""
        orders_response = self.paradex.api_client.fetch_orders({"market": contract_id, "status": "OPEN"})
        if not orders_response or 'results' not in orders_response:
            self.logger.log("Failed to get orders", "ERROR")
            raise ValueError("Failed to get orders")

        return orders_response['results']

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract using official SDK."""
        order_list = await self._fetch_orders_with_retry(contract_id)

        # Filter orders for the specific market
        contract_orders = []
        for order in order_list:
            contract_orders.append(OrderInfo(
                order_id=order.get('id', ''),
                side=order.get('side', '').lower(),
                size=Decimal(order.get('remaining_size', 0)),  # FIXME: This is wrong. Should be size
                price=Decimal(order.get('price', 0)),
                status=order.get('status', ''),
                filled_size=Decimal(order.get('size', 0)) - Decimal(order.get('remaining_size', 0)),
                remaining_size=Decimal(order.get('remaining_size', 0))
            ))

        return contract_orders

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_positions_with_retry(self) -> List[Dict[str, Any]]:
        """Get positions using official SDK."""
        positions_response = self.paradex.api_client.fetch_positions()
        if not positions_response or 'results' not in positions_response:
            self.logger.log("Failed to get positions", "ERROR")
            raise ValueError("Failed to get positions")

        return positions_response['results']

    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        # Get account info which includes positions
        positions = await self._fetch_positions_with_retry()

        # Find position for current market
        for position in positions:
            if isinstance(position, dict) and position.get('market') == self.config.contract_id and position.get('status') == 'OPEN':
                if position.get('side') == 'LONG' and self.config.direction == 'sell':
                    raise ValueError("Long position found for sell direction")
                elif position.get('side') == 'SHORT' and self.config.direction == 'buy':
                    raise ValueError("Short position found for buy direction")

                return abs(Decimal(position.get('size', 0)).quantize(self.order_size_increment, rounding=ROUND_HALF_UP))

        return Decimal(0)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_market_with_retry(self, symbol: str) -> Dict[str, Any]:
        """Get market using official SDK."""
        market_response = self.paradex.api_client.fetch_markets({"market": symbol})
        if not market_response or 'results' not in market_response:
            self.logger.log("Failed to get markets", "ERROR")
            raise ValueError("Failed to get markets")

        if not market_response['results']:
            self.logger.log("Failed to get markets list", "ERROR")
            raise ValueError("Failed to get markets list")

        market = market_response['results'][0]

        return market

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_markets_summary_with_retry(self, symbol: str) -> Dict[str, Any]:
        """Get markets summary using official SDK."""
        market_summary_response = self.paradex.api_client.fetch_markets_summary({"market": symbol})
        if not market_summary_response or 'results' not in market_summary_response:
            self.logger.log("Failed to get markets summary", "ERROR")
            raise ValueError("Failed to get markets summary")
        market_summary = market_summary_response['results'][0]
        return market_summary

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID for a ticker."""
        ticker = self.config.ticker
        if len(ticker) == 0:
            self.logger.log("Ticker is empty", "ERROR")
            raise ValueError("Ticker is empty")

        symbol = f"{ticker}-USD-PERP"

        market = await self._fetch_market_with_retry(symbol)
        market_summary = await self._fetch_markets_summary_with_retry(symbol)

        last_price = Decimal(market_summary.get('mark_price', 0))

        # Set contract_id to market name (Paradex uses market names as identifiers)
        self.config.contract_id = symbol
        try:
            min_notional = Decimal(market.get('min_notional'))
        except Exception:
            self.logger.log("Failed to get min notional", "ERROR")
            raise ValueError("Failed to get min notional")

        try:
            self.order_size_increment = Decimal(market.get('order_size_increment'))
        except Exception:
            self.logger.log("Failed to get min quantity", "ERROR")
            raise ValueError("Failed to get min quantity")

        order_notional = last_price * self.config.quantity
        if order_notional < min_notional:
            self.logger.log(f"Order notional is less than min notional: {order_notional} < {min_notional}", "ERROR")
            raise ValueError(f"Order notional is less than min notional: {order_notional} < {min_notional}")

        try:
            self.config.tick_size = Decimal(market.get('price_tick_size'))
        except Exception:
            self.logger.log("Failed to get tick size", "ERROR")
            raise ValueError("Failed to get tick size")

        return self.config.contract_id, self.config.tick_size
