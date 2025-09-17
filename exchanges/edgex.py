"""
EdgeX exchange client implementation.
"""

import os
import asyncio
import json
import traceback
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from edgex_sdk import Client, OrderSide, WebSocketManager, CancelOrderParams, GetOrderBookDepthParams, GetActiveOrderParams

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger


class EdgeXClient(BaseExchangeClient):
    """EdgeX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize EdgeX client."""
        super().__init__(config)

        # EdgeX credentials from environment
        self.account_id = os.getenv('EDGEX_ACCOUNT_ID')
        self.stark_private_key = os.getenv('EDGEX_STARK_PRIVATE_KEY')
        self.base_url = os.getenv('EDGEX_BASE_URL', 'https://pro.edgex.exchange')
        self.ws_url = os.getenv('EDGEX_WS_URL', 'wss://quote.edgex.exchange')

        if not self.account_id or not self.stark_private_key:
            raise ValueError("EDGEX_ACCOUNT_ID and EDGEX_STARK_PRIVATE_KEY must be set in environment variables")

        # Initialize EdgeX client using official SDK
        self.client = Client(
            base_url=self.base_url,
            account_id=int(self.account_id),
            stark_private_key=self.stark_private_key
        )

        # Initialize WebSocket manager using official SDK
        self.ws_manager = WebSocketManager(
            base_url=self.ws_url,
            account_id=int(self.account_id),
            stark_pri_key=self.stark_private_key
        )

        # Initialize logger using the same format as helpers
        self.logger = TradingLogger(exchange="edgex", ticker=self.config.ticker, log_to_console=False)

        self._order_update_handler = None

    def _validate_config(self) -> None:
        """Validate EdgeX configuration."""
        required_env_vars = ['EDGEX_ACCOUNT_ID', 'EDGEX_STARK_PRIVATE_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    async def connect(self) -> None:
        """Connect to EdgeX WebSocket."""
        self.ws_manager.connect_private()
        # Wait a moment for connection to establish
        await asyncio.sleep(2)

    async def disconnect(self) -> None:
        """Disconnect from EdgeX."""
        try:
            if hasattr(self, 'client') and self.client:
                await self.client.close()
            if hasattr(self, 'ws_manager'):
                self.ws_manager.disconnect_all()
        except Exception as e:
            self.logger.log(f"Error during EdgeX disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "edgex"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Parse the message structure
                if isinstance(message, str):
                    message = json.loads(message)

                # Check if this is a trade-event with ORDER_UPDATE
                content = message.get("content", {})
                event = content.get("event", "")
                if event == "ORDER_UPDATE":
                    # Extract order data from the nested structure
                    data = content.get('data', {})
                    orders = data.get('order', [])

                    if orders and len(orders) > 0:
                        order = orders[0]  # Get the first order
                        order_id = order.get('id')
                        status = order.get('status')
                        side = order.get('side', '').lower()
                        filled_size = order.get('cumMatchSize')

                        if side == self.config.close_order_side:
                            order_type = "CLOSE"
                        else:
                            order_type = "OPEN"

                        # edgex returns TWO filled events for the same order; take the first one
                        if status == "FILLED" and len(data.get('collateral', [])):
                            return

                        # ignore canceled close orders
                        if status == "CANCELED" and order_type == "CLOSE":
                            return

                        # edgex returns partially filled events as "OPEN" orders
                        if status == "OPEN" and Decimal(filled_size) > 0:
                            status = "PARTIALLY_FILLED"

                        if status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED']:
                            if self._order_update_handler:
                                self._order_update_handler({
                                    'order_id': order_id,
                                    'side': side,
                                    'order_type': order_type,
                                    'status': status,
                                    'size': order.get('size'),
                                    'price': order.get('price'),
                                    'contract_id': order.get('contractId'),
                                    'filled_size': filled_size
                                })

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        try:
            private_client = self.ws_manager.get_private_client()
            private_client.on_message("trade-event", order_update_handler)
        except Exception as e:
            self.logger.log(f"Could not add trade-event handler: {e}", "ERROR")

    @query_retry(default_return=(0, 0))
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        depth_params = GetOrderBookDepthParams(contract_id=contract_id, limit=15)
        order_book = await self.client.quote.get_order_book_depth(depth_params)
        order_book_data = order_book['data']

        # Get the first (and should be only) order book entry
        order_book_entry = order_book_data[0]

        # Extract bids and asks from the entry
        bids = order_book_entry.get('bids', [])
        asks = order_book_entry.get('asks', [])

        # Best bid is the highest price someone is willing to buy at
        best_bid = Decimal(bids[0]['price']) if bids and len(bids) > 0 else 0
        # Best ask is the lowest price someone is willing to sell at
        best_ask = Decimal(asks[0]['price']) if asks and len(asks) > 0 else 0
        return best_bid, best_ask

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with EdgeX using official SDK with retry logic for POST_ONLY rejections."""
        max_retries = 15
        retry_count = 0

        while retry_count < max_retries:
            try:
                best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

                if direction == 'buy':
                    # For buy orders, place slightly below best ask to ensure execution
                    order_price = best_ask - self.config.tick_size
                    side = OrderSide.BUY
                else:
                    # For sell orders, place slightly above best bid to ensure execution
                    order_price = best_bid + self.config.tick_size
                    side = OrderSide.SELL

                # Place the order using official SDK (post-only to ensure maker order)
                order_result = await self.client.create_limit_order(
                    contract_id=contract_id,
                    size=str(quantity),
                    price=str(self.round_to_tick(order_price)),
                    side=side,
                    post_only=True
                )

                if not order_result or 'data' not in order_result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = order_result['data'].get('orderId')
                if not order_id:
                    return OrderResult(success=False, error_message='No order ID in response')

                # Check order status after a short delay to see if it was rejected
                await asyncio.sleep(0.01)
                order_info = await self.get_order_info(order_id)

                if order_info:
                    if order_info.status == 'CANCELED':
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            continue
                        else:
                            return OrderResult(success=False, error_message=f'Order rejected after {max_retries} attempts')
                    elif order_info.status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED']:
                        # Order successfully placed
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            side=side.value,
                            size=quantity,
                            price=order_price,
                            status=order_info.status
                        )
                    else:
                        return OrderResult(success=False, error_message=f'Unexpected order status: {order_info.status}')
                else:
                    # Assume order is successful if we can't get info
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        side=side.value,
                        size=quantity,
                        price=order_price
                    )

            except Exception as e:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(0.1)  # Wait before retry
                    continue
                else:
                    return OrderResult(success=False, error_message=str(e))

        return OrderResult(success=False, error_message='Max retries exceeded')

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with EdgeX using official SDK with retry logic for POST_ONLY rejections."""
        max_retries = 15
        retry_count = 0

        while retry_count < max_retries:
            try:
                best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

                # Convert side string to OrderSide enum
                order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

                # Adjust order price based on market conditions and side
                adjusted_price = price

                if side.lower() == 'sell':
                    # For sell orders, ensure price is above best bid to be a maker order
                    if price <= best_bid:
                        adjusted_price = best_bid + self.config.tick_size
                elif side.lower() == 'buy':
                    # For buy orders, ensure price is below best ask to be a maker order
                    if price >= best_ask:
                        adjusted_price = best_ask - self.config.tick_size

                adjusted_price = self.round_to_tick(adjusted_price)
                # Place the order using official SDK (post-only to avoid taker fees)
                order_result = await self.client.create_limit_order(
                    contract_id=contract_id,
                    size=str(quantity),
                    price=str(adjusted_price),
                    side=order_side,
                    post_only=True
                )

                if not order_result or 'data' not in order_result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = order_result['data'].get('orderId')
                if not order_id:
                    return OrderResult(success=False, error_message='No order ID in response')

                # Check order status after a short delay to see if it was rejected
                await asyncio.sleep(0.01)
                order_info = await self.get_order_info(order_id)

                if order_info:
                    if order_info.status == 'CANCELED':
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            continue
                        else:
                            return OrderResult(success=False, error_message=f'Close order rejected after {max_retries} attempts')
                    elif order_info.status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED']:
                        # Order successfully placed
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            side=side,
                            size=quantity,
                            price=adjusted_price,
                            status=order_info.status
                        )
                    else:
                        return OrderResult(success=False, error_message=f'Unexpected close order status: {order_info.status}')
                else:
                    # Assume order is successful if we can't get info
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        side=side,
                        size=quantity,
                        price=adjusted_price
                    )

            except Exception as e:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(0.1)  # Wait before retry
                    continue
                else:
                    return OrderResult(success=False, error_message=str(e))

        return OrderResult(success=False, error_message='Max retries exceeded for close order')

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with EdgeX using official SDK."""
        try:
            # Create cancel parameters using official SDK
            cancel_params = CancelOrderParams(order_id=order_id)

            # Cancel the order using official SDK
            cancel_result = await self.client.cancel_order(cancel_params)

            if not cancel_result or 'data' not in cancel_result:
                return OrderResult(success=False, error_message='Failed to cancel order')

            return OrderResult(success=True)

        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from EdgeX using official SDK."""
        # Use the newly created get_order_by_id method
        order_result = await self.client.order.get_order_by_id(order_id_list=[order_id])

        if not order_result or 'data' not in order_result:
            return None

        # The API returns a list of orders, get the first (and should be only) one
        order_list = order_result['data']
        if order_list and len(order_list) > 0:
            order_data = order_list[0]
            return OrderInfo(
                order_id=order_data.get('id', ''),
                side=order_data.get('side', '').lower(),
                size=Decimal(order_data.get('size', 0)),
                price=Decimal(order_data.get('price', 0)),
                status=order_data.get('status', ''),
                filled_size=Decimal(order_data.get('cumMatchSize', 0)),
                remaining_size=Decimal(order_data.get('size', 0)) - Decimal(order_data.get('cumMatchSize', 0))
            )

        return None

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract using official SDK."""
        # Get active orders using official SDK
        params = GetActiveOrderParams(size="100", offset_data="")
        active_orders = await self.client.get_active_orders(params)

        if not active_orders or 'data' not in active_orders:
            return []

        # Filter orders for the specific contract and ensure they are dictionaries
        # The API returns orders under 'dataList' key, not 'orderList'
        order_list = active_orders['data'].get('dataList', [])
        contract_orders = []

        for order in order_list:
            if isinstance(order, dict) and order.get('contractId') == contract_id:
                contract_orders.append(OrderInfo(
                    order_id=order.get('id', ''),
                    side=order.get('side', '').lower(),
                    size=Decimal(order.get('size', 0)),
                    price=Decimal(order.get('price', 0)),
                    status=order.get('status', ''),
                    filled_size=Decimal(order.get('cumMatchSize', 0)),
                    remaining_size=Decimal(order.get('size', 0)) - Decimal(order.get('cumMatchSize', 0))
                ))

        return contract_orders

    @query_retry(default_return=0)
    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        positions_data = await self.client.get_account_positions()
        if not positions_data or 'data' not in positions_data:
            self.logger.log("No positions or failed to get positions", "WARNING")
            position_amt = 0
        else:
            # The API returns positions under data.positionList
            positions = positions_data.get('data', {}).get('positionList', [])
            if positions:
                # Find position for current contract
                position = None
                for p in positions:
                    if isinstance(p, dict) and p.get('contractId') == self.config.contract_id:
                        position = p
                        break

                if position:
                    position_amt = abs(Decimal(position.get('openSize', 0)))
                else:
                    position_amt = 0
            else:
                position_amt = 0
        return position_amt

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID for a ticker."""
        ticker = self.config.ticker
        if len(ticker) == 0:
            self.logger.log("Ticker is empty", "ERROR")
            raise ValueError("Ticker is empty")

        response = await self.client.get_metadata()
        data = response.get('data', {})
        if not data:
            self.logger.log("Failed to get metadata", "ERROR")
            raise ValueError("Failed to get metadata")

        contract_list = data.get('contractList', [])
        if not contract_list:
            self.logger.log("Failed to get contract list", "ERROR")
            raise ValueError("Failed to get contract list")

        current_contract = None
        for c in contract_list:
            if c.get('contractName') == ticker+'USD':
                current_contract = c
                break

        if not current_contract:
            self.logger.log("Failed to get contract ID for ticker", "ERROR")
            raise ValueError("Failed to get contract ID for ticker")

        self.config.contract_id = current_contract.get('contractId')
        min_quantity = Decimal(current_contract.get('minOrderSize'))
        if self.config.quantity < min_quantity:
            self.logger.log(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}", "ERROR")
            raise ValueError(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}")

        self.config.tick_size = Decimal(current_contract.get('tickSize'))

        return self.config.contract_id, self.config.tick_size
