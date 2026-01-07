"""
Nado exchange client implementation.
"""

import os
import asyncio
import json
import traceback
import time
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from nado_protocol.client import create_nado_client, NadoClientMode
from nado_protocol.utils.subaccount import SubaccountParams
from nado_protocol.engine_client.types import OrderParams
from nado_protocol.utils.bytes32 import subaccount_to_hex
from nado_protocol.utils.expiration import get_expiration_timestamp
from nado_protocol.utils.math import to_x18, from_x18
from nado_protocol.utils.nonce import gen_order_nonce
from nado_protocol.utils.order import build_appendix, OrderType
from nado_protocol.engine_client.types.execute import CancelOrdersParams

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger


class NadoClient(BaseExchangeClient):
    """Nado exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Nado client."""
        super().__init__(config)

        # Nado credentials from environment
        self.private_key = os.getenv('NADO_PRIVATE_KEY')
        self.mode = os.getenv('NADO_MODE', 'MAINNET').upper()
        self.subaccount_name = os.getenv('NADO_SUBACCOUNT_NAME', 'default')
        self.symbol = self.config.ticker + '-PERP'
        
        if not self.private_key:
            raise ValueError("NADO_PRIVATE_KEY must be set in environment variables")

        # Map mode string to NadoClientMode enum
        mode_map = {
            'MAINNET': NadoClientMode.MAINNET,
            'DEVNET': NadoClientMode.DEVNET,
        }
        client_mode = mode_map.get(self.mode, NadoClientMode.MAINNET)

        # Initialize Nado client using official SDK
        self.client = create_nado_client(client_mode, self.private_key)
        self.owner = self.client.context.engine_client.signer.address

        # Initialize logger
        self.logger = TradingLogger(exchange="nado", ticker=self.config.ticker, log_to_console=False)

        self._order_update_handler = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_stop = asyncio.Event()

    def _validate_config(self) -> None:
        """Validate Nado configuration."""
        required_env_vars = ['NADO_PRIVATE_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    async def connect(self) -> None:
        """Connect to Nado (setup WebSocket if needed)."""
        # Nado SDK may handle WebSocket internally, but we'll set up order monitoring
        # For now, we'll use polling for order updates if WebSocket is not available
        self.logger.log("Connected to Nado", "INFO")

    async def disconnect(self) -> None:
        """Disconnect from Nado."""
        try:
            self._ws_stop.set()
            if self._ws_task and not self._ws_task.done():
                await self._ws_task
        except Exception as e:
            self.logger.log(f"Error during Nado disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "nado"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler
        # Nado SDK may provide WebSocket callbacks, but for now we'll use polling
        # This can be enhanced if Nado SDK provides WebSocket support

    @query_retry(default_return=(0, 0))
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/offer prices from Nado."""
        try:
            # Get order book depth
            order_book = self.client.context.engine_client.get_orderbook(ticker_id=contract_id, depth=1)
            
            if not order_book:
                return Decimal(0), Decimal(0)

            # Extract best bid and ask
            bids = order_book.bids
            asks = order_book.asks

            if not bids or not asks:
                return Decimal(0), Decimal(0)

            # Best bid is highest price (first in sorted list)
            best_bid = Decimal(str(bids[0][0]))
            # Best ask is lowest price (first in sorted list)
            best_ask = Decimal(str(asks[0][0]))

            return best_bid, best_ask

        except Exception as e:
            self.logger.log(f"Error fetching BBO prices: {e}", "ERROR")
            return Decimal(0), Decimal(0)

    def _get_product_id_from_contract(self, contract_id: str) -> int:
        """Convert contract_id (ticker) to product_id."""
        # Try to parse as int first
        try:
            return int(contract_id)
        except ValueError:
            # If it's a ticker like "BTC", we need to look it up
            # For now, use a simple mapping (this should be improved)
            ticker_to_product_id = {
                'BTC': 1,
                'ETH': 2,
            }
            ticker = contract_id.upper()
            return ticker_to_product_id.get(ticker, 1)  # Default to BTC

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price of an order with Nado."""
        best_bid, best_ask = await self.fetch_bbo_prices(self.symbol + '_USDT0')
        if best_bid <= 0 or best_ask <= 0:
            self.logger.log("Invalid bid/ask prices", "ERROR")
            raise ValueError("Invalid bid/ask prices")

        if direction == 'buy':
            # For buy orders, place slightly below best ask to ensure execution
            order_price = best_ask - self.config.tick_size
        else:
            # For sell orders, place slightly above best bid to ensure execution
            order_price = best_bid + self.config.tick_size
        return self.round_to_tick(order_price)

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with Nado using official SDK."""
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                best_bid, best_ask = await self.fetch_bbo_prices(self.symbol + '_USDT0')

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

                # Determine order price
                if direction == 'buy':
                    order_price = best_ask - self.config.tick_size
                else:
                    order_price = best_bid + self.config.tick_size

                # Build order parameters
                order = OrderParams(
                    sender=SubaccountParams(
                        subaccount_owner=self.owner,
                        subaccount_name=self.subaccount_name,
                    ),
                    priceX18=to_x18(float(str(order_price))),
                    amount=to_x18(float(str(quantity))) if direction == 'buy' else -to_x18(float(str(quantity))),
                    expiration=get_expiration_timestamp(60*60*24*30),
                    nonce=gen_order_nonce(),
                    appendix=build_appendix(order_type=OrderType.POST_ONLY)
                )

                # Place the order
                result = self.client.market.place_order({"product_id": int(contract_id), "order": order})

                if not result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = result.data.digest

                # Order successfully placed
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    side=direction,
                    size=quantity,
                    price=order_price,
                    status='OPEN'
                )

            except Exception as e:
                self.logger.log(f"Error placing open order: {e}", "ERROR")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return OrderResult(success=False, error_message=str(e))

        return OrderResult(success=False, error_message='Max retries exceeded')

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with Nado using official SDK."""
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                best_bid, best_ask = await self.fetch_bbo_prices(self.symbol + '_USDT0')

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

                # Adjust order price based on market conditions
                adjusted_price = price
                if side.lower() == 'sell':
                    # For sell orders, ensure price is above best bid to be a maker order
                    if price <= best_bid:
                        adjusted_price = best_bid + self.config.tick_size
                elif side.lower() == 'buy':
                    # For buy orders, ensure price is below best ask to be a maker order
                    if price >= best_ask:
                        adjusted_price = best_ask - self.config.tick_size

                # Build order parameters
                order = OrderParams(
                    sender=SubaccountParams(
                        subaccount_owner=self.owner,
                        subaccount_name=self.subaccount_name,
                    ),
                    priceX18=to_x18(float(adjusted_price)),
                    amount=to_x18(float(quantity)) if side.lower() == 'buy' else -to_x18(float(quantity)),
                    expiration=get_expiration_timestamp(3600),  # 1 hour expiration
                    nonce=gen_order_nonce(),
                    appendix=build_appendix(order_type=OrderType.POST_ONLY)
                )

                # Place the order
                result = self.client.market.place_order({"product_id": int(contract_id), "order": order})

                if not result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = result.data.digest
                if not order_id:
                    await asyncio.sleep(0.1)
                    return OrderResult(
                        success=True,
                        side=side,
                        size=quantity,
                        price=adjusted_price,
                        status='OPEN'
                    )

                # Order successfully placed
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    side=side,
                    size=quantity,
                    price=adjusted_price,
                    status='OPEN'
                )

            except Exception as e:
                self.logger.log(f"Error placing close order: {e}", "ERROR")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(0.1)
                    continue
                else:
                    return OrderResult(success=False, error_message=str(e))

        return OrderResult(success=False, error_message='Max retries exceeded for close order')

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with Nado using official SDK."""
        try:
            sender = subaccount_to_hex(SubaccountParams(
                    subaccount_owner=self.owner,
                    subaccount_name=self.subaccount_name,
                ))
            # Cancel order using Nado SDK
            result = self.client.market.cancel_orders(
                CancelOrdersParams(productIds=[self.config.contract_id], digests=[order_id], sender=sender)
            )

            if not result:
                return OrderResult(success=False, error_message='Failed to cancel order')

            order_info = await self.get_order_info(order_id)

            filled_size = order_info.filled_size if order_info is not None else Decimal(0)
            price = order_info.price if order_info is not None else Decimal(0)

            return OrderResult(success=True, filled_size=filled_size, price=price)

        except Exception as e:
            self.logger.log(f"Error canceling order: {e}", "ERROR")
            return OrderResult(success=False, error_message=str(e))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from Nado using official SDK."""
        try:
            # Get order info from Nado SDK
            # Note: Adjust method name if SDK uses different API
            order = self.client.context.engine_client.get_order(product_id=self.config.contract_id, digest=order_id)
            price_x18 = getattr(order, 'price_x18', None)
            amount_x18 = getattr(order, 'amount', None)
            unfilled_x18 = getattr(order, 'unfilled_amount', None)
            order_id = str(getattr(order, 'digest', None))

            size = Decimal(str(from_x18(amount_x18))) if amount_x18 else Decimal(0)
            remaining_size = Decimal(str(from_x18(unfilled_x18))) if unfilled_x18 else Decimal(0)
            filled_size = size - remaining_size

            side = 'buy' if size > 0 else 'sell'

            return OrderInfo(
                order_id=order_id,
                side=side,
                size=size,
                price=Decimal(str(from_x18(price_x18))),
                status='OPEN',
                filled_size=filled_size,
                remaining_size=remaining_size
            )

        except Exception as e:
            attempt = 0
            while attempt < 4:
                attempt += 1
                self.logger.log(f"Attempt {attempt} to get archived order info", "INFO")
                try:
                    order_result = self.client.context.indexer_client.get_historical_orders_by_digest([order_id])
                    if order_result.orders != []:
                        order = order_result.orders[0]
                        # Parse order data
                        price_x18 = getattr(order, 'price_x18', None)
                        amount_x18 = getattr(order, 'amount', None)
                        filled_x18 = getattr(order, 'base_filled', None)
                        order_id = str(getattr(order, 'digest', None))

                        if order.base_filled == order.amount:
                            status = 'FILLED'
                        else:
                            status = 'CANCELLED'

                        size = Decimal(str(from_x18(amount_x18))) if amount_x18 else Decimal(0)
                        filled_size = Decimal(str(from_x18(filled_x18))) if filled_x18 else Decimal(0)
                        remaining_size = size - filled_size

                        side = 'buy' if size > 0 else 'sell'

                        return OrderInfo(
                            order_id=order_id,
                            side=side,
                            size=size,
                            price=Decimal(str(from_x18(price_x18))),
                            status=status,
                            filled_size=filled_size,
                            remaining_size=remaining_size
                        )
                except Exception as e:
                    self.logger.log(f"Error getting order info after retry: {e}", "ERROR")

                await asyncio.sleep(0.5)

            return OrderInfo(
                order_id=order_id,
                side='',
                size=Decimal(0),
                price=Decimal(0),
                status='CANCELLED',
                filled_size=Decimal(0),
                remaining_size=Decimal(0)
            )

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract using official SDK."""
        try:            
            # Get subaccount open orders from Nado SDK
            sender = subaccount_to_hex(SubaccountParams(
                    subaccount_owner=self.owner,
                    subaccount_name=self.subaccount_name,
                ))

            orders_data = self.client.market.get_subaccount_open_orders(
                product_id=contract_id,
                sender=sender)

            if not orders_data:
                return []

            orders = []
            # Handle both list and object with orders attribute
            order_list = orders_data if isinstance(orders_data, list) else getattr(orders_data, 'orders', [])

            for order in order_list:
                price_x18 = getattr(order, 'price_x18', None)
                amount_x18 = getattr(order, 'amount', None)
                unfilled_x18 = getattr(order, 'unfilled_amount', None)
                order_id = str(getattr(order, 'digest', None))

                size = Decimal(str(from_x18(amount_x18))) if amount_x18 else Decimal(0)
                remaining_size = Decimal(str(from_x18(unfilled_x18))) if unfilled_x18 else Decimal(0)
                filled_size = size - remaining_size

                side = 'buy' if size > 0 else 'sell'

                orders.append(OrderInfo(
                    order_id=str(order_id),
                    side=side,
                    size=size,
                    price=Decimal(str(from_x18(price_x18))) if price_x18 else Decimal(0),
                    status='OPEN',
                    filled_size=filled_size,
                    remaining_size=remaining_size
                ))

            return orders

        except Exception as e:
            self.logger.log(f"Error getting active orders: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return []

    @query_retry(default_return=0)
    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        try:
            # Get subaccount identifier
            resolved_subaccount = subaccount_to_hex(self.client.context.signer.address, self.subaccount_name)
            
            # Get isolated positions from Nado SDK (requires subaccount parameter)
            account_data = self.client.context.engine_client.get_subaccount_info(resolved_subaccount)
            position_data = account_data.perp_balances

            # Find position for current contract
            product_id = self.config.contract_id
            
            for position in position_data:
                if position.product_id == product_id:
                    position_size = position.balance.amount
                    return Decimal(str(from_x18(position_size)))

            return Decimal(0)

        except Exception as e:
            self.logger.log(f"Error getting account positions: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return Decimal(0)

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker
        if len(ticker) == 0:
            self.logger.log("Ticker is empty", "ERROR")
            raise ValueError("Ticker is empty")

        try:
            # Get markets/products from Nado SDK
            symbols = self.client.market.get_all_product_symbols()
            product_id = None
            for symbol in symbols:
                symbol_str = symbol.symbol if hasattr(symbol, 'symbol') else str(symbol)
                if symbol_str == f"{ticker.upper()}-PERP":
                    product_id = symbol.product_id if hasattr(symbol, 'product_id') else symbol
                    self.config.contract_id = product_id
                    break
            all_markets = self.client.market.get_all_engine_markets()
            markets = all_markets.perp_products
            current_market = None
            for market in markets:
                if market.product_id == product_id:
                    current_market = market
                    break

            if current_market is None:
                self.logger.log(f"Failed to get market for ticker {ticker}", "ERROR")
                raise ValueError(f"Failed to get market for ticker {ticker}")

            # Get tick size and min quantity
            tick_size_x18 = current_market.book_info.price_increment_x18
            min_quantity_x18 = current_market.book_info.size_increment

            self.config.tick_size = Decimal(str(from_x18(tick_size_x18)))

            min_quantity = Decimal(str(from_x18(min_quantity_x18)))

            if self.config.quantity < min_quantity:
                self.logger.log(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}", "ERROR")
                raise ValueError(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}")

            return self.config.contract_id, self.config.tick_size

        except Exception as e:
            self.logger.log(f"Error getting contract attributes: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise

