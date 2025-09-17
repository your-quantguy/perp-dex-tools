"""
Simplified Paradex exchange client implementation - L2 credentials only.
"""

import os
import asyncio
import json
import time
import traceback
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from paradex_py import Paradex
from paradex_py.environment import Environment
from paradex_py.common.order import Order, OrderType, OrderSide, OrderStatus
from paradex_py.api.ws_client import ParadexWebsocketChannel
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .base import BaseExchangeClient, OrderResult, OrderInfo
from helpers.logger import TradingLogger


class ParadexClient(BaseExchangeClient):
    """Simplified Paradex exchange client - L2 credentials only."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Paradex client with L2 credentials only."""
        super().__init__(config)

        # Paradex credentials from environment - L1 address + L2 private key
        self.l1_address = os.getenv('PARADEX_L1_ADDRESS')
        self.l2_private_key = os.getenv('PARADEX_L2_PRIVATE_KEY')
        self.l2_address = os.getenv('PARADEX_L2_ADDRESS')
        self.environment = os.getenv('PARADEX_ENVIRONMENT', 'testnet')

        # Validate that required credentials are provided
        if not self.l1_address:
            raise ValueError(
                "PARADEX_L1_ADDRESS must be set in environment variables.\n"
                "This is your Ethereum L1 address."
            )
        
        if not self.l2_private_key:
            raise ValueError(
                "PARADEX_L2_PRIVATE_KEY must be set in environment variables.\n"
                "Run 'python get_paradex_api_key.py' to generate L2 credentials from L1 credentials."
            )

        # Convert environment string to string (Paradex SDK expects strings, not enums)
        env_map = {
            'prod': 'prod',
            'testnet': 'testnet',
            'nightly': 'nightly'
        }
        self.env = env_map.get(self.environment.lower(), 'testnet')

        # Initialize logger
        self.logger = TradingLogger(exchange="paradex", ticker=self.config.ticker, log_to_console=False)

        # Initialize Paradex client with L2 credentials only
        self._initialize_paradex_client()

        self._order_update_handler = None
        self.order_size_increment=''

    def _initialize_paradex_client(self) -> None:
        """Initialize the Paradex client with L2 credentials only."""
        try:
            # Initialize Paradex client (without credentials first)
            self.paradex = Paradex(env=self.env)
            
            # Initialize account with L1 address and L2 private key
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
        if not self.l2_private_key:
            raise ValueError("L2 private key is required for trading operations")

    async def connect(self) -> None:
        """Connect to Paradex WebSocket."""
        await self.paradex.ws_client.connect()
        # Wait a moment for connection to establish
        await asyncio.sleep(2)

    async def disconnect(self) -> None:
        """Disconnect from Paradex."""
        try:
            if hasattr(self, 'paradex') and self.paradex:
                await self.paradex.ws_client.disconnect()
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
            try:
                # Parse the message structure
                if isinstance(message, dict) and "params" in message:
                    params = message["params"]
                    data = params.get("data", {})
                    
                    # Extract order data
                    order_id = data.get("id")
                    status = data.get("status")
                    side = data.get("side", "").lower()
                    filled_size = data.get("filled_size", 0)
                    size = data.get("size", 0)
                    price = data.get("price", 0)
                    market = data.get("market", "")

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
                                    'market': market,
                                    'filled_size': filled_size
                                })

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Subscribe to orders channel for the specific market
        market = self.config.contract_id
        asyncio.create_task(
            self.paradex.ws_client.subscribe(
                ParadexWebsocketChannel.ORDERS,
                callback=order_update_handler,
                params={"market": market}
            )
        )


    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_orderbook_with_retry(self, market: str) -> Dict[str, Any]:
        """Get orderbook using official SDK."""
        orderbook_data = self.paradex.api_client.fetch_orderbook(market, {"depth": 1})
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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _submit_order_with_retry(self, order: Order) -> OrderResult:
        """Submit an order with Paradex using official SDK."""
            # Submit order using official SDK
            order_result = self.paradex.api_client.submit_order(order)

            # Extract order ID from response
            order_id = order_result.get('id')
            if not order_id:
                return OrderResult(success=False, error_message='No order ID in response')
            return order_result


    async def place_post_only_order(self, market: str, quantity: Decimal, price: Decimal,
                                    side: str) -> OrderResult:
        """Place a post only order with Paradex using official SDK."""

        # Create order using Paradex SDK
        order = Order(
            market=market,
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
        while order_status in ['NEW'] and time.time() - order_status_start_time < 10:
            # Check order status after a short delay
            await asyncio.sleep(0.01)
            order_info = await self.get_order_info(order_id)
            order_status = order_info.get('status')

        if order_status == 'NEW':
            raise Exception('Paradex Server Error: Order not processed after 10 seconds')
        else:
            return order_result

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with Paradex using official SDK."""
        market = contract_id
        while True:
            # Get current market prices
            best_bid, best_ask = await self._fetch_orderbook_with_retry(market)
            
            # Determine order side and price
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
            order_result = await self.place_post_only_order(market, quantity, order_price, order_side)
            order_status = order_result.get('status')
            order_id = order_result.get('id')

            if order_status == 'CLOSED':
                remaining_size = Decimal(order_result.get('remaining_size'))
                cancel_reason = order_result.get('cancel_reason')
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
        else:
            raise Exception(f"[OPEN] [{order_id}] Unexpected order status: {order_status}")


    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with Paradex using official SDK."""
        # Get current market prices
        market = self.config.contract_id
        while True:
            # Get current market prices
            best_bid, best_ask = await self._fetch_orderbook_with_retry(market)

            # Convert side string to OrderSide enum
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
            order_result = await self.place_post_only_order(market, quantity, adjusted_price, order_side)
            order_status = order_result.get('status')
            order_id = order_result.get('id')

            if order_status == 'CLOSED':
                remaining_size = Decimal(order_result.get('remaining_size'))
                cancel_reason = order_result.get('cancel_reason')
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
            order_result = self.paradex.api_client.fetch_order(order_id)

            if not order_result or 'data' not in order_result:
                return None

            order_data = order_result['data']
            return OrderInfo(
                order_id=order_data.get('id', ''),
                side=order_data.get('side', '').lower(),
                size=Decimal(order_data.get('size', 0)).quantize(self.order_size_increment, rounding=ROUND_HALF_UP),
                price=Decimal(order_data.get('price', 0)),
                status=order_data.get('status', ''),
                filled_size=Decimal(order_data.get('filled_size', 0)),
                remaining_size=Decimal(order_data.get('remaining_size', 0))
            )

        except Exception:
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _fetch_orders_with_retry(self, market: str) -> List[Dict[str, Any]]:
        """Get orders using official SDK."""
        orders_response = self.paradex.api_client.fetch_orders({"market": market, "status": "OPEN"})
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
                size=Decimal(order.get('remaining_size', 0)), #FIXME: This is wrong. Should be size
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
        positions_response = await self.paradex.api_client.fetch_positions()
        if not positions_response or 'results' not in positions_response:
            self.logger.log("Failed to get positions", "ERROR")
            raise ValueError("Failed to get positions")

        return positions_response['results']

    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        # Get account info which includes positions
        positions = await self._fetch_positions_with_retry()

        # Find position for current market
        market = self.config.contract_id
        for position in positions:
            if isinstance(position, dict) and position.get('market') == market and position.get('status') == 'OPEN':
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

        order_notional = last_price * self.order_size_increment
        if order_notional < min_notional:
            self.logger.log(f"Order notional is less than min notional: {order_notional} < {min_notional}", "ERROR")
            raise ValueError(f"Order notional is less than min notional: {order_notional} < {min_notional}")

        try:
            self.config.tick_size = Decimal(market.get('price_tick_size'))
        except Exception:
            self.logger.log("Failed to get tick size", "ERROR")
            raise ValueError("Failed to get tick size")

        return self.config.contract_id, self.config.tick_size
