# coding: utf-8
"""
Apex exchange client implementation.
"""

import os
import asyncio
import json
import time
import traceback
import types
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from apexomni import constants as apex_constants, FailedRequestError
from apexomni._websocket_stream import _ApexWebSocketManager, PRIVATE_WSS
from apexomni.http_private_sign import HttpPrivateSign
from apexomni.websocket_api import WebSocket as ApexWebSocketClient

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger


class ApexClient(BaseExchangeClient):
    """Apex exchange client implementation"""

    def __init__(self, config: Dict[str, any]):
        """Initialize Apex client."""
        super().__init__(config)

        # Apex credentials from environment
        self.api_key = os.getenv('APEX_API_KEY')
        self.api_key_passphrase = os.getenv('APEX_API_KEY_PASSPHRASE')
        self.api_key_secret = os.getenv('APEX_API_KEY_SECRET')
        self.omni_key_seed = os.getenv('APEX_OMNI_KEY_SEED')
        self.api_key_credentials = {
            'key': self.api_key, 'secret': self.api_key_secret,
            'passphrase': self.api_key_passphrase
        }
        self.environment = os.getenv('APEX_ENVIRONMENT', 'prod')

        assert self.environment in {'prod', 'test'}, 'Apex environment can only be prod or test.'
        if self.environment == 'prod':
            self.http_base_url = apex_constants.APEX_OMNI_HTTP_MAIN
            self.ws_base_url = apex_constants.APEX_OMNI_WS_MAIN
            self.network_id = apex_constants.NETWORKID_OMNI_MAIN_ARB
        else:
            self.http_base_url = apex_constants.APEX_OMNI_HTTP_TEST
            self.ws_base_url = apex_constants.APEX_OMNI_WS_TEST
            self.network_id = apex_constants.NETWORKID_TEST

        # Initialize logger
        self.logger = TradingLogger(exchange="apex", ticker=self.config.ticker, log_to_console=False)

        # Initialize Apex clients
        self._initialize_apex_clients()

        self._order_update_handler = None
        self.account_handler = None

        # --- reconnection state ---
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_stop = asyncio.Event()
        self._ws_disconnected = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _initialize_apex_clients(self) -> None:
        """Initialize Apex REST and Websocket clients"""
        try:
            self.rest_client = HttpPrivateSign(
                self.http_base_url, network_id=self.network_id,
                zk_seeds=self.omni_key_seed, zk_l2Key='',
                api_key_credentials=self.api_key_credentials
            )

            # According to official SDK, need to request for user data first:
            # https://github.com/ApeX-Protocol/apexpro-openapi/blob/main/README_V3.md#zkkey-sign-create-order-method
            # This will store user info in HttpPrivateSign instance
            self.rest_client.configs_v3()
            self.rest_client.get_account_v3()

            self.ws_client = ApexWebSocketClient(
                endpoint=self.ws_base_url,
                api_key_credentials=self.api_key_credentials
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Apex client: {e}")

    def _validate_config(self) -> None:
        """Validate Apex configuration."""
        required_env_vars = ['APEX_API_KEY', 'APEX_API_KEY_PASSPHRASE',
                             'APEX_API_KEY_SECRET', 'APEX_OMNI_KEY_SEED']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        return

    async def connect(self) -> None:
        """Connect to Apex Websocket with auto-reconnect."""
        self._loop = asyncio.get_running_loop()
        try:
            # Patch _on_close/_on_open (SDK did nothing but logging)
            self.ws_client._on_close = types.MethodType(
                lambda _: self._loop.call_soon_threadsafe(self._ws_disconnected.set),
                self.ws_client
            )
            self.ws_client._on_open = types.MethodType(
                lambda _: self.logger.log("[WS] connected", "INFO"),
                self.ws_client
            )
        except Exception as e:
            self.logger.log(f"[WS] failed to set hooks: {e}", "ERROR")

        if not self._ws_task or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._run_private_ws())

    async def _run_private_ws(self):
        """Tiny reconnect loop with exponential backoff."""
        backoff = 1.0
        while not self._ws_stop.is_set():
            try:
                # connect
                self.ws_client.ws_private = _ApexWebSocketManager(**self.ws_client.kwargs)
                self.ws_client.ws_private._connect(self.ws_client.endpoint + PRIVATE_WSS)
                self.ws_client.account_info_stream_v3(self.account_handler)
                self.logger.log("[WS] private connected", "INFO")
                backoff = 1.0

                # wait until either disconnect or stop
                self._ws_disconnected.clear()
                done, _ = await asyncio.wait(
                    {asyncio.create_task(self._ws_stop.wait()),
                     asyncio.create_task(self._ws_disconnected.wait()),},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._ws_stop.is_set():
                    break

                self.logger.log(
                    "[WS] disconnected; attempting to reconnect...", "WARNING"
                )
            except Exception as e:
                self.logger.log(f"[WS] connect error: {e}", "ERROR")
            finally:
                # ensure socket is closed before retry
                try:
                    self.ws_client.exit()
                except Exception:
                    pass

            # backoff and retry
            await asyncio.sleep(backoff)
            backoff = min(60.0, backoff * 2)

        # Final cleanup (on stop)
        try:
            self.ws_client.exit()
        except Exception:
            pass

    async def disconnect(self) -> None:
        """Disconnect from Apex."""
        try:
            self._ws_stop.set()
            if self._ws_task:
                await self._ws_task
        except Exception:
            pass

        try:
            if hasattr(self, "private_rest_client") and self.rest_client:
                self.rest_client._exit()
            if hasattr(self, "ws_client"):
                self.ws_client.exit()
        except Exception as e:
            self.logger.log(f"Error during Apex disconnect: {e}", "ERROR")

    # ---------------------------
    # Utility / Name
    # ---------------------------

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "apex"

    # ---------------------------
    # WS Handlers
    # ---------------------------

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
                content = message.get("contents", {})
                topic = message.get("topic", "")
                if topic != "ws_zk_accounts_v3":
                    return
                # Extract order data from the nested structure
                orders = content.get('orders', [])

                # on websocket starting, Apex will send the historical filled orders, could be confusing
                if len(orders) > 1 and not content.get('fills'):
                    return

                if not orders:
                    return

                order = orders[0]  # Get the first order
                if order.get('symbol') != self.config.contract_id:
                    return

                order_id = order.get('id')
                status = order.get('status')
                side = order.get('side', '').lower()
                filled_size = order.get('cumSuccessFillSize')
                remaining_size = order.get('remainingSize')

                if side == self.config.close_order_side:
                    order_type = "CLOSE"
                else:
                    order_type = "OPEN"

                if status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED']:
                    if self._order_update_handler:
                        self._order_update_handler({
                            'order_id': order_id,
                            'side': side,
                            'order_type': order_type,
                            'status': status,
                            'size': order.get('size'),
                            'price': order.get('price'),
                            'contract_id': order.get('symbol'),
                            'filled_size': filled_size
                        })

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        try:
            self.account_handler = order_update_handler
        except Exception as e:
            self.logger.log(f"Could not add trade-event handler: {e}", "ERROR")

    # ---------------------------
    # REST-ish helpers
    # ---------------------------

    @query_retry(default_return=(0, 0))
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid and ask price using official SDK"""
        order_book = self.rest_client.depth_v3(symbol=contract_id)
        order_book_data = order_book['data']

        # Extract bids and asks from the entry
        bids = order_book_data.get('b', [])
        asks = order_book_data.get('a', [])

        # Best bid is the highest price someone is willing to buy at
        best_bid = Decimal(max(bids, key=lambda x: Decimal(x[0]))[0])
        # Best ask is the lowest price someone is willing to sell at
        best_ask = Decimal(min(asks, key=lambda x: Decimal(x[0]))[0])
        return best_bid, best_ask

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price of an order with Apex using official SDK."""
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
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
        """Place an open order with Apex using official SDK with retry logic for POST_ONLY rejections."""
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
                    side = 'buy'
                else:
                    # For sell orders, place slightly above best bid to ensure execution
                    order_price = best_bid + self.config.tick_size
                    side = 'sell'

                # Place the order using official SDK (post-only to ensure maker order)
                order_result = self.rest_client.create_order_v3(
                    symbol=contract_id,
                    size=str(quantity),
                    price=str(self.round_to_tick(order_price)),
                    side=side.upper(),
                    type='LIMIT',
                    timestampSeconds=time.time(),
                    timeInForce='POST_ONLY',
                )

                if not order_result or 'data' not in order_result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = order_result['data'].get('id')
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
                    elif order_info.status in ['OPEN', 'PENDING', 'PARTIALLY_FILLED', 'FILLED']:
                        # Order successfully placed
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            side=side,
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
                        side=side,
                        size=quantity,
                        price=order_price,
                        status='OPEN'
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
        """Place a close order with Apex using official SDK with retry logic for POST_ONLY rejections."""
        max_retries = 15
        retry_count = 0

        while retry_count < max_retries:
            try:
                best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

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
                order_result = self.rest_client.create_order_v3(
                    symbol=contract_id,
                    size=str(quantity),
                    price=str(adjusted_price),
                    side=side.upper(),
                    type='LIMIT',
                    timestampSeconds=time.time(),
                    timeInForce='POST_ONLY',
                )

                if not order_result or 'data' not in order_result:
                    return OrderResult(success=False, error_message='Failed to place order')

                # Extract order ID from response
                order_id = order_result['data'].get('id')
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
                    elif order_info.status in ['OPEN', 'PENDING', 'PARTIALLY_FILLED', 'FILLED']:
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
        """Cancel an order with Apex using official SDK."""
        try:
            # Cancel the order using official SDK
            cancel_result = self.rest_client.delete_order_v3(id=order_id)

            if not cancel_result or 'data' not in cancel_result:
                return OrderResult(success=False, error_message='Failed to cancel order')

            return OrderResult(success=True)
        except FailedRequestError as e:
            # Apex's API return non-JSON response when trying to cancel a filled order
            if 'Could not decode JSON' in e.message:
                return OrderResult(success=False, error_message='Order has been filled')
            else:
                return OrderResult(success=False, error_message=e.message)
        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from Apex using official SDK."""
        order_result = self.rest_client.get_order_v3(id=order_id)
        if not order_result or 'data' not in order_result:
            return None

        order_data = order_result['data']
        return OrderInfo(
            order_id=order_data.get('id', ''),
            side=order_data.get('side', '').lower(),
            size=Decimal(order_data.get('size', 0)),
            price=Decimal(order_data.get('price', 0)),
            status=order_data.get('status', ''),
            filled_size=Decimal(order_data.get('cumSuccessFillSize', 0)),
            remaining_size=Decimal(order_data.get('size', 0)) - Decimal(order_data.get('cumSuccessFillSize', 0))
        )

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a symbol using official SDK."""
        # Get active orders using official SDK
        active_orders = self.rest_client.open_orders_v3()

        if not active_orders or 'data' not in active_orders:
            return []

        # Filter orders for the specific contract and ensure they are dictionaries
        # The API returns orders under 'dataList' key, not 'orderList'
        order_list = active_orders['data']
        contract_orders = []

        for order in order_list:
            if isinstance(order, dict) and order.get('symbol') == contract_id:
                contract_orders.append(OrderInfo(
                    order_id=order.get('id', ''),
                    side=order.get('side', '').lower(),
                    size=Decimal(order.get('size', 0)),
                    price=Decimal(order.get('price', 0)),
                    status=order.get('status', ''),
                    filled_size=Decimal(order.get('cumSuccessFillSize', 0)),
                    remaining_size=Decimal(order.get('size', 0)) - Decimal(order.get('cumSuccessFillSize', 0))
                ))
        return contract_orders

    @query_retry(default_return=0)
    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        account_data = self.rest_client.get_account_v3()
        if not account_data or 'positions' not in account_data:
            self.logger.log("No positions or failed to get positions", "WARNING")
            position_amt = 0
        else:
            # The API returns positions under data.positionList
            positions = account_data.get('positions', [])
            if positions:
                # Find position for current contract
                position = None
                for p in positions:
                    if isinstance(p, dict) and p.get('symbol') == self.config.contract_id:
                        position = p
                        break

                if position:
                    position_amt = abs(Decimal(position.get('size', 0)))
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

        response = self.rest_client.configs_v3(symbol=ticker)
        data = response.get('data', {})
        if not data:
            self.logger.log("Failed to get metadata", "ERROR")
            raise ValueError("Failed to get metadata")

        contract_list = data.get('contractConfig', {}).get('perpetualContract', [])
        if not contract_list:
            self.logger.log("Failed to get contract list", "ERROR")
            raise ValueError("Failed to get contract list")

        current_contract = None
        for c in contract_list:
            if c.get('crossSymbolName') == ticker + 'USDT':
                current_contract = c
                break

        if not current_contract:
            self.logger.log("Failed to get contract ID for ticker", "ERROR")
            raise ValueError("Failed to get contract ID for ticker")

        self.config.contract_id = current_contract.get('symbol')
        min_quantity = Decimal(current_contract.get('minOrderSize'))
        if self.config.quantity < min_quantity:
            self.logger.log(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}", "ERROR")
            raise ValueError(f"Order quantity is less than min quantity: {self.config.quantity} < {min_quantity}")

        self.config.tick_size = Decimal(current_contract.get('tickSize'))

        return self.config.contract_id, self.config.tick_size
