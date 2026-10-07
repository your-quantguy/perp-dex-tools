"""
Lighter exchange client implementation.
"""

import os
import asyncio
import time
import logging
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple

from .base import BaseExchangeClient, ExchangeConfig, MarketId, OrderOutcomeUnknown, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger

# Import official Lighter SDK for API client
from lighter.api.account_api import AccountApi
from lighter.api.order_api import OrderApi
from lighter.api_client import ApiClient
from lighter.configuration import Configuration
from lighter.signer_client import CODE_OK, SignerClient

if TYPE_CHECKING:
    from lighter.models.account_position import AccountPosition
    from lighter.models.order import Order

# Import custom WebSocket implementation
from .lighter_custom_websocket import LighterCustomWebSocketManager

# Suppress Lighter SDK debug logs
logging.getLogger('lighter').setLevel(logging.WARNING)
# Also suppress root logger DEBUG messages that might be coming from Lighter SDK
root_logger = logging.getLogger()
if root_logger.level == logging.DEBUG:
    root_logger.setLevel(logging.WARNING)


class LighterClient(BaseExchangeClient):
    """Lighter exchange client implementation."""

    ORDER_CONFIRM_TIMEOUT = 10

    @property
    def network_id(self) -> str:
        return "mainnet"

    def get_quantity_step(self) -> Decimal:
        base_multiplier, _ = self._multipliers()
        return Decimal(1) / Decimal(base_multiplier)

    def get_account_description(self) -> tuple[str, ...]:
        return (f"Account index: {self.account_index}", f"API key index: {self.api_key_index}")

    def __init__(self, config: ExchangeConfig):
        """Initialize Lighter client."""
        super().__init__(config)

        # Lighter credentials from environment
        self.api_key_private_key = os.getenv('API_KEY_PRIVATE_KEY')
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX', '0'))
        self.base_url = "https://mainnet.zklighter.elliot.ai"

        if not self.api_key_private_key:
            raise ValueError("API_KEY_PRIVATE_KEY must be set in environment variables")

        # Initialize logger
        self.logger = TradingLogger(exchange="lighter", ticker=self.config.ticker, log_to_console=False)
        self._order_update_handler = None

        # Initialize Lighter client (will be done in connect)
        self.lighter_client = None

        # Initialize API client (will be done in connect)
        self.api_client = None

        # Market configuration
        self.base_amount_multiplier = None
        self.price_multiplier = None
        self.orders_cache = {}
        self.current_order_client_id = None
        self.current_order = None
        self._hedge_order_index = int(time.time() * 1000) % 2_000_000_000

    def _validate_config(self) -> None:
        """Validate Lighter configuration."""
        required_env_vars = ['API_KEY_PRIVATE_KEY', 'LIGHTER_ACCOUNT_INDEX', 'LIGHTER_API_KEY_INDEX']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def _market_id(self) -> int:
        market_id = self.config.contract_id
        if not isinstance(market_id, int):
            raise RuntimeError("Lighter market metadata is not loaded")
        return market_id

    def _multipliers(self) -> tuple[int, int]:
        if self.base_amount_multiplier is None or self.price_multiplier is None:
            raise RuntimeError("Lighter market metadata is not loaded")
        return self.base_amount_multiplier, self.price_multiplier

    def _signer(self) -> SignerClient:
        if self.lighter_client is None:
            raise RuntimeError("Lighter signer is not initialized")
        return self.lighter_client

    async def _get_market_config(self, ticker: str) -> Tuple[int, int, int]:
        """Get market configuration for a ticker using official SDK."""
        try:
            # Use shared API client
            order_api = OrderApi(self.api_client)

            # Get order books to find market info
            order_books = await order_api.order_books()

            for market in order_books.order_books:
                if market.symbol == ticker:
                    market_id = market.market_id
                    base_multiplier = pow(10, market.supported_size_decimals)
                    price_multiplier = pow(10, market.supported_price_decimals)

                    # Store market info for later use
                    self.market_info = market

                    self.logger.log(
                        f"Market config for {ticker}: ID={market_id}, "
                        f"Base multiplier={base_multiplier}, Price multiplier={price_multiplier}",
                        "INFO"
                    )
                    return market_id, base_multiplier, price_multiplier

            raise Exception(f"Ticker {ticker} not found in available markets")

        except Exception as e:
            self.logger.log(f"Error getting market config: {e}", "ERROR")
            raise

    async def _initialize_lighter_client(self):
        """Initialize the Lighter client using official SDK."""
        if self.lighter_client is None:
            signer = None
            try:
                private_key = self.api_key_private_key
                if private_key is None:
                    raise ValueError("API_KEY_PRIVATE_KEY must be set")
                signer = SignerClient(
                    url=self.base_url,
                    account_index=self.account_index,
                    api_private_keys={self.api_key_index: private_key}
                )

                # Check client
                err = signer.check_client()
                if err is not None:
                    raise Exception(f"CheckClient error: {err}")

                self.lighter_client = signer
                self.logger.log("Lighter client initialized successfully", "INFO")
            except Exception as e:
                if signer is not None:
                    try:
                        await signer.close()
                    except Exception as close_error:
                        self.logger.log(f"Error closing failed Lighter signer: {close_error}", "ERROR")
                self.logger.log(f"Failed to initialize Lighter client: {e}", "ERROR")
                raise
        return self.lighter_client

    async def connect(self) -> None:
        """Connect to Lighter."""
        try:
            # Initialize shared API client
            if self.api_client is None:
                self.api_client = ApiClient(configuration=Configuration(host=self.base_url))

            # Initialize Lighter client
            await self._initialize_lighter_client()

            # Initialize WebSocket manager (using custom implementation)
            self.ws_manager = LighterCustomWebSocketManager(
                market_index=self._market_id(),
                account_index=self.account_index,
                lighter_client=self._signer(),
                order_update_callback=self._handle_websocket_order_update
            )

            # Set logger for WebSocket manager
            self.ws_manager.set_logger(self.logger)

            # Start WebSocket connection in background task
            asyncio.create_task(self.ws_manager.connect())
            # Wait a moment for connection to establish
            await asyncio.sleep(2)

        except Exception as e:
            self.logger.log(f"Error connecting to Lighter: {e}", "ERROR")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Lighter."""
        try:
            if hasattr(self, 'ws_manager') and self.ws_manager:
                await self.ws_manager.disconnect()
        except Exception as e:
            self.logger.log(f"Error disconnecting Lighter WebSocket: {e}", "ERROR")
        for attr in ("api_client", "lighter_client"):
            client = getattr(self, attr, None)
            if client is not None:
                try:
                    await client.close()
                except Exception as e:
                    self.logger.log(f"Error closing Lighter {attr}: {e}", "ERROR")
                setattr(self, attr, None)

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "lighter"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

    @staticmethod
    def _order_status(raw_status: str, filled: Decimal) -> str:
        status = raw_status.lower().replace("_", "-")
        if status == "filled":
            return "FILLED"
        if status.startswith("canceled"):
            return "CANCELED"
        if status in {"open", "pending", "in-progress"}:
            return "PARTIALLY_FILLED" if filled > 0 else "OPEN"
        raise ValueError(f"Unsupported Lighter order status: {raw_status}")

    @classmethod
    def _order_info(cls, raw: Any) -> OrderInfo:
        size = Decimal(raw.initial_base_amount)
        filled = Decimal(raw.filled_base_amount)
        reported_remaining = Decimal(raw.remaining_base_amount)
        if (not all(value.is_finite() for value in (size, filled, reported_remaining))
                or size <= 0 or filled < 0 or filled > size
                or reported_remaining < 0 or reported_remaining > size):
            raise ValueError("Lighter order quantities are inconsistent")
        return OrderInfo(
            order_id=str(raw.client_order_index),
            side="sell" if raw.is_ask else "buy",
            size=size,
            price=Decimal(raw.price),
            status=cls._order_status(raw.status, filled),
            filled_size=filled,
            remaining_size=size - filled,
            cancel_reason=raw.status if raw.status.lower().startswith("canceled") else "",
        )

    def _handle_websocket_order_update(self, order_data_list: List[Dict[str, Any]]):
        """Handle order updates from WebSocket."""
        for order_data in order_data_list:
            if order_data['market_index'] != self.config.contract_id:
                continue
            order = self._order_info(SimpleNamespace(**order_data))
            order_id = order.order_id
            order_type = "CLOSE" if order.side == self.config.close_order_side else "OPEN"
            previous = self.orders_cache.get(order_id)
            if previous == (order.status, order.filled_size):
                continue
            if order.status in {"FILLED", "CANCELED"}:
                self.orders_cache.pop(order_id, None)
            else:
                self.orders_cache[order_id] = (order.status, order.filled_size)
            self.logger.log(f"[{order_type}] [{order_id}] {order.status} "
                            f"{order.filled_size} @ {order.price}", "INFO")
            if str(order_data['client_order_index']) == str(self.current_order_client_id):
                self.current_order = order
            if self._order_update_handler is not None:
                self._order_update_handler({
                    "contract_id": self.config.contract_id, "order_id": order_id,
                    "side": order.side, "order_type": order_type, "status": order.status,
                    "size": order.size, "price": order.price,
                    "filled_size": order.filled_size,
                })
            if order.status in {"FILLED", "CANCELED"}:
                self.logger.log_transaction(order_id, order.side, order.filled_size, order.price, order.status)

    @query_retry(default_return=(0, 0))
    async def fetch_bbo_prices(self, contract_id: MarketId) -> Tuple[Decimal, Decimal]:
        """Get orderbook using official SDK."""
        # Use WebSocket data if available
        if (hasattr(self, 'ws_manager') and
                self.ws_manager.best_bid and self.ws_manager.best_ask):
            best_bid = Decimal(str(self.ws_manager.best_bid))
            best_ask = Decimal(str(self.ws_manager.best_ask))

            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                self.logger.log("Invalid bid/ask prices", "ERROR")
                raise ValueError("Invalid bid/ask prices")
        else:
            self.logger.log("Unable to get bid/ask prices from WebSocket.", "ERROR")
            raise ValueError("WebSocket not running. No bid/ask prices available")

        return best_bid, best_ask

    async def get_spread_book(self, contract_id: MarketId):
        from exchanges.market import BookSnapshot

        if contract_id != self.config.contract_id or not hasattr(self, "ws_manager"):
            raise ConnectionError("Lighter order book is unavailable")
        manager = self.ws_manager
        async with manager.order_book_lock:
            if not manager.snapshot_loaded or manager.order_book_sequence_gap:
                raise ConnectionError("Lighter order book snapshot is not synchronized")
            bids = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in manager.order_book["bids"].items())
            asks = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in manager.order_book["asks"].items())
            return BookSnapshot(bids, asks, manager.book_updated_at)

    async def get_margin_snapshot(self):
        from exchanges.market import MarginPosition, MarginSnapshot

        response = await AccountApi(self.api_client).account(by="index", value=str(self.account_index))
        if not response or not response.accounts:
            raise ValueError("Lighter account risk data is unavailable")
        account = response.accounts[0]
        equity = Decimal(account.total_asset_value)
        cross_equity = Decimal(account.cross_asset_value)
        notional = sum((abs(Decimal(p.position_value)) for p in account.positions), Decimal(0))
        maintenance = Decimal(account.cross_maintenance_margin_requirement)
        positions = tuple(
            MarginPosition(
                market_id=int(p.market_id),
                mode="CROSS" if p.margin_mode == 0 else "ISOLATED",
                notional=abs(Decimal(p.position_value)),
            )
            for p in account.positions if Decimal(p.position) != 0
        )
        return MarginSnapshot(equity, notional, maintenance, positions=positions,
                              cross_equity=cross_equity)

    async def _submit_order_with_retry(self, order_params: Dict[str, Any]) -> OrderResult:
        """Return only a confirmed SDK submission response."""
        # Ensure client is initialized
        if self.lighter_client is None:
            # This is a sync method, so we need to handle this differently
            # For now, raise an error if client is not initialized
            raise ValueError("Lighter client not initialized. Call connect() first.")

        # Create order using official SDK
        client_id = str(order_params['client_order_index'])
        try:
            _, response, error = await self.lighter_client.create_order(**order_params)
        except Exception as exc:
            raise OrderOutcomeUnknown(f"Lighter submit outcome is unknown for {client_id}", client_id) from exc
        if error is not None:
            return OrderResult(
                success=False, order_id=client_id,
                error_message=f"Order creation error: {error}")
        if response is None:
            raise OrderOutcomeUnknown(f"Lighter submit outcome is unknown for {client_id}", client_id)
        if response.code != CODE_OK:
            return OrderResult(False, client_id, error_message=getattr(response, "message", "Lighter rejected order"))
        return OrderResult(True, client_id, status="SUBMITTED")

    def _next_client_order_index(self) -> int:
        index = max(time.time_ns() // 1_000_000, getattr(self, "_last_client_order_index", 0) + 1)
        self._last_client_order_index = index
        return index

    async def place_limit_order(self, contract_id: MarketId, quantity: Decimal, price: Decimal,
                                side: str, reduce_only: bool = False) -> OrderResult:
        """Place a good-till-time limit order with Lighter using the official SDK."""
        # Ensure client is initialized
        if self.lighter_client is None:
            await self._initialize_lighter_client()

        # Determine order side and price
        if side.lower() == 'buy':
            is_ask = False
        elif side.lower() == 'sell':
            is_ask = True
        else:
            raise Exception(f"Invalid side: {side}")

        # Generate unique client order index
        client_order_index = self._next_client_order_index()
        self.current_order_client_id = client_order_index
        base_multiplier, price_multiplier = self._multipliers()
        signer = self._signer()

        # Create order parameters
        order_params = {
            'market_index': self._market_id(),
            'client_order_index': client_order_index,
            'base_amount': int(quantity * base_multiplier),
            'price': int(price * price_multiplier),
            'is_ask': is_ask,
            'order_type': signer.ORDER_TYPE_LIMIT,
            'time_in_force': signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
            'reduce_only': reduce_only,
            'trigger_price': 0,
        }

        order_result = await self._submit_order_with_retry(order_params)
        return order_result

    async def _wait_for_order(self, order_id: str, timeout: float) -> OrderInfo:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            current = self.current_order
            if current is not None and current.order_id == order_id:
                return current
            try:
                order = await self.get_order_info(order_id)
            except Exception as exc:
                last_error = exc
            else:
                if order is not None:
                    self.current_order = order
                    return order
            await asyncio.sleep(0.25)
        raise OrderOutcomeUnknown(
            f"Lighter order {order_id} was accepted but not found: {last_error}", order_id,
        )

    async def place_open_order(self, contract_id: MarketId, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with Lighter using official SDK."""

        self.current_order = None
        self.current_order_client_id = None
        order_price = await self.get_order_price(direction)

        order_price = self.round_to_tick(order_price)
        order_result = await self.place_limit_order(contract_id, quantity, order_price, direction)
        if not order_result.success:
            return order_result
        if order_result.order_id is None:
            raise OrderOutcomeUnknown("Lighter submit succeeded without a client order ID")
        order = await self._wait_for_order(order_result.order_id, self.ORDER_CONFIRM_TIMEOUT)
        accepted = order.status != "CANCELED" or order.filled_size > 0
        return OrderResult(
            success=accepted,
            order_id=order.order_id,
            side=direction,
            size=quantity,
            price=order_price,
            status=order.status,
            error_message=order.cancel_reason or None,
            filled_size=order.filled_size,
        )

    async def _get_active_close_orders(self, contract_id: MarketId) -> int:
        """Get active close orders for a contract using official SDK."""
        active_orders = await self.get_active_orders(contract_id)
        active_close_orders = 0
        for order in active_orders:
            if order.side == self.config.close_order_side:
                active_close_orders += 1
        return active_close_orders

    async def place_close_order(self, contract_id: MarketId, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with Lighter using official SDK."""
        self.current_order = None
        self.current_order_client_id = None
        order_result = await self.place_limit_order(contract_id, quantity, price, side, reduce_only=True)
        if not order_result.success:
            return order_result
        if order_result.order_id is None:
            raise OrderOutcomeUnknown("Lighter close submit succeeded without a client order ID")
        order = await self._wait_for_order(order_result.order_id, self.ORDER_CONFIRM_TIMEOUT)
        return OrderResult(order.status != "CANCELED", order.order_id, side, quantity, price,
                           order.status, order.cancel_reason or None, order.filled_size)

    async def place_hedge_market_order(self, quantity: Decimal, side: str,
                                       max_slippage_percent: Decimal,
                                       reduce_only: bool = False) -> OrderResult:
        """Submit one bounded IOC market hedge; the caller must confirm the position change."""
        if side not in {"buy", "sell"}:
            raise ValueError("Invalid Lighter hedge side or quantity")
        if not Decimal(0) < max_slippage_percent <= Decimal(5):
            raise ValueError("Lighter hedge slippage limit must be within (0, 5] percent")
        self.validate_hedge_quantity(quantity)
        base_multiplier, price_multiplier = self._multipliers()
        base_units = quantity * base_multiplier
        bid, ask = await self.fetch_bbo_prices(self.config.contract_id)
        if bid <= 0 or ask <= 0 or bid >= ask:
            raise ConnectionError("Lighter order book is not ready for a hedge")
        factor = max_slippage_percent / 100
        if side == "buy":
            price = ask * (1 + factor)
            rounding = ROUND_CEILING
        else:
            price = bid * (1 - factor)
            rounding = ROUND_FLOOR
        price_units = (price * price_multiplier).to_integral_value(rounding=rounding)
        if price_units <= 0:
            raise ValueError("Invalid Lighter hedge price")
        order_index = self._next_client_order_index()
        try:
            _, response, error = await self._signer().create_market_order(
                market_index=self._market_id(),
                client_order_index=order_index,
                base_amount=int(base_units),
                avg_execution_price=int(price_units),
                is_ask=side == "sell",
                reduce_only=reduce_only,
                api_key_index=self.api_key_index,
            )
        except Exception as exc:
            raise OrderOutcomeUnknown(f"Lighter hedge submission outcome is unknown for {order_index}", str(order_index)) from exc
        if error is not None:
            return OrderResult(False, str(order_index), error_message=str(error))
        if response is None:
            raise OrderOutcomeUnknown(f"Lighter hedge submission outcome is unknown for {order_index}", str(order_index))
        if response.code != CODE_OK:
            return OrderResult(False, str(order_index),
                               error_message=getattr(response, "message", "Lighter rejected hedge order"))
        return OrderResult(True, str(order_index), side, quantity, price, "SUBMITTED")

    def validate_hedge_quantity(self, quantity: Decimal) -> None:
        """Lighter sizes are integer multiples of the market's base unit."""
        base_multiplier, _ = self._multipliers()
        units = quantity * base_multiplier
        if not quantity.is_finite() or quantity <= 0 or units != units.to_integral_value():
            raise ValueError(f"Lighter cannot represent exact hedge quantity {quantity}")
    
    async def get_order_price(self, side: str = '') -> Decimal:
        """Get the price of an order with Lighter using official SDK."""
        # Get current market prices
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            self.logger.log("Invalid bid/ask prices", "ERROR")
            raise ValueError("Invalid bid/ask prices")

        order_price = (best_bid + best_ask) / 2

        active_orders = await self.get_active_orders(self.config.contract_id)
        close_orders = [order for order in active_orders if order.side == self.config.close_order_side]
        for order in close_orders:
            if side == 'buy':
                order_price = min(order_price, order.price - self.config.tick_size)
            else:
                order_price = max(order_price, order.price + self.config.tick_size)

        return order_price

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel by client order ID and return the confirmed terminal fill."""
        raw = await self._fetch_order_by_client_id(order_id)
        if raw is None:
            raise OrderOutcomeUnknown(f"Lighter order {order_id} is absent from available history", order_id)
        order = self._order_info(raw)
        if order.status not in {"FILLED", "CANCELED"}:
            cancel_error: Exception | None = None
            try:
                _, response, error = await self._signer().cancel_order(
                    market_index=self._market_id(),
                    order_index=raw.order_index,
                    api_key_index=self.api_key_index,
                )
            except Exception as exc:
                cancel_error = exc
            else:
                if error is not None:
                    cancel_error = RuntimeError(str(error))
                elif response is None:
                    cancel_error = RuntimeError("No Lighter cancel response")
                elif response.code != CODE_OK:
                    cancel_error = RuntimeError(getattr(response, "message", "Lighter rejected cancellation"))
            deadline = time.monotonic() + self.ORDER_CONFIRM_TIMEOUT
            last_read_error: Exception | None = None
            while True:
                try:
                    raw = await self._fetch_order_by_client_id(order_id)
                except Exception as exc:
                    last_read_error = exc
                else:
                    if raw is not None:
                        order = self._order_info(raw)
                        if order.status in {"FILLED", "CANCELED"}:
                            break
                if time.monotonic() >= deadline:
                    raise OrderOutcomeUnknown(
                        f"Lighter cancel terminal state is unknown for {order_id}: "
                        f"cancel_error={cancel_error}, read_error={last_read_error}", order_id,
                    )
                await asyncio.sleep(0.25)
        return OrderResult(True, order_id, order.side, order.size, order.price,
                           order.status, filled_size=order.filled_size)

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Query a specific client order ID; position size is never an order state."""
        raw = await self._fetch_order_by_client_id(order_id)
        return self._order_info(raw) if raw is not None else None

    def _auth_token(self) -> str:
        token, error = self._signer().create_auth_token_with_expiry(api_key_index=self.api_key_index)
        if error is not None or not token:
            raise ValueError(f"Error creating Lighter auth token: {error or 'empty token'}")
        return token

    @query_retry(reraise=True)
    async def _fetch_order_by_client_id(self, order_id: str) -> Any | None:
        response = await OrderApi(self.api_client).account_orders(
            authorization=self._auth_token(), client_order_indexes=order_id,
            account_index=self.account_index,
        )
        if response is None or response.orders is None:
            raise ValueError("Lighter returned an invalid order response")
        for raw in response.orders:
            if str(raw.client_order_index) == order_id and raw.market_index == self._market_id():
                return raw
        return None

    @query_retry(reraise=True)
    async def _fetch_orders_with_retry(self) -> List["Order"]:
        """Get orders using official SDK."""
        # Ensure client is initialized
        if self.lighter_client is None:
            await self._initialize_lighter_client()

        # Generate auth token for API call
        auth_token = self._auth_token()

        # Use OrderApi to get active orders
        order_api = OrderApi(self.api_client)

        # Get active orders for the specific market
        orders_response = await order_api.account_active_orders(
            account_index=self.account_index,
            market_id=self._market_id(),
            authorization=auth_token,
        )

        if orders_response is None or orders_response.orders is None:
            self.logger.log("Failed to get orders", "ERROR")
            raise ValueError("Failed to get orders")

        return orders_response.orders

    async def get_active_orders(self, contract_id: MarketId) -> List[OrderInfo]:
        """Get active orders for a contract using official SDK."""
        if contract_id != self._market_id():
            raise ValueError("Lighter contract does not match configured market")
        order_list = await self._fetch_orders_with_retry()
        orders = [self._order_info(raw) for raw in order_list if raw.market_index == contract_id]
        return [order for order in orders if order.status in {"OPEN", "PARTIALLY_FILLED"}
                and order.remaining_size > 0]

    @query_retry(reraise=True)
    async def _fetch_positions_with_retry(self) -> List["AccountPosition"]:
        """Get positions using official SDK."""
        # Use shared API client
        account_api = AccountApi(self.api_client)

        # Get account info
        account_data = await account_api.account(by="index", value=str(self.account_index))

        if not account_data or not account_data.accounts:
            self.logger.log("Failed to get positions", "ERROR")
            raise ValueError("Failed to get positions")

        return account_data.accounts[0].positions

    async def get_account_positions(self) -> Decimal:
        """Get account positions using official SDK."""
        # Get account info which includes positions
        positions = await self._fetch_positions_with_retry()

        # Find position for current market
        for position in positions:
            if position.market_id == self.config.contract_id:
                return Decimal(position.position) * Decimal(position.sign)

        return Decimal(0)

    async def get_contract_attributes(self) -> Tuple[int, Decimal]:
        """Get contract ID for a ticker."""
        ticker = self.config.ticker
        if len(ticker) == 0:
            self.logger.log("Ticker is empty", "ERROR")
            raise ValueError("Ticker is empty")

        if self.api_client is None:
            self.api_client = ApiClient(configuration=Configuration(host=self.base_url))

        order_api = OrderApi(self.api_client)
        # Get all order books to find the market for our ticker
        order_books = await order_api.order_books()

        # Find the market that matches our ticker
        market_info = None
        for market in order_books.order_books:
            if market.symbol == ticker:
                market_info = market
                break

        if market_info is None:
            self.logger.log("Failed to get markets", "ERROR")
            raise ValueError("Failed to get markets")

        market_summary = await order_api.order_book_details(market_id=market_info.market_id)
        order_book_details = market_summary.order_book_details[0]
        # Set contract_id to market name (Lighter uses market IDs as identifiers)
        self.config.contract_id = market_info.market_id
        self.base_amount_multiplier = pow(10, market_info.supported_size_decimals)
        self.price_multiplier = pow(10, market_info.supported_price_decimals)

        try:
            self.config.tick_size = Decimal("1") / (Decimal("10") ** order_book_details.price_decimals)
        except Exception:
            self.logger.log("Failed to get tick size", "ERROR")
            raise ValueError("Failed to get tick size")

        return self.config.contract_id, self.config.tick_size
