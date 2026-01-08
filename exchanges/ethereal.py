"""
Ethereal exchange client skeleton built on the official ethereal-sdk.
"""

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from .base import BaseExchangeClient, OrderResult, OrderInfo
from helpers.logger import TradingLogger
from ethereal import AsyncRESTClient, AsyncWSClient


class EtherealClient(BaseExchangeClient):
    """
    Ethereal exchange client wired to ethereal-sdk.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ethereal client with environment-driven defaults."""
        # REST + chain config (all optional; defaults follow the SDK docs)
        self.base_url = os.getenv("ETHEREAL_BASE_URL", "https://api.ethereal.trade")
        self.rpc_url = os.getenv("ETHEREAL_RPC_URL", "https://rpc.ethereal.trade")
        self.private_key = os.getenv("ETHEREAL_PRIVATE_KEY")  # required for signing
        self.chain_id = int(os.getenv("ETHEREAL_CHAIN_ID", "42161"))
        self.ws_url = os.getenv("ETHEREAL_WS_URL", "wss://ws.ethereal.trade")

        # Account / trading context
        self.subaccount_id = os.getenv("ETHEREAL_SUBACCOUNT_ID")
        self.account_name = os.getenv("ETHEREAL_ACCOUNT_NAME", "primary")
        self.subaccount_hex = os.getenv("ETHEREAL_SUBACCOUNT")
        if not self.subaccount_hex and self.account_name:
            try:
                self.subaccount_hex = self._encode_subaccount_name(self.account_name)
            except Exception:
                self.subaccount_hex = None
        self._warned_missing_subaccount = False

        # Clients are created lazily in connect()
        self._rest_client: Optional["AsyncRESTClient"] = None
        self._ws_client: Optional["AsyncWSClient"] = None
        self._order_update_handler = None
        self._product_cache: Dict[str, Any] = {}

        self.logger = TradingLogger(
            exchange="ethereal",
            ticker=getattr(config, "ticker", ""),
            log_to_console=False
        )

        # Base init last: sets self.config and runs _validate_config
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate Ethereal configuration."""
        # Ticker is required; contract_id is optional (will default to ticker if missing).
        if not getattr(self.config, "ticker", None):
            raise ValueError("Missing required config field: ticker")
        if not getattr(self.config, "contract_id", None):
            self.config.contract_id = self.config.ticker

        # Signing key is optional for read-only calls, but required for trading.
        if not getattr(self, "private_key", None):
            self.logger.log("ETHEREAL_PRIVATE_KEY not set; trading calls will fail.", "WARN")

    def _build_rest_config(self) -> Dict[str, Any]:
        """
        Assemble the config dict expected by AsyncRESTClient.create().

        Example from the SDK docs:
            await AsyncRESTClient.create({
                "base_url": "https://api.ethereal.trade",
                "chain_config": {
                    "rpc_url": "https://rpc.ethereal.trade",
                    "private_key": "your_private_key",
                }
            })
        """
        config: Dict[str, Any] = {
            "base_url": self.base_url,
            "chain_config": {
                "rpc_url": self.rpc_url,
                "chain_id": self.chain_id,
            },
        }
        if self.private_key:
            config["chain_config"]["private_key"] = self.private_key
        return config

    async def _ensure_rest_client(self) -> None:
        """Create the AsyncRESTClient on-demand."""
        if self._rest_client is not None:
            return

        if AsyncRESTClient is None:
            raise ImportError("ethereal-sdk not installed or not available in this environment")

        self._rest_client = await AsyncRESTClient.create(self._build_rest_config())

    async def _ensure_products(self) -> Dict[str, Any]:
        """Cache products by ticker for quick lookups."""
        if self._product_cache:
            return self._product_cache
        await self._ensure_rest_client()
        products = await self._rest_client.products_by_ticker()
        # store uppercase keys
        self._product_cache = {k.upper(): v for k, v in products.items()}
        return self._product_cache

    async def _start_ws(self) -> None:
        """Start AsyncWSClient and subscribe to order stream."""
        if AsyncWSClient is None:
            raise ImportError("ethereal-sdk websocket client not available")

        self._ws_client = AsyncWSClient({"base_url": self.ws_url})

        # Set callbacks
        self._ws_client.callbacks["OrderUpdate"] = [self._handle_ws_message_order_update]

        await self._ws_client.open(namespaces=["/", "/v1/stream"])
        # connected flag kept for future use
        self._ws_connected = True

        # Best-effort subscribe
        try:
            if self.subaccount_id:
                await self._ws_client.subscribe(
                    stream_type="OrderUpdate",
                    subaccount_id=self.subaccount_id,
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.log(f"[ethereal] WS subscribe failed: {exc}", "WARNING")


    async def _get_product_by_ticker(self, ticker: Any) -> Optional[Any]:
        products = await self._ensure_products()
        if ticker is None:
            return None

        # Normalize ticker/id to string for comparison
        if isinstance(ticker, UUID):
            ticker_str = str(ticker)
        else:
            ticker_str = str(ticker)

        # First try matching by product id
        for item in products.values():
            if str(getattr(item, "id", "")) == ticker_str:
                return item

        # Then try ticker-based lookup (expects e.g. BTCUSD)
        key = ticker_str.upper()
        if key.endswith("USD"):
            prod = products.get(key)
        else:
            prod = products.get(f"{key}USD")
        if prod:
            return prod

        return None

    def _encode_subaccount_name(self, name: str) -> str:
        """Encode subaccount name to bytes32 hex (0x...)."""
        if not name:
            raise ValueError("Subaccount name is empty")
        b = name.encode("utf-8")
        if len(b) > 32:
            raise ValueError("Subaccount name too long for bytes32")
        return "0x" + b.ljust(32, b"\x00").hex()

    async def _handle_ws_message_order_update(self, data: Dict[str, Any]):
        """Bridge WS order events to the trading bot handler."""
        if not self._order_update_handler:
            return

        try:
            # Stream data type:
            # https://docs.ethereal.trade/developer-guides/trading-api/websocket-gateway#subscribe
            payload = data.get("data") if isinstance(data, dict) and "data" in data else data
            if not isinstance(payload, list):
                return

            for order in payload:
                if not isinstance(order, dict):
                    continue
                order_id = order.get("id")
                status = self._normalize_status(order.get("status") or order.get("state"))
                side_val = order.get("side")
                side = "buy" if side_val in (0, "buy", "BUY") else "sell"
                filled_size = order.get("filled") or 0
                size = order.get("quantity") or order.get("availableQuantity") or filled_size or 0
                price = order.get("price") or 0
                contract_id = order.get("productId") or self.config.contract_id
                order_type = "CLOSE" if side == self.config.close_order_side else "OPEN"

                if not order_id:
                    continue

                self._order_update_handler(
                    {
                        "contract_id": contract_id,
                        "order_id": str(order_id),
                        "status": status or "",
                        "side": side,
                        "order_type": order_type,
                        "filled_size": str(filled_size),
                        "size": str(size),
                        "price": str(price),
                    }
                )
        except Exception:
            pass


    async def connect(self) -> None:
        """
        Initialize REST client and prepare WS hooks.

        The ethereal-sdk exposes AsyncRESTClient (for trading / queries) and
        AsyncWSClient (for streaming order updates).
        """
        await self._ensure_rest_client()
        try:
            await self._ensure_products()
        except Exception as exc:
            self.logger.log(f"Failed to preload products: {exc}", "WARNING")

        # Try to open WS for order updates; no REST polling fallback to avoid delays.
        if self._order_update_handler:
            try:
                await self._start_ws()
            except Exception as exc:
                self.logger.log(f"WS connect failed: {exc}", "ERROR")

    async def disconnect(self) -> None:
        """Tear down REST/WS clients."""
        try:
            if self._ws_client:
                await self._ws_client.close()
        except Exception as exc:
            self.logger.log(f"Error closing Ethereal WS client: {exc}", "ERROR")
        self._ws_connected = False
        self._ws_client = None

        try:
            if self._rest_client and hasattr(self._rest_client, "close"):
                await self._rest_client.close()
        except Exception as exc:
            self.logger.log(f"Error closing Ethereal REST client: {exc}", "ERROR")

        self._ws_client = None
        self._rest_client = None

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "ethereal"

    def setup_order_update_handler(self, handler) -> None:
        """
        Register a callback for order updates.

        AsyncWSClient supports a callbacks dict keyed by stream type
        (see docs for subscribe/unsubscribe). Once WS wiring is added,
        self._order_update_handler will be invoked from the WS message handler.
        """
        self._order_update_handler = handler

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """
        Place an open order using AsyncRESTClient.create_order().
        """
        try:
            await self._ensure_rest_client()
        except ImportError as exc:
            return OrderResult(success=False, error_message=str(exc))

        if not self.private_key or not self.subaccount_id:
            return OrderResult(
                success=False,
                error_message="Missing ETHEREAL_PRIVATE_KEY or ETHEREAL_SUBACCOUNT_ID"
            )

        product = await self._get_product_by_ticker(contract_id or self.config.contract_id)
        if not product:
            return OrderResult(success=False, error_message="Product not found for ticker")

        best_bid, best_ask = await self._fetch_bbo(product.id)
        if best_bid == 0 and best_ask == 0:
            return OrderResult(success=False, error_message="Failed to fetch order book")

        side = 0 if direction == "buy" else 1
        price = self._maker_price(direction, best_bid, best_ask)

        try:
            order = await self._rest_client.create_order(
                order_type="LIMIT",
                quantity=float(quantity),
                side=side,
                price=float(price),
                product_id=product.id,
                subaccount=self.subaccount_hex,
                time_in_force="GTD",
                post_only=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return OrderResult(success=False, error_message=str(exc))

        order_id = getattr(order, "id", None) or (order.get("id") if isinstance(order, dict) else None)
        return OrderResult(
            success=bool(order_id),
            order_id=str(order_id) if order_id else None,
            side="buy" if side == 0 else "sell",
            size=Decimal(str(quantity)),
            price=Decimal(str(price)),
            status=getattr(order, "status", "OPEN") if order_id else "FAILED",
        )

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """
        Place a close/reduce order using AsyncRESTClient.create_order().
        """
        try:
            await self._ensure_rest_client()
        except ImportError as exc:
            return OrderResult(success=False, error_message=str(exc))

        if not self.private_key or not self.subaccount_id:
            return OrderResult(
                success=False,
                error_message="Missing ETHEREAL_PRIVATE_KEY or ETHEREAL_SUBACCOUNT_ID"
            )

        product = await self._get_product_by_ticker(contract_id or self.config.contract_id)
        if not product:
            return OrderResult(success=False, error_message="Product not found for ticker")

        side_val = 0 if side.lower() == "buy" else 1
        try:
            order = await self._rest_client.create_order(
                order_type="LIMIT",
                quantity=float(quantity),
                side=side_val,
                price=float(price),
                product_id=product.id,
                subaccount=self.subaccount_hex,
                time_in_force="GTD",
                reduce_only=True,
                post_only=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return OrderResult(success=False, error_message=str(exc))

        order_id = getattr(order, "id", None) or (order.get("id") if isinstance(order, dict) else None)
        return OrderResult(
            success=bool(order_id),
            order_id=str(order_id) if order_id else None,
            side="buy" if side_val == 0 else "sell",
            size=Decimal(str(quantity)),
            price=Decimal(str(price)),
            status=getattr(order, "status", "OPEN") if order_id else "FAILED",
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an order.

        The SDK exposes AsyncRESTClient.cancel_orders([...]) / cancel_all_orders().
        """
        try:
            await self._ensure_rest_client()
        except ImportError as exc:
            return OrderResult(success=False, error_message=str(exc))

        if not self.private_key or not self.subaccount_id:
            return OrderResult(
                success=False,
                error_message="Missing ETHEREAL_PRIVATE_KEY or ETHEREAL_SUBACCOUNT_ID"
            )

        try:
            await self._rest_client.cancel_orders(
                order_ids=[order_id],
                sender=self._rest_client.chain.address if hasattr(self._rest_client, "chain") else None,
                subaccount=self.subaccount_hex,
            )
            return OrderResult(success=True, order_id=order_id)
        except Exception as exc:  # pylint: disable=broad-except
            return OrderResult(success=False, error_message=str(exc))

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """
        Fetch order details via AsyncRESTClient.get_order().
        """
        try:
            await self._ensure_rest_client()
        except ImportError:
            return None

        try:
            order = await self._rest_client.get_order(id=order_id)
        except Exception:
            order = None

        if not order:
            return None

        side = getattr(order, "side", None)
        size = getattr(order, "quantity", None)
        price = getattr(order, "price", None)
        status = self._normalize_status(getattr(order, "status", "UNKNOWN"))
        filled = getattr(order, "filled", None)
        remaining = getattr(order, "available_quantity", None)

        return OrderInfo(
            order_id=order_id,
            side="buy" if side in (0, "buy", "BUY") else "sell",
            size=Decimal(str(size or "0")),
            price=Decimal(str(price or "0")),
            status=status,
            filled_size=Decimal(str(filled or "0")),
            remaining_size=Decimal(str(remaining or "0")),
        )

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """
        List open orders for the configured contract.

        The SDK provides AsyncRESTClient.list_orders() / list_trades().
        """
        try:
            await self._ensure_rest_client()
        except ImportError:
            return []

        if not self.subaccount_id:
            self.logger.log("ETHEREAL_SUBACCOUNT_ID not set; cannot fetch active orders.", "WARNING")
            return []

        product = await self._get_product_by_ticker(contract_id)
        product_id = getattr(product, "id", None) if product else None

        try:
            orders = await self._rest_client.list_orders(
                subaccount_id=self.subaccount_id,
                product_ids=[product_id] if product_id else None,
                statuses=["NEW", "PENDING", "FILLED_PARTIAL"],
                is_working=True,
            )
        except Exception:
            orders = []

        order_infos: List[OrderInfo] = []
        for order in orders or []:
            try:
                status = self._normalize_status(getattr(order, "status", ""))
                # Only keep active orders
                if status not in ("OPEN", "PARTIALLY_FILLED"):
                    continue

                order_infos.append(
                    OrderInfo(
                        order_id=str(getattr(order, "id", "")),
                        side="buy" if getattr(order, "side", 0) in (0, "buy", "BUY") else "sell",
                        size=Decimal(str(getattr(order, "quantity", "0"))),
                        price=Decimal(str(getattr(order, "price", "0"))),
                        status=status,
                        filled_size=Decimal(str(getattr(order, "filled", "0"))),
                        remaining_size=Decimal(str(getattr(order, "available_quantity", getattr(order, "quantity", "0")))),
                    )
                )
            except Exception:
                continue

        return order_infos

    async def get_order_price(self, direction: str) -> Decimal:
        """
        Provide a placeholder price used by TradingBot when pacing orders.

        Uses current BBO to place near-touching maker orders.
        """
        product = await self._get_product_by_ticker(self.config.contract_id or self.config.ticker)
        if not product:
            raise ValueError("Product not found for ticker")
        best_bid, best_ask = await self._fetch_bbo(product.id)
        return self._maker_price(direction, best_bid, best_ask)

    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Expose BBO for TradingBot stop/price checks."""
        product = await self._get_product_by_ticker(contract_id or self.config.contract_id)
        if not product:
            return Decimal(0), Decimal(0)
        return await self._fetch_bbo(product.id)

    async def _fetch_bbo(self, product_id: Any) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask via get_market_liquidity."""
        try:
            liquidity = await self._rest_client.get_market_liquidity(product_id=product_id)
        except Exception as exc:
            # Log richer context so we know why BBO is missing
            self.logger.log(
                f"[ethereal] get_market_liquidity failed for {product_id}: "
                f"{type(exc).__name__}: {exc!r}",
                "WARNING",
            )
            return Decimal(0), Decimal(0)

        bids = getattr(liquidity, "bids", None) or []
        asks = getattr(liquidity, "asks", None) or []
        best_bid = Decimal(str(bids[0][0])) if bids else Decimal(0)
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal(0)
        return best_bid, best_ask

    def _maker_price(self, direction: str, best_bid: Decimal, best_ask: Decimal) -> Decimal:
        """Choose a maker-friendly price using tick_size to avoid post_only rejection."""
        tick = self.config.tick_size or Decimal(0)
        price: Decimal
        if direction == "buy":
            # Aim just below best ask; if no ask, fall back to best bid.
            candidate = best_ask - tick if best_ask > 0 and tick > 0 else best_bid
            price = candidate if candidate > 0 else best_ask or best_bid
        else:
            # Aim just above best bid; if no bid, fall back to best ask.
            candidate = best_bid + tick if best_bid > 0 and tick > 0 else best_ask
            price = candidate if candidate > 0 else best_bid or best_ask
        return self.round_to_tick(price)

    async def list_positions(self, subaccount_id: Optional[str] = None) -> List[Any]:
        """
        Fetch positions for a subaccount using any available SDK method.

        - subaccount_id: overrides the default from ENV if provided.
        Returns a list (can be empty).
        """
        try:
            await self._ensure_rest_client()
        except ImportError as exc:
            self.logger.log(f"ethereal-sdk not installed: {exc}", "ERROR")
            return []

        rest = self._rest_client

        sid = subaccount_id or self.subaccount_id
        if not sid:
            if not self._warned_missing_subaccount:
                self.logger.log("ETHEREAL_SUBACCOUNT_ID not set; cannot fetch positions.", "WARN")
                self._warned_missing_subaccount = True
            return []

        if hasattr(rest, "list_positions"):
            try:
                resp = await rest.list_positions(subaccount_id=sid)
                return self._extract_positions(resp)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.log(f"[ethereal] list_positions failed: {exc}", "WARNING")

        self.logger.log("No position endpoint available on Ethereal client", "ERROR")
        return []

    def _extract_positions(self, resp: Any) -> List[Any]:
        """Normalize common Ethereal SDK response shapes to a list."""
        if resp is None:
            return []
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            for key in ("positions", "data", "result", "results"):
                if key in resp and isinstance(resp[key], list):
                    return resp[key]
        if hasattr(resp, "positions"):
            maybe = getattr(resp, "positions")
            if isinstance(maybe, list):
                return maybe
        return []

    def _normalize_status(self, status_raw: Any) -> str:
        """Normalize SDK status to bot-friendly keywords."""
        if status_raw is None:
            return ""
        s = str(status_raw)
        if "." in s:
            s = s.split(".")[-1]
        s_up = s.upper()
        if s_up in ("NEW", "PENDING", "OPEN"):
            return "OPEN"
        if s_up in ("FILLED_PARTIAL", "FILLED_PARTIALLY", "PARTIALLY_FILLED"):
            return "PARTIALLY_FILLED"
        if s_up == "FILLED":
            return "FILLED"
        if s_up in ("CANCELED", "CANCELLED", "EXPIRED"):
            return "CANCELED"
        return s_up

    async def get_account_positions(self) -> Decimal:
        """
        Retrieve current position size for the configured contract.

        The SDK provides AsyncRESTClient.list_positions() and get_position().
        """
        await self._ensure_rest_client()
        positions = await self.list_positions()
        target_id = str(self.config.contract_id)
        for pos in positions:
            product_id = (
                pos.get("product_id")
                if isinstance(pos, dict)
                else getattr(pos, "product_id", None)
            )
            if not product_id:
                product_id = (
                    pos.get("productId")
                    if isinstance(pos, dict)
                    else getattr(pos, "productId", None)
                )

            if product_id and str(product_id) == target_id:
                size_val = (
                    pos.get("size")
                    if isinstance(pos, dict)
                    else getattr(pos, "size", None)
                )
                try:
                    return abs(Decimal(str(size_val or "0")))
                except Exception:
                    return Decimal(0)

        return Decimal(0)

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """
        Retrieve contract identifier and tick size for a ticker.

        Uses products_by_ticker; falls back to ticker if lookup fails.
        """
        ticker = self.config.ticker
        if not ticker:
            raise ValueError("Ticker is empty")

        product = await self._get_product_by_ticker(ticker)
        if product:
            self.config.contract_id = getattr(product, "id", ticker.upper())
            tick_size_val = getattr(product, "tick_size", None)
            self.config.tick_size = Decimal(str(tick_size_val)) if tick_size_val else Decimal(0)
        else:
            self.config.contract_id = ticker.upper()
            self.config.tick_size = Decimal(0)

        return self.config.contract_id, self.config.tick_size
