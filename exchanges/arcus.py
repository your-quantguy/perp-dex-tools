"""Arcus perpetuals: Ed25519 API signing, REST orders and streaming snapshots.

Protocol references: https://docs.arcus.xyz/api-reference/authentication and
https://docs.arcus.xyz/api-reference/channels. Never retries a mutating request.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from urllib.parse import quote

import aiohttp
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .base import (BaseExchangeClient, LIVE_ORDER_STATUSES, OrderInfo,
                   OrderOutcomeUnknown, OrderResult, TERMINAL_ORDER_STATUSES)

logger = logging.getLogger(__name__)
TERMINAL = TERMINAL_ORDER_STATUSES
LIVE = LIVE_ORDER_STATUSES


class ArcusAPIError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"Arcus HTTP {status}: {message}")


class ArcusClient(BaseExchangeClient):
    @property
    def network_id(self) -> str:
        return self.network

    def get_quantity_step(self) -> Decimal:
        if not hasattr(self, "step") or self.step <= 0:
            raise RuntimeError("Arcus market quantity step is unavailable")
        return self.step

    def get_account_description(self) -> tuple[str, ...]:
        return (f"Network: {self.network}", f"Trading account: {self.address}",
                f"Subaccount: {self.account_index}", f"API key: {self.api_key}")

    def validate_hedge_quantity(self, quantity: Decimal) -> None:
        self._check_quantity(quantity)

    def _validate_config(self):
        if not os.getenv("ARCUS_API_SIGNING_KEY", "").strip():
            raise ValueError("ARCUS_API_SIGNING_KEY is required (not your Ethereum private key)")
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", os.getenv("ARCUS_ADDRESS", "").strip()):
            raise ValueError("ARCUS_ADDRESS must be your master wallet's 0x address")
        if os.getenv("ARCUS_NETWORK", "mainnet") not in {"mainnet", "testnet"}:
            raise ValueError("ARCUS_NETWORK must be mainnet or testnet")
        try:
            index = int(os.getenv("ARCUS_ACCOUNT_INDEX", "0"))
        except ValueError:
            raise ValueError("ARCUS_ACCOUNT_INDEX must be an integer from 0 to 9") from None
        if not 0 <= index <= 9:
            raise ValueError("ARCUS_ACCOUNT_INDEX must be an integer from 0 to 9")

    def __init__(self, config):
        super().__init__(config)
        self.address = os.environ["ARCUS_ADDRESS"].strip().lower()
        self.account_index = int(os.getenv("ARCUS_ACCOUNT_INDEX", "0"))
        self.network = os.getenv("ARCUS_NETWORK", "mainnet")
        host = "api.arcus.xyz" if self.network == "mainnet" else "api.testnet.arcus.xyz"
        self.api_url, self.ws_url = f"https://{host}/v1", f"wss://{host}/v1/ws"
        secret = os.environ["ARCUS_API_SIGNING_KEY"].strip().removeprefix("0x")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", secret):
            raise ValueError("ARCUS_API_SIGNING_KEY must be the 32-byte hex API Signing Key")
        self._signer = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(secret))
        self.api_key = self._signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        expected = os.getenv("ARCUS_API_KEY", "").strip().removeprefix("0x").lower()
        if expected and expected != self.api_key:
            raise ValueError("ARCUS_API_KEY does not match ARCUS_API_SIGNING_KEY")
        ticker = config.ticker.upper()
        self.symbol = ticker if ticker.endswith("-USD") else f"{ticker}-USD"
        self.market = None
        self.http = None
        self._ws_http = None
        self._ws = None
        self._task = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._fatal = None
        self._book = None
        self._depth_bids = ()
        self._depth_asks = ()
        self._book_at = 0.0
        self._orders = {}
        self._versions = {}
        self._tracked = set()
        self._clients = set()
        self._handler = None
        self._changed = asyncio.Event()
        self._nonce = 0

    def get_exchange_name(self):
        return "arcus"

    def setup_order_update_handler(self, handler):
        self._handler = handler

    def _scope(self, **extra):
        return {"address": self.address, "accountIndex": self.account_index, **extra}

    def _session(self):
        if self.http is None or self.http.closed:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self.http

    async def _request(self, method, path, **kwargs):
        headers = {"X-API-Key": self.api_key, **kwargs.pop("headers", {})}
        async with self._session().request(method, self.api_url + path, headers=headers, **kwargs) as r:
            data = await r.json(content_type=None)
            if r.status >= 400:
                # Do not include request bodies/headers (signing material) in errors.
                raise ArcusAPIError(r.status, data.get("error", "request failed"))
            if not isinstance(data, dict):
                raise ValueError("Arcus returned a non-object response")
            return data

    async def _get(self, path, **params):
        for attempt in range(3):
            try:
                return await self._request("GET", path, params=params)
            except (aiohttp.ClientError, TimeoutError, ArcusAPIError) as exc:
                if isinstance(exc, ArcusAPIError) and exc.status not in {429, 500, 502, 503, 504}:
                    raise
                if attempt == 2:
                    raise
                await asyncio.sleep(attempt + 1)
        raise RuntimeError("Arcus GET retry loop ended without a response")

    def _market_data(self):
        if self.market is None:
            raise RuntimeError("Arcus market metadata is not loaded")
        return self.market

    async def get_contract_attributes(self):
        data = await self._get("/markets")
        self.market = next((m for m in data["markets"] if m["marketDisplayName"] == self.symbol), None)
        if not self.market or self.market["status"] != "ONLINE":
            raise ValueError(f"Arcus market {self.symbol} is not ONLINE")
        self.market_id = int(self.market["marketId"])
        self.tick = Decimal(self.market["tickSize"])
        self.step = Decimal(self.market["stepSize"])
        if self.tick <= 0 or self.step <= 0:
            raise ValueError("Arcus market has invalid tick/step sizes")
        self._check_quantity(self.config.quantity)
        if self.config.quantity < Decimal(self.market["minOrderSize"]):
            raise ValueError(f"Arcus minimum order size is {self.market['minOrderSize']}")
        self.config.contract_id, self.config.tick_size = self.symbol, self.tick
        return self.symbol, self.tick

    def _check_quantity(self, quantity):
        if not quantity.is_finite() or quantity <= 0 or quantity % self.step:
            raise ValueError(f"Arcus quantity must be positive and a multiple of {self.step}")
        market = self._market_data()
        if quantity > Decimal(market["maxOrderSize"]):
            raise ValueError(f"Arcus maximum order size is {market['maxOrderSize']}")

    def _price_tick(self, price):
        for tier in self._market_data().get("tickTiers", []):
            if "upToPrice" not in tier or price < Decimal(tier["upToPrice"]):
                return Decimal(tier["tick"])
        return self.tick

    def _snap(self, price, side):
        rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
        # Rounding up can cross a tier boundary, which changes the grid again.
        for _ in range(len(self._market_data().get("tickTiers", [])) + 2):
            tick = self._price_tick(price)
            price = (price / tick).to_integral_value(rounding=rounding) * tick
            if price % self._price_tick(price) == 0:
                return price
        raise ValueError("Cannot align Arcus price to its tier")

    def _timestamp(self):
        self._nonce = max(time.time_ns(), self._nonce + 1)
        return self._nonce

    def _headers(self, payload, ts):
        message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {"X-Timestamp": str(ts), "X-Signature": self._signer.sign(message).hex()}

    def _place_payload(self, quantity, price, side, reduce_only, client_id, ts, expiry):
        # Signature divisor is always BASE tickSize, not the price-tier tick.
        return {"ad": self.address, "ai": self.account_index, "c": client_id,
                "ct": ts, "g": expiry * 1000, "m": self.market_id, "op": 1,
                "p": self._units(price, self.tick), "q": self._units(quantity, self.step),
                "r": int(reduce_only), "s": 0 if side == "buy" else 1, "t": 3, "v": 1}

    @staticmethod
    def _units(value, unit):
        n = value / unit
        if n != n.to_integral_value():
            raise ValueError("Arcus price/quantity is not exactly aligned")
        return int(n)

    async def connect(self):
        await self._validate_api_key()
        await self._get("/account", **self._scope())  # Fail clearly for unfunded accounts.
        existing = await self.get_active_orders(self.symbol)
        if any(o.side == self.config.direction for o in existing):
            raise ValueError("Arcus has pre-existing opening orders; reconcile/cancel them before starting")
        self._stop.clear()
        self._task = asyncio.create_task(self._socket_loop())
        await self._wait_ready()

    async def _validate_api_key(self):
        data = await self._get("/apiKeys", address=self.address)
        keys = data.get("apiKeys")
        if not isinstance(keys, list):
            raise ValueError("Arcus API key listing is malformed")
        key = next((k for k in keys if k.get("apiKey", "").lower() == self.api_key), None)
        if not key or key.get("status") != "ACTIVE":
            raise ValueError("Arcus API Signing Key is not registered/active on this network and wallet")
        expiry = int(key["validUntil"])
        if expiry and expiry <= time.time_ns() // 1_000_000:
            raise ValueError("Arcus API key has expired; authorize a new key in the web app")
        if not key.get("allSubaccounts") and key.get("accountIndex") != self.account_index:
            raise ValueError("Arcus API key is not authorized for ARCUS_ACCOUNT_INDEX")

    async def _wait_ready(self):
        if self._fatal:
            raise RuntimeError(f"Arcus recovery needs attention: {self._fatal}")
        try:
            await asyncio.wait_for(self._ready.wait(), 30)
        except TimeoutError:
            raise ConnectionError("Arcus streams are not ready; no new orders submitted") from None
        if self._fatal:
            raise RuntimeError(f"Arcus recovery needs attention: {self._fatal}")

    async def _socket_loop(self):
        backoff = 1
        while not self._stop.is_set():
            self._ready.clear()
            self._book = None
            try:
                if self._ws_http is None or self._ws_http.closed:
                    self._ws_http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
                async with self._ws_http.ws_connect(self.ws_url, heartbeat=15, receive_timeout=45) as ws:
                    self._ws = ws
                    await ws.send_json({"type": "subscribe", "channel": "l2Orderbook",
                                        "id": self.symbol, "nLevels": 20})
                    await ws.send_json({"type": "subscribe", "channel": "orders", "id": self.address,
                                        "accountIndex": self.account_index, "market": self.symbol})
                    have_orders = False
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            raise ConnectionError("Arcus WebSocket error")
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if data.get("type") == "error":
                            raise ValueError("Arcus rejected a stream subscription")
                        if data.get("type") not in {"subscribed", "channel_data"}:
                            continue
                        contents = data.get("contents", {})
                        if data.get("channel") == "l2Orderbook" and data.get("id") == self.symbol:
                            self._handle_book(contents)
                        elif data.get("channel") == "orders" and data.get("id", "").lower() == self.address:
                            if data.get("accountIndex", self.account_index) != self.account_index:
                                continue
                            if contents.get("isSnapshot"):
                                await self._reconcile(contents)
                                have_orders = True
                            else:
                                self._remember(contents)
                        if have_orders and self._book:
                            self._ready.set()
                            backoff = 1
            except asyncio.CancelledError:
                raise
            except ValueError as exc:
                self._fatal = str(exc)
                self._ready.set()  # Wake any waiters, which check _fatal.
                logger.error("Arcus recovery needs attention: %s", exc)
                return
            except Exception as exc:
                logger.warning("Arcus stream disconnected; reconnecting: %s", exc)
            finally:
                self._ws = None
                if not self._fatal:
                    self._ready.clear()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)

    def _handle_book(self, contents):
        bids, asks = contents["bids"], contents["asks"]
        if not bids or not asks:
            self._book = None
            return
        bid, ask = Decimal(bids[0][0]), Decimal(asks[0][0])
        if bid <= 0 or ask <= bid:
            self._book = None
            return
        self._book, self._book_at = (bid, ask), time.monotonic()
        self._depth_bids = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in bids)
        self._depth_asks = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in asks)

    async def get_spread_book(self, contract_id):
        from exchanges.market import BookSnapshot

        await self.fetch_bbo_prices(contract_id)
        return BookSnapshot(self._depth_bids, self._depth_asks, self._book_at)

    async def get_spread_risk(self):
        from exchanges.market import AccountRisk

        data = await self._get("/account", **self._scope())
        equity = Decimal(str(data["equity"]))
        positions = data.get("positions")
        if not isinstance(positions, dict):
            raise ValueError("Arcus account positions are unavailable")
        if any(p.get("marginMode") != "CROSS" for p in positions.values()):
            raise ValueError("Spread risk does not support Arcus isolated positions")
        notional = sum((abs(Decimal(str(p["positionValueNotional"]))) for p in positions.values()), Decimal(0))
        return AccountRisk(equity, None, notional)

    async def _reconcile(self, contents):
        if not isinstance(contents.get("openOrders"), list) or not isinstance(contents.get("recentClosedOrders"), list):
            raise ValueError("Arcus orders snapshot is malformed")
        for raw in contents["openOrders"] + contents["recentClosedOrders"]:
            self._remember(raw)
        # A tracked order may fall outside the recent-closed snapshot window.
        for oid in list(self._tracked):
            current = self._orders.get(oid)
            if current is None or current.status not in TERMINAL:
                order = await self.get_order_info(oid, refresh=True)
                if order is None:
                    raise ValueError(f"Cannot reconcile tracked Arcus order {oid}")

    def _remember(self, raw):
        if raw.get("marketDisplayName") != self.symbol or int(raw["marketId"]) != self.market_id:
            return None
        oid = str(raw["orderId"])
        size, remaining = Decimal(raw["originalSize"]), Decimal(raw["remainingSize"])
        filled = Decimal(raw.get("filledSize", size - remaining))
        status = raw.get("state") or raw["status"]
        if raw["status"] in {"MARGIN_CANCELED", "TPSL_CANCELED"}:
            status = "CANCELED"
        if raw.get("cancelReason") == "MODIFY_CANCELED":
            # External modify is not a terminal cancel; do not enable a replacement.
            status = "PENDING"
        if status == "OPEN" and filled > 0:
            status = "PARTIALLY_FILLED"
        if raw["side"] not in {"BUY", "SELL"} or status not in LIVE | TERMINAL | {"PENDING", "UNTRIGGERED", "TPSL_PLACED", "TPSL_TRIGGERED", "ACK", "CANCEL_PENDING"}:
            raise ValueError(f"Unsupported Arcus order state for {oid}")
        if (not all(n.is_finite() for n in (size, remaining, filled)) or size <= 0
                or remaining < 0 or filled < 0 or filled + remaining != size):
            raise ValueError("Invalid Arcus order quantities")
        prior = self._orders.get(oid)
        version = (int(raw.get("sequenceNumber", 0)), int(raw.get("updatedAt", 0)))
        previous_version = self._versions.get(oid, (0, 0))
        if prior and (prior.status in TERMINAL or filled < prior.filled_size
                      or (version[0] and previous_version[0] and version[0] < previous_version[0])
                      or (not version[0] and version[1] < previous_version[1])):
            return prior
        order = OrderInfo(oid, raw["side"].lower(), size, Decimal(raw["price"]), status,
                          filled, remaining, raw.get("rejectionReason") or raw.get("cancelReason", ""))
        self._orders[oid], self._versions[oid] = order, max(version, previous_version)
        self._changed.set()
        owned = oid in self._tracked or raw.get("clientId") in self._clients
        if owned:
            self._tracked.add(oid)
        if owned and order != prior and self._handler:
            self._handler({"contract_id": self.symbol, "order_id": oid, "side": order.side,
                           "order_type": "OPEN" if order.side == self.config.direction else "CLOSE",
                           "status": status, "size": str(size), "filled_size": str(filled),
                           "price": str(order.price)})
        if status in TERMINAL:
            self._clients.discard(raw.get("clientId"))
        # Bound terminal history for long-running bots. Keep live orders intact.
        if len(self._orders) > 2048:
            for old_id in list(self._orders):
                if len(self._orders) <= 2048:
                    break
                if old_id != oid and self._orders[old_id].status in TERMINAL:
                    self._orders.pop(old_id)
                    self._versions.pop(old_id, None)
                    self._tracked.discard(old_id)
        return order

    async def fetch_bbo_prices(self, contract_id):
        if contract_id != self.symbol:
            raise ValueError("Arcus contract does not match configured market")
        await self._wait_ready()
        if self._book is None or time.monotonic() - self._book_at > 10:
            # A live socket can remain open without a usable book. Invalidate
            # readiness and force a new snapshot plus order reconciliation.
            self._ready.clear()
            if self._ws is not None and not self._ws.closed:
                try:
                    await self._ws.close()
                except Exception as exc:
                    logger.warning("Arcus stale socket could not be closed: %s", exc)
            raise ConnectionError("Arcus order book is stale or empty")
        return self._book

    async def get_order_price(self, direction):
        bid, ask = await self.fetch_bbo_prices(self.symbol)
        # One tier-aware increment inside the opposite quote, never crossing.
        price = ask - self._price_tick(ask) if direction == "buy" else bid + self._price_tick(bid)
        return self._snap(price, direction)

    async def place_open_order(self, contract_id, quantity, direction):
        await self.fetch_bbo_prices(contract_id)
        return await self._place(quantity, await self.get_order_price(direction), direction, False)

    async def place_post_only_order(self, contract_id, quantity, price, side, reduce_only):
        await self.fetch_bbo_prices(contract_id)
        return await self._place(quantity, price, side, reduce_only)

    async def place_reduce_only_ioc(self, contract_id, quantity, side, max_slippage_percent):
        bid, ask = await self.fetch_bbo_prices(contract_id)
        factor = max_slippage_percent / 100
        price = ask * (1 + factor) if side == "buy" else bid * (1 - factor)
        return await self._place(quantity, self._snap(price, side), side, True, tif="IOC")

    async def place_hedge_market_order(self, quantity: Decimal, side: str,
                                       max_slippage_percent: Decimal,
                                       reduce_only: bool = False) -> OrderResult:
        """Bounded IOC limit order; the caller confirms its resulting position."""
        if side not in {"buy", "sell"} or not Decimal(0) < max_slippage_percent <= Decimal(5):
            raise ValueError("Invalid Arcus taker side or slippage limit")
        self.validate_hedge_quantity(quantity)
        bid, ask = await self.fetch_bbo_prices(self.config.contract_id)
        factor = max_slippage_percent / 100
        protection = ask * (1 + factor) if side == "buy" else bid * (1 - factor)
        result = await self._place(quantity, self._snap(protection, side), side, reduce_only, tif="IOC")
        # A partially filled IOC may end CANCELED. It is still a known outcome;
        # the pair executor will read the resulting position and reject a short fill.
        if result.order_id and result.status == "CANCELED" and result.filled_size:
            result.success = True
        return result

    async def place_close_order(self, contract_id, quantity, price, side):
        # Only a confirmed, zero-fill post-only rejection is safe to resubmit.
        # A lost acknowledgement or an unconfirmed order must remain unknown.
        for attempt in range(5):
            bid, ask = await self.fetch_bbo_prices(contract_id)
            tick = self._price_tick(ask if side == "sell" else bid)
            if side == "sell":
                adjusted = max(price, ask + tick if attempt else ask)
            else:
                adjusted = min(price, bid - tick if attempt else bid)
            result = await self._place(quantity, self._snap(adjusted, side), side, True)
            if result.success:
                return result
            if (not result.order_id
                    or result.status not in {"REJECTED", "CANCELED"}
                    or result.filled_size != 0
                    or result.error_message != "POST_ONLY_WOULD_CROSS"):
                return result
        return OrderResult(False, result.order_id, side, quantity, result.price,
                           result.status, "Arcus close rejected after 5 POST_ONLY_WOULD_CROSS attempts",
                           result.filled_size)

    async def _place(self, quantity, price, side, reduce_only, tif="ALO"):
        self._check_quantity(quantity)
        if side not in {"buy", "sell"} or price <= 0 or not price.is_finite():
            raise ValueError("Arcus order side/price is invalid")
        market = self._market_data()
        if not reduce_only and (quantity < Decimal(market["minOrderSize"])
                                or quantity * price < Decimal(market["minOrderNotional"])):
            raise ValueError(f"Arcus minimum opening notional is {market['minOrderNotional']} USD")
        cid = "pdt-" + uuid.uuid4().hex
        self._clients.add(cid)
        ts = self._timestamp()
        expiry = ts // 1000 + 40 * 86400 * 1_000_000
        body = self._scope(marketId=self.market_id, orderSide=side.upper(), orderType="LIMIT",
                           quantity=format(quantity, "f"), price=format(price, "f"), timeInForce=tif,
                           goodTilTime=str(expiry), timestamp=ts, clientId=cid, reduceOnly=reduce_only)
        signed = self._place_payload(quantity, price, side, reduce_only, cid, ts, expiry)
        try:
            ack = await self._request("POST", "/placeOrder", params=self._scope(), json=body,
                                      headers=self._headers(signed, ts))
        except ArcusAPIError as exc:
            if exc.status < 500:
                self._clients.discard(cid)
                return OrderResult(False, error_message=str(exc), filled_size=Decimal(0))
            raise OrderOutcomeUnknown(f"Arcus submit outcome is unknown for clientId {cid}") from exc
        except Exception as exc:
            raise OrderOutcomeUnknown(f"Arcus submit outcome is unknown for clientId {cid}") from exc
        oid = ack.get("orderId")
        if not oid:
            raise OrderOutcomeUnknown(f"Arcus submit outcome is unknown for clientId {cid}: missing orderId")
        self._tracked.add(oid)
        try:
            order = await self._confirm(oid, LIVE | TERMINAL)
        except Exception as exc:
            raise OrderOutcomeUnknown(f"Arcus submit outcome is unknown for {oid}: {exc}", oid) from exc
        if order.status in TERMINAL:
            self._clients.discard(cid)
        return OrderResult(order.status in LIVE | {"FILLED"}, oid, side, quantity, price,
                           order.status, order.cancel_reason, order.filled_size)

    async def _confirm(self, oid, states):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self._changed.clear()
            order = self._orders.get(oid)
            if order and order.status in states:
                return order
            try:
                await asyncio.wait_for(self._changed.wait(), 1)
            except TimeoutError:
                try:
                    order = await self.get_order_info(oid, refresh=True)
                except (ArcusAPIError, aiohttp.ClientError, TimeoutError):
                    continue
                if order and order.status in states:
                    return order
        raise RuntimeError(f"Arcus order outcome is unknown for {oid}; check orders before restarting")

    async def cancel_order(self, order_id):
        order = await self.get_order_info(order_id)
        if order is None:
            raise OrderOutcomeUnknown(f"Arcus cancel outcome is unknown for {order_id}", order_id)
        if order.status not in TERMINAL:
            await self._wait_ready()
            ts = self._timestamp()
            signed = {"ad": self.address, "ai": self.account_index, "ct": ts,
                      "id": order_id, "m": self.market_id, "op": 2, "v": 1}
            try:
                await self._request("POST", "/cancelOrder", params=self._scope(),
                                    json=self._scope(kind="orderId", orderId=order_id,
                                                     marketId=self.market_id, timestamp=ts),
                                    headers=self._headers(signed, ts))
            except Exception:
                # A racing fill or a lost ACK still needs authoritative confirmation.
                pass
            try:
                order = await self._confirm(order_id, TERMINAL)
            except Exception as exc:
                raise OrderOutcomeUnknown(f"Arcus cancel outcome is unknown for {order_id}: {exc}", order_id) from exc
        return OrderResult(True, order_id, order.side, order.size, order.price,
                           order.status, filled_size=order.filled_size)

    async def get_order_info(self, order_id, refresh=False):
        order = self._orders.get(order_id)
        if order and (order.status in TERMINAL or not refresh):
            return order
        try:
            raw = await self._get("/order/" + quote(order_id, safe=""), **self._scope())
        except ArcusAPIError as exc:
            if exc.status == 404:
                if order_id in self._tracked and not refresh:
                    raise RuntimeError(f"Arcus order outcome is unknown for {order_id}: not found") from exc
                return None
            if order_id in self._tracked and not refresh:
                raise RuntimeError(f"Arcus order outcome is unknown for {order_id}: query failed") from exc
            raise
        except Exception as exc:
            if order_id in self._tracked and not refresh:
                raise RuntimeError(f"Arcus order outcome is unknown for {order_id}: query failed") from exc
            raise
        return self._remember(raw)

    async def get_order_for_recovery(self, order_id):
        """Read an in-flight order authoritatively while market data recovers."""
        try:
            raw = await self._get("/order/" + quote(order_id, safe=""), **self._scope())
        except (ArcusAPIError, aiohttp.ClientError, TimeoutError, OSError, ValueError, TypeError) as exc:
            raise ConnectionError(f"Arcus order {order_id} cannot yet be confirmed") from exc
        try:
            order = self._remember(raw)
        except (ValueError, TypeError, KeyError) as exc:
            raise ConnectionError(f"Arcus order {order_id} has an invalid recovery response") from exc
        if order is None or order.order_id != order_id or order.status not in LIVE | TERMINAL:
            raise ConnectionError(f"Arcus order {order_id} has no confirmed terminal/live state")
        return order

    async def get_active_orders(self, contract_id):
        if contract_id != self.symbol:
            raise ValueError("Arcus contract does not match configured market")
        data = await self._get("/openOrders", **self._scope(market=self.symbol, status="OPEN", limit=1000))
        rows = data.get("orders")
        if not isinstance(rows, list) or len(rows) >= 1000:
            raise ValueError("Arcus open-order response is malformed or capped; cannot safely count orders")
        result = []
        for raw in rows:
            order = self._remember(raw)
            if order and order.status in LIVE:
                result.append(order)
        return result

    async def get_account_positions(self):
        data = await self._get("/positions", **self._scope(market=self.symbol))
        positions = data.get("positions")
        if not isinstance(positions, dict):
            raise ValueError("Arcus positions response is malformed")
        position = positions.get(str(self.market_id))
        return Decimal(position["size"]) if position else Decimal(0)

    async def disconnect(self):
        self._stop.set()
        self._ready.clear()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for session in (self.http, self._ws_http):
            if session:
                await session.close()
        self.http = self._ws_http = None
