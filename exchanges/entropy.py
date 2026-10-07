"""Entropy HIP-3 perpetuals on Hyperliquid, as a bounded IOC taker.

The official SDK signs and submits orders. All SDK calls run off the event
loop because its HTTP client is synchronous. Mutating calls are never retried.
"""

import asyncio
import os
import re
import time
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError, ServerError
from hyperliquid.utils.types import Cloid

from .base import BaseExchangeClient, MarketId, OrderInfo, OrderOutcomeUnknown, OrderResult
from .market import BookSnapshot, MarginPosition, MarginSnapshot


class EntropyClient(BaseExchangeClient):
    """The ``io`` perp DEX; the account belongs to the Hyperliquid address."""

    def _validate_config(self) -> None:
        if not re.fullmatch(r"0x[0-9a-fA-F]{64}", os.getenv("ENTROPY_API_PRIVATE_KEY", "").strip()):
            raise ValueError("ENTROPY_API_PRIVATE_KEY must be a 32-byte API wallet private key")
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", os.getenv("ENTROPY_ACCOUNT_ADDRESS", "").strip()):
            raise ValueError("ENTROPY_ACCOUNT_ADDRESS must be the main Hyperliquid account address")
        if os.getenv("ENTROPY_NETWORK", "mainnet") not in {"mainnet", "testnet"}:
            raise ValueError("ENTROPY_NETWORK must be mainnet or testnet")

    def __init__(self, config) -> None:
        super().__init__(config)
        self.network = os.getenv("ENTROPY_NETWORK", "mainnet")
        self.url = (constants.MAINNET_API_URL if self.network == "mainnet"
                    else constants.TESTNET_API_URL)
        self.address = os.environ["ENTROPY_ACCOUNT_ADDRESS"].strip().lower()
        self.wallet = Account.from_key(os.environ["ENTROPY_API_PRIVATE_KEY"].strip())
        self.coin = f"io:{config.ticker.upper().removesuffix('-USD')}"
        self.info: Info | None = None
        self.exchange: Exchange | None = None
        self.step = Decimal(0)
        self.sz_decimals = 0
        self._handler = None

    @property
    def network_id(self) -> str:
        return self.network

    def get_exchange_name(self) -> str:
        return "entropy"

    def get_account_description(self) -> tuple[str, ...]:
        return (f"Network: {self.network}", f"Hyperliquid account: {self.address}",
                f"API wallet: {self.wallet.address}", "Perp DEX: io")

    def setup_order_update_handler(self, handler) -> None:
        self._handler = handler

    def _info(self) -> Info:
        if self.info is None:
            raise RuntimeError("Entropy market metadata has not been loaded")
        return self.info

    @staticmethod
    async def _read(method, *args):
        try:
            return await asyncio.to_thread(method, *args)
        except ServerError as exc:
            raise ConnectionError(f"Entropy read returned HTTP {exc.status_code}") from exc
        except ClientError as exc:
            if exc.status_code == 429:
                raise ConnectionError("Entropy read was rate limited") from exc
            raise

    async def get_contract_attributes(self) -> tuple[MarketId, Decimal]:
        self.info = await asyncio.to_thread(Info, self.url, True, None, None, ["io"], 10)
        metadata = await self._read(self.info.meta, "io")
        market = next((m for m in metadata["universe"] if m["name"] == self.coin), None)
        if market is None or market.get("isDelisted"):
            raise ValueError(f"Entropy market {self.coin} is unavailable")
        self.sz_decimals = int(market["szDecimals"])
        if not 0 <= self.sz_decimals <= 8:
            raise ValueError("Entropy market has an invalid quantity grid")
        self.step = Decimal(1).scaleb(-self.sz_decimals)
        self.validate_hedge_quantity(self.config.quantity)
        # HIP-3 prices have a dynamic five-significant-figure grid.
        book = await self.get_spread_book(self.coin)
        tick = self._price_step(book.mid)
        self.config.contract_id, self.config.tick_size = self.coin, tick
        return self.coin, tick

    async def connect(self) -> None:
        info = self._info()
        role = await self._read(
            info.post, "/info", {"type": "userRole", "user": self.wallet.address})
        if (role.get("role") != "agent"
                or role.get("data", {}).get("user", "").lower() != self.address):
            raise ValueError("Entropy API wallet is not authorized for ENTROPY_ACCOUNT_ADDRESS")
        self.exchange = await asyncio.to_thread(
            Exchange, self.wallet, self.url, None, None, self.address, None, ["io"], 10)
        await self.get_margin_snapshot()
        if await self.get_active_orders(self.coin):
            raise ValueError("Entropy has pre-existing orders in this market; reconcile them before starting")

    async def disconnect(self) -> None:
        if self.exchange is not None:
            self.exchange.info.session.close()
        for client in (self.exchange, self.info):
            if client is not None:
                client.session.close()
        self.exchange = None
        self.info = None

    def get_quantity_step(self) -> Decimal:
        if self.step <= 0:
            raise RuntimeError("Entropy quantity grid is unavailable")
        return self.step

    def validate_hedge_quantity(self, quantity: Decimal) -> None:
        step = self.get_quantity_step()
        if not quantity.is_finite() or quantity <= 0 or quantity % step:
            raise ValueError(f"Entropy quantity must be positive and a multiple of {step}")

    def _price_step(self, price: Decimal) -> Decimal:
        if not price.is_finite() or price <= 0:
            raise ValueError("Entropy price is invalid")
        return max(Decimal(1).scaleb(price.adjusted() - 4),
                   Decimal(1).scaleb(self.sz_decimals - 6))

    def _bounded_price(self, best: Decimal, side: str, slippage: Decimal) -> Decimal:
        bound = best * (1 + slippage / 100 if side == "buy" else 1 - slippage / 100)
        # The dynamic significant-figure step can change at a power-of-ten edge.
        for _ in range(4):
            step = self._price_step(bound)
            rounded = (bound / step).to_integral_value(
                rounding=ROUND_FLOOR if side == "buy" else ROUND_CEILING) * step
            if rounded % self._price_step(rounded) == 0:
                return rounded
            bound = rounded
        raise ValueError("Entropy price cannot be aligned within slippage bound")

    async def get_spread_book(self, contract_id: MarketId) -> BookSnapshot:
        if contract_id != self.coin:
            raise ValueError("Entropy market identifier mismatch")
        data = await self._read(self._info().l2_snapshot, self.coin)
        if data.get("coin") != self.coin:
            raise ValueError("Entropy book is for another market")
        levels = data["levels"]
        if len(levels) != 2:
            raise ValueError("Entropy book is malformed")
        age_seconds = (time.time() * 1000 - int(data["time"])) / 1000
        if age_seconds > 5 or age_seconds < -2:
            raise ConnectionError("Entropy order book is stale")
        bids, asks = (tuple((Decimal(level["px"]), Decimal(level["sz"])) for level in side)
                      for side in levels)
        return BookSnapshot(bids, asks, time.monotonic() - max(0, age_seconds))

    async def fetch_bbo_prices(self, contract_id: MarketId) -> tuple[Decimal, Decimal]:
        book = await self.get_spread_book(contract_id)
        return book.bid, book.ask

    async def get_account_positions(self) -> Decimal:
        data = await self._read(self._info().user_state, self.address, "io")
        return sum((Decimal(item["position"]["szi"])
                    for item in data["assetPositions"] if item["position"]["coin"] == self.coin), Decimal(0))

    async def get_active_orders(self, contract_id: MarketId) -> list[OrderInfo]:
        if contract_id != self.coin:
            raise ValueError("Entropy market identifier mismatch")
        data = await self._read(self._info().open_orders, self.address, "io")
        return [OrderInfo(str(item["oid"]), "buy" if item["side"] == "B" else "sell",
                          Decimal(item["sz"]), Decimal(item["limitPx"]), "OPEN",
                          remaining_size=Decimal(item["sz"]))
                for item in data if item["coin"] == self.coin]

    async def get_margin_snapshot(self) -> MarginSnapshot:
        info = self._info()
        data = await self._read(info.user_state, self.address, "io")
        abstraction = await self._read(
            info.post, "/info", {"type": "userAbstraction", "user": self.address})
        market_data = await self._read(
            info.post, "/info", {"type": "activeAssetData", "user": self.address, "coin": self.coin})
        summary = data["marginSummary"]
        positions = []
        for item in data["assetPositions"]:
            position = item["position"]
            if Decimal(position["szi"]) == 0:
                continue
            mode = position["leverage"]["type"].upper()
            pnl = Decimal(position["unrealizedPnl"])
            # Hyperliquid reports the isolated margin balance as marginUsed;
            # leverage.rawUsd is a signed account offset, not posted collateral.
            posted = (Decimal(position["marginUsed"]) - pnl if mode == "ISOLATED" else None)
            liquidation = position.get("liquidationPx")
            positions.append(MarginPosition(
                market_id=position["coin"], mode=mode,
                notional=abs(Decimal(position["positionValue"])),
                posted_collateral=posted,
                unrealized_pnl=pnl,
                liquidation_price=Decimal(liquidation) if liquidation is not None else None,
            ))
        return MarginSnapshot(
            equity=Decimal(summary["accountValue"]),
            total_notional=Decimal(summary["totalNtlPos"]),
            maintenance_requirement=None,
            available_collateral=Decimal(data["withdrawable"]) if "withdrawable" in data else None,
            positions=tuple(positions),
            account_mode=str(abstraction).upper(),
            market_margin_mode=str(market_data["leverage"]["type"]).upper(),
            market_leverage=Decimal(str(market_data["leverage"]["value"])),
        )

    async def place_hedge_market_order(
        self, quantity: Decimal, side: str, max_slippage_percent: Decimal,
        reduce_only: bool = False,
    ) -> OrderResult:
        self.validate_hedge_quantity(quantity)
        if side not in {"buy", "sell"} or not Decimal(0) < max_slippage_percent <= Decimal(5):
            raise ValueError("Invalid Entropy IOC side or slippage limit")
        if self.exchange is None:
            raise RuntimeError("Entropy order session is not connected")
        book = await self.get_spread_book(self.coin)
        price = self._bounded_price(book.ask if side == "buy" else book.bid,
                                    side, max_slippage_percent)
        client_id = Cloid.from_str("0x" + uuid.uuid4().hex)
        submission = asyncio.create_task(asyncio.to_thread(
            self.exchange.order, self.coin, side == "buy", float(quantity), float(price),
            {"limit": {"tif": "Ioc"}}, reduce_only, client_id))
        try:
            response = await asyncio.shield(submission)
        except asyncio.CancelledError:
            try:
                await submission
            except Exception:
                pass
            raise OrderOutcomeUnknown("Entropy IOC was interrupted; inspect the account before restarting",
                                      client_id.to_raw()) from None
        except Exception as exc:
            raise OrderOutcomeUnknown("Entropy IOC result is unknown; inspect the account before retrying",
                                      client_id.to_raw()) from exc
        if response.get("status") != "ok":
            return OrderResult(False, side=side, size=quantity,
                               error_message=str(response.get("response", response)))
        statuses = response.get("response", {}).get("data", {}).get("statuses", [])
        if len(statuses) != 1:
            raise OrderOutcomeUnknown("Entropy IOC response lacks one definitive status", client_id.to_raw())
        status = statuses[0]
        if "error" in status:
            return OrderResult(False, side=side, size=quantity, error_message=str(status["error"]))
        if "filled" in status:
            fill = status["filled"]
            return OrderResult(True, str(fill["oid"]), side, quantity, price, "FILLED",
                               filled_size=Decimal(fill["totalSz"]))
        # A resting IOC or unrecognized status is unsafe to resubmit.
        raise OrderOutcomeUnknown("Entropy IOC did not return a final fill or rejection", client_id.to_raw())
