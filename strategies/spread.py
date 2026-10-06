"""Bounded maker-first convergence strategy for one market and two accounts."""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import aiohttp

from exchanges import ExchangeFactory
from exchanges.base import HedgeExecutionClient, OrderResult
from exchanges.config import VenueConfig
from execution.pair import MakerFirstPairExecutor
from helpers import TradingLogger
from helpers.notifications import Notifier
from strategies.runner import StrategyRunner
from strategies.spread_model import (
    BPS, AccountRisk, BookSnapshot, basis_bps, closing_signal, common_size_step, opening_signal,
    opposite_band_exit, rounded_size,
)


@dataclass(frozen=True)
class SpreadConfig:
    venue_a: str
    venue_b: str
    maker_exchange: str
    ticker: str
    center_bps: Decimal
    entry_band_bps: Decimal
    exit_band_bps: Decimal
    size: Decimal
    max_position: Decimal
    maker_fee_bps: Decimal
    taker_fee_bps: Decimal
    exit_reserve_bps: Decimal
    carry_reserve_bps: Decimal
    min_profit_bps: Decimal
    maintenance_fraction_a: Decimal
    maintenance_fraction_b: Decimal
    warning_buffer_fraction: Decimal
    critical_buffer_fraction: Decimal
    state_file: Path
    observation_file: Path | None = None
    exit_rule: str = "center"
    max_leverage: Decimal = Decimal("2.5")
    max_slippage: Decimal = Decimal("0.05")
    fill_timeout: float = 5
    target_timeout: float = 30
    recovery_timeout: float = 15
    poll_interval: float = 5
    max_book_age: float = 5
    max_book_skew: float = 2
    resume: bool = False
    execution_mode: str = "maker_taker"

    @property
    def exchange(self) -> str:
        return self.maker_exchange

    @property
    def hedge_exchange(self) -> str:
        return self.venue_b if self.maker_exchange == self.venue_a else self.venue_a

    def __post_init__(self) -> None:
        if {self.venue_a, self.venue_b} not in ({"bulk", "lighter"}, {"arcus", "lighter"}):
            raise ValueError("Live spread strategy supports Bulk/Lighter or Arcus/Lighter")
        if self.maker_exchange not in {"bulk", "arcus"} or self.hedge_exchange != "lighter":
            raise ValueError("This maker-taker executor requires Bulk or Arcus maker and Lighter taker")
        if self.execution_mode != "maker_taker":
            raise ValueError("Only maker_taker execution is implemented")
        if self.exit_rule not in {"center", "opposite_band"}:
            raise ValueError("exit_rule must be center or opposite_band")
        if (self.entry_band_bps <= self.exit_band_bps or self.exit_band_bps < 0
                or self.size <= 0 or self.max_position < self.size or self.max_leverage <= 1
                or not Decimal(0) < self.max_slippage <= Decimal(5)):
            raise ValueError("Invalid entry/exit, size, leverage or slippage limits")
        costs = (self.maker_fee_bps, self.taker_fee_bps, self.exit_reserve_bps,
                 self.carry_reserve_bps, self.min_profit_bps)
        if any(x < 0 for x in costs):
            raise ValueError("Cost reserves and minimum profit must be nonnegative")
        if not (Decimal(0) < self.critical_buffer_fraction < self.warning_buffer_fraction < 1):
            raise ValueError("Risk buffers must satisfy 0 < critical < warning < 1")
        if any(not Decimal(0) <= x < 1 for x in
               (self.maintenance_fraction_a, self.maintenance_fraction_b)):
            raise ValueError("Maintenance fractions must be in [0, 1)")
        if min(self.fill_timeout, self.recovery_timeout, self.poll_interval,
               self.max_book_age, self.max_book_skew) <= 0:
            raise ValueError("Timeouts and book limits must be positive")


@dataclass
class SpreadPosition:
    maker_position: Decimal = Decimal(0)
    entry_cash_per_unit: Decimal = Decimal(0)
    opened_at: float = 0


@dataclass(frozen=True)
class SpreadQuote:
    quantity: Decimal
    maker_side: str
    maker_price: Decimal
    hedge_price: Decimal
    basis_bps: Decimal
    cash_per_unit: Decimal
    fees_per_unit: Decimal


class SpreadExecution(Protocol):
    async def quote(self, maker_side: str, quantity_limit: Decimal,
                    maker_book: BookSnapshot, hedge_book: BookSnapshot) -> SpreadQuote | None: ...
    async def execute_once(self, quote: SpreadQuote, position: Decimal,
                           reduce_only: bool) -> Decimal: ...


@runtime_checkable
class SpreadDataClient(Protocol):
    async def get_spread_book(self, contract_id) -> BookSnapshot: ...
    async def get_spread_risk(self) -> AccountRisk: ...


@runtime_checkable
class SpreadMakerClient(SpreadDataClient, Protocol):
    async def place_reduce_only_ioc(self, contract_id, quantity: Decimal, side: str,
                                     max_slippage_percent: Decimal) -> OrderResult: ...


class MakerTakerExecution:
    """One expiring post-only attempt, reusing partial-fill reconciliation."""

    def __init__(self, pair: MakerFirstPairExecutor, config: SpreadConfig):
        self.pair = pair
        self.config = config

    @staticmethod
    def _depth(book: BookSnapshot, side: str, limit_percent: Decimal) -> Decimal:
        best = book.ask if side == "buy" else book.bid
        bound = best * (1 + limit_percent / 100) if side == "buy" else best * (1 - limit_percent / 100)
        return sum((q for p, q in (book.asks if side == "buy" else book.bids)
                    if (p <= bound if side == "buy" else p >= bound)), Decimal(0))

    def _step(self) -> Decimal:
        maker_step = getattr(self.pair.maker, "_lot_size", None) or getattr(self.pair.maker, "step", None)
        hedge_multiplier = getattr(self.pair.hedge, "base_amount_multiplier", None)
        if not maker_step or not hedge_multiplier:
            raise RuntimeError("Venue quantity precision is unavailable")
        return common_size_step(Decimal(str(maker_step)), Decimal(1) / Decimal(hedge_multiplier))

    async def quote(self, maker_side: str, quantity_limit: Decimal,
                    maker_book: BookSnapshot, hedge_book: BookSnapshot) -> SpreadQuote | None:
        if quantity_limit <= 0:
            return None
        if (time.monotonic() - maker_book.observed_at > self.config.max_book_age
                or time.monotonic() - hedge_book.observed_at > self.config.max_book_age):
            return None
        hedge_side = "sell" if maker_side == "buy" else "buy"
        quantity = min(quantity_limit, self._depth(hedge_book, hedge_side, self.config.max_slippage))
        quantity = rounded_size(quantity, self._step()) if quantity > 0 else Decimal(0)
        if quantity <= 0:
            return None
        maker_price = await self.pair.maker.get_order_price(maker_side)
        if time.monotonic() - hedge_book.observed_at > self.config.max_book_age:
            return None
        if hedge_book.sweep(hedge_side, quantity, self.config.max_slippage) is None:
            return None
        hedge_price = (hedge_book.ask * (1 + self.config.max_slippage / 100) if hedge_side == "buy"
                       else hedge_book.bid * (1 - self.config.max_slippage / 100))
        a, b = ((maker_price, hedge_price) if self.config.venue_a == self.config.maker_exchange
                else (hedge_price, maker_price))
        basis = (a / b - 1) * BPS
        cash = hedge_price - maker_price if maker_side == "buy" else maker_price - hedge_price
        fees = (maker_price * self.config.maker_fee_bps + hedge_price * self.config.taker_fee_bps) / BPS
        return SpreadQuote(quantity, maker_side, maker_price, hedge_price, basis, cash, fees)

    async def execute_once(self, quote: SpreadQuote, position: Decimal, reduce_only: bool) -> Decimal:
        return await self.pair.execute_maker_order(
            quote.maker_side, quote.quantity, position, reduce_only, quote.maker_price)


class SpreadStrategy:
    def __init__(self, config: SpreadConfig, maker=None, hedge=None, notifier=None):
        self.config = config
        self.logger = TradingLogger("spread", config.ticker, log_to_console=True)
        self.maker_config = VenueConfig(config.ticker, "", Decimal(0), config.size, "buy")
        self.hedge_config = VenueConfig(config.ticker, "", Decimal(0), config.size, "sell")
        self.maker = maker or ExchangeFactory.create_exchange(config.maker_exchange, self.maker_config)
        selected = hedge or ExchangeFactory.create_exchange(config.hedge_exchange, self.hedge_config)
        if not isinstance(selected, HedgeExecutionClient):
            raise TypeError("Taker venue lacks bounded immediate orders")
        self.hedge = selected
        maker_capability = self.maker
        hedge_capability = self.hedge
        if not isinstance(maker_capability, SpreadMakerClient) or not isinstance(hedge_capability, SpreadDataClient):
            raise TypeError("Spread venue lacks depth, risk or emergency-close capabilities")
        network = getattr(self.maker, "signature_domain", None) or getattr(self.maker, "network", "mainnet")
        if network != "mainnet":
            raise ValueError("Both spread venues must be on mainnet")
        self.maker.setup_order_update_handler(lambda _message: None)
        self.hedge.setup_order_update_handler(lambda _message: None)
        self.pair = MakerFirstPairExecutor(config, self.maker, self.hedge,
                                           self.maker_config, self.hedge_config, self.logger)
        self.execution: SpreadExecution = MakerTakerExecution(self.pair, config)
        self.notifier = notifier or Notifier()
        self.position = SpreadPosition()
        self._warning_sent = False
        self._data_warning_sent = False

    @property
    def venues(self):
        return self.maker, self.hedge

    async def prepare(self):
        self.maker_config.contract_id, self.maker_config.tick_size = await self.maker.get_contract_attributes()
        self.hedge_config.contract_id, self.hedge_config.tick_size = await self.hedge.get_contract_attributes()

    def _save(self):
        path = self.config.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"venue_a": self.config.venue_a, "venue_b": self.config.venue_b,
                "maker_exchange": self.config.maker_exchange, "ticker": self.config.ticker,
                "maker_position": str(self.position.maker_position),
                "entry_cash_per_unit": str(self.position.entry_cash_per_unit),
                "opened_at": self.position.opened_at}
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w") as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    async def _record(self, basis: Decimal, maker: BookSnapshot, hedge: BookSnapshot, buffer: Decimal):
        if self.config.observation_file is None:
            return
        quotes = {}
        for side in ("buy", "sell"):
            try:
                quote = await self.execution.quote(side, self.config.size, maker, hedge)
            except (ConnectionError, TimeoutError, ValueError):
                quote = None
            quotes[side] = (None if quote is None else {
                "quantity": str(quote.quantity), "cash_per_unit": str(quote.cash_per_unit),
                "fees_per_unit": str(quote.fees_per_unit), "basis_bps": str(quote.basis_bps),
            })
        path = self.config.observation_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"time": time.time(), "basis_bps": str(basis),
                "maker_bid": str(maker.bid), "maker_ask": str(maker.ask),
                "hedge_bid": str(hedge.bid), "hedge_ask": str(hedge.ask),
                "margin_buffer": str(buffer), "maker_position": str(self.position.maker_position),
                "quotes": quotes}
        with path.open("a") as stream:
            stream.write(json.dumps(data) + "\n")

    def _load(self):
        data = json.loads(self.config.state_file.read_text())
        for field in ("venue_a", "venue_b", "maker_exchange", "ticker"):
            if data[field] != getattr(self.config, field):
                raise RuntimeError(f"State file {field} differs from this strategy")
        self.position = SpreadPosition(Decimal(data["maker_position"]),
                                       Decimal(data["entry_cash_per_unit"]), float(data["opened_at"]))
        if self.position.maker_position and not 0 < self.position.opened_at <= time.time():
            raise RuntimeError("Saved spread entry time is invalid")

    async def _account_position(self) -> Decimal:
        return await self.pair.positions()

    async def _data_unavailable(self, exc: Exception):
        self.logger.log(f"Spread data unavailable; trading paused: {exc}", "WARNING")
        if not self._data_warning_sent:
            await self.notifier.send(f"Spread data unavailable; no new orders: {exc}")
            self._data_warning_sent = True

    async def _books(self) -> tuple[BookSnapshot, BookSnapshot, Decimal]:
        maker_book = await cast(SpreadMakerClient, self.maker).get_spread_book(self.maker_config.contract_id)
        hedge_book = await cast(SpreadDataClient, self.hedge).get_spread_book(self.hedge_config.contract_id)
        a, b = ((maker_book, hedge_book) if self.config.venue_a == self.config.maker_exchange
                else (hedge_book, maker_book))
        return maker_book, hedge_book, basis_bps(a, b, self.config.max_book_age, self.config.max_book_skew)

    async def _risks(self) -> tuple[AccountRisk, AccountRisk, Decimal]:
        maker = await cast(SpreadMakerClient, self.maker).get_spread_risk()
        hedge = await cast(SpreadDataClient, self.hedge).get_spread_risk()
        a, b = ((maker, hedge) if self.config.venue_a == self.config.maker_exchange
                else (hedge, maker))
        buffer = min(a.buffer_fraction(self.config.maintenance_fraction_a),
                     b.buffer_fraction(self.config.maintenance_fraction_b))
        return maker, hedge, buffer

    async def _emergency_close(self):
        self.logger.log("Critical margin buffer; attempting immediate reduce-only close", "ERROR")
        await self.notifier.send("Spread strategy critical margin buffer: attempting reduce-only close on both venues now.")
        for _ in range(3):
            current = await self._account_position()
            if current == 0:
                self.position = SpreadPosition()
                self._save()
                return
            if await self.maker.get_active_orders(self.maker_config.contract_id):
                raise RuntimeError("Active maker order blocks emergency close; manual review required")
            if await self.hedge.get_active_orders(self.hedge_config.contract_id):
                raise RuntimeError("Active hedge order blocks emergency close; manual review required")
            side = "sell" if current > 0 else "buy"
            result = await cast(SpreadMakerClient, self.maker).place_reduce_only_ioc(
                self.maker_config.contract_id, abs(current), side, self.config.max_slippage)
            # IOC finality and account state must be known before any repeat.
            if result.order_id:
                order = await self.maker.get_order_info(result.order_id)
                if order is None or order.status not in {"FILLED", "CANCELED"}:
                    raise RuntimeError("Emergency IOC outcome is unconfirmed")
            elif not result.success:
                raise RuntimeError(f"Emergency maker close rejected: {result.error_message}")
            maker_actual = await self.maker.get_account_positions()
            await self.pair.hedge_to(-maker_actual)
        if await self._account_position() != 0:
            raise RuntimeError("Emergency close was incomplete; inspect both venues immediately")
        self.position = SpreadPosition()
        self._save()

    async def _tick(self):
        actual = await self._account_position()
        if actual != self.position.maker_position:
            raise RuntimeError("Recorded and exchange positions differ; manual review required")
        try:
            maker_risk, hedge_risk, buffer = await self._risks()
        except (ConnectionError, TimeoutError, OSError, aiohttp.ClientError) as exc:
            await self._data_unavailable(exc)
            return
        if buffer <= self.config.critical_buffer_fraction and actual != 0:
            await self._emergency_close()
            return
        if buffer <= self.config.warning_buffer_fraction:
            if not self._warning_sent:
                await self.notifier.send(
                    f"Spread margin buffer {buffer:.3%} is below warning threshold; "
                    "add collateral or review the position. Opening is paused.")
                self._warning_sent = True
        else:
            self._warning_sent = False
        try:
            maker_book, hedge_book, basis = await self._books()
        except (ConnectionError, TimeoutError, OSError, ValueError, aiohttp.ClientError) as exc:
            await self._data_unavailable(exc)
            return
        self._data_warning_sent = False
        await self._record(basis, maker_book, hedge_book, buffer)
        if await self.maker.get_active_orders(self.maker_config.contract_id):
            raise RuntimeError("Unexpected maker order; manual review required")
        if await self.hedge.get_active_orders(self.hedge_config.contract_id):
            raise RuntimeError("Unexpected hedge order; manual review required")

        if buffer > self.config.warning_buffer_fraction:
            a_direction = opening_signal(basis, self.config.center_bps, self.config.entry_band_bps)
            maker_direction = a_direction if self.config.maker_exchange == self.config.venue_a else -a_direction
            if maker_direction and (actual == 0 or actual * maker_direction > 0):
                maker_side = "buy" if maker_direction > 0 else "sell"
                quantity = min(self.config.size, self.config.max_position - abs(actual),
                               maker_risk.opening_room(maker_book.mid, self.config.max_leverage),
                               hedge_risk.opening_room(hedge_book.mid, self.config.max_leverage))
                quote = await self.execution.quote(maker_side, quantity, maker_book, hedge_book)
                if quote is not None:
                    available_edge = (self.config.center_bps - quote.basis_bps if a_direction > 0
                                      else quote.basis_bps - self.config.center_bps)
                    required = (self.config.maker_fee_bps + self.config.taker_fee_bps
                                + self.config.exit_reserve_bps + self.config.carry_reserve_bps
                                + self.config.min_profit_bps)
                    if available_edge > required:
                        self.maker_config.direction = maker_side
                        filled = await self.execution.execute_once(quote, actual, False)
                        if filled:
                            entry_cash = quote.cash_per_unit - quote.fees_per_unit
                            new_size = abs(actual) + filled
                            weighted = (abs(actual) * self.position.entry_cash_per_unit
                                        + filled * entry_cash) / new_size
                            self.position = SpreadPosition(
                                (Decimal(1) if maker_side == "buy" else Decimal(-1)) * new_size,
                                weighted, self.position.opened_at or time.time())
                            self._save()
                            self.logger.log(f"Opened {filled} at conservative basis {quote.basis_bps:.2f} bps", "INFO")
                        return

        if actual == 0:
            return

        a_direction = (1 if actual > 0 else -1) * (1 if self.config.maker_exchange == self.config.venue_a else -1)
        exit_ready = (closing_signal(basis, self.config.center_bps, self.config.exit_band_bps, a_direction)
                      if self.config.exit_rule == "center" else
                      opposite_band_exit(basis, self.config.center_bps, self.config.entry_band_bps, a_direction))
        if not exit_ready:
            return
        maker_side = "sell" if actual > 0 else "buy"
        quote = await self.execution.quote(maker_side, min(abs(actual), self.config.size),
                                           maker_book, hedge_book)
        if quote is None:
            return
        held_days = max(Decimal(1), Decimal(str(time.time() - self.position.opened_at)) / Decimal(86400))
        reference_price = max(maker_book.mid, hedge_book.mid)
        net_per_unit = (self.position.entry_cash_per_unit
                        + quote.cash_per_unit - quote.fees_per_unit
                        - reference_price * self.config.carry_reserve_bps * held_days / BPS)
        if net_per_unit * BPS / reference_price < self.config.min_profit_bps:
            return
        self.maker_config.direction = maker_side
        filled = await self.execution.execute_once(quote, actual, True)
        if filled:
            remaining = abs(actual) - filled
            self.position = SpreadPosition(
                (Decimal(1) if actual > 0 else Decimal(-1)) * remaining,
                self.position.entry_cash_per_unit if remaining else Decimal(0),
                self.position.opened_at if remaining else 0)
            self._save()
            self.logger.log(f"Closed {filled}; conservative net {net_per_unit:.6f} per unit", "INFO")

    async def execute(self):
        if await self.maker.get_active_orders(self.maker_config.contract_id):
            raise RuntimeError("Maker account has active orders")
        if await self.hedge.get_active_orders(self.hedge_config.contract_id):
            raise RuntimeError("Hedge account has active orders")
        actual = await self._account_position()
        if actual:
            if not self.config.resume:
                raise RuntimeError("Open spread position requires manual review and --resume")
            self._load()
            if self.position.maker_position != actual:
                raise RuntimeError("Saved spread position differs from exchange positions")
        else:
            self.position = SpreadPosition()
            self._save()
        while True:
            await self._tick()
            await asyncio.sleep(self.config.poll_interval)

    async def on_failure(self, exc: BaseException):
        message = (f"[{self.config.venue_a}/{self.config.venue_b} {self.config.ticker}] "
                   f"Spread strategy stopped: {type(exc).__name__}: {exc}. "
                   "Inspect both venues and the state file before restarting.")
        self.logger.log(message, "ERROR")
        try:
            await self.notifier.send(message)
        except Exception as notify_error:
            self.logger.log(f"Notification failed: {notify_error}", "ERROR")

    async def run(self):
        await StrategyRunner().run(self)
