"""Pure signal, order-book and account-risk calculations for basis trading."""

import time
from math import lcm
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


BPS = Decimal(10_000)


@dataclass(frozen=True)
class BookSnapshot:
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    observed_at: float

    def __post_init__(self) -> None:
        for levels in (self.bids, self.asks):
            if (not levels or not any(q > 0 for _, q in levels)
                    or any(p <= 0 or q < 0 or not p.is_finite() or not q.is_finite() for p, q in levels)):
                raise ValueError("Order book lacks valid levels")
        if self.bid >= self.ask or self.observed_at <= 0:
            raise ValueError("Order book is crossed or has no timestamp")

    @property
    def bid(self) -> Decimal:
        return max(p for p, q in self.bids if q > 0)

    @property
    def ask(self) -> Decimal:
        return min(p for p, q in self.asks if q > 0)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    def sweep(self, side: str, quantity: Decimal, max_slippage_percent: Decimal) -> Decimal | None:
        """VWAP across all available levels inside the bounded taker price."""
        if quantity <= 0 or side not in {"buy", "sell"}:
            raise ValueError("Invalid taker sweep")
        best = self.ask if side == "buy" else self.bid
        limit = best * (1 + max_slippage_percent / 100) if side == "buy" else best * (1 - max_slippage_percent / 100)
        levels = sorted(self.asks if side == "buy" else self.bids, reverse=side == "sell")
        remaining = quantity
        cost = Decimal(0)
        for price, available in levels:
            if (side == "buy" and price > limit) or (side == "sell" and price < limit):
                break
            used = min(remaining, available)
            cost += used * price
            remaining -= used
            if remaining == 0:
                return cost / quantity
        return None


@dataclass(frozen=True)
class AccountRisk:
    equity: Decimal
    maintenance: Decimal | None
    notional: Decimal

    def buffer_fraction(self, fallback_maintenance_fraction: Decimal) -> Decimal:
        if self.equity <= 0 or self.notional < 0:
            raise ValueError("Account risk values are invalid")
        if self.notional == 0:
            return Decimal(1)
        maintenance = self.maintenance
        if maintenance is None:
            maintenance = self.notional * fallback_maintenance_fraction
        return (self.equity - maintenance) / self.notional

    def opening_room(self, price: Decimal, max_leverage: Decimal) -> Decimal:
        if self.equity <= 0 or self.notional < 0:
            raise ValueError("Account risk values are invalid")
        return max(Decimal(0), (self.equity * max_leverage - self.notional) / price)


def basis_bps(a: BookSnapshot, b: BookSnapshot, max_age: float, max_skew: float) -> Decimal:
    now = time.monotonic()
    if (now - a.observed_at > max_age or now - b.observed_at > max_age
            or abs(a.observed_at - b.observed_at) > max_skew):
        raise ConnectionError("Venue books are stale or not synchronized")
    return (a.mid / b.mid - 1) * BPS


def opening_signal(basis: Decimal, center: Decimal, entry_band: Decimal) -> int:
    """+1 is long A/short B; -1 is short A/long B."""
    if basis <= center - entry_band:
        return 1
    if basis >= center + entry_band:
        return -1
    return 0


def closing_signal(basis: Decimal, center: Decimal, exit_band: Decimal, a_direction: int) -> bool:
    return basis >= center - exit_band if a_direction > 0 else basis <= center + exit_band


def opposite_band_exit(basis: Decimal, center: Decimal, entry_band: Decimal, a_direction: int) -> bool:
    return basis >= center + entry_band if a_direction > 0 else basis <= center - entry_band


def rounded_size(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("Invalid quantity step")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def common_size_step(a: Decimal, b: Decimal) -> Decimal:
    """Smallest size representable exactly by both venue quantity grids."""
    if a <= 0 or b <= 0:
        raise ValueError("Quantity steps must be positive")
    exponent_a = a.as_tuple().exponent
    exponent_b = b.as_tuple().exponent
    if not isinstance(exponent_a, int) or not isinstance(exponent_b, int):
        raise ValueError("Quantity steps must be finite")
    scale = 10 ** max(-exponent_a, -exponent_b, 0)
    return Decimal(lcm(int(a * scale), int(b * scale))) / scale
