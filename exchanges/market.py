"""Venue-neutral market depth and account risk snapshots."""

from dataclasses import dataclass
from decimal import Decimal


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
