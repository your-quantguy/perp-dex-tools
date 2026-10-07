"""Risk policy for independently margined legs of a cross-venue spread."""

from dataclasses import dataclass
from decimal import Decimal

from exchanges.base import MarketId
from exchanges.market import MarginSnapshot


@dataclass(frozen=True)
class LegRisk:
    buffer_fraction: Decimal
    opening_notional_room: Decimal


def evaluate_cross_venue_leg(
    snapshot: MarginSnapshot, market_id: MarketId, max_leverage: Decimal,
    fallback_maintenance_fraction: Decimal,
) -> LegRisk:
    """Use the weaker of account and isolated-position buffers.

    Each venue is assessed on its own; unrealized gains on the other venue
    never enter this calculation. Unknown isolated equity stops new trades.
    """
    if (not snapshot.equity.is_finite()
            or not snapshot.total_notional.is_finite() or snapshot.total_notional < 0
            or not max_leverage.is_finite() or max_leverage <= 1
            or not fallback_maintenance_fraction.is_finite()
            or not 0 <= fallback_maintenance_fraction < 1):
        raise ValueError("Margin snapshot or risk parameters are invalid")
    if snapshot.account_mode not in {"STANDARD", "DEFAULT", "DISABLED"}:
        raise ValueError(f"Account margin mode {snapshot.account_mode} needs a separate risk policy")
    equity = snapshot.equity
    if snapshot.cross_equity is not None:
        if not snapshot.cross_equity.is_finite():
            raise ValueError("Cross equity is invalid")
        equity = min(equity, snapshot.cross_equity)
    maintenance = snapshot.maintenance_requirement
    if maintenance is None:
        maintenance = snapshot.total_notional * fallback_maintenance_fraction
        if snapshot.margin_used is not None:
            if not snapshot.margin_used.is_finite() or snapshot.margin_used < 0:
                raise ValueError("Margin used is invalid")
            maintenance = max(maintenance, snapshot.margin_used)
    if not maintenance.is_finite() or maintenance < 0:
        raise ValueError("Account maintenance requirement is invalid")
    buffer = ((equity - maintenance) / snapshot.total_notional
              if snapshot.total_notional else Decimal(1))
    room = max(Decimal(0), equity * max_leverage - snapshot.total_notional)
    if snapshot.market_margin_mode == "ISOLATED":
        if snapshot.market_leverage is None or not snapshot.market_leverage.is_finite():
            raise ValueError("Isolated market leverage is unavailable")
        if snapshot.market_leverage > max_leverage:
            room = Decimal(0)

    for position in snapshot.positions:
        if position.market_id != market_id:
            continue
        if position.mode == "CROSS":
            continue
        if position.mode != "ISOLATED":
            raise ValueError(f"Unknown margin mode: {position.mode}")
        if position.posted_collateral is None or position.unrealized_pnl is None:
            raise ValueError("Isolated position equity is unavailable; manual risk review required")
        equity = position.posted_collateral + position.unrealized_pnl
        requirement = position.maintenance_requirement
        if requirement is None:
            requirement = position.notional * fallback_maintenance_fraction
        if (not equity.is_finite() or not requirement.is_finite()
                or not position.notional.is_finite() or position.notional <= 0
                or requirement < 0):
            raise ValueError("Isolated margin facts are invalid")
        buffer = min(buffer, (equity - requirement) / position.notional)
        room = min(room, max(Decimal(0), equity * max_leverage - position.notional))
    return LegRisk(buffer, room)
