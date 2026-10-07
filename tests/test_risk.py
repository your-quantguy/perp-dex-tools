"""Cross-venue policy uses each leg's own collateral and never guesses it."""

from decimal import Decimal as D

import pytest

from exchanges.market import MarginPosition, MarginSnapshot
from strategies.risk import evaluate_cross_venue_leg


def test_isolated_leg_uses_its_own_equity_even_if_account_is_well_funded():
    snapshot = MarginSnapshot(
        equity=D(1000), total_notional=D(200), maintenance_requirement=D(10),
        positions=(MarginPosition("ETH-USD", "ISOLATED", D(200),
                                  posted_collateral=D(25), unrealized_pnl=D(-5),
                                  maintenance_requirement=D(10)),),
    )
    risk = evaluate_cross_venue_leg(snapshot, "ETH-USD", D("2.5"), D("0.05"))
    assert risk.buffer_fraction == D("0.05")
    assert risk.opening_notional_room == 0


def test_missing_isolated_equity_stops_policy_instead_of_using_account_equity():
    snapshot = MarginSnapshot(
        equity=D(1000), total_notional=D(200), maintenance_requirement=None,
        positions=(MarginPosition("ETH-USD", "ISOLATED", D(200), posted_collateral=D(100)),),
    )
    with pytest.raises(ValueError, match="Isolated position equity is unavailable"):
        evaluate_cross_venue_leg(snapshot, "ETH-USD", D("2.5"), D("0.05"))


def test_unrelated_market_does_not_supply_target_leg_collateral():
    snapshot = MarginSnapshot(
        equity=D(100), total_notional=D(50), maintenance_requirement=None,
        positions=(MarginPosition("BTC-USD", "ISOLATED", D(50),
                                  posted_collateral=D(1000), unrealized_pnl=D(0)),),
    )
    risk = evaluate_cross_venue_leg(snapshot, "ETH-USD", D("2.5"), D("0.05"))
    assert risk.opening_notional_room == D(200)
    assert risk.buffer_fraction == D("1.95")


def test_isolated_market_leverage_above_policy_stops_opening():
    snapshot = MarginSnapshot(D(1000), D(0), None,
                              market_margin_mode="ISOLATED", market_leverage=D(20))
    risk = evaluate_cross_venue_leg(snapshot, "io:SNDK", D("2.5"), D("0.05"))
    assert risk.opening_notional_room == 0


def test_unified_account_needs_different_risk_policy():
    snapshot = MarginSnapshot(D(1000), D(0), None, account_mode="UNIFIEDACCOUNT")
    with pytest.raises(ValueError, match="separate risk policy"):
        evaluate_cross_venue_leg(snapshot, "io:SNDK", D("2.5"), D("0.05"))


def test_negative_equity_still_exposes_critical_buffer_for_emergency_close():
    snapshot = MarginSnapshot(D(-5), D(100), D(5))
    risk = evaluate_cross_venue_leg(snapshot, "io:SNDK", D("2.5"), D("0.05"))
    assert risk.buffer_fraction == D("-0.1")
    assert risk.opening_notional_room == 0


def test_strategy_conservatively_uses_reported_margin_used():
    snapshot = MarginSnapshot(D(100), D(100), None, margin_used=D(20))
    risk = evaluate_cross_venue_leg(snapshot, "ETH-USD", D("2.5"), D("0.05"))
    assert risk.buffer_fraction == D("0.8")


def test_strategy_uses_smaller_cross_equity_without_adapter_policy():
    snapshot = MarginSnapshot(D(100), D(100), D(5), cross_equity=D(50))
    risk = evaluate_cross_venue_leg(snapshot, "ETH-USD", D("2.5"), D("0.05"))
    assert risk.buffer_fraction == D("0.45")
    assert risk.opening_notional_room == D(25)
