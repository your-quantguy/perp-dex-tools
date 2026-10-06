import asyncio
import time
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from exchanges.config import VenueConfig
from exchanges.arcus import ArcusClient
from exchanges.bulk import BulkClient
from exchanges.base import OrderResult
from scripts.compare_spread_exits import compare
from spread_mode import load_config
from strategies.spread import MakerTakerExecution, SpreadConfig, SpreadPosition, SpreadStrategy
from strategies.spread_model import AccountRisk, BookSnapshot, basis_bps, closing_signal, common_size_step, opening_signal


def book(bids, asks, age=0):
    return BookSnapshot(tuple((D(p), D(q)) for p, q in bids),
                        tuple((D(p), D(q)) for p, q in asks), time.monotonic() - age)


def config(tmp_path: Path):
    return SpreadConfig(
        venue_a="bulk", venue_b="lighter", maker_exchange="bulk", ticker="ETH",
        center_bps=D(0), entry_band_bps=D(100), exit_band_bps=D(5),
        size=D(1), max_position=D(2), maker_fee_bps=D(1), taker_fee_bps=D(2),
        exit_reserve_bps=D(5), carry_reserve_bps=D(2), min_profit_bps=D(3),
        maintenance_fraction_a=D("0.05"), maintenance_fraction_b=D("0.05"),
        warning_buffer_fraction=D("0.15"), critical_buffer_fraction=D("0.08"),
        state_file=tmp_path / "state.json",
    )


def test_multi_level_sweep_and_synchronized_basis():
    a = book([("90", "1")], [("91", "1")])
    b = book([("100", "0.4"), ("99.98", "0.6")], [("101", "1")])
    assert b.sweep("sell", D(1), D("0.05")) == D("99.988")
    assert b.sweep("sell", D("1.1"), D("0.05")) is None
    assert opening_signal(basis_bps(a, b, 5, 2), D(0), D(100)) == 1
    assert closing_signal(D(-4), D(0), D(5), 1)
    with pytest.raises(ConnectionError):
        basis_bps(book([("90", "1")], [("91", "1")], age=10), b, 5, 2)


def test_example_config_loads():
    loaded = load_config(Path("spread_config.example.json"), resume=False)
    assert loaded.maintenance_fraction_a == D("0.05")


def test_risk_limits_use_full_account_notional():
    risk = AccountRisk(D(100), D(5), D(200))
    assert risk.opening_room(D(100), D("2.5")) == D("0.5")
    assert risk.buffer_fraction(D("0.1")) == D("0.475")
    assert AccountRisk(D(100), None, D(200)).buffer_fraction(D("0.05")) == D("0.45")
    assert common_size_step(D("0.03"), D("0.02")) == D("0.06")


def strategy(tmp_path, maker_book, hedge_book, maker_risk=None) -> Any:
    subject: Any = SpreadStrategy.__new__(SpreadStrategy)
    subject.config = config(tmp_path)
    subject.maker_config = VenueConfig("ETH", "ETH-USD", D("0.01"), D(1), "buy")
    subject.hedge_config = VenueConfig("ETH", 1, D("0.01"), D(1), "sell")
    subject.maker = SimpleNamespace(
        _lot_size=D("0.1"), get_spread_book=AsyncMock(return_value=maker_book),
        get_spread_risk=AsyncMock(return_value=maker_risk or AccountRisk(D(1000), D(0), D(0))),
        get_active_orders=AsyncMock(return_value=[]), get_order_price=AsyncMock(return_value=D("90.5")),
    )
    subject.hedge = SimpleNamespace(
        base_amount_multiplier=10, get_spread_book=AsyncMock(return_value=hedge_book),
        get_spread_risk=AsyncMock(return_value=AccountRisk(D(1000), D(0), D(0))),
        get_active_orders=AsyncMock(return_value=[]),
    )
    subject.pair = SimpleNamespace(
        positions=AsyncMock(return_value=D(0)), maker=subject.maker, hedge=subject.hedge,
        execute_maker_order=AsyncMock(return_value=D("0.5")))
    subject.execution = MakerTakerExecution(cast(Any, subject.pair), subject.config)
    subject.position = SpreadPosition()
    subject.notifier = SimpleNamespace(send=AsyncMock())
    subject.logger = SimpleNamespace(log=Mock())
    subject._warning_sent = False
    subject._data_warning_sent = False
    return subject


def test_open_once_then_accumulate_only_within_cap(tmp_path):
    maker = book([("90", "10")], [("91", "10")])
    hedge = book([("100", "10")], [("101", "10")])
    subject = strategy(tmp_path, maker, hedge)
    asyncio.run(subject._tick())
    assert subject.position.maker_position == D("0.5")
    assert subject.pair.execute_maker_order.call_count == 1
    assert subject.config.state_file.is_file()
    subject.pair.positions.return_value = D("0.5")
    asyncio.run(subject._tick())
    assert subject.position.maker_position == D(1)
    assert subject.pair.execute_maker_order.call_count == 2


def test_warning_pauses_opening_and_sends_once(tmp_path):
    subject = strategy(tmp_path, book([("90", "1")], [("91", "1")]),
                       book([("100", "1")], [("101", "1")]),
                       AccountRisk(D(100), D(90), D(100)))
    asyncio.run(subject._tick())
    asyncio.run(subject._tick())
    subject.pair.execute_maker_order.assert_not_called()
    subject.notifier.send.assert_awaited_once()


def test_stale_book_pauses_and_recovers_without_order(tmp_path):
    maker = book([("90", "1")], [("91", "1")], age=10)
    subject = strategy(tmp_path, maker, book([("100", "1")], [("101", "1")]))
    asyncio.run(subject._tick())
    subject.pair.execute_maker_order.assert_not_called()
    subject.notifier.send.assert_awaited_once()
    subject.maker.get_spread_book.return_value = book([("90", "1")], [("91", "1")])
    asyncio.run(subject._tick())
    assert subject.pair.execute_maker_order.await_count == 1


def test_critical_buffer_invokes_emergency_before_signal(tmp_path):
    subject = strategy(tmp_path, book([("90", "1")], [("91", "1")]),
                       book([("100", "1")], [("101", "1")]),
                       AccountRisk(D(100), D(93), D(100)))
    subject.position = SpreadPosition(D(1), D(10), time.time())
    subject.pair.positions.return_value = D(1)
    subject._emergency_close = AsyncMock()
    asyncio.run(subject._tick())
    subject._emergency_close.assert_awaited_once()
    subject.pair.execute_maker_order.assert_not_called()


def test_near_center_closes_only_when_conservative_profit_positive(tmp_path):
    subject = strategy(tmp_path, book([("100", "10")], [("101", "10")]),
                       book([("100", "10")], [("101", "10")]))
    subject.position = SpreadPosition(D(1), D(10), time.time())
    subject.pair.positions.return_value = D(1)
    subject.maker.get_order_price.return_value = D("100.5")
    asyncio.run(subject._tick())
    assert subject.position.maker_position == D("0.5")
    assert subject.pair.execute_maker_order.await_args.args[0] == "sell"
    subject.position = SpreadPosition(D("0.5"), D(0), time.time())
    subject.pair.positions.return_value = D("0.5")
    subject.pair.execute_maker_order.reset_mock()
    asyncio.run(subject._tick())
    subject.pair.execute_maker_order.assert_not_called()


def test_exit_rules_compare_same_entry_signal(tmp_path):
    rows = [{"time": i * 86400, "basis_bps": str(value)}
            for i, value in enumerate((-150, -50, 0, 110))]
    result = compare(config(tmp_path), rows)
    assert result["center"]["completed_signals"] == 1
    assert result["opposite_band"]["completed_signals"] == 1
    assert D(result["opposite_band"]["median_indicative_net_bps"]) > D(
        result["center"]["median_indicative_net_bps"])


def test_exit_comparison_uses_recorded_execution_quotes(tmp_path):
    rows = []
    for index, (basis, sell_cash) in enumerate(((-150, "0"), (0, "-1"), (110, "2"))):
        rows.append({
            "time": index * 86400, "basis_bps": str(basis),
            "maker_bid": "100", "maker_ask": "100", "hedge_bid": "100", "hedge_ask": "100",
            "quotes": {
                "buy": {"quantity": "1", "cash_per_unit": "10", "fees_per_unit": "1", "basis_bps": str(basis)},
                "sell": {"quantity": "1", "cash_per_unit": sell_cash, "fees_per_unit": "1", "basis_bps": str(basis)},
            },
        })
    result = compare(config(tmp_path), rows)
    assert D(result["center"]["median_indicative_net_bps"]) == D(698)
    assert D(result["opposite_band"]["median_indicative_net_bps"]) == D(996)


def test_emergency_adapters_submit_reduce_only_ioc():
    async def run():
        bulk: Any = BulkClient.__new__(BulkClient)
        bulk.config = SimpleNamespace(tick_size=D("0.01"))
        bulk.fetch_bbo_prices = AsyncMock(return_value=(D(100), D(101)))
        bulk._place_limit = AsyncMock(return_value=OrderResult(True, "id"))
        await bulk.place_reduce_only_ioc("ETH-USD", D(1), "sell", D("0.5"))
        bulk_call = bulk._place_limit.await_args
        assert bulk_call is not None
        assert bulk_call.args[3] is True
        assert bulk_call.kwargs["tif"] == "IOC"

        arcus: Any = ArcusClient.__new__(ArcusClient)
        arcus.fetch_bbo_prices = AsyncMock(return_value=(D(100), D(101)))
        arcus._snap = lambda price, side: price
        arcus._place = AsyncMock(return_value=OrderResult(True, "id"))
        await arcus.place_reduce_only_ioc("ETH-USD", D(1), "buy", D("0.5"))
        arcus_call = arcus._place.await_args
        assert arcus_call is not None
        assert arcus_call.args[3] is True
        assert arcus_call.kwargs["tif"] == "IOC"
        assert arcus_call.args[1] > D(101)

    asyncio.run(run())
