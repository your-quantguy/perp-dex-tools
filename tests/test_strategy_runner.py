import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from strategies.runner import StrategyRunner
from trading_bot import TradingBot, TradingConfig


class FakeVenue:
    def __init__(self, events: list[str], name: str, fail_connect: bool = False):
        self.events = events
        self.name = name
        self.fail_connect = fail_connect

    async def connect(self) -> None:
        self.events.append(f"connect:{self.name}")
        if self.fail_connect:
            raise ConnectionError("connect failed")

    async def disconnect(self) -> None:
        self.events.append(f"disconnect:{self.name}")


class FakeStrategy:
    def __init__(self, events: list[str], *venues: FakeVenue):
        self.events = events
        self.venues = venues

    async def prepare(self) -> None:
        self.events.append("prepare")

    async def execute(self) -> None:
        self.events.append("execute")

    async def on_failure(self, exc: BaseException) -> None:
        self.events.append(f"failure:{exc}")


class StrategyRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepares_connects_executes_and_disconnects(self):
        events: list[str] = []
        strategy = FakeStrategy(events, FakeVenue(events, "maker"), FakeVenue(events, "hedge"))
        await StrategyRunner().run(strategy)
        self.assertEqual(events, [
            "prepare", "connect:maker", "connect:hedge", "execute",
            "disconnect:maker", "disconnect:hedge",
        ])

    async def test_connect_failure_reports_and_cleans_up_both_venues(self):
        events: list[str] = []
        strategy = FakeStrategy(
            events, FakeVenue(events, "maker"), FakeVenue(events, "hedge", fail_connect=True),
        )
        with self.assertRaisesRegex(ConnectionError, "connect failed"):
            await StrategyRunner().run(strategy)
        self.assertEqual(events, [
            "prepare", "connect:maker", "connect:hedge", "failure:connect failed",
            "disconnect:maker", "disconnect:hedge",
        ])

    async def test_take_profit_strategy_uses_runner_lifecycle(self):
        strategy = cast(Any, TradingBot.__new__(TradingBot))
        strategy.config = TradingConfig(
            ticker="ETH", contract_id="", tick_size=Decimal(0),
            quantity=Decimal("0.1"), direction="buy", take_profit=Decimal("0.02"),
            max_orders=1, wait_time=1, exchange="lighter", grid_step=Decimal(-100),
            stop_price=Decimal(-1), pause_price=Decimal(-1),
        )
        strategy.logger = SimpleNamespace(log=lambda *_args: None)
        strategy.exchange_client = SimpleNamespace(
            get_contract_attributes=AsyncMock(return_value=(1, Decimal("0.01"))),
            connect=AsyncMock(), disconnect=AsyncMock(),
        )
        strategy.execute = AsyncMock()
        await strategy.run()
        strategy.exchange_client.connect.assert_awaited_once()
        strategy.execute.assert_awaited_once()
        strategy.exchange_client.disconnect.assert_awaited_once()
        self.assertEqual(strategy.config.contract_id, 1)

    async def test_failure_handler_error_does_not_hide_trading_error(self):
        events: list[str] = []
        strategy = FakeStrategy(events, FakeVenue(events, "maker"))

        async def trading_error() -> None:
            raise RuntimeError("order outcome unknown")

        async def notification_error(_exc: BaseException) -> None:
            raise ConnectionError("notification unavailable")

        cast(Any, strategy).execute = trading_error
        cast(Any, strategy).on_failure = notification_error
        with self.assertRaisesRegex(RuntimeError, "order outcome unknown"):
            await StrategyRunner().run(strategy)
        self.assertEqual(events[-1], "disconnect:maker")
