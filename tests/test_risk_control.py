import asyncio
import os
import sys
import types
from types import SimpleNamespace
from decimal import Decimal

import pytest

# Ensure project root is on sys.path for direct module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Stub heavy dependencies to avoid installing external SDKs
exchanges_pkg = types.ModuleType("exchanges")
lighter_mod = types.ModuleType("exchanges.lighter")
paradex_mod = types.ModuleType("exchanges.paradex")
helpers_pkg = types.ModuleType("helpers")
logger_mod = types.ModuleType("helpers.logger")

class _DummyLogger:
    def __init__(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        return None

class _DummyLighterClient:
    def __init__(self, config):
        self.config = SimpleNamespace(contract_id="FAKE-LIGHTER", tick_size=Decimal("0.1"))
    async def connect(self):
        return True
    async def disconnect(self):
        return True
    async def get_contract_attributes(self):
        return True
    async def fetch_bbo_prices(self, contract_id):
        return Decimal("100"), Decimal("101")
    def round_to_tick(self, price: Decimal) -> Decimal:
        return price

class _DummyParadexClient:
    def __init__(self, config):
        self.config = SimpleNamespace(contract_id="FAKE-PARADEX", tick_size=Decimal("0.1"))
    async def connect(self):
        return True
    async def disconnect(self):
        return True
    async def get_contract_attributes(self):
        return True
    async def fetch_bbo_prices(self, contract_id):
        return Decimal("100"), Decimal("101")
    def round_to_tick(self, price: Decimal) -> Decimal:
        return price

logger_mod.TradingLogger = _DummyLogger
lighter_mod.LighterClient = _DummyLighterClient
paradex_mod.ParadexClient = _DummyParadexClient
exchanges_pkg.lighter = lighter_mod
exchanges_pkg.paradex = paradex_mod
helpers_pkg.logger = logger_mod

sys.modules["exchanges"] = exchanges_pkg
sys.modules["exchanges.lighter"] = lighter_mod
sys.modules["exchanges.paradex"] = paradex_mod
sys.modules["helpers"] = helpers_pkg
sys.modules["helpers.logger"] = logger_mod

from hedge_strategy import HedgeStrategy


class LighterFake:
    def __init__(self, bid: Decimal, ask: Decimal, pos_signed: Decimal, liq_price: Decimal):
        self._bid = bid
        self._ask = ask
        self._pos_signed = pos_signed
        self.close_calls = []
        self.cancel_calls = 0
        self.config = SimpleNamespace(
            contract_id="ETH-PERP-MARKET-ID",
            tick_size=Decimal("0.1"),
            liquidation_price=liq_price,
            ticker="ETH-PERP",
        )

    async def get_account_positions(self) -> Decimal:
        return self._pos_signed

    async def fetch_bbo_prices(self, contract_id):
        return self._bid, self._ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        return price

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str):
        self.close_calls.append((contract_id, quantity, price, side))
        return SimpleNamespace(success=True)

    async def cancel_order(self, order_id: str):
        self.cancel_calls += 1
        return True

    async def disconnect(self):
        return True


class ParadexFake:
    def __init__(self, bid: Decimal, ask: Decimal, pos_size: Decimal, pos_side: str, liq_price: Decimal):
        self._bid = bid
        self._ask = ask
        self._pos_size = pos_size
        self._pos_side = pos_side  # 'LONG' or 'SHORT' or None
        self.close_calls = []
        self.config = SimpleNamespace(
            contract_id="ETH-USD-PERP",
            tick_size=Decimal("0.1"),
            liquidation_price=liq_price,
            quantity=Decimal("1")
        )

    async def get_account_positions(self) -> Decimal:
        return abs(self._pos_size)

    async def _fetch_positions_with_retry(self):
        if self._pos_size > 0 and self._pos_side:
            return [{
                "market": self.config.contract_id,
                "status": "OPEN",
                "side": self._pos_side,
                "size": str(self._pos_size)
            }]
        return []

    async def fetch_bbo_prices(self, contract_id):
        return self._bid, self._ask

    def round_to_tick(self, price: Decimal) -> Decimal:
        return price

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str):
        self.close_calls.append((contract_id, quantity, price, side))
        return SimpleNamespace(success=True)

    async def disconnect(self):
        return True


def test_risk_triggers_and_closes_both_sides():
    # Current market mid ~ 1001; liq price 950 within 10% (threshold 100)
    lighter = LighterFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_signed=Decimal("1"), liq_price=Decimal("950"))
    paradex = ParadexFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_size=Decimal("1"), pos_side="SHORT", liq_price=Decimal("960"))

    cfg = {
        "ticker": "ETH-PERP",
        "quantity": Decimal("1"),
        "side": "buy",
        "lighter": {},
        "paradex": {},
        "risk_enabled": True,
        "risk_threshold_pct": 0.10,
    }
    strategy = HedgeStrategy(cfg)
    strategy.lighter_client = lighter
    strategy.paradex_client = paradex

    triggered = asyncio.run(strategy.monitor_liquidation_risk())
    assert triggered is True

    # Lighter had long position -> close with sell
    assert len(lighter.close_calls) == 1
    assert lighter.close_calls[0][3] == "sell"
    assert lighter.close_calls[0][1] == Decimal("1")

    # Paradex had short position -> close with buy
    assert len(paradex.close_calls) == 1
    assert paradex.close_calls[0][3] == "buy"
    assert paradex.close_calls[0][1] == Decimal("1")


def test_risk_not_triggered_when_far_from_liq():
    lighter = LighterFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_signed=Decimal("1"), liq_price=Decimal("800"))
    paradex = ParadexFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_size=Decimal("1"), pos_side="LONG", liq_price=Decimal("780"))

    strategy = HedgeStrategy({
        "ticker": "ETH-PERP",
        "quantity": Decimal("1"),
        "side": "buy",
        "lighter": {},
        "paradex": {},
        "risk_enabled": True,
        "risk_threshold_pct": 0.10,
    })
    strategy.lighter_client = lighter
    strategy.paradex_client = paradex

    triggered = asyncio.run(strategy.monitor_liquidation_risk())
    assert triggered is False
    assert len(lighter.close_calls) == 0
    assert len(paradex.close_calls) == 0


def test_risk_triggers_close_only_one_side():
    # Paradex no position; Lighter long and liq near
    lighter = LighterFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_signed=Decimal("2"), liq_price=Decimal("950"))
    paradex = ParadexFake(bid=Decimal("1000"), ask=Decimal("1002"), pos_size=Decimal("0"), pos_side=None, liq_price=Decimal("980"))

    strategy = HedgeStrategy({
        "ticker": "ETH-PERP",
        "quantity": Decimal("2"),
        "side": "buy",
        "lighter": {},
        "paradex": {},
        "risk_enabled": True,
        "risk_threshold_pct": 0.10,
    })
    strategy.lighter_client = lighter
    strategy.paradex_client = paradex

    triggered = asyncio.run(strategy.monitor_liquidation_risk())
    assert triggered is True
    assert len(lighter.close_calls) == 1
    assert lighter.close_calls[0][3] == "sell"
    assert lighter.close_calls[0][1] == Decimal("2")
    assert len(paradex.close_calls) == 0