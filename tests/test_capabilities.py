"""A strategy accepts venues by behavior rather than by exchange name."""

import asyncio
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exchanges.arcus import ArcusClient
from exchanges.base import OrderResult
from exchanges.bulk import BulkClient
from exchanges.capabilities import PairMaker, PairTaker, SpreadMaker, SpreadTaker, TakeProfitVenue
from exchanges.entropy import EntropyClient
from exchanges.factory import ExchangeFactory
from exchanges.lighter import LighterClient
from spread_mode import load_config
from strategies.spread import SpreadStrategy


def test_strategy_capability_matrix():
    bulk = BulkClient.__new__(BulkClient)
    arcus = ArcusClient.__new__(ArcusClient)
    lighter = LighterClient.__new__(LighterClient)
    entropy = EntropyClient.__new__(EntropyClient)
    for venue in (bulk, arcus, lighter):
        assert isinstance(venue, TakeProfitVenue)
        assert isinstance(venue, PairTaker)
        assert isinstance(venue, SpreadTaker)
    for venue in (bulk, arcus):
        assert isinstance(venue, PairMaker)
        assert isinstance(venue, SpreadMaker)
    assert not isinstance(lighter, PairMaker)
    assert not isinstance(lighter, SpreadMaker)
    assert isinstance(entropy, PairTaker)
    assert isinstance(entropy, SpreadTaker)
    assert not isinstance(entropy, PairMaker)
    assert not isinstance(entropy, TakeProfitVenue)
    assert "entropy" in ExchangeFactory.get_capable_exchanges(SpreadTaker)
    assert "entropy" not in ExchangeFactory.get_capable_exchanges(TakeProfitVenue)


def test_spread_accepts_arcus_as_taker_and_reversed_roles(tmp_path):
    bulk: Any = BulkClient.__new__(BulkClient)
    bulk.signature_domain = "mainnet"
    arcus: Any = ArcusClient.__new__(ArcusClient)
    arcus.network = "mainnet"
    for maker_name, maker, taker_name, taker in (
        ("bulk", bulk, "arcus", arcus),
        ("arcus", arcus, "bulk", bulk),
    ):
        settings = replace(load_config(Path("spread_config.example.json"), resume=False),
                           venue_a=maker_name, venue_b=taker_name,
                           maker_exchange=maker_name, state_file=tmp_path / "state.json")
        strategy = SpreadStrategy(settings, maker=maker, hedge=taker,
                                  notifier=SimpleNamespace())
        assert strategy.venues == (maker, taker)


def test_spread_accepts_arcus_maker_entropy_taker(tmp_path):
    arcus: Any = ArcusClient.__new__(ArcusClient)
    arcus.network = "mainnet"
    entropy: Any = EntropyClient.__new__(EntropyClient)
    entropy.network = "mainnet"
    settings = replace(load_config(Path("spread_config.example.json"), resume=False),
                       venue_a="arcus", venue_b="entropy", maker_exchange="arcus",
                       ticker="SNDK", state_file=tmp_path / "state.json")
    strategy = SpreadStrategy(settings, maker=arcus, hedge=entropy, notifier=SimpleNamespace())
    assert strategy.venues == (arcus, entropy)


@pytest.mark.parametrize("side,expected", [("buy", D("101.49")), ("sell", D("99.51"))])
def test_arcus_taker_is_bounded_ioc(side, expected):
    async def run():
        arcus: Any = ArcusClient.__new__(ArcusClient)
        arcus.config = SimpleNamespace(contract_id="ETH-USD")
        arcus.market = {"tickTiers": [{"tick": "0.03"}]}
        arcus.tick = D("0.03")
        arcus.validate_hedge_quantity = lambda quantity: None
        arcus.fetch_bbo_prices = AsyncMock(return_value=(D(100), D(101)))
        arcus._place = AsyncMock(return_value=OrderResult(True, "id"))
        await arcus.place_hedge_market_order(D(1), side, D("0.5"), reduce_only=True)
        call = arcus._place.await_args
        assert call is not None
        assert call.args == (D(1), expected, side, True)
        assert call.kwargs == {"tif": "IOC"}

    asyncio.run(run())


@pytest.mark.parametrize("side,expected", [("buy", D("101.49")), ("sell", D("99.51"))])
def test_bulk_taker_tick_never_exceeds_slippage_bound(side, expected):
    async def run():
        bulk: Any = BulkClient.__new__(BulkClient)
        bulk.config = SimpleNamespace(contract_id="ETH-USD", tick_size=D("0.03"))
        bulk.validate_hedge_quantity = lambda quantity: None
        bulk.fetch_bbo_prices = AsyncMock(return_value=(D(100), D(101)))
        bulk._place_limit = AsyncMock(return_value=OrderResult(True, "id"))
        await bulk.place_hedge_market_order(D(1), side, D("0.5"))
        call = bulk._place_limit.await_args
        assert call is not None
        price = call.args[1]
        assert price == expected
        if side == "buy":
            assert price <= D("101.505")
        else:
            assert price >= D("99.5")
        assert price % D("0.03") == 0
        assert call.kwargs == {"tif": "IOC"}

    asyncio.run(run())
