"""Entropy adapter maps official SDK facts and preserves IOC finality."""

import asyncio
from decimal import Decimal as D
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exchanges.base import OrderOutcomeUnknown
from exchanges.entropy import EntropyClient
from strategies.risk import evaluate_cross_venue_leg


def client_with_info(position=None):
    client = EntropyClient.__new__(EntropyClient)
    client.coin = "io:SNDK"
    client.address = "0x" + "a" * 40
    client.step = D("0.0001")
    client.sz_decimals = 4
    client.info = MagicMock()
    client.info.user_state.return_value = {
        "assetPositions": [] if position is None else [{"position": position}],
        "marginSummary": {"accountValue": "1000", "totalNtlPos": "100", "totalMarginUsed": "50"},
        "withdrawable": "900",
    }
    client.info.post.side_effect = ["default", {"leverage": {"type": "isolated", "value": 2}}]
    return client


def test_isolated_margin_mapping_uses_margin_balance_not_signed_raw_usd():
    async def run():
        position = {
            "coin": "io:SNDK", "szi": "1", "positionValue": "100",
            "leverage": {"type": "isolated", "value": 2, "rawUsd": "-95"},
            "marginUsed": "45", "unrealizedPnl": "-5", "liquidationPx": "50",
        }
        snapshot = await client_with_info(position).get_margin_snapshot()
        assert snapshot.positions[0].posted_collateral == D(50)
        assert snapshot.positions[0].unrealized_pnl == D(-5)
        assert snapshot.positions[0].posted_collateral is not None
        assert snapshot.positions[0].unrealized_pnl is not None
        assert snapshot.positions[0].posted_collateral + snapshot.positions[0].unrealized_pnl == D(45)
        assert snapshot.account_mode == "DEFAULT"
        assert snapshot.market_leverage == D(2)
        risk = evaluate_cross_venue_leg(snapshot, "io:SNDK", D("2.5"), D("0.05"))
        assert risk.opening_notional_room == D("12.5")

    asyncio.run(run())


@pytest.mark.parametrize("side,expected", [("buy", D("101.5")), ("sell", D("99.5"))])
def test_ioc_price_stays_inside_slippage_limit(side, expected):
    client = client_with_info()
    assert client._bounded_price(D(101 if side == "buy" else 100), side, D("0.5")) == expected


def test_ioc_unknown_response_is_not_retried():
    async def run():
        client: Any = client_with_info()
        order = MagicMock(return_value={
            "status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 123}}]}},
        })
        client.exchange = SimpleNamespace(order=order)
        client.get_spread_book = AsyncMock(return_value=SimpleNamespace(bid=D(100), ask=D(101)))
        with pytest.raises(OrderOutcomeUnknown):
            await client.place_hedge_market_order(D(1), "buy", D("0.5"))
        order.assert_called_once()

    asyncio.run(run())


def test_ioc_filled_result_preserves_confirmed_quantity():
    async def run():
        client: Any = client_with_info()
        order = MagicMock(return_value={
            "status": "ok", "response": {"data": {"statuses": [
                {"filled": {"oid": 12, "totalSz": "0.5", "avgPx": "101"}},
            ]}},
        })
        client.exchange = SimpleNamespace(order=order)
        client.get_spread_book = AsyncMock(return_value=SimpleNamespace(bid=D(100), ask=D(101)))
        result = await client.place_hedge_market_order(D(1), "buy", D("0.5"))
        assert result.success and result.filled_size == D("0.5")
        assert result.order_id == "12"
        assert order.call_args.args[4] == {"limit": {"tif": "Ioc"}}

    asyncio.run(run())


def test_ioc_transport_failure_keeps_client_id_for_manual_reconciliation():
    async def run():
        client: Any = client_with_info()
        order = MagicMock(side_effect=TimeoutError("lost acknowledgement"))
        client.exchange = SimpleNamespace(order=order)
        client.get_spread_book = AsyncMock(return_value=SimpleNamespace(bid=D(100), ask=D(101)))
        with pytest.raises(OrderOutcomeUnknown) as error:
            await client.place_hedge_market_order(D(1), "buy", D("0.5"))
        assert error.value.order_id is not None and error.value.order_id.startswith("0x")
        order.assert_called_once()

    asyncio.run(run())


def test_connect_rejects_unapproved_api_wallet_before_any_order_session():
    async def run():
        client: Any = client_with_info()
        client.wallet = SimpleNamespace(address="0x" + "b" * 40)
        client.info.post.side_effect = None
        client.info.post.return_value = {"role": "missing"}
        with pytest.raises(ValueError, match="not authorized"):
            await client.connect()

    asyncio.run(run())
