import asyncio
import os
import time
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import base58

from exchanges.base import OrderOutcomeUnknown, OrderResult
from exchanges.bulk import BulkClient


def _resting(order_id: str) -> dict:
    return {"response": {"data": {"statuses": [{"resting": {"oid": order_id}}]}}}


async def _timeout_wait(awaitable, timeout):
    awaitable.close()
    raise asyncio.TimeoutError


class BulkClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = base58.b58encode(bytes(range(32))).decode()
        self.environment = patch.dict(os.environ, {"BULK_PRIVATE_KEY": self.key}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config: Any = type("Config", (), {
            "ticker": "ETH", "contract_id": "", "tick_size": Decimal(0),
            "quantity": Decimal("0.1"), "direction": "buy",
        })()
        self.client: Any = BulkClient(self.config)

    def _market_is_connected(self):
        self.client._ws_task = asyncio.get_running_loop().create_future()
        self.client._ws_ready.set()
        self.client._best_bid = Decimal("1999")
        self.client._best_ask = Decimal("2000")
        self.client._book_update_time = time.monotonic()

    def test_mainnet_is_the_default_signature_domain(self):
        self.assertEqual(self.client.api_url, BulkClient.API_URL)
        self.assertEqual(self.client.ws_url, BulkClient.WS_URL)
        self.assertEqual(self.client.signature_domain, "mainnet")
        subscription = self.client.subscription_payload()["subscription"]
        self.assertEqual(subscription[0], {"type": "account", "user": self.client.public_key})
        self.assertEqual(subscription[1]["type"], "l2Snapshot")
        self.assertEqual(subscription[1]["symbol"], "ETH-USD")

    def test_testnet_endpoints_sign_for_testnet(self):
        with patch.dict(os.environ, {
            "BULK_PRIVATE_KEY": self.key,
            "BULK_API_URL": BulkClient.TESTNET_API_URL,
            "BULK_WS_URL": BulkClient.TESTNET_WS_URL,
        }):
            client = BulkClient(self.config)
        self.assertEqual(client.signature_domain, "testnet")

    def test_keychain_signs_an_alo_limit_for_mainnet(self):
        signed = self.client._signer.sign({
            "type": "order",
            "symbol": "ETH-USD",
            "is_buy": True,
            "price": 2000.0,
            "size": 0.1,
            "reduce_only": False,
            "order_type": {"type": "limit", "tif": "ALO"},
        })
        self.assertEqual(signed["account"], self.client.public_key)
        self.assertEqual(signed["signer"], self.client.public_key)
        self.assertTrue(signed["signature"])
        self.assertEqual(signed["actions"][0]["l"]["tif"], "ALO")
        self.assertFalse(signed["actions"][0]["l"]["r"])

    def test_mixed_networks_are_rejected(self):
        with patch.dict(os.environ, {
            "BULK_PRIVATE_KEY": self.key,
            "BULK_API_URL": BulkClient.API_URL,
            "BULK_WS_URL": BulkClient.TESTNET_WS_URL,
        }):
            with self.assertRaisesRegex(ValueError, "different networks"):
                BulkClient(self.config)

    async def test_connect_waits_for_account_and_book(self):
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})

        async def socket():
            self.client._handle_message({
                "type": "l2Snapshot",
                "data": {"book": {
                    "symbol": "ETH-USD",
                    "levels": [[{"px": "1999", "sz": "1"}], [{"px": "2000", "sz": "1"}]],
                }},
            })
            self.client._handle_message({"type": "account", "data": {"type": "snapshot"}})
            self.client._ws_ready.set()
            await asyncio.Event().wait()

        with patch.object(self.client, "_run_market_socket", socket):
            await self.client.connect()
        self.assertEqual(await self.client.fetch_bbo_prices("ETH-USD"), (Decimal("1999"), Decimal("2000")))
        assert self.client._ws_task is not None
        self.client._ws_task.cancel()

    async def test_market_metadata_validates_lot_and_uses_actual_api_list(self):
        self.client._request = AsyncMock(return_value=[{
            "symbol": "ETH-USD", "status": "TRADING", "tickSize": 0.001,
            "lotSize": 0.0001, "minNotional": 50.0,
        }])
        self.assertEqual(await self.client.get_contract_attributes(), ("ETH-USD", Decimal("0.001")))
        self.config.quantity = Decimal("0.10001")
        with self.assertRaisesRegex(ValueError, "multiple"):
            await self.client.get_contract_attributes()

    async def test_entry_and_close_are_post_only_and_close_is_reduce_only(self):
        self._market_is_connected()
        self.client._lot_size = Decimal("0.0001")
        self.client._min_notional = Decimal("50")
        self.config.tick_size = Decimal("0.001")
        signed = []

        class StubSigner:
            def sign(self, payload):
                signed.append(payload)
                order_id = "entry" if len(signed) == 1 else "exit"
                return {
                    "actions": [{"l": {"tif": "ALO"}}], "nonce": 1,
                    "account": "a", "signer": "a", "signature": "s", "order_id": order_id,
                }

        self.client._signer = StubSigner()
        self.client._post_signed = AsyncMock(side_effect=[_resting("entry"), _resting("exit")])
        entry = await self.client.place_open_order("ETH-USD", Decimal("0.1"), "buy")
        close = await self.client.place_close_order("ETH-USD", Decimal("0.1"), Decimal("2001"), "sell")
        self.assertTrue(entry.success and close.success)
        self.assertEqual(entry.price, Decimal("1999.999"))
        self.assertEqual(signed[0]["order_type"], {"type": "limit", "tif": "ALO"})
        self.assertFalse(signed[0]["reduce_only"])
        self.assertTrue(signed[1]["is_buy"] is False)
        self.assertTrue(signed[1]["reduce_only"])
        posted = self.client._post_signed.await_args_list[0].args[0]
        self.assertEqual(set(posted), {"actions", "nonce", "account", "signer", "signature", "order_id"})

    async def test_lost_submit_ack_exposes_signed_order_id_for_recovery(self):
        self.client._lot_size = Decimal("0.1")
        self.client._min_notional = Decimal(0)
        self.client._wait_market_ready = AsyncMock()
        self.client._sign_action = lambda _action: {"order_id": "entry"}
        self.client._post_signed = AsyncMock(side_effect=TimeoutError())
        with self.assertRaisesRegex(OrderOutcomeUnknown, "outcome is unknown") as caught:
            await self.client._place_limit(Decimal("0.1"), Decimal(2000), "buy", False)
        self.assertEqual(caught.exception.order_id, "entry")
        self.client._post_signed.assert_awaited_once()

        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        self.client._filled_size_from_history = AsyncMock(return_value=Decimal("0.1"))
        recovered = await self.client.get_order_info("entry")
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, "FILLED")
        self.assertEqual(recovered.filled_size, Decimal("0.1"))

    async def test_lost_submit_ack_with_incomplete_history_stays_unknown(self):
        self.client._uncertain_orders["entry"] = SimpleNamespace(
            order_id="entry", side="buy", size=Decimal("0.1"), price=Decimal(2000),
            status="OPEN", filled_size=Decimal(0), remaining_size=Decimal("0.1"),
        )
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        self.client._filled_size_from_history = AsyncMock(return_value=Decimal("0.04"))
        with self.assertRaisesRegex(RuntimeError, "absent without a confirmed final state"):
            await self.client.get_order_info("entry")

    async def test_close_retries_when_post_only_order_would_cross(self):
        self._market_is_connected()
        self.client._lot_size = Decimal("0.0001")
        self.client._min_notional = Decimal("50")
        self.config.tick_size = Decimal("0.001")
        prices = []

        async def place(quantity, price, side, reduce_only):
            prices.append(price)
            if len(prices) == 1:
                return OrderResult(success=False, error_message="rejectedCrossing")
            return OrderResult(success=True, order_id="close", side=side, size=quantity, price=price, status="OPEN")

        self.client._place_limit = place
        self.client.fetch_bbo_prices = AsyncMock(side_effect=[
            (Decimal("2671.7"), Decimal("2671.8")),
            (Decimal("2671.2"), Decimal("2671.3")),
        ])
        result = await self.client.place_close_order("ETH-USD", Decimal("0.02"), Decimal("2671.626"), "sell")
        self.assertTrue(result.success)
        self.assertEqual(prices, [Decimal("2671.701"), Decimal("2671.626")])

    async def test_terminal_event_before_open_ack_cannot_be_revived(self):
        self._market_is_connected()
        self.client._lot_size = Decimal("0.0001")
        self.client._min_notional = Decimal("50")
        self.config.tick_size = Decimal("0.001")
        self.client._signer = type("StubSigner", (), {
            "sign": staticmethod(lambda payload: {
                "actions": [{"l": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s", "order_id": "entry",
            }),
        })()

        async def post(_signed):
            self.client._handle_account_data({
                "type": "orderUpdate", "sym": "ETH-USD", "oid": "entry",
                "isBuy": True, "sz": "0", "origSz": "0.1", "fillSz": "0.1",
                "px": "1999.999", "status": "filled",
            })
            return _resting("entry")

        self.client._post_signed = post
        result = await self.client.place_open_order("ETH-USD", Decimal("0.1"), "buy")
        self.assertEqual(result.status, "FILLED")
        self.assertEqual(result.filled_size, Decimal("0.1"))
        self.assertNotIn("entry", self.client._owned_open_order_ids)

    async def test_cancel_returns_confirmed_partial_fill(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": 1, "origSz": 1,
            "fillSz": 0, "px": 2000, "status": "placed", "isBuy": True,
        })
        self.client._account = AsyncMock(return_value={
            "openOrders": [{"sym": "ETH-USD", "oid": "entry", "sz": 1,
                            "origSz": 1, "fillSz": 0, "px": 2000, "status": "placed", "isBuy": True}],
            "positions": [],
        })
        events = []
        self.client.setup_order_update_handler(events.append)

        async def post(_signed):
            self.client._handle_account_data({
                "type": "orderUpdate",
                "sym": "ETH-USD",
                "oid": "entry",
                "isBuy": True,
                "sz": "0.6",
                "origSz": "1",
                "fillSz": "0.4",
                "px": "2000",
                "status": "cancelled",
            })
            return {"response": {"data": {"statuses": [{"cancelled": {"oid": "entry"}}]}}}

        self.client._signer = type("StubSigner", (), {
            "pubkey": self.client.public_key,
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = post
        result = await self.client.cancel_order("entry")
        self.assertTrue(result.success)
        self.assertEqual(result.filled_size, Decimal("0.4"))
        self.assertEqual(events[-1]["status"], "CANCELED")
        self.assertEqual(events[-1]["filled_size"], "0.4")

    async def test_cancel_without_event_checks_account_and_fills(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        })
        self.client.get_order_info = AsyncMock(return_value=self.client._orders["entry"])
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        self.client._request = AsyncMock(return_value={"data": [{
            "symbol": "ETH-USD", "maker": self.client.public_key,
            "taker": "other", "orderIdMaker": "entry", "amount": "0.4",
        }], "page": {"hasMore": False}})
        self.client._signer = type("StubSigner", (), {
            "pubkey": self.client.public_key,
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = AsyncMock(return_value={
            "response": {"data": {"statuses": [{"cancelled": {"oid": "entry"}}]}}
        })
        with patch("exchanges.bulk.asyncio.wait_for", new=_timeout_wait):
            result = await self.client.cancel_order("entry")
        self.assertEqual(result.filled_size, Decimal("0.4"))
        self.assertEqual(self.client._orders["entry"].status, "CANCELED")

    async def test_cancel_does_not_invent_terminal_state_while_order_is_live(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        })
        self.client.get_order_info = AsyncMock(return_value=self.client._orders["entry"])
        self.client._account = AsyncMock(return_value={"openOrders": [{
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        }], "positions": []})
        self.client._signer = type("StubSigner", (), {
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = AsyncMock(return_value={
            "response": {"data": {"statuses": [{"placed": {"oid": "entry"}}]}}
        })
        with patch("exchanges.bulk.asyncio.wait_for", new=_timeout_wait):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                await self.client.cancel_order("entry")
        self.assertEqual(self.client._orders["entry"].status, "OPEN")

    async def test_cancel_needs_terminal_ack_even_if_order_is_absent(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        })
        self.client.get_order_info = AsyncMock(return_value=self.client._orders["entry"])
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        self.client._signer = type("StubSigner", (), {
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = AsyncMock(return_value={
            "response": {"data": {"statuses": [{"placed": {"oid": "entry"}}]}}
        })
        with patch("exchanges.bulk.asyncio.wait_for", new=_timeout_wait):
            with self.assertRaisesRegex(RuntimeError, "did not confirm a terminal state"):
                await self.client.cancel_order("entry")
        self.assertEqual(self.client._orders["entry"].status, "OPEN")

    async def test_cancel_keeps_terminal_event_arriving_during_account_check(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        })
        self.client.get_order_info = AsyncMock(return_value=self.client._orders["entry"])

        async def account():
            self.client._remember(self.client._parse_order({
                "sym": "ETH-USD", "oid": "entry", "sz": "0", "origSz": "1",
                "fillSz": "1", "px": "2000", "status": "filled", "isBuy": True,
            }))
            return {"openOrders": [], "positions": []}

        self.client._account = account
        self.client._signer = type("StubSigner", (), {
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = AsyncMock(return_value={
            "response": {"data": {"statuses": [{"cancelled": {"oid": "entry"}}]}}
        })
        self.client._request = AsyncMock()
        with patch("exchanges.bulk.asyncio.wait_for", new=_timeout_wait):
            result = await self.client.cancel_order("entry")
        self.assertEqual(result.filled_size, Decimal("1"))
        self.assertEqual(self.client._orders["entry"].status, "FILLED")
        self.client._request.assert_not_awaited()

    async def test_reconnect_requires_reconciliation_of_live_open_order(self):
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": "1", "origSz": "1",
            "px": "2000", "status": "placed", "isBuy": True,
        })
        self.client._owned_open_order_ids.add("entry")
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        with self.assertRaisesRegex(ValueError, "disappeared"):
            await self.client._reconcile_after_reconnect()
        self.client._account.return_value["openOrders"] = [{
            "sym": "ETH-USD", "oid": "entry", "sz": "0.6", "origSz": "1",
            "fillSz": "0.4", "px": "2000", "status": "partiallyFilled", "isBuy": True,
        }]
        await self.client._reconcile_after_reconnect()
        self.assertEqual(self.client._orders["entry"].filled_size, Decimal("0.4"))

    async def test_market_socket_reconnects_and_refreshes_the_book(self):
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})
        account = SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data='{"type":"account","data":{"type":"snapshot"}}')
        book = SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=(
            '{"type":"l2Snapshot","data":{"book":{"symbol":"ETH-USD",'
            '"levels":[[{"px":"1999"}],[{"px":"2000"}]]}}}'
        ))
        closed = SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)

        class FakeSocket:
            def __init__(self, messages):
                self.messages = iter(messages)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def send_json(self, _payload):
                pass

            async def receive(self):
                try:
                    return next(self.messages)
                except StopIteration:
                    await asyncio.Event().wait()

        class FakeSession:
            closed = False

            def __init__(self):
                self.connects = 0

            def ws_connect(self, *_args, **_kwargs):
                self.connects += 1
                messages = [account, book, closed] if self.connects == 1 else [account, book]
                return FakeSocket(messages)

            async def close(self):
                self.closed = True

        session = FakeSession()
        self.client._ws_http = session
        self.client._ws_task = asyncio.create_task(self.client._run_market_socket())
        try:
            async def reconnected():
                while session.connects < 2 or not self.client._ws_ready.is_set():
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(reconnected(), timeout=3)
            self.assertEqual(await self.client.fetch_bbo_prices("ETH-USD"),
                             (Decimal("1999"), Decimal("2000")))
            self.assertEqual(self.client._account.await_count, 2)
        finally:
            await self.client.disconnect()

    async def test_market_read_waits_until_reconnect_is_ready(self):
        self.client._ws_task = asyncio.get_running_loop().create_future()
        self.client._best_bid = Decimal("1999")
        self.client._best_ask = Decimal("2000")
        self.client._book_update_time = time.monotonic()
        read = asyncio.create_task(self.client.fetch_bbo_prices("ETH-USD"))
        await asyncio.sleep(0)
        self.assertFalse(read.done())
        self.client._ws_ready.set()
        self.assertEqual(await read, (Decimal("1999"), Decimal("2000")))

    async def test_account_read_retries_a_timeout(self):
        calls = {"n": 0}

        async def request(_method, _path, **_kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError()
            return [{"fullAccount": {
                "openOrders": [],
                "positions": [{"symbol": "ETH-USD", "size": "0.02"}],
            }}]

        self.client._request = request
        with patch("exchanges.bulk.asyncio.sleep", new=AsyncMock()):
            position = await self.client.get_account_positions()
        self.assertEqual(position, Decimal("0.02"))
        self.assertEqual(calls["n"], 3)

    async def test_account_read_failure_does_not_look_like_empty_position(self):
        self.client._request = AsyncMock(return_value={"error": "unavailable"})
        with self.assertRaisesRegex(ValueError, "invalid account"):
            await self.client.get_account_positions()

    async def test_live_orders_expose_remaining_size_and_signed_position(self):
        self.client._request = AsyncMock(return_value=[{"fullAccount": {
            "openOrders": [{
                "oid": "close", "sym": "ETH-USD", "sz": -0.06,
                "origSz": 0.1, "fillSz": 0.04, "px": 2001,
                "status": "partiallyFilled",
            }],
            "positions": [{"symbol": "ETH-USD", "size": -0.06}],
        }}])
        orders = await self.client.get_active_orders("ETH-USD")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "sell")
        self.assertEqual(orders[0].status, "PARTIALLY_FILLED")
        self.assertEqual(orders[0].remaining_size, Decimal("0.06"))
        self.assertEqual(await self.client.get_account_positions(), Decimal("-0.06"))


if __name__ == "__main__":
    unittest.main()
