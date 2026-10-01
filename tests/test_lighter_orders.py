import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from exchanges.base import OrderInfo, OrderOutcomeUnknown, OrderResult
from exchanges.lighter import LighterClient


def sdk_order(status="open", filled="0.04", remaining="0.06", client_id=42):
    return SimpleNamespace(
        order_index=987,
        client_order_index=client_id,
        market_index=1,
        initial_base_amount="0.1",
        filled_base_amount=filled,
        remaining_base_amount=remaining,
        price="2000",
        is_ask=False,
        status=status,
    )


class LighterOrderContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = cast(Any, LighterClient.__new__(LighterClient))
        self.client.config = SimpleNamespace(contract_id=1, close_order_side="sell")
        self.client.account_index = 2
        self.client.api_key_index = 3
        self.client.api_client = object()
        self.client.lighter_client = SimpleNamespace(
            create_auth_token_with_expiry=lambda **_kwargs: ("token", None),
        )
        self.client.logger = SimpleNamespace(log=lambda *_args: None, log_transaction=lambda *_args: None)
        self.client.orders_cache = {}
        self.client.current_order = None
        self.client.current_order_client_id = 42
        self.client._order_update_handler = None

    async def test_order_lookup_uses_client_id_and_cumulative_fill(self):
        with patch("exchanges.lighter.OrderApi") as order_api:
            order_api.return_value.account_orders = AsyncMock(
                return_value=SimpleNamespace(orders=[sdk_order()]),
            )
            order = await self.client.get_order_info("42")
        self.assertEqual(order, OrderInfo(
            "42", "buy", Decimal("0.1"), Decimal("2000"), "PARTIALLY_FILLED",
            Decimal("0.04"), Decimal("0.06"),
        ))
        order_api.return_value.account_orders.assert_awaited_once_with(
            authorization="token", client_order_indexes="42", account_index=2,
        )

    async def test_absent_order_is_not_inferred_from_position(self):
        with patch("exchanges.lighter.OrderApi") as order_api:
            order_api.return_value.account_orders = AsyncMock(
                return_value=SimpleNamespace(orders=[]),
            )
            self.assertIsNone(await self.client.get_order_info("42"))

    async def test_order_lookup_failure_is_not_reported_as_absence(self):
        self.client._fetch_order_by_client_id = AsyncMock(side_effect=ConnectionError("read failed"))
        with self.assertRaisesRegex(ConnectionError, "read failed"):
            await self.client.get_order_info("42")

    async def test_active_orders_keep_original_size_and_client_id(self):
        self.client._fetch_orders_with_retry = AsyncMock(return_value=[sdk_order()])
        orders = await self.client.get_active_orders(1)
        self.assertEqual(orders[0].order_id, "42")
        self.assertEqual(orders[0].size, Decimal("0.1"))
        self.assertEqual(orders[0].filled_size, Decimal("0.04"))
        self.assertEqual(orders[0].remaining_size, Decimal("0.06"))

    async def test_canceled_order_retains_unfilled_quantity_even_if_sdk_reports_zero_remaining(self):
        order = self.client._order_info(sdk_order("canceled-post-only", "0", "0"))
        self.assertEqual(order.status, "CANCELED")
        self.assertEqual(order.remaining_size, Decimal("0.1"))

    async def test_websocket_event_uses_same_client_id_as_rest(self):
        updates = []
        self.client.setup_order_update_handler(updates.append)
        self.client._handle_websocket_order_update([vars(sdk_order("filled", "0.1", "0"))])
        self.assertEqual(self.client.current_order.order_id, "42")
        self.assertEqual(updates[0]["order_id"], "42")
        self.assertEqual(updates[0]["status"], "FILLED")

    async def test_cancel_confirms_terminal_fill_and_resolves_exchange_id(self):
        self.client._fetch_order_by_client_id = AsyncMock(side_effect=[
            sdk_order(), sdk_order("canceled", "0.04", "0.06"),
        ])
        cancel = AsyncMock(return_value=(None, SimpleNamespace(code=200), None))
        self.client.lighter_client.cancel_order = cancel
        result = await self.client.cancel_order("42")
        self.assertTrue(result.success)
        self.assertEqual(result.status, "CANCELED")
        self.assertEqual(result.filled_size, Decimal("0.04"))
        cancel.assert_awaited_once_with(market_index=1, order_index=987, api_key_index=3)

    async def test_cancel_lost_ack_remains_unknown(self):
        self.client.ORDER_CONFIRM_TIMEOUT = 0.001
        self.client._fetch_order_by_client_id = AsyncMock(return_value=sdk_order())
        self.client.lighter_client.cancel_order = AsyncMock(side_effect=TimeoutError())
        with self.assertRaisesRegex(OrderOutcomeUnknown, "terminal state is unknown"):
            await self.client.cancel_order("42")

    async def test_cancel_lost_ack_recovers_confirmed_terminal_order(self):
        self.client._fetch_order_by_client_id = AsyncMock(side_effect=[
            sdk_order(), sdk_order("filled", "0.1", "0"),
        ])
        self.client.lighter_client.cancel_order = AsyncMock(side_effect=TimeoutError())
        result = await self.client.cancel_order("42")
        self.assertTrue(result.success)
        self.assertEqual(result.status, "FILLED")
        self.assertEqual(result.filled_size, Decimal("0.1"))

    async def test_close_order_is_reduce_only_and_rejected_cancel_is_failure(self):
        self.client.place_limit_order = AsyncMock(return_value=OrderResult(
            True, "42", status="SUBMITTED",
        ))
        self.client._wait_for_order = AsyncMock(return_value=OrderInfo(
            "42", "sell", Decimal("0.1"), Decimal("2001"), "CANCELED",
            Decimal(0), Decimal("0.1"), "canceled-post-only",
        ))
        result = await self.client.place_close_order(1, Decimal("0.1"), Decimal("2001"), "sell")
        self.client.place_limit_order.assert_awaited_once_with(
            1, Decimal("0.1"), Decimal("2001"), "sell", reduce_only=True,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, "CANCELED")

    async def test_sdk_submit_ack_is_not_a_fill(self):
        self.client.lighter_client.create_order = AsyncMock(return_value=(
            None, SimpleNamespace(code=200), None,
        ))
        result = await self.client._submit_order_with_retry({"client_order_index": 42})
        self.assertTrue(result.success)
        self.assertEqual(result.status, "SUBMITTED")
        self.assertIsNone(result.filled_size)

    async def test_hedge_accepts_sdk_success_code(self):
        self.client.base_amount_multiplier = 1000
        self.client.price_multiplier = 100
        self.client.fetch_bbo_prices = AsyncMock(return_value=(Decimal(100), Decimal(101)))
        self.client.lighter_client.create_market_order = AsyncMock(return_value=(
            None, SimpleNamespace(code=200), None,
        ))
        result = await self.client.place_hedge_market_order(Decimal("0.1"), "sell", Decimal("0.5"))
        self.assertTrue(result.success)
        self.client.lighter_client.create_market_order.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
