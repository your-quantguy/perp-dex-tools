import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from exchanges.base import OrderInfo, OrderOutcomeUnknown, OrderResult
from exchanges.lighter import LighterClient
from hedge_mode import HedgeBot, HedgeConfig


class FakeMaker:
    def __init__(self):
        self.network = "mainnet"
        self.position = Decimal(0)
        self.orders = []
        self.cancels = []
        self.filled_by_id = {}

    def setup_order_update_handler(self, handler):
        self.handler = handler

    @property
    def network_id(self):
        return self.network

    async def get_contract_attributes(self):
        return "ETH-USD", Decimal("0.01")

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_active_orders(self, contract_id):
        return []

    async def get_account_positions(self):
        return self.position

    async def fetch_bbo_prices(self, contract_id):
        return Decimal(100), Decimal(101)

    async def get_order_price(self, direction):
        return Decimal(100)

    async def place_post_only_order(self, contract_id, quantity, price, side, reduce_only):
        return self._fill(quantity, side, reduce_only)

    async def place_open_order(self, contract_id, quantity, side):
        return self._fill(quantity, side, False)

    async def place_close_order(self, contract_id, quantity, price, side):
        return self._fill(quantity, side, True)

    def _fill(self, quantity, side, reduce_only):
        self.orders.append((side, quantity, reduce_only))
        self.filled_by_id[str(len(self.orders))] = (side, quantity)
        self.position += quantity if side == "buy" else -quantity
        return OrderResult(True, str(len(self.orders)), side, quantity, Decimal(100),
                           "FILLED", filled_size=quantity)

    async def cancel_order(self, order_id):
        self.cancels.append(order_id)
        return OrderResult(True, order_id, filled_size=Decimal(0))

    async def get_order_info(self, order_id):
        side, quantity = self.filled_by_id[order_id]
        return OrderInfo(order_id, side, quantity, Decimal(100), "FILLED",
                         filled_size=quantity, remaining_size=Decimal(0))


class FakeLighter:
    base_amount_multiplier = 1000

    def __init__(self):
        self.position = Decimal(0)
        self.orders = []

    @property
    def network_id(self):
        return "mainnet"

    def setup_order_update_handler(self, handler):
        self.handler = handler

    async def get_contract_attributes(self):
        return 1, Decimal("0.01")

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_active_orders(self, contract_id):
        return []

    async def fetch_bbo_prices(self, contract_id):
        return Decimal(100), Decimal(101)

    async def get_account_positions(self):
        return self.position

    def validate_hedge_quantity(self, quantity):
        units = quantity * self.base_amount_multiplier
        if quantity <= 0 or units != units.to_integral_value():
            raise ValueError("Hedge quantity is not exact")

    async def place_hedge_market_order(self, quantity, side, max_slippage_percent,
                                       reduce_only=False):
        self.orders.append((side, quantity, reduce_only))
        self.position += quantity if side == "buy" else -quantity
        return OrderResult(True, str(len(self.orders)), side, quantity, status="SUBMITTED")


class HedgeModeTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, max_position=Decimal(0), exchange="bulk"):
        maker, lighter = FakeMaker(), FakeLighter()
        bot = HedgeBot(HedgeConfig(exchange, "ETH", Decimal("0.1"), 1,
                                   max_position=max_position), maker, lighter)
        bot._notify = AsyncMock()
        return bot, maker, lighter

    async def test_flat_round_trip_hedges_both_fills(self):
        bot, maker, lighter = self.make_bot()
        await bot.run()
        self.assertEqual(maker.position, Decimal(0))
        self.assertEqual(lighter.position, Decimal(0))
        self.assertEqual(maker.orders, [("buy", Decimal("0.1"), False),
                                        ("sell", Decimal("0.1"), True)])
        self.assertEqual(lighter.orders, [("sell", Decimal("0.1"), False),
                                          ("buy", Decimal("0.1"), True)])

    async def test_arcus_uses_the_same_confirmed_hedge_flow(self):
        bot, maker, lighter = self.make_bot(exchange="arcus")
        await bot.run()
        self.assertEqual((maker.position, lighter.position), (Decimal(0), Decimal(0)))

    async def test_pair_executor_reaches_target_without_cycle_policy(self):
        bot, maker, lighter = self.make_bot()
        await bot.pair.reach_target(Decimal("0.1"))
        self.assertEqual((maker.position, lighter.position),
                         (Decimal("0.1"), Decimal("-0.1")))

    async def test_mixed_networks_are_rejected_before_any_order(self):
        maker = FakeMaker()
        maker.network = "testnet"
        with self.assertRaisesRegex(ValueError, "same network"):
            HedgeBot(HedgeConfig("arcus", "ETH", Decimal("0.1"), 1),
                     maker, FakeLighter())
        self.assertEqual(maker.orders, [])

    async def test_arcus_can_be_selected_as_taker(self):
        config = HedgeConfig("bulk", "ETH", Decimal("0.1"), 1, hedge_exchange="arcus")
        maker, taker = FakeMaker(), FakeLighter()
        bot = HedgeBot(config, maker, taker)
        bot._notify = AsyncMock()
        await bot.run()
        self.assertEqual((maker.position, taker.position), (Decimal(0), Decimal(0)))
        self.assertEqual(len(taker.orders), 2)

    async def test_hedge_client_must_implement_execution_capability(self):
        with self.assertRaisesRegex(TypeError, "does not support bounded market hedges"):
            HedgeBot(HedgeConfig("bulk", "ETH", Decimal("0.1"), 1), FakeMaker(), object())

    async def test_selected_hedge_exchange_is_passed_to_factory(self):
        with patch("hedge_mode.ExchangeFactory.create_exchange",
                   side_effect=[FakeMaker(), FakeLighter()]) as create:
            bot = HedgeBot(HedgeConfig("bulk", "ETH", Decimal("0.1"), 1,
                                       hedge_exchange="lighter"))
        self.assertEqual([call.args[0] for call in create.call_args_list], ["bulk", "lighter"])
        self.assertIs(create.call_args_list[1].args[1], bot.hedge_config)
        self.assertEqual(bot.hedge_config.quantity, Decimal("0.1"))

    async def test_inexact_hedge_size_stops_before_maker_order(self):
        maker, lighter = FakeMaker(), FakeLighter()
        bot = HedgeBot(HedgeConfig("bulk", "ETH", Decimal("0.0005"), 1), maker, lighter)
        bot._notify = AsyncMock()
        with self.assertRaisesRegex(ValueError, "Hedge quantity is not exact"):
            await bot.run()
        self.assertEqual(maker.orders, [])
        self.assertEqual(lighter.orders, [])

    async def test_transient_position_read_recovers_before_new_order(self):
        bot, maker, lighter = self.make_bot()
        cast(Any, maker).get_account_positions = AsyncMock(side_effect=[
            ConnectionError("temporary account read failure"), Decimal(0),
        ])
        self.assertEqual(await bot._positions(), Decimal(0))
        self.assertEqual(lighter.orders, [])

    async def test_position_mismatch_stops_after_recovery_window(self):
        bot, maker, lighter = self.make_bot()
        maker.position = Decimal("0.1")
        bot.config.recovery_timeout = 0.001
        with self.assertRaisesRegex(RuntimeError, "Unmatched or unreadable positions"):
            await bot._positions()
        self.assertEqual(lighter.orders, [])

    async def test_max_position_accumulates_then_reverses_and_flattens(self):
        bot, maker, lighter = self.make_bot(Decimal("0.2"))
        await bot.run()
        self.assertEqual((maker.position, lighter.position), (Decimal(0), Decimal(0)))
        self.assertEqual(len(maker.orders), 8)
        self.assertEqual(len(lighter.orders), 8)
        self.assertEqual([side for side, _, _ in maker.orders],
                         ["buy", "buy", "sell", "sell", "sell", "sell", "buy", "buy"])

    async def test_partial_fill_is_hedged_before_confirmed_cancel(self):
        bot, maker, lighter = self.make_bot()
        maker.position = Decimal("0.04")
        maker.place_open_order = AsyncMock(return_value=OrderResult(
            True, "partial", "buy", Decimal("0.1"), Decimal(100), "OPEN",
            filled_size=Decimal("0.04")))
        maker.cancel_order = AsyncMock(return_value=OrderResult(
            True, "partial", filled_size=Decimal("0.04")))
        async def order_info(_order_id):
            status = "CANCELED" if cast(AsyncMock, maker.cancel_order).await_count else "OPEN"
            return OrderInfo("partial", "buy", Decimal("0.1"), Decimal(100), status,
                             filled_size=Decimal("0.04"), remaining_size=Decimal("0.06"))
        cast(Any, maker).get_order_info = order_info
        bot.config.fill_timeout = 0.001
        filled = await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        self.assertEqual(filled, Decimal("0.04"))
        self.assertEqual(lighter.orders, [("sell", Decimal("0.04"), False)])
        maker.cancel_order.assert_awaited_once_with("partial")

    async def test_hedge_failure_cancels_maker_and_stops(self):
        bot, maker, lighter = self.make_bot()
        lighter.place_hedge_market_order = AsyncMock(side_effect=RuntimeError("unknown ACK"))
        with self.assertRaisesRegex(RuntimeError, "unknown ACK"):
            await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        self.assertEqual(maker.cancels, ["1"])
        lighter.place_hedge_market_order.assert_awaited_once()

    async def test_lost_maker_ack_with_order_id_recovers_confirmed_fill(self):
        bot, maker, lighter = self.make_bot()
        original_place = maker.place_open_order

        async def lose_ack(contract_id, quantity, side):
            result = await original_place(contract_id, quantity, side)
            raise OrderOutcomeUnknown("Bulk ACK lost", result.order_id)

        maker.place_open_order = lose_ack
        filled = await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        self.assertEqual(filled, Decimal("0.1"))
        self.assertEqual(lighter.position, Decimal("-0.1"))
        self.assertEqual(len(maker.orders), 1)

    async def test_transient_maker_read_recovers_without_new_order(self):
        bot, maker, lighter = self.make_bot()
        original_read = maker.get_order_info
        reads = 0

        async def flaky_read(order_id):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise ConnectionError("temporary read failure")
            return await original_read(order_id)

        cast(Any, maker).get_order_info = flaky_read
        filled = await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        self.assertEqual(filled, Decimal("0.1"))
        self.assertEqual(lighter.position, Decimal("-0.1"))
        self.assertEqual(len(maker.orders), 1)

    async def test_lost_maker_ack_without_order_id_stops_without_resubmitting(self):
        bot, maker, lighter = self.make_bot(exchange="arcus")
        maker.place_open_order = AsyncMock(side_effect=OrderOutcomeUnknown("Arcus client ID only"))
        with self.assertRaisesRegex(OrderOutcomeUnknown, "client ID only"):
            await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        maker.place_open_order.assert_awaited_once()
        self.assertEqual(lighter.orders, [])

    async def test_lost_cancel_ack_recovers_late_partial_fill(self):
        bot, maker, lighter = self.make_bot()
        maker.place_open_order = AsyncMock(return_value=OrderResult(
            True, "partial", "buy", Decimal("0.1"), Decimal(100), "OPEN", filled_size=Decimal(0),
        ))
        maker.position = Decimal("0.04")
        canceled = False

        async def read_order(_order_id):
            return OrderInfo("partial", "buy", Decimal("0.1"), Decimal(100),
                             "CANCELED" if canceled else "PARTIALLY_FILLED",
                             filled_size=Decimal("0.04"), remaining_size=Decimal("0.06"))

        async def lose_cancel_ack(_order_id):
            nonlocal canceled
            canceled = True
            raise ConnectionError("cancel ACK lost")

        cast(Any, maker).get_order_info = read_order
        cancel_mock = AsyncMock(side_effect=lose_cancel_ack)
        cast(Any, maker).cancel_order = cancel_mock
        bot.config.fill_timeout = 0.001
        filled = await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        self.assertEqual(filled, Decimal("0.04"))
        self.assertEqual(lighter.orders, [("sell", Decimal("0.04"), False)])
        cancel_mock.assert_awaited_once_with("partial")

    async def test_unconfirmed_maker_terminal_stops_without_another_order(self):
        bot, maker, lighter = self.make_bot()
        bot.config.fill_timeout = 0.001
        bot.config.recovery_timeout = 0.001
        maker.place_open_order = AsyncMock(return_value=OrderResult(
            True, "stuck", "buy", Decimal("0.1"), Decimal(100), "OPEN", filled_size=Decimal(0),
        ))
        cast(Any, maker).get_order_info = AsyncMock(return_value=OrderInfo(
            "stuck", "buy", Decimal("0.1"), Decimal(100), "OPEN",
            filled_size=Decimal(0), remaining_size=Decimal("0.1"),
        ))
        cast(Any, maker).cancel_order = AsyncMock(return_value=OrderResult(
            True, "stuck", filled_size=Decimal(0),
        ))
        with self.assertRaisesRegex(RuntimeError, "terminal state is unconfirmed"):
            await bot._execute_maker_order("buy", Decimal("0.1"), Decimal(0), False)
        maker.place_open_order.assert_awaited_once()
        self.assertEqual(lighter.orders, [])

    async def test_lost_lighter_ack_recovers_confirmed_target_without_resubmission(self):
        bot, _, lighter = self.make_bot()

        async def lose_ack(quantity, side, max_slippage_percent, reduce_only=False):
            lighter.position -= quantity
            raise OrderOutcomeUnknown("Lighter ACK lost", "123")

        lighter.place_hedge_market_order = AsyncMock(side_effect=lose_ack)
        await bot._hedge_to(Decimal("-0.1"))
        self.assertEqual(lighter.position, Decimal("-0.1"))
        lighter.place_hedge_market_order.assert_awaited_once()

    async def test_transient_lighter_position_read_recovers_before_hedge(self):
        bot, _, lighter = self.make_bot()
        original_read = lighter.get_account_positions
        reads = 0

        async def flaky_read():
            nonlocal reads
            reads += 1
            if reads == 1:
                raise ConnectionError("temporary position read failure")
            return await original_read()

        cast(Any, lighter).get_account_positions = flaky_read
        await bot._hedge_to(Decimal("-0.1"))
        self.assertEqual(lighter.position, Decimal("-0.1"))
        self.assertEqual(len(lighter.orders), 1)

    async def test_lost_lighter_ack_without_target_stops_without_resubmission(self):
        bot, _, lighter = self.make_bot()
        bot.config.recovery_timeout = 0.001
        lighter.place_hedge_market_order = AsyncMock(side_effect=OrderOutcomeUnknown("ACK lost", "123"))
        with self.assertRaisesRegex(RuntimeError, "not fully confirmed"):
            await bot._hedge_to(Decimal("-0.1"))
        lighter.place_hedge_market_order.assert_awaited_once()

    async def test_telegram_notification_remains_available(self):
        bot, _, _ = self.make_bot()
        del bot._notify
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token",
                                  "TELEGRAM_CHAT_ID": "chat", "LARK_TOKEN": ""}):
            with patch("helpers.notifications.TelegramBot") as telegram:
                await HedgeBot._notify(bot, "check positions")
        telegram.return_value.__enter__.return_value.send_text.assert_called_once_with(
            "check positions")


class LighterHedgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_quantity_capability_rejects_inexact_size(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.base_amount_multiplier = 1000
        client.price_multiplier = 100
        client.validate_hedge_quantity(Decimal("0.1"))
        with self.assertRaisesRegex(ValueError, "cannot represent exact"):
            client.validate_hedge_quantity(Decimal("0.0005"))

    async def test_failed_signer_check_closes_sdk_session(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.lighter_client = None
        client.api_key_private_key = "test-key"
        client.base_url = "https://example.invalid"
        client.account_index = 1
        client.api_key_index = 0
        client.logger = SimpleNamespace(log=lambda *_args: None)
        with patch("exchanges.lighter.SignerClient") as signer_class:
            signer_class.return_value.check_client.return_value = "invalid key"
            signer_class.return_value.close = AsyncMock()
            with self.assertRaisesRegex(Exception, "CheckClient error"):
                await client._initialize_lighter_client()
        signer_class.return_value.close.assert_awaited_once()
        self.assertIsNone(client.lighter_client)

    async def test_disconnect_closes_both_sdk_sessions(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.api_client = SimpleNamespace(close=AsyncMock())
        client.lighter_client = SimpleNamespace(close=AsyncMock())
        api_session, signer_session = client.api_client, client.lighter_client
        client.logger = SimpleNamespace(log=lambda *_args: None)
        await client.disconnect()
        api_session.close.assert_awaited_once()
        signer_session.close.assert_awaited_once()
        self.assertIsNone(client.api_client)
        self.assertIsNone(client.lighter_client)

    async def test_active_orders_pass_sdk_authorization_header(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.config = SimpleNamespace(contract_id=1)
        client.account_index = 2
        client.api_key_index = 3
        client.api_client = object()
        client.lighter_client = SimpleNamespace(
            create_auth_token_with_expiry=lambda **_kwargs: ("token", None),
        )
        with patch("exchanges.lighter.OrderApi") as order_api:
            order_api.return_value.account_active_orders = AsyncMock(
                return_value=SimpleNamespace(orders=[]),
            )
            self.assertEqual(await client._fetch_orders_with_retry(), [])
        order_api.return_value.account_active_orders.assert_awaited_once_with(
            account_index=2, market_id=1, authorization="token",
        )

    async def test_market_metadata_initializes_api_client_before_connect(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.config = SimpleNamespace(ticker="ETH")
        client.api_client = None
        client.base_url = "https://example.invalid"
        client.logger = SimpleNamespace(log=lambda *_args: None)
        market = SimpleNamespace(symbol="ETH", market_id=1,
                                 supported_size_decimals=3, supported_price_decimals=2)
        details = SimpleNamespace(price_decimals=2)
        with patch("exchanges.lighter.ApiClient", return_value=object()) as api_client:
            with patch("exchanges.lighter.OrderApi") as order_api:
                order_api.return_value.order_books = AsyncMock(return_value=SimpleNamespace(
                    order_books=[market]))
                order_api.return_value.order_book_details = AsyncMock(return_value=SimpleNamespace(
                    order_book_details=[details]))
                contract_id, tick = await client.get_contract_attributes()
        self.assertEqual((contract_id, tick), (1, Decimal("0.01")))
        self.assertEqual(client.base_amount_multiplier, 1000)
        api_client.assert_called_once()

    async def test_market_hedge_uses_ioc_and_slippage_bound(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.config = SimpleNamespace(contract_id=1)
        client.base_amount_multiplier = 1000
        client.price_multiplier = 100
        client.api_key_index = 0
        client._hedge_order_index = 1
        client.fetch_bbo_prices = AsyncMock(return_value=(Decimal(100), Decimal(101)))
        client.lighter_client = SimpleNamespace(create_market_order=AsyncMock(
            return_value=(None, SimpleNamespace(code=200), None)))
        result = await client.place_hedge_market_order(
            Decimal("0.1"), "buy", Decimal("0.5"), reduce_only=True)
        self.assertTrue(result.success)
        args = client.lighter_client.create_market_order.await_args.kwargs
        self.assertEqual(args["base_amount"], 100)
        self.assertEqual(args["avg_execution_price"], 10151)
        self.assertFalse(args["is_ask"])
        self.assertTrue(args["reduce_only"])

    async def test_position_sign_is_preserved(self):
        client = cast(Any, LighterClient.__new__(LighterClient))
        client.config = SimpleNamespace(contract_id=1)
        client._fetch_positions_with_retry = AsyncMock(return_value=[
            SimpleNamespace(market_id=1, position="0.1", sign=-1),
        ])
        self.assertEqual(await client.get_account_positions(), Decimal("-0.1"))


if __name__ == "__main__":
    unittest.main()
