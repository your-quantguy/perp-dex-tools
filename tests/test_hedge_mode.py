import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from exchanges.base import OrderInfo, OrderResult
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

    async def test_mixed_networks_are_rejected_before_any_order(self):
        maker = FakeMaker()
        maker.network = "testnet"
        with self.assertRaisesRegex(ValueError, "mainnet"):
            HedgeBot(HedgeConfig("arcus", "ETH", Decimal("0.1"), 1),
                     maker, FakeLighter())
        self.assertEqual(maker.orders, [])

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

    async def test_telegram_notification_remains_available(self):
        bot, _, _ = self.make_bot()
        del bot._notify
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "token",
                                  "TELEGRAM_CHAT_ID": "chat", "LARK_TOKEN": ""}):
            with patch("hedge_mode.TelegramBot") as telegram:
                await HedgeBot._notify(bot, "check positions")
        telegram.return_value.__enter__.return_value.send_text.assert_called_once_with(
            "check positions")


class LighterHedgeTests(unittest.IsolatedAsyncioTestCase):
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
            return_value=(None, SimpleNamespace(code=0), None)))
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
