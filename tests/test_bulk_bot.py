import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from exchanges.base import OrderResult
from trading_bot import TradingBot


class BulkBotOrderEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot: Any = TradingBot.__new__(TradingBot)
        self.handler: Any = None
        self.bot.config = SimpleNamespace(
            exchange="bulk", contract_id="ETH-USD", direction="buy",
            close_order_side="sell", quantity=Decimal("0.1"),
            take_profit=Decimal("0.02"),
        )
        self.bot.logger = SimpleNamespace(log=lambda *_args: None, log_transaction=lambda *_args: None)
        self.bot.exchange_client = SimpleNamespace(
            setup_order_update_handler=lambda handler: setattr(self, "handler", handler),
        )
        self.bot.loop = None
        self.bot.order_filled_event = asyncio.Event()
        self.bot.order_canceled_event = asyncio.Event()
        self.bot.current_order_status = None
        self.bot.order_filled_amount = Decimal(0)
        self.bot._bulk_open_order_id = None
        self.bot._bulk_waiting_for_open_ack = False
        self.bot._bulk_early_open_updates = {}
        self.bot._setup_websocket_handlers()

    @staticmethod
    def _filled(order_id):
        return {
            "contract_id": "ETH-USD", "order_id": order_id, "side": "buy",
            "order_type": "OPEN", "status": "FILLED", "size": "0.1",
            "price": "2000", "filled_size": "0.1",
        }

    async def test_late_old_fill_does_not_complete_current_open_order(self):
        self.bot._bulk_open_order_id = "new"
        self.handler(self._filled("old"))
        await asyncio.sleep(0)
        self.assertFalse(self.bot.order_filled_event.is_set())
        self.assertEqual(self.bot.order_filled_amount, Decimal(0))
        self.handler(self._filled("new"))
        await asyncio.sleep(0)
        self.assertTrue(self.bot.order_filled_event.is_set())

    async def test_fill_before_order_ack_is_replayed_by_its_id(self):
        async def place_open_order(*_args):
            self.handler(self._filled("new"))
            return OrderResult(True, "new", "buy", Decimal("0.1"),
                               Decimal("2000"), "OPEN")

        self.bot.exchange_client.place_open_order = place_open_order
        self.bot._handle_order_result = AsyncMock(return_value=True)
        result = await self.bot._place_and_monitor_open_order()
        self.assertTrue(result)
        self.assertTrue(self.bot.order_filled_event.is_set())
        self.assertEqual(self.bot.order_filled_amount, Decimal("0.1"))
        self.bot._handle_order_result.assert_awaited_once()

    async def test_bulk_cancel_uses_confirmed_filled_size(self):
        self.bot.order_filled_event.clear()
        self.bot.exchange_client.get_order_price = AsyncMock(return_value=Decimal("2001"))
        self.bot.exchange_client.get_order_info = AsyncMock(return_value=SimpleNamespace(status="OPEN"))
        self.bot.exchange_client.cancel_order = AsyncMock(return_value=OrderResult(
            True, "entry", filled_size=Decimal("0.04"),
        ))
        self.bot.exchange_client.place_close_order = AsyncMock(return_value=OrderResult(
            True, "exit", "sell", Decimal("0.04"), Decimal("2000.4"), "OPEN",
        ))
        opened = OrderResult(True, "entry", "buy", Decimal("0.1"), Decimal("2000"), "OPEN")
        self.assertTrue(await self.bot._handle_order_result(opened))
        close_call = self.bot.exchange_client.place_close_order.await_args
        assert close_call is not None
        self.assertEqual(close_call.args[1], Decimal("0.04"))
