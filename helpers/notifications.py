"""Shared alerts for trading strategies."""

import asyncio
import os

from helpers.lark_bot import LarkBot
from helpers.telegram_bot import TelegramBot


class Notifier:
    """Send configured alerts without blocking the trading event loop."""

    async def send(self, message: str) -> None:
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            async with LarkBot(lark_token) as bot:
                await bot.send_text(message)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            def send_telegram() -> None:
                with TelegramBot(telegram_token, telegram_chat_id) as bot:
                    bot.send_text(message)

            await asyncio.to_thread(send_telegram)
