#!/usr/bin/env python3
"""Maker trades on Bulk or Arcus, matched by market hedges on Lighter."""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

import dotenv

from exchanges import ExchangeFactory
from exchanges.base import BaseExchangeClient, OrderInfo, OrderResult
from helpers import TradingLogger
from helpers.lark_bot import LarkBot
from helpers.telegram_bot import TelegramBot
from trading_bot import TradingConfig


TERMINAL = {"FILLED", "CANCELED", "REJECTED"}


class LighterHedgeClient(Protocol):
    base_amount_multiplier: Decimal

    def setup_order_update_handler(self, handler) -> None: ...
    async def get_contract_attributes(self) -> tuple[str, Decimal]: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def fetch_bbo_prices(self, contract_id: str) -> tuple[Decimal, Decimal]: ...
    async def get_active_orders(self, contract_id: str) -> list[OrderInfo]: ...
    async def get_account_positions(self) -> Decimal: ...
    async def place_hedge_market_order(
        self, quantity: Decimal, side: str, max_slippage_percent: Decimal,
        reduce_only: bool = False,
    ) -> OrderResult: ...


@dataclass
class HedgeConfig:
    exchange: str
    ticker: str
    size: Decimal
    iterations: int
    fill_timeout: float = 5
    sleep_time: float = 0
    max_position: Decimal = Decimal(0)
    max_slippage: Decimal = Decimal("0.5")
    target_timeout: float = 600

    def __post_init__(self):
        if self.exchange not in {"bulk", "arcus"}:
            raise ValueError("Hedge maker exchange must be bulk or arcus")
        if self.size <= 0 or self.iterations <= 0 or self.fill_timeout <= 0 or self.target_timeout <= 0:
            raise ValueError("Size, iterations, fill timeout and target timeout must be positive")
        if self.sleep_time < 0 or not Decimal(0) < self.max_slippage <= Decimal(5):
            raise ValueError("Sleep must be nonnegative and slippage must be within (0, 5] percent")
        if self.max_position < 0 or (self.max_position and self.max_position < self.size):
            raise ValueError("Max position must be zero or at least one order size")
        if self.max_position and self.max_position % self.size:
            raise ValueError("Max position must be an exact multiple of order size")


class HedgeBot:
    def __init__(self, config: HedgeConfig, maker_client=None, lighter_client=None):
        self.config = config
        self.logger = TradingLogger(f"{config.exchange}_lighter_hedge", config.ticker, log_to_console=True)
        self.maker_config = TradingConfig(
            ticker=config.ticker, contract_id="", tick_size=Decimal(0),
            quantity=config.size, take_profit=Decimal(0), direction="buy",
            max_orders=1, wait_time=0, exchange=config.exchange,
            grid_step=Decimal(-100), stop_price=Decimal(-1), pause_price=Decimal(-1),
        )
        self.lighter_config = TradingConfig(
            ticker=config.ticker, contract_id="", tick_size=Decimal(0),
            quantity=config.size, take_profit=Decimal(0), direction="sell",
            max_orders=1, wait_time=0, exchange="lighter",
            grid_step=Decimal(-100), stop_price=Decimal(-1), pause_price=Decimal(-1),
        )
        self.maker: BaseExchangeClient = maker_client or ExchangeFactory.create_exchange(
            config.exchange, self.maker_config,
        )
        self.lighter = cast(LighterHedgeClient, lighter_client or ExchangeFactory.create_exchange(
            "lighter", self.lighter_config,
        ))
        maker_network = getattr(self.maker, "signature_domain", None) or getattr(self.maker, "network", "mainnet")
        if maker_network != "mainnet":
            raise ValueError("Lighter hedge mode requires the maker exchange on mainnet too")
        self.maker.setup_order_update_handler(lambda _message: None)
        self.lighter.setup_order_update_handler(lambda _message: None)

    async def _notify(self, message: str):
        token = os.getenv("LARK_TOKEN")
        if token:
            async with LarkBot(token) as bot:
                await bot.send_text(message)
        token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            def send():
                with TelegramBot(token, chat_id) as bot:
                    bot.send_text(message)
            await asyncio.to_thread(send)

    async def _positions(self) -> Decimal:
        maker = await self.maker.get_account_positions()
        lighter = await self.lighter.get_account_positions()
        if maker + lighter != 0:
            raise RuntimeError(f"Unmatched positions: {self.config.exchange}={maker}, Lighter={lighter}")
        return maker

    async def _confirm_lighter_target(self, target: Decimal):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            position = await self.lighter.get_account_positions()
            if position == target:
                return
            await asyncio.sleep(0.25)
        position = await self.lighter.get_account_positions()
        raise RuntimeError(f"Lighter hedge not fully confirmed: target={target}, actual={position}")

    async def _hedge_to(self, target: Decimal):
        current = await self.lighter.get_account_positions()
        difference = target - current
        if difference == 0:
            return
        quantity = abs(difference)
        if quantity * self.lighter.base_amount_multiplier != (
            quantity * self.lighter.base_amount_multiplier
        ).to_integral_value():
            raise RuntimeError(f"Lighter cannot represent exact hedge quantity {quantity}")
        side = "buy" if difference > 0 else "sell"
        reduce_only = current != 0 and current * difference < 0 and quantity <= abs(current)
        result = await self.lighter.place_hedge_market_order(
            quantity, side, self.config.max_slippage, reduce_only=reduce_only,
        )
        if not result.success:
            raise RuntimeError(f"Lighter hedge rejected: {result.error_message}")
        await self._confirm_lighter_target(target)

    async def _confirmed_maker_fill(self, starting_position: Decimal, side: str,
                                    quantity: Decimal, filled: Decimal,
                                    previous: Decimal) -> Decimal:
        if filled < previous or filled > quantity:
            raise RuntimeError("Maker order cumulative fill is inconsistent")
        if filled > previous:
            direction = Decimal(1) if side == "buy" else Decimal(-1)
            await self._hedge_to(-(starting_position + direction * filled))
            self.logger.log(f"Hedged {filled - previous} {self.config.ticker} after maker fill", "INFO")
        return filled

    async def _execute_maker_order(self, side: str, quantity: Decimal,
                                   starting_position: Decimal, reduce_only: bool) -> Decimal:
        if quantity * self.lighter.base_amount_multiplier != (
            quantity * self.lighter.base_amount_multiplier
        ).to_integral_value():
            raise ValueError("Order quantity cannot be hedged exactly on Lighter")
        if reduce_only:
            bid, ask = await self.maker.fetch_bbo_prices(self.maker_config.contract_id)
            if bid <= 0 or ask <= 0 or bid >= ask:
                raise ConnectionError("Maker order book is not ready")
            price = bid if side == "buy" else ask
            result = await self.maker.place_close_order(
                self.maker_config.contract_id, quantity, price, side,
            )
        else:
            result = await self.maker.place_open_order(
                self.maker_config.contract_id, quantity, side,
            )
        if not result.order_id:
            if result.filled_size:
                raise RuntimeError("Maker order reports a fill without an order ID")
            if not result.success:
                self.logger.log(f"Maker order rejected: {result.error_message}", "WARNING")
                return Decimal(0)
            raise RuntimeError("Maker order has no ID")

        order_id = result.order_id
        filled = Decimal(0)
        terminal = False
        try:
            reported = Decimal(result.filled_size or 0)
            filled = await self._confirmed_maker_fill(starting_position, side, quantity, reported, filled)
            status = result.status or "OPEN"
            deadline = time.monotonic() + self.config.fill_timeout
            while status not in TERMINAL and time.monotonic() < deadline:
                order = await self.maker.get_order_info(order_id)
                if order is None:
                    raise RuntimeError(f"Maker order {order_id} state is unknown")
                filled = await self._confirmed_maker_fill(
                    starting_position, side, quantity, Decimal(order.filled_size), filled,
                )
                status = order.status
                if status not in TERMINAL:
                    await asyncio.sleep(0.25)
            if status not in TERMINAL:
                canceled = await self.maker.cancel_order(order_id)
                if not canceled.success:
                    raise RuntimeError(f"Maker cancel not confirmed: {canceled.error_message}")
                filled = await self._confirmed_maker_fill(
                    starting_position, side, quantity, Decimal(canceled.filled_size or 0), filled,
                )
            confirmed = await self.maker.get_order_info(order_id)
            if confirmed is None or confirmed.status not in TERMINAL:
                raise RuntimeError(f"Maker order {order_id} has no confirmed terminal state")
            filled = await self._confirmed_maker_fill(
                starting_position, side, quantity, Decimal(confirmed.filled_size), filled,
            )
            terminal = True
            return filled
        finally:
            if not terminal:
                try:
                    await self.maker.cancel_order(order_id)
                except Exception as exc:
                    self.logger.log(f"Maker order {order_id} still needs manual cancellation check: {exc}", "ERROR")

    async def _reach_target(self, target: Decimal):
        deadline = time.monotonic() + self.config.target_timeout
        while time.monotonic() < deadline:
            position = await self._positions()
            if position == target:
                return
            if await self.maker.get_active_orders(self.maker_config.contract_id):
                raise RuntimeError("Unexpected active maker order; review it before continuing")
            difference = target - position
            side = "buy" if difference > 0 else "sell"
            reducing = position != 0 and position * difference < 0
            quantity = min(self.config.size, abs(difference))
            if reducing:
                quantity = min(quantity, abs(position))
            self.maker_config.direction = side
            filled = await self._execute_maker_order(side, quantity, position, reducing)
            if filled == 0:
                await asyncio.sleep(1)
            else:
                expected = position + (filled if side == "buy" else -filled)
                settle_deadline = time.monotonic() + 10
                while time.monotonic() < settle_deadline:
                    actual = await self.maker.get_account_positions()
                    if actual == expected:
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise RuntimeError(f"Maker position did not settle: expected={expected}, actual={actual}")
        raise TimeoutError(f"Target maker position {target} was not reached in time")

    async def _wait_for_books(self):
        deadline = time.monotonic() + 20
        for client, contract_id, name in (
            (self.maker, self.maker_config.contract_id, self.config.exchange),
            (self.lighter, self.lighter_config.contract_id, "Lighter"),
        ):
            while time.monotonic() < deadline:
                try:
                    bid, ask = await asyncio.wait_for(client.fetch_bbo_prices(contract_id), 5)
                    if bid > 0 and ask > 0 and bid < ask:
                        break
                except (ConnectionError, TimeoutError, ValueError):
                    pass
                await asyncio.sleep(0.5)
            else:
                raise ConnectionError(f"{name} order book was not ready before hedging")

    async def run(self):
        try:
            self.maker_config.contract_id, self.maker_config.tick_size = (
                await self.maker.get_contract_attributes()
            )
            self.lighter_config.contract_id, self.lighter_config.tick_size = (
                await self.lighter.get_contract_attributes()
            )
            await self.maker.connect()
            await self.lighter.connect()
            await self._wait_for_books()
            if self.config.size * self.lighter.base_amount_multiplier != (
                self.config.size * self.lighter.base_amount_multiplier
            ).to_integral_value():
                raise ValueError("Configured order size cannot be hedged exactly on Lighter")
            if await self.maker.get_active_orders(self.maker_config.contract_id):
                raise RuntimeError("Maker account has active orders; start with a dedicated flat account")
            if await self.lighter.get_active_orders(self.lighter_config.contract_id):
                raise RuntimeError("Lighter account has active orders; start with a dedicated flat account")
            if await self._positions() != 0:
                raise RuntimeError("Hedge mode requires flat positions at startup")

            cap = self.config.max_position or self.config.size
            for iteration in range(self.config.iterations):
                self.logger.log(f"Hedge cycle {iteration + 1}/{self.config.iterations}", "INFO")
                await self._reach_target(cap)
                if self.config.sleep_time:
                    await asyncio.sleep(self.config.sleep_time)
                await self._reach_target(-cap if self.config.max_position else Decimal(0))
            if self.config.max_position:
                await self._reach_target(Decimal(0))
            self.logger.log("Hedge cycles complete; both positions are flat", "INFO")
        except BaseException as exc:
            message = (f"[{self.config.exchange.upper()}+LIGHTER {self.config.ticker}] "
                       f"Hedge mode stopped: {type(exc).__name__}: {exc}. "
                       "Check both exchanges' orders and positions before restarting.")
            self.logger.log(message, "ERROR")
            try:
                await self._notify(message)
            except Exception as notify_error:
                self.logger.log(f"Hedge notification failed: {notify_error}", "ERROR")
            raise
        finally:
            await asyncio.gather(self.maker.disconnect(), self.lighter.disconnect(),
                                 return_exceptions=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", required=True, choices=("bulk", "arcus"),
                        help="Maker exchange; Lighter is always the hedge leg")
    parser.add_argument("--ticker", default="ETH")
    parser.add_argument("--size", type=Decimal, required=True, help="Underlying amount per maker order")
    parser.add_argument("--iter", type=int, required=True, help="Number of hedge cycles")
    parser.add_argument("--fill-timeout", type=float, default=5,
                        help="Seconds before canceling an unfilled maker order")
    parser.add_argument("--sleep", type=float, default=0,
                        help="Seconds to wait at the positive position target")
    parser.add_argument("--max-position", type=Decimal, default=Decimal(0),
                        help="Absolute maker position cap; zero means one flat round trip per cycle")
    parser.add_argument("--max-slippage", type=Decimal, default=Decimal("0.5"),
                        help="Maximum Lighter market hedge slippage in percent (default: 0.5)")
    parser.add_argument("--target-timeout", type=float, default=600,
                        help="Seconds allowed to reach each maker position target")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    if not Path(args.env_file).exists():
        raise FileNotFoundError(f"Env file not found: {args.env_file}")
    dotenv.load_dotenv(args.env_file)
    config = HedgeConfig(
        exchange=args.exchange, ticker=args.ticker.upper(), size=args.size,
        iterations=args.iter, fill_timeout=args.fill_timeout, sleep_time=args.sleep,
        max_position=args.max_position, max_slippage=args.max_slippage,
        target_timeout=args.target_timeout,
    )
    await HedgeBot(config).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Hedge mode stopped: {exc}", file=sys.stderr)
        sys.exit(1)
