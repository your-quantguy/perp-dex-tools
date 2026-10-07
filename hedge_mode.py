#!/usr/bin/env python3
"""Maker trades on Bulk or Arcus, matched by a selected hedge exchange."""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import dotenv

from exchanges import ExchangeFactory
from exchanges.capabilities import PairMaker, PairTaker
from exchanges.config import VenueConfig
from execution.pair import MakerFirstPairExecutor
from helpers import TradingLogger
from helpers.notifications import Notifier
from strategies.runner import StrategyRunner, VenueLifecycle


@dataclass
class HedgeCycleConfig:
    exchange: str
    ticker: str
    size: Decimal
    iterations: int
    fill_timeout: float = 5
    sleep_time: float = 0
    max_position: Decimal = Decimal(0)
    max_slippage: Decimal = Decimal("0.5")
    target_timeout: float = 600
    recovery_timeout: float = 15
    hedge_exchange: str = "lighter"

    def __post_init__(self):
        if not self.exchange or not self.hedge_exchange:
            raise ValueError("Both exchange names are required")
        if self.exchange == self.hedge_exchange:
            raise ValueError("Maker and hedge exchanges must differ")
        if (self.size <= 0 or self.iterations <= 0 or self.fill_timeout <= 0
                or self.target_timeout <= 0 or self.recovery_timeout <= 0):
            raise ValueError("Size, iterations and timeouts must be positive")
        if self.sleep_time < 0 or not Decimal(0) < self.max_slippage <= Decimal(5):
            raise ValueError("Sleep must be nonnegative and slippage must be within (0, 5] percent")
        if self.max_position < 0 or (self.max_position and self.max_position < self.size):
            raise ValueError("Max position must be zero or at least one order size")
        if self.max_position and self.max_position % self.size:
            raise ValueError("Max position must be an exact multiple of order size")


HedgeConfig = HedgeCycleConfig


class HedgeCycleStrategy:
    def __init__(self, config: HedgeCycleConfig, maker_client=None, hedge_client=None):
        self.config = config
        self.logger = TradingLogger(
            f"{config.exchange}_{config.hedge_exchange}_hedge", config.ticker, log_to_console=True,
        )
        self.maker_config = VenueConfig(
            ticker=config.ticker, contract_id="", tick_size=Decimal(0),
            quantity=config.size, direction="buy",
        )
        self.hedge_config = VenueConfig(
            ticker=config.ticker, contract_id="", tick_size=Decimal(0),
            quantity=config.size, direction="sell",
        )
        selected_maker = maker_client or ExchangeFactory.create_exchange(
            config.exchange, self.maker_config,
        )
        selected_hedge = hedge_client or ExchangeFactory.create_exchange(
            config.hedge_exchange, self.hedge_config,
        )
        if not isinstance(selected_maker, PairMaker):
            raise TypeError(f"{config.exchange} does not support post-only maker orders and order reconciliation")
        if not isinstance(selected_hedge, PairTaker):
            raise TypeError(f"{config.hedge_exchange} does not support bounded market hedges")
        self.maker = selected_maker
        self.hedge = selected_hedge
        if self.maker.network_id != self.hedge.network_id:
            raise ValueError("Hedge venues must use the same network")
        self.maker.setup_order_update_handler(lambda _message: None)
        self.hedge.setup_order_update_handler(lambda _message: None)
        self.notifier = Notifier()
        self.pair = MakerFirstPairExecutor(
            config, self.maker, self.hedge,
            self.maker_config, self.hedge_config, self.logger,
        )

    async def _notify(self, message: str):
        await self.notifier.send(message)

    async def _positions(self) -> Decimal:
        return await self.pair.positions()

    async def _hedge_to(self, target: Decimal) -> None:
        await self.pair.hedge_to(target)

    async def _execute_maker_order(self, side: str, quantity: Decimal,
                                   starting_position: Decimal, reduce_only: bool) -> Decimal:
        return await self.pair.execute_maker_order(side, quantity, starting_position, reduce_only)

    async def _reach_target(self, target: Decimal) -> None:
        await self.pair.reach_target(target)

    async def _wait_for_books(self):
        deadline = time.monotonic() + 20
        for client, contract_id, name in (
            (self.maker, self.maker_config.contract_id, self.config.exchange),
            (self.hedge, self.hedge_config.contract_id, self.config.hedge_exchange),
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

    @property
    def venues(self) -> tuple[VenueLifecycle, ...]:
        return self.maker, self.hedge

    async def prepare(self) -> None:
        self.maker_config.contract_id, self.maker_config.tick_size = (
            await self.maker.get_contract_attributes()
        )
        self.hedge_config.contract_id, self.hedge_config.tick_size = (
            await self.hedge.get_contract_attributes()
        )

    async def execute(self) -> None:
        await self._wait_for_books()
        self.hedge.validate_hedge_quantity(self.config.size)
        if await self.maker.get_active_orders(self.maker_config.contract_id):
            raise RuntimeError("Maker account has active orders; start with a dedicated flat account")
        if await self.hedge.get_active_orders(self.hedge_config.contract_id):
            raise RuntimeError("Hedge account has active orders; start with a dedicated flat account")
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

    async def on_failure(self, exc: BaseException) -> None:
        message = (f"[{self.config.exchange.upper()}+{self.config.hedge_exchange.upper()} "
                   f"{self.config.ticker}] "
                   f"Hedge mode stopped: {type(exc).__name__}: {exc}. "
                   "Check both exchanges' orders and positions before restarting.")
        self.logger.log(message, "ERROR")
        try:
            await self._notify(message)
        except Exception as notify_error:
            self.logger.log(f"Hedge notification failed: {notify_error}", "ERROR")

    async def run(self) -> None:
        await StrategyRunner().run(self)


HedgeBot = HedgeCycleStrategy


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", required=True, choices=ExchangeFactory.get_capable_exchanges(PairMaker),
                        help="Maker exchange")
    parser.add_argument("--hedge-exchange", choices=ExchangeFactory.get_capable_exchanges(PairTaker), default="lighter",
                        help="Exchange with bounded immediate-order capability")
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
                        help="Maximum hedge market order slippage in percent (default: 0.5)")
    parser.add_argument("--target-timeout", type=float, default=600,
                        help="Seconds allowed to reach each maker position target")
    parser.add_argument("--recovery-timeout", type=float, default=15,
                        help="Seconds allowed to reconcile a known maker order after a read or cancel failure")
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    if not Path(args.env_file).exists():
        raise FileNotFoundError(f"Env file not found: {args.env_file}")
    dotenv.load_dotenv(args.env_file)
    config = HedgeCycleConfig(
        exchange=args.exchange, ticker=args.ticker.upper(), size=args.size,
        iterations=args.iter, hedge_exchange=args.hedge_exchange,
        fill_timeout=args.fill_timeout, sleep_time=args.sleep,
        max_position=args.max_position, max_slippage=args.max_slippage,
        target_timeout=args.target_timeout, recovery_timeout=args.recovery_timeout,
    )
    await HedgeCycleStrategy(config).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Hedge mode stopped: {exc}", file=sys.stderr)
        sys.exit(1)
