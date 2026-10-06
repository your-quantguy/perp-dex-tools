"""Maker-first execution of two opposite positions."""

import asyncio
import time
from decimal import Decimal
from typing import Protocol

from exchanges.base import (
    BaseExchangeClient, ExchangeConfig, HedgeExecutionClient,
    LIVE_ORDER_STATUSES, OrderInfo, OrderOutcomeUnknown, OrderResult,
    TERMINAL_ORDER_STATUSES,
)
from helpers.logger import TradingLogger

LIVE = LIVE_ORDER_STATUSES
TERMINAL = TERMINAL_ORDER_STATUSES


class PairExecutionSettings(Protocol):
    @property
    def exchange(self) -> str: ...
    @property
    def hedge_exchange(self) -> str: ...
    @property
    def ticker(self) -> str: ...
    @property
    def size(self) -> Decimal: ...
    @property
    def fill_timeout(self) -> float: ...
    @property
    def target_timeout(self) -> float: ...
    @property
    def recovery_timeout(self) -> float: ...
    @property
    def max_slippage(self) -> Decimal: ...


class MakerFirstPairExecutor:
    """Reach a maker position target while keeping the opposite hedge leg aligned."""

    def __init__(
        self, config: PairExecutionSettings, maker: BaseExchangeClient,
        hedge: HedgeExecutionClient, maker_config: ExchangeConfig,
        hedge_config: ExchangeConfig, logger: TradingLogger,
    ) -> None:
        self.config = config
        self.maker = maker
        self.hedge = hedge
        self.maker_config = maker_config
        self.hedge_config = hedge_config
        self.logger = logger

    async def positions(self) -> Decimal:
        deadline = time.monotonic() + self.config.recovery_timeout
        maker: Decimal | None = None
        hedge: Decimal | None = None
        last_error: Exception | None = None
        while True:
            try:
                maker = await self.maker.get_account_positions()
                hedge = await self.hedge.get_account_positions()
            except Exception as exc:
                last_error = exc
            else:
                if maker + hedge == 0:
                    return maker
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Unmatched or unreadable positions: {self.config.exchange}={maker}, "
                    f"{self.config.hedge_exchange}={hedge}, read_error={last_error}"
                )
            await asyncio.sleep(0.25)

    async def _confirm_hedge_target(self, target: Decimal):
        deadline = time.monotonic() + self.config.recovery_timeout
        position: Decimal | None = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                position = await self.hedge.get_account_positions()
            except Exception as exc:
                last_error = exc
            else:
                if position == target:
                    return
            await asyncio.sleep(0.25)
        try:
            position = await self.hedge.get_account_positions()
        except Exception as exc:
            last_error = exc
        else:
            if position == target:
                return
        raise RuntimeError(
            f"{self.config.hedge_exchange} hedge not fully confirmed: "
            f"target={target}, actual={position}, read_error={last_error}"
        )

    async def hedge_to(self, target: Decimal):
        deadline = time.monotonic() + self.config.recovery_timeout
        while True:
            try:
                current = await self.hedge.get_account_positions()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.25)
        difference = target - current
        if difference == 0:
            return
        quantity = abs(difference)
        self.hedge.validate_hedge_quantity(quantity)
        side = "buy" if difference > 0 else "sell"
        reduce_only = current != 0 and current * difference < 0 and quantity <= abs(current)
        try:
            result = await self.hedge.place_hedge_market_order(
                quantity, side, self.config.max_slippage, reduce_only=reduce_only,
            )
        except OrderOutcomeUnknown as exc:
            # An ACK can be lost after execution. A confirmed target position
            # resolves that case without sending another market order.
            self.logger.log(f"{self.config.hedge_exchange} hedge acknowledgement unknown; "
                            f"checking target position: {exc}", "WARNING")
            await self._confirm_hedge_target(target)
            if await self.hedge.get_active_orders(self.hedge_config.contract_id):
                raise RuntimeError("Hedge acknowledgement unknown and an active order remains")
            return
        if not result.success:
            raise RuntimeError(f"{self.config.hedge_exchange} hedge rejected: {result.error_message}")
        await self._confirm_hedge_target(target)

    async def _confirmed_maker_fill(self, starting_position: Decimal, side: str,
                                    quantity: Decimal, filled: Decimal,
                                    previous: Decimal) -> Decimal:
        if filled < previous or filled > quantity:
            raise RuntimeError("Maker order cumulative fill is inconsistent")
        if filled > previous:
            direction = Decimal(1) if side == "buy" else Decimal(-1)
            await self.hedge_to(-(starting_position + direction * filled))
            self.logger.log(f"Hedged {filled - previous} {self.config.ticker} after maker fill", "INFO")
        return filled

    async def _read_maker_order(self, order_id: str) -> OrderInfo | None:
        # Arcus's ordinary getter can return a cached live snapshot. Recovery
        # needs its authoritative REST read before placing another order.
        recovery_read = getattr(self.maker, "get_order_for_recovery", None)
        if self.config.exchange == "arcus" and recovery_read is not None:
            return await recovery_read(order_id)
        return await self.maker.get_order_info(order_id)

    async def _observe_maker_order(self, order: OrderInfo, order_id: str, side: str,
                                   quantity: Decimal, starting_position: Decimal,
                                   filled: Decimal) -> Decimal:
        if order.order_id != order_id or order.side != side or order.status not in LIVE | TERMINAL:
            raise RuntimeError(f"Maker order {order_id} returned an inconsistent snapshot")
        return await self._confirmed_maker_fill(
            starting_position, side, quantity, Decimal(order.filled_size), filled,
        )

    async def _recover_maker_terminal(self, order_id: str, side: str, quantity: Decimal,
                                      starting_position: Decimal, filled: Decimal) -> Decimal:
        """Cancel a known order, hedge late fills, and wait for a confirmed terminal state."""
        deadline = time.monotonic() + self.config.recovery_timeout
        last_error: Exception | None = None
        read_after_cancel = False
        last_cancel_at = 0.0
        while time.monotonic() < deadline or read_after_cancel:
            read_after_cancel = False
            try:
                order = await self._read_maker_order(order_id)
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.25)
                continue
            if order is not None:
                filled = await self._observe_maker_order(
                    order, order_id, side, quantity, starting_position, filled,
                )
                if order.status in TERMINAL:
                    return filled
                if time.monotonic() - last_cancel_at >= 2:
                    last_cancel_at = time.monotonic()
                    read_after_cancel = True
                    try:
                        canceled = await self.maker.cancel_order(order_id)
                    except Exception as exc:
                        # A lost cancel ACK may still have canceled or filled
                        # the order; the next read resolves it when possible.
                        last_error = exc
                    else:
                        if canceled.order_id and canceled.order_id != order_id:
                            raise RuntimeError("Maker cancel returned a different order ID")
                        if canceled.filled_size is not None:
                            filled = await self._confirmed_maker_fill(
                                starting_position, side, quantity, Decimal(canceled.filled_size), filled,
                            )
                        if not canceled.success:
                            last_error = RuntimeError(f"Maker cancel not confirmed: {canceled.error_message}")
            await asyncio.sleep(0.25)
        raise RuntimeError(f"Maker order {order_id} terminal state is unconfirmed: {last_error}")

    async def execute_maker_order(self, side: str, quantity: Decimal,
                                   starting_position: Decimal, reduce_only: bool,
                                   price: Decimal | None = None) -> Decimal:
        self.hedge.validate_hedge_quantity(quantity)
        try:
            if price is not None:
                submit = getattr(self.maker, "place_post_only_order", None)
                if submit is None:
                    raise TypeError("Maker venue does not support an explicit post-only price")
                result = await submit(self.maker_config.contract_id, quantity, price, side, reduce_only)
            elif reduce_only:
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
        except OrderOutcomeUnknown as exc:
            if not exc.order_id:
                raise
            self.logger.log(f"Maker acknowledgement unknown; reconciling order {exc.order_id}", "WARNING")
            result = OrderResult(True, exc.order_id, side, quantity, status="OPEN")
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
                try:
                    order = await self._read_maker_order(order_id)
                except Exception as exc:
                    self.logger.log(f"Maker order {order_id} read failed; retrying: {exc}", "WARNING")
                    order = None
                if order is not None:
                    filled = await self._observe_maker_order(
                        order, order_id, side, quantity, starting_position, filled,
                    )
                    status = order.status
                if status not in TERMINAL:
                    await asyncio.sleep(0.25)
            filled = await self._recover_maker_terminal(
                order_id, side, quantity, starting_position, filled,
            )
            terminal = True
            return filled
        finally:
            if not terminal:
                try:
                    await self.maker.cancel_order(order_id)
                except Exception as exc:
                    self.logger.log(f"Maker order {order_id} still needs manual cancellation check: {exc}", "ERROR")

    async def reach_target(self, target: Decimal):
        deadline = time.monotonic() + self.config.target_timeout
        while time.monotonic() < deadline:
            position = await self.positions()
            if position == target:
                return
            try:
                active_orders = await self.maker.get_active_orders(self.maker_config.contract_id)
            except (ConnectionError, TimeoutError, OSError) as exc:
                self.logger.log(f"Maker active-order read failed; retrying: {exc}", "WARNING")
                await asyncio.sleep(1)
                continue
            if active_orders:
                raise RuntimeError("Unexpected active maker order; review it before continuing")
            difference = target - position
            side = "buy" if difference > 0 else "sell"
            reducing = position != 0 and position * difference < 0
            quantity = min(self.config.size, abs(difference))
            if reducing:
                quantity = min(quantity, abs(position))
            self.maker_config.direction = side
            filled = await self.execute_maker_order(side, quantity, position, reducing)
            if filled == 0:
                await asyncio.sleep(1)
            else:
                expected = position + (filled if side == "buy" else -filled)
                settle_deadline = time.monotonic() + 10
                actual: Decimal | None = None
                while time.monotonic() < settle_deadline:
                    try:
                        actual = await self.maker.get_account_positions()
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        self.logger.log(f"Maker position read failed; retrying: {exc}", "WARNING")
                        await asyncio.sleep(0.25)
                        continue
                    if actual == expected:
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise RuntimeError(f"Maker position did not settle: expected={expected}, actual={actual}")
        raise TimeoutError(f"Target maker position {target} was not reached in time")
