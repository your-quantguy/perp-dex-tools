"""Common lifecycle for trading strategies."""

import asyncio
import logging
from typing import Protocol


class VenueLifecycle(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...


class Strategy(Protocol):
    """A strategy owns decisions; the runner owns its venue lifecycle."""

    @property
    def venues(self) -> tuple[VenueLifecycle, ...]: ...

    async def prepare(self) -> None: ...
    async def execute(self) -> None: ...
    async def on_failure(self, exc: BaseException) -> None: ...


class StrategyRunner:
    async def run(self, strategy: Strategy) -> None:
        venues = strategy.venues
        try:
            await strategy.prepare()
            for venue in venues:
                await venue.connect()
            await strategy.execute()
        except BaseException as exc:
            try:
                await strategy.on_failure(exc)
            except BaseException:
                # Preserve the trading failure, which may require manual review.
                logging.getLogger(__name__).exception("Strategy failure handler failed")
            raise
        finally:
            results = await asyncio.gather(
                *(venue.disconnect() for venue in venues), return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logging.getLogger(__name__).error("Venue disconnect failed: %s", result)
