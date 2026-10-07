"""Composable exchange capabilities consumed by trading strategies.

Runtime checks verify method presence. Implementations must also honor the
documented order semantics; adapter tests verify those guarantees.
"""

from decimal import Decimal
from typing import Protocol, runtime_checkable

from exchanges.base import MarketId, OrderInfo, OrderResult
from exchanges.market import AccountRisk, BookSnapshot


@runtime_checkable
class VenueSession(Protocol):
    def setup_order_update_handler(self, handler) -> None: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def get_contract_attributes(self) -> tuple[MarketId, Decimal]: ...
    async def fetch_bbo_prices(self, contract_id: MarketId) -> tuple[Decimal, Decimal]: ...


@runtime_checkable
class NetworkIdentity(Protocol):
    @property
    def network_id(self) -> str: ...


@runtime_checkable
class AccountDescription(Protocol):
    def get_account_description(self) -> tuple[str, ...]: ...


@runtime_checkable
class PositionReader(Protocol):
    async def get_account_positions(self) -> Decimal: ...


@runtime_checkable
class ActiveOrderReader(Protocol):
    async def get_active_orders(self, contract_id: MarketId) -> list[OrderInfo]: ...


@runtime_checkable
class OrderTracker(Protocol):
    async def get_order_info(self, order_id: str) -> OrderInfo | None: ...
    async def cancel_order(self, order_id: str) -> OrderResult: ...


@runtime_checkable
class AuthoritativeOrderRecovery(Protocol):
    """Read an order from the authoritative venue source, bypassing live caches."""

    async def get_order_for_recovery(self, order_id: str) -> OrderInfo: ...


@runtime_checkable
class TakeProfitOrders(Protocol):
    async def place_open_order(self, contract_id: MarketId, quantity: Decimal, direction: str) -> OrderResult: ...
    async def place_close_order(self, contract_id: MarketId, quantity: Decimal,
                                price: Decimal, side: str) -> OrderResult: ...
    async def get_order_price(self, direction: str) -> Decimal: ...


@runtime_checkable
class PostOnlyMaker(Protocol):
    """An explicit limit price must either rest without taking or be rejected."""

    async def place_post_only_order(self, contract_id: MarketId, quantity: Decimal,
                                    price: Decimal, side: str, reduce_only: bool) -> OrderResult: ...
    async def get_order_price(self, direction: str) -> Decimal: ...


@runtime_checkable
class BoundedTaker(Protocol):
    """Immediate order with a price bound; ambiguous outcomes must raise."""

    def validate_hedge_quantity(self, quantity: Decimal) -> None: ...
    async def place_hedge_market_order(self, quantity: Decimal, side: str,
                                       max_slippage_percent: Decimal,
                                       reduce_only: bool = False) -> OrderResult: ...


@runtime_checkable
class QuantityGrid(Protocol):
    def get_quantity_step(self) -> Decimal: ...


@runtime_checkable
class OrderBookFeed(Protocol):
    async def get_spread_book(self, contract_id: MarketId) -> BookSnapshot: ...


@runtime_checkable
class MarginRiskFeed(Protocol):
    async def get_spread_risk(self) -> AccountRisk: ...


@runtime_checkable
class EmergencyReduce(Protocol):
    async def place_reduce_only_ioc(self, contract_id: MarketId, quantity: Decimal,
                                     side: str, max_slippage_percent: Decimal) -> OrderResult: ...


@runtime_checkable
class PairMaker(VenueSession, NetworkIdentity, PositionReader, ActiveOrderReader, OrderTracker,
                TakeProfitOrders, PostOnlyMaker, Protocol):
    """Maker capabilities shared by hedge cycling and convergence."""


@runtime_checkable
class PairTaker(VenueSession, NetworkIdentity, PositionReader, ActiveOrderReader,
                BoundedTaker, Protocol):
    """Taker capabilities shared by hedge cycling and convergence."""


@runtime_checkable
class TakeProfitVenue(VenueSession, PositionReader, ActiveOrderReader,
                      OrderTracker, TakeProfitOrders, Protocol):
    """Capabilities required by the single-venue take-profit strategy."""


@runtime_checkable
class SpreadMaker(PairMaker, QuantityGrid, OrderBookFeed,
                  MarginRiskFeed, EmergencyReduce, Protocol):
    """All capabilities for the first leg of maker/taker convergence."""


@runtime_checkable
class SpreadTaker(PairTaker, QuantityGrid, OrderBookFeed,
                  MarginRiskFeed, Protocol):
    """All capabilities for the immediate leg of maker/taker convergence."""
