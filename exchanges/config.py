"""Market configuration shared by exchange sessions."""

from dataclasses import dataclass
from decimal import Decimal

from exchanges.base import MarketId


@dataclass
class VenueConfig:
    ticker: str
    contract_id: MarketId
    tick_size: Decimal
    quantity: Decimal
    direction: str

    @property
    def close_order_side(self) -> str:
        return "buy" if self.direction == "sell" else "sell"
