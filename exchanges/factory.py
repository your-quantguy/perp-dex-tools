"""
Exchange factory for creating exchange clients dynamically.
"""

from typing import Dict, Any
from .base import BaseExchangeClient
from .edgex import EdgeXClient
from .backpack import BackpackClient
from .paradex import ParadexClient


class ExchangeFactory:
    """Factory class for creating exchange clients."""

    _registered_exchanges = {
        'edgex': EdgeXClient,
        'backpack': BackpackClient,
        'paradex': ParadexClient,
    }

    @classmethod
    def create_exchange(cls, exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient:
        """Create an exchange client instance.

        Args:
            exchange_name: Name of the exchange (e.g., 'edgex')
            config: Configuration dictionary for the exchange

        Returns:
            Exchange client instance

        Raises:
            ValueError: If the exchange is not supported
        """
        exchange_name = exchange_name.lower()

        if exchange_name not in cls._registered_exchanges:
            available_exchanges = ', '.join(cls._registered_exchanges.keys())
            raise ValueError(f"Unsupported exchange: {exchange_name}. Available exchanges: {available_exchanges}")

        exchange_class = cls._registered_exchanges[exchange_name]
        return exchange_class(config)

    @classmethod
    def get_supported_exchanges(cls) -> list:
        """Get list of supported exchanges.

        Returns:
            List of supported exchange names
        """
        return list(cls._registered_exchanges.keys())

    @classmethod
    def register_exchange(cls, name: str, exchange_class: type) -> None:
        """Register a new exchange client.

        Args:
            name: Exchange name
            exchange_class: Exchange client class that inherits from BaseExchangeClient
        """
        if not issubclass(exchange_class, BaseExchangeClient):
            raise ValueError("Exchange class must inherit from BaseExchangeClient")

        cls._registered_exchanges[name.lower()] = exchange_class
