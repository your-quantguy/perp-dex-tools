"""
Exchange factory for creating exchange clients dynamically.
"""

from typing import Dict, Any, Type
from .base import BaseExchangeClient


class ExchangeFactory:
    """Factory class for creating exchange clients."""

    _registered_exchanges = {
        'edgex': 'exchanges.edgex.EdgeXClient',
        'backpack': 'exchanges.backpack.BackpackClient',
        'paradex': 'exchanges.paradex.ParadexClient',
        'aster': 'exchanges.aster.AsterClient',
        'lighter': 'exchanges.lighter.LighterClient',
        'grvt': 'exchanges.grvt.GrvtClient',
        'extended': 'exchanges.extended.ExtendedClient',
        'apex': 'exchanges.apex.ApexClient',
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

        # Dynamically import the exchange class only when needed
        exchange_class_path = cls._registered_exchanges[exchange_name]
        exchange_class = cls._import_exchange_class(exchange_class_path)
        return exchange_class(config)

    @classmethod
    def _import_exchange_class(cls, class_path: str) -> Type[BaseExchangeClient]:
        """Dynamically import an exchange class.

        Args:
            class_path: Full module path to the exchange class (e.g., 'exchanges.edgex.EdgeXClient')

        Returns:
            The exchange class

        Raises:
            ImportError: If the class cannot be imported
            ValueError: If the class does not inherit from BaseExchangeClient
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            exchange_class = getattr(module, class_name)
            
            if not issubclass(exchange_class, BaseExchangeClient):
                raise ValueError(f"Exchange class {class_name} must inherit from BaseExchangeClient")
            
            return exchange_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import exchange class {class_path}: {e}")

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

        # Convert class to module path for lazy loading
        module_name = exchange_class.__module__
        class_name = exchange_class.__name__
        class_path = f"{module_name}.{class_name}"
        
        cls._registered_exchanges[name.lower()] = class_path
