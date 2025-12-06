"""
Integration Helper for Perp-DEX-Tools
Simplifies adding security features to existing code
"""

import logging
from typing import Callable, Any, Optional, Tuple, Dict
from functools import wraps

# Conditional import for SecureKeyManager (requires cryptography)
try:
    from .secure_key_manager import SecureKeyManager
    _HAS_SECURE_KEY_MANAGER = True
except ImportError:
    SecureKeyManager = None
    _HAS_SECURE_KEY_MANAGER = False

from .rate_limiter import get_manager as get_rate_limit_manager, CircuitState
from .risk_manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)


class SecurityIntegration:
    """
    Helper class to easily integrate security features into existing code

    Usage:
        security = SecurityIntegration(use_encrypted_keys=True)

        # Wrap exchange calls
        @security.with_rate_limit('edgex')
        def get_orderbook(symbol):
            return exchange.get_orderbook(symbol)

        # Check circuit breakers
        if security.can_trade('edgex'):
            place_order()
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        use_encrypted_keys: bool = False,
        initial_capital: float = 10000,
        risk_config: Optional[RiskConfig] = None
    ):
        """
        Initialize security integration

        Args:
            use_encrypted_keys: Use encrypted key storage
            initial_capital: Starting capital for risk manager
            risk_config: Risk management configuration
        """
        if hasattr(self, '_initialized'):
            return

        self.use_encrypted_keys = use_encrypted_keys
        self.key_manager = None
        self.rate_limit_manager = get_rate_limit_manager()
        self.risk_manager = RiskManager(
            config=risk_config or RiskConfig(),
            initial_capital=initial_capital
        )

        if use_encrypted_keys:
            if not _HAS_SECURE_KEY_MANAGER:
                logger.warning("SecureKeyManager not available (cryptography library not installed)")
                logger.warning("Falling back to environment variables")
            else:
                self.key_manager = SecureKeyManager()
                try:
                    self.key_manager.load_keys()
                    logger.info("✅ Loaded encrypted API keys")
                except Exception as e:
                    logger.warning(f"Failed to load encrypted keys: {e}")

        self._initialized = True

    def get_api_key(self, key_name: str, default: str = None) -> Optional[str]:
        """
        Get API key (from encrypted storage or env)

        Args:
            key_name: Name of the API key
            default: Default value if not found

        Returns:
            API key value
        """
        if self.use_encrypted_keys and self.key_manager:
            return self.key_manager.get_key(key_name, default)

        # Fallback to environment variables
        import os
        return os.getenv(key_name, default)

    def with_rate_limit(self, exchange: str):
        """
        Decorator to add rate limiting to a function

        Usage:
            @security.with_rate_limit('edgex')
            def fetch_data():
                return exchange.get_data()
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.rate_limit_manager.execute_with_protection(
                    exchange,
                    func,
                    *args,
                    **kwargs
                )
            return wrapper
        return decorator

    def can_trade(self, exchange: str) -> bool:
        """
        Check if trading is allowed on exchange (circuit breaker check)

        Args:
            exchange: Exchange name

        Returns:
            True if circuit is closed (safe to trade)
        """
        circuit = self.rate_limit_manager.get_circuit_breaker(exchange)
        is_closed = circuit.is_closed()

        if not is_closed:
            logger.warning(f"⚠️ Circuit breaker OPEN for {exchange}")

        return is_closed

    def record_trade(self, exchange: str, amount: float, pnl: float):
        """
        Record a trade for circuit breaker monitoring

        Args:
            exchange: Exchange name
            amount: Trade amount (USD)
            pnl: Profit/Loss (USD)
        """
        circuit = self.rate_limit_manager.get_circuit_breaker(exchange)
        circuit.record_trade(amount, pnl)

    def get_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning percentage
            avg_loss: Average losing percentage

        Returns:
            Recommended position size in USD
        """
        return self.risk_manager.calculate_position_size(
            win_rate, avg_win, avg_loss
        )

    def check_risk_limits(
        self,
        exchange: str,
        symbol: str,
        side: str,
        entry_price: float,
        size_usd: float
    ) -> Tuple[bool, str]:
        """
        Check if a trade passes risk limits

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            size_usd: Position size in USD

        Returns:
            (allowed, reason)
        """
        # Check daily loss limit
        metrics = self.risk_manager.get_portfolio_metrics()

        if metrics['daily_pnl'] < 0:
            daily_loss_pct = abs(metrics['daily_pnl']) / self.risk_manager.current_capital * 100
            max_daily = self.risk_manager.config.max_daily_loss_pct

            if daily_loss_pct >= max_daily:
                return False, f"Daily loss limit reached: {daily_loss_pct:.1f}% >= {max_daily}%"

        # Check drawdown limit
        if metrics['drawdown_pct'] >= self.risk_manager.config.max_drawdown_pct:
            return False, f"Max drawdown reached: {metrics['drawdown_pct']:.1f}%"

        # Check position size limit
        if size_usd > self.risk_manager.config.max_position_size_usd:
            return False, f"Position size too large: ${size_usd:.2f}"

        # Check total exposure
        total_exposure = metrics['total_exposure'] + size_usd
        if total_exposure > self.risk_manager.config.max_total_exposure_usd:
            return False, f"Total exposure limit exceeded: ${total_exposure:.2f}"

        return True, "OK"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with rate limiter, circuit breaker, and risk stats
        """
        return {
            'rate_limiters': self.rate_limit_manager.get_all_stats(),
            'portfolio': self.risk_manager.get_portfolio_metrics(),
        }

    def print_status(self):
        """Print current security status"""
        print("\n" + "=" * 60)
        print("🔒 SECURITY STATUS")
        print("=" * 60)

        # Portfolio metrics
        metrics = self.risk_manager.get_portfolio_metrics()
        print(f"\n💼 Portfolio:")
        print(f"  Capital: ${metrics['current_capital']:,.2f}")
        print(f"  Total PnL: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)")
        print(f"  Daily PnL: ${metrics['daily_pnl']:,.2f}")
        print(f"  Drawdown: {metrics['drawdown_pct']:.2f}%")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")

        # Circuit breakers
        print(f"\n🔌 Circuit Breakers:")
        stats = self.rate_limit_manager.get_all_stats()
        for exchange, data in stats.items():
            if data['circuit_breaker']:
                cb = data['circuit_breaker']
                state_emoji = "✅" if cb['state'] == 'closed' else "❌"
                print(f"  {state_emoji} {exchange}: {cb['state'].upper()} | PnL: ${cb['total_pnl']:.2f}")

        # Rate limiters
        print(f"\n⏱️ Rate Limiters:")
        for exchange, data in stats.items():
            if data['rate_limiter']:
                rl = data['rate_limiter']
                print(f"  {exchange}: {rl['current_rate']:.1f} req/s (max {rl['max_requests']}/{rl['window_seconds']}s)")

        print("\n" + "=" * 60)


# Global instance
_security = None


def get_security(
    use_encrypted_keys: bool = False,
    initial_capital: float = 10000
) -> SecurityIntegration:
    """
    Get global SecurityIntegration instance

    Args:
        use_encrypted_keys: Use encrypted key storage
        initial_capital: Starting capital

    Returns:
        SecurityIntegration instance
    """
    global _security
    if _security is None:
        _security = SecurityIntegration(use_encrypted_keys, initial_capital)
    return _security


# Convenience decorators
def with_rate_limit(exchange: str):
    """Convenience decorator for rate limiting"""
    return get_security().with_rate_limit(exchange)


def require_circuit_closed(exchange: str):
    """
    Decorator to check circuit breaker before executing

    Usage:
        @require_circuit_closed('edgex')
        def place_order():
            return exchange.place_order(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_security()
            if not security.can_trade(exchange):
                raise Exception(f"Circuit breaker OPEN for {exchange}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage in existing code
def example_integration():
    """
    Example showing how to integrate security into existing code
    """
    # Initialize security
    security = get_security(use_encrypted_keys=True, initial_capital=10000)

    # Get API keys
    api_key = security.get_api_key('EDGEX_ACCOUNT_ID')

    # Check if can trade
    if security.can_trade('edgex'):
        print("Safe to trade on edgex")

    # Calculate position size
    position_size = security.get_position_size(
        win_rate=0.6,
        avg_win=2.0,
        avg_loss=1.0
    )
    print(f"Recommended position size: ${position_size:.2f}")

    # Check risk limits before trading
    allowed, reason = security.check_risk_limits(
        exchange='edgex',
        symbol='BTC-USD',
        side='long',
        entry_price=50000,
        size_usd=position_size
    )

    if allowed:
        print("Trade passes risk checks")
    else:
        print(f"Trade rejected: {reason}")

    # Record trade
    security.record_trade('edgex', amount=1000, pnl=25.50)

    # Print status
    security.print_status()


if __name__ == "__main__":
    example_integration()
