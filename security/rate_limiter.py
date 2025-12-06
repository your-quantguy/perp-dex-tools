"""
Rate Limiter and Circuit Breaker for Perp-DEX-Tools
Protects against API rate limits and trading anomalies
"""

import time
import logging
from collections import deque
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, blocking requests
    HALF_OPEN = "half_open"  # Testing if system recovered


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an exchange"""
    max_requests: int  # Maximum requests per window
    window_seconds: int  # Time window in seconds
    burst_size: int = None  # Max burst size (defaults to max_requests)
    adaptive: bool = True  # Enable adaptive rate limiting

    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.max_requests


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close circuit
    timeout_seconds: int = 60  # Time before attempting half-open
    max_loss_percentage: float = 5.0  # Max loss % before tripping
    monitoring_window: int = 300  # Monitor last 5 minutes


class RateLimiter:
    """
    Token bucket rate limiter with adaptive behavior

    Features:
    - Token bucket algorithm
    - Per-exchange limits
    - Burst handling
    - Adaptive rate adjustment
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = config.max_requests  # Start with full bucket
        self.last_update = time.time()
        self.lock = threading.Lock()

        # Adaptive rate limiting
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.total_requests = 0

        logger.info(
            f"RateLimiter initialized: {config.max_requests} req/{config.window_seconds}s"
        )

    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update

        # Calculate tokens to add
        tokens_to_add = (elapsed / self.config.window_seconds) * self.config.max_requests
        self.tokens = min(self.config.max_requests, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire tokens to make a request

        Args:
            tokens: Number of tokens to acquire
            blocking: Wait if tokens not available
            timeout: Maximum time to wait (seconds)

        Returns:
            True if tokens acquired, False otherwise
        """
        start_time = time.time()

        with self.lock:
            self._refill_tokens()

            # Check if tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.request_times.append(time.time())
                self.total_requests += 1
                return True

            # Non-blocking mode
            if not blocking:
                return False

        # Blocking mode - wait for tokens
        while True:
            with self.lock:
                self._refill_tokens()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.request_times.append(time.time())
                    self.total_requests += 1
                    return True

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning("Rate limit acquire timeout")
                return False

            # Sleep briefly
            time.sleep(0.1)

    def record_error(self):
        """Record an API error (for adaptive limiting)"""
        with self.lock:
            self.error_count += 1

            # Adaptive: reduce rate on errors
            if self.config.adaptive and self.error_count > 3:
                old_max = self.config.max_requests
                self.config.max_requests = max(1, int(self.config.max_requests * 0.8))
                logger.warning(
                    f"Adaptive rate limit: reduced from {old_max} to "
                    f"{self.config.max_requests} req/{self.config.window_seconds}s"
                )
                self.error_count = 0  # Reset counter

    def record_success(self):
        """Record a successful request (for adaptive limiting)"""
        # Adaptive: gradually increase rate on success
        if self.config.adaptive and self.total_requests > 100:
            # Increase by 1% every 100 successful requests
            if self.total_requests % 100 == 0:
                old_max = self.config.max_requests
                self.config.max_requests = min(
                    self.config.burst_size,
                    int(self.config.max_requests * 1.01)
                )
                if self.config.max_requests != old_max:
                    logger.info(
                        f"Adaptive rate limit: increased to {self.config.max_requests} "
                        f"req/{self.config.window_seconds}s"
                    )

    def get_current_rate(self) -> float:
        """
        Get current request rate (requests per second)

        Returns:
            Current rate
        """
        if not self.request_times:
            return 0.0

        # Calculate rate over last minute
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t <= 60]

        if not recent_requests:
            return 0.0

        time_span = now - recent_requests[0]
        if time_span == 0:
            return 0.0

        return len(recent_requests) / time_span

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics

        Returns:
            Statistics dictionary
        """
        with self.lock:
            return {
                'max_requests': self.config.max_requests,
                'window_seconds': self.config.window_seconds,
                'tokens_available': self.tokens,
                'total_requests': self.total_requests,
                'error_count': self.error_count,
                'current_rate': round(self.get_current_rate(), 2),
            }


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures and excessive losses

    Features:
    - Automatic failure detection
    - Loss-based tripping
    - Timeout and recovery
    - Half-open testing
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name (e.g., exchange name)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()

        # Trading metrics
        self.trades = deque(maxlen=100)  # Last 100 trades
        self.total_pnl = 0.0

        self.lock = threading.Lock()

        logger.info(f"CircuitBreaker '{name}' initialized in CLOSED state")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.state != CircuitState.OPEN:
            return False

        if not self.last_failure_time:
            return False

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout_seconds

    def _check_loss_threshold(self) -> bool:
        """
        Check if losses exceed threshold

        Returns:
            True if losses exceed threshold
        """
        if len(self.trades) < 10:  # Need minimum trades
            return False

        # Calculate recent PnL
        recent_trades = [t for t in self.trades if t['timestamp'] > time.time() - self.config.monitoring_window]

        if not recent_trades:
            return False

        total_pnl = sum(t['pnl'] for t in recent_trades)
        total_volume = sum(abs(t['amount']) for t in recent_trades)

        if total_volume == 0:
            return False

        loss_percentage = (total_pnl / total_volume) * 100

        return loss_percentage < -self.config.max_loss_percentage

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open
        """
        with self.lock:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                logger.info(f"CircuitBreaker '{self.name}' attempting reset to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0

            # Block if circuit is open
            if self.state == CircuitState.OPEN:
                raise Exception(
                    f"CircuitBreaker '{self.name}' is OPEN. "
                    f"Will retry after {self.config.timeout_seconds}s timeout."
                )

        # Execute function
        try:
            result = func(*args, **kwargs)

            # Record success
            with self.lock:
                self.success_count += 1
                self.failure_count = 0  # Reset failures on success

                # Close circuit if enough successes
                if self.state == CircuitState.HALF_OPEN:
                    if self.success_count >= self.config.success_threshold:
                        logger.info(f"CircuitBreaker '{self.name}' closing (recovery successful)")
                        self.state = CircuitState.CLOSED
                        self.last_state_change = time.time()

            return result

        except Exception as e:
            # Record failure
            with self.lock:
                self.failure_count += 1
                self.success_count = 0  # Reset successes on failure
                self.last_failure_time = time.time()

                # Open circuit if threshold exceeded
                if self.failure_count >= self.config.failure_threshold:
                    if self.state != CircuitState.OPEN:
                        logger.error(
                            f"CircuitBreaker '{self.name}' opening due to {self.failure_count} failures"
                        )
                        self.state = CircuitState.OPEN
                        self.last_state_change = time.time()

                # Immediately open if in half-open
                if self.state == CircuitState.HALF_OPEN:
                    logger.error(f"CircuitBreaker '{self.name}' reopening (recovery failed)")
                    self.state = CircuitState.OPEN
                    self.last_state_change = time.time()

            raise

    def record_trade(self, amount: float, pnl: float):
        """
        Record a trade for loss monitoring

        Args:
            amount: Trade amount (USD)
            pnl: Profit/Loss (USD)
        """
        with self.lock:
            self.trades.append({
                'timestamp': time.time(),
                'amount': amount,
                'pnl': pnl,
            })
            self.total_pnl += pnl

            # Check loss threshold
            if self._check_loss_threshold():
                logger.error(
                    f"CircuitBreaker '{self.name}' opening due to excessive losses "
                    f"({self.config.max_loss_percentage}% threshold)"
                )
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.last_state_change = time.time()

    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            logger.info(f"CircuitBreaker '{self.name}' manually reset to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_state_change = time.time()

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self.state == CircuitState.CLOSED

    def get_stats(self) -> Dict[str, Any]:
        """
        Get circuit breaker statistics

        Returns:
            Statistics dictionary
        """
        with self.lock:
            recent_trades = [t for t in self.trades if t['timestamp'] > time.time() - 300]
            recent_pnl = sum(t['pnl'] for t in recent_trades) if recent_trades else 0

            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_trades': len(self.trades),
                'total_pnl': round(self.total_pnl, 2),
                'recent_pnl_5min': round(recent_pnl, 2),
                'uptime_seconds': round(time.time() - self.last_state_change, 2),
            }


class ExchangeRateLimitManager:
    """
    Manages rate limiters and circuit breakers for multiple exchanges

    Features:
    - Per-exchange rate limiting
    - Per-exchange circuit breakers
    - Centralized management
    """

    # Default rate limits for supported exchanges
    DEFAULT_LIMITS = {
        'edgex': RateLimitConfig(max_requests=10, window_seconds=1),
        'backpack': RateLimitConfig(max_requests=20, window_seconds=1),
        'paradex': RateLimitConfig(max_requests=10, window_seconds=1),
        'aster': RateLimitConfig(max_requests=15, window_seconds=1),
        'lighter': RateLimitConfig(max_requests=20, window_seconds=1),
        'grvt': RateLimitConfig(max_requests=10, window_seconds=1),
        'extended': RateLimitConfig(max_requests=10, window_seconds=1),
        'apex': RateLimitConfig(max_requests=10, window_seconds=1),
    }

    def __init__(self):
        """Initialize exchange rate limit manager"""
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()

        logger.info("ExchangeRateLimitManager initialized")

    def get_rate_limiter(self, exchange: str) -> RateLimiter:
        """
        Get or create rate limiter for exchange

        Args:
            exchange: Exchange name

        Returns:
            RateLimiter instance
        """
        with self.lock:
            if exchange not in self.rate_limiters:
                config = self.DEFAULT_LIMITS.get(
                    exchange.lower(),
                    RateLimitConfig(max_requests=10, window_seconds=1)
                )
                self.rate_limiters[exchange] = RateLimiter(config)
                logger.info(f"Created rate limiter for {exchange}")

            return self.rate_limiters[exchange]

    def get_circuit_breaker(self, exchange: str) -> CircuitBreaker:
        """
        Get or create circuit breaker for exchange

        Args:
            exchange: Exchange name

        Returns:
            CircuitBreaker instance
        """
        with self.lock:
            if exchange not in self.circuit_breakers:
                self.circuit_breakers[exchange] = CircuitBreaker(
                    name=exchange,
                    config=CircuitBreakerConfig()
                )
                logger.info(f"Created circuit breaker for {exchange}")

            return self.circuit_breakers[exchange]

    def execute_with_protection(
        self,
        exchange: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with rate limiting and circuit breaker protection

        Args:
            exchange: Exchange name
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        # Acquire rate limit token
        rate_limiter = self.get_rate_limiter(exchange)
        if not rate_limiter.acquire(timeout=30):
            raise Exception(f"Rate limit timeout for {exchange}")

        # Execute with circuit breaker
        circuit_breaker = self.get_circuit_breaker(exchange)
        try:
            result = circuit_breaker.call(func, *args, **kwargs)
            rate_limiter.record_success()
            return result
        except Exception as e:
            rate_limiter.record_error()
            raise

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all exchanges

        Returns:
            Dictionary of statistics per exchange
        """
        stats = {}

        for exchange in set(list(self.rate_limiters.keys()) + list(self.circuit_breakers.keys())):
            stats[exchange] = {
                'rate_limiter': self.rate_limiters[exchange].get_stats() if exchange in self.rate_limiters else None,
                'circuit_breaker': self.circuit_breakers[exchange].get_stats() if exchange in self.circuit_breakers else None,
            }

        return stats


# Global instance
_manager = None


def get_manager() -> ExchangeRateLimitManager:
    """Get global ExchangeRateLimitManager instance"""
    global _manager
    if _manager is None:
        _manager = ExchangeRateLimitManager()
    return _manager
