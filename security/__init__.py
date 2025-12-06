"""
Security module for Perp-DEX-Tools
Provides encryption, rate limiting, and risk management
"""

# Import core modules that don't require cryptography
from .rate_limiter import RateLimiter, CircuitBreaker, get_manager
from .risk_manager import RiskManager, RiskConfig, RiskLevel

# Try to import cryptography-dependent modules
try:
    from .secure_key_manager import SecureKeyManager
    _has_crypto = True
except ImportError as e:
    SecureKeyManager = None
    _has_crypto = False
    import warnings
    warnings.warn(
        "SecureKeyManager unavailable: cryptography library not installed. "
        "Install with: pip install cryptography>=41.0.0"
    )

try:
    from .integration_helper import (
        SecurityIntegration,
        get_security,
        with_rate_limit,
        require_circuit_closed
    )
except ImportError:
    # Fallback if integration_helper fails
    SecurityIntegration = None
    get_security = None
    with_rate_limit = None
    require_circuit_closed = None

__all__ = [
    'SecureKeyManager',
    'RateLimiter',
    'CircuitBreaker',
    'RiskManager',
    'RiskConfig',
    'RiskLevel',
    'SecurityIntegration',
    'get_security',
    'get_manager',
    'with_rate_limit',
    'require_circuit_closed',
]
