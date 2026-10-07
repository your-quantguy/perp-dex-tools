"""Pure signal, order-book and account-risk calculations for basis trading."""

import time
from math import lcm
from decimal import Decimal, ROUND_DOWN

from exchanges.market import BookSnapshot

BPS = Decimal(10_000)


def basis_bps(a: BookSnapshot, b: BookSnapshot, max_age: float, max_skew: float) -> Decimal:
    now = time.monotonic()
    if (now - a.observed_at > max_age or now - b.observed_at > max_age
            or abs(a.observed_at - b.observed_at) > max_skew):
        raise ConnectionError("Venue books are stale or not synchronized")
    return (a.mid / b.mid - 1) * BPS


def opening_signal(basis: Decimal, center: Decimal, entry_band: Decimal) -> int:
    """+1 is long A/short B; -1 is short A/long B."""
    if basis <= center - entry_band:
        return 1
    if basis >= center + entry_band:
        return -1
    return 0


def closing_signal(basis: Decimal, center: Decimal, exit_band: Decimal, a_direction: int) -> bool:
    return basis >= center - exit_band if a_direction > 0 else basis <= center + exit_band


def opposite_band_exit(basis: Decimal, center: Decimal, entry_band: Decimal, a_direction: int) -> bool:
    return basis >= center + entry_band if a_direction > 0 else basis <= center - entry_band


def rounded_size(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("Invalid quantity step")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def common_size_step(a: Decimal, b: Decimal) -> Decimal:
    """Smallest size representable exactly by both venue quantity grids."""
    if a <= 0 or b <= 0:
        raise ValueError("Quantity steps must be positive")
    exponent_a = a.as_tuple().exponent
    exponent_b = b.as_tuple().exponent
    if not isinstance(exponent_a, int) or not isinstance(exponent_b, int):
        raise ValueError("Quantity steps must be finite")
    scale = 10 ** max(-exponent_a, -exponent_b, 0)
    return Decimal(lcm(int(a * scale), int(b * scale))) / scale
