"""Small statistics helpers.

Avoids a SciPy dependency for one formula. Wilson is the right CI for binomial
proportions at small N — Normal (Wald) breaks at edges (`p=0` or `p=1`) and
under-covers when N is tiny, both of which describe our typical compare data.
"""
from __future__ import annotations

import math

# 95% z-score. We deliberately don't expose this as a knob — confidence-interval
# level is a project-wide policy decision, not a per-call tuning parameter.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Return (lower, upper) of the Wilson score interval for a binomial proportion.

    With n=0 returns (0.0, 1.0) — total uncertainty, no observations.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def intervals_disjoint(
    a: tuple[float, float], b: tuple[float, float], slack: float = 0.0
) -> bool:
    """True iff the two intervals don't overlap (allowing optional slack)."""
    a_lo, a_hi = a
    b_lo, b_hi = b
    return a_hi + slack < b_lo or b_hi + slack < a_lo
