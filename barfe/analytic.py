"""Closed-form displacement of the tapered bar.

Integrating u'(x) = N(x) / (E A(x)) with N(x) = P + q (L - x) and
A(x) = A0 (1 - c x), c = taper / L, gives

    u(x) = [ (P + qL) * g1(x) - q * g2(x) ] / (E A0)

    g1(x) = -ln(1 - c x) / c                 -> x        as c -> 0
    g2(x) = (-ln(1 - c x) - c x) / c^2       -> x^2 / 2  as c -> 0

For a prismatic bar (c = 0) this reduces to the textbook
u(x) = [P x + q (L x - x^2 / 2)] / (E A0).
"""

import numpy as np

from .problem import Bar

# Below this value of c*L the log form loses precision to cancellation,
# so the Taylor series is used instead.
_SERIES_CUTOFF = 1e-3
_SERIES_TERMS = 8


def _g1_g2(x, c):
    x = np.asarray(x, dtype=float)
    if c * x.max(initial=0.0) < _SERIES_CUTOFF:
        # -ln(1-t) = sum t^k / k  =>  g1 = sum c^(k-1) x^k / k,  g2 = sum_{k>=2} c^(k-2) x^k / k
        g1 = sum(c ** (k - 1) * x ** k / k for k in range(1, _SERIES_TERMS + 1))
        g2 = sum(c ** (k - 2) * x ** k / k for k in range(2, _SERIES_TERMS + 2))
        return g1, g2
    t = c * x
    neg_log = -np.log1p(-t)
    return neg_log / c, (neg_log - t) / c ** 2


def displacement(bar: Bar, x):
    """Exact axial displacement u(x) [m]."""
    c = bar.taper / bar.L
    g1, g2 = _g1_g2(x, c)
    return ((bar.P + bar.q * bar.L) * g1 - bar.q * g2) / (bar.E * bar.A0)


def tip_displacement(bar: Bar):
    return float(displacement(bar, np.array([bar.L]))[0])
