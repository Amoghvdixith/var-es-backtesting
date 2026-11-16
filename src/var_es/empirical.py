"""Common empirical VaR / ES engine (equal-weight empirical loss distribution).

Convention (losses ``x``, sorted ascending ``x_(1) <= ... <= x_(n)``, confidence ``c``,
tail probability ``p = 1 - c``):

* **VaR_c** = inverse empirical CDF at ``c`` = ``inf{x : F_n(x) >= c}`` = ``x_(k)``
  with ``k = ceil(n c)``. A small relative guard is applied to ``n c`` so that, e.g.,
  ``n = 100, c = 0.95`` gives ``k = 95`` despite floating-point representation.
* **ES_c** = average of the worst ``p`` probability mass of the empirical
  distribution: ``m = n p`` observations' worth of mass is taken from the top. With
  ``j = floor(m)``, ``ES = [sum of the j largest losses + (m - j) * x_(n-j)] / m``.
  The boundary observation receives fractional weight ``m - j``. This equals the
  Acerbi-Tasche / Rockafellar-Uryasev definition applied to the empirical measure
  and is exact under ties. ``ES_c >= VaR_c`` always holds.

Exceptions elsewhere in the package are strict: ``realized_loss > VaR``.
"""

from __future__ import annotations

import math

import numpy as np

from .validation import DataValidationError, validate_confidence

_GUARD = 1e-9


def _prepare(losses: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(losses, dtype=float).ravel()
    if x.size == 0:
        raise DataValidationError("empirical VaR/ES requires at least one observation")
    if not np.all(np.isfinite(x)):
        raise DataValidationError("losses must be finite")
    return np.sort(x)


def _var_index(n: int, c: float) -> int:
    """1-based order-statistic index ``k = ceil(n c)`` with a floating-point guard."""
    k = math.ceil(n * c - _GUARD * max(1.0, n * c))
    return min(max(k, 1), n)


def empirical_var(losses, confidence: float) -> float:
    """Empirical VaR: the ``ceil(n c)``-th smallest loss."""
    c = validate_confidence(confidence)
    x = _prepare(losses)
    return float(x[_var_index(x.size, c) - 1])


def empirical_es(losses, confidence: float) -> float:
    """Empirical ES: mean of the worst ``1 - c`` probability mass with fractional weighting."""
    c = validate_confidence(confidence)
    x = _prepare(losses)
    return _es_sorted(x, c)


def _es_sorted(x: np.ndarray, c: float) -> float:
    n = x.size
    m = n * (1.0 - c)
    # guard against representation error, e.g. 100 * 0.05 = 5.000000000000004
    m_round = round(m)
    if abs(m - m_round) < _GUARD * max(1.0, m):
        m = float(m_round)
    j = int(math.floor(m))
    total = x[n - j :].sum() if j > 0 else 0.0
    frac = m - j
    if frac > 0:
        total += frac * x[n - j - 1]
    return float(total / m)


def empirical_var_es(losses, confidence_levels) -> dict[float, tuple[float, float]]:
    """VaR and ES for several confidence levels from one sorted sample.

    Returns ``{c: (VaR_c, ES_c)}``.
    """
    x = _prepare(losses)
    out: dict[float, tuple[float, float]] = {}
    for c in confidence_levels:
        c = validate_confidence(c)
        var = float(x[_var_index(x.size, c) - 1])
        es = _es_sorted(x, c)
        out[c] = (var, es)
    return out
