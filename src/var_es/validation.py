"""Input validation helpers.

All checks raise :class:`DataValidationError` with an explicit message rather than
silently repairing data. Nothing in this module reweights, fills, or clips values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

WEIGHT_SUM_TOL = 1e-9


class DataValidationError(ValueError):
    """Raised when input data or configuration violates a documented requirement."""


def validate_confidence(confidence: float) -> float:
    """Return ``confidence`` as float after checking ``0 < confidence < 1``."""
    c = float(confidence)
    if not np.isfinite(c) or not 0.0 < c < 1.0:
        raise DataValidationError(f"confidence must lie strictly in (0, 1); got {confidence!r}")
    return c


def validate_weights(
    weights: Mapping[str, float],
    assets: Sequence[str],
    *,
    long_only: bool = True,
    tol: float = WEIGHT_SUM_TOL,
) -> np.ndarray:
    """Validate portfolio weights and return them aligned to ``assets``.

    Checks: no duplicate asset names, identical name sets (no missing or extra
    assets), finite values, sum to one within ``tol``, and non-negativity when
    ``long_only`` is True. Weights are never renormalised.
    """
    assets = list(assets)
    if len(set(assets)) != len(assets):
        raise DataValidationError(f"duplicate asset names in data: {assets}")
    missing = [a for a in assets if a not in weights]
    extra = [k for k in weights if k not in assets]
    if missing or extra:
        raise DataValidationError(
            f"weights do not align with assets; missing weights for {missing}, "
            f"weights for unknown assets {extra}"
        )
    w = np.array([float(weights[a]) for a in assets], dtype=float)
    if not np.all(np.isfinite(w)):
        raise DataValidationError("weights must be finite")
    if abs(w.sum() - 1.0) > tol:
        raise DataValidationError(f"weights must sum to 1 within {tol}; sum is {w.sum()!r}")
    if long_only and np.any(w < 0):
        raise DataValidationError("long-only portfolio has negative weights")
    return w


def validate_dates(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Check that dates parse, are unique and strictly increasing."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.hasnans:
        raise DataValidationError("dates contain missing values")
    if not idx.is_unique:
        dupes = idx[idx.duplicated()].unique()[:5]
        raise DataValidationError(f"dates are not unique; examples: {list(dupes.astype(str))}")
    if not idx.is_monotonic_increasing:
        raise DataValidationError("dates must be sorted in strictly increasing order")
    return idx


def validate_simple_returns(returns: pd.DataFrame | np.ndarray, *, name: str = "returns") -> None:
    """Check simple returns are finite and strictly greater than -1."""
    arr = np.asarray(returns, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise DataValidationError(f"{name} contain non-finite values")
    if np.any(arr <= -1.0):
        raise DataValidationError(f"{name} contain simple returns <= -1 (loss of 100% or more)")


def validate_min_history(n_obs: int, required: int, *, what: str = "observations") -> None:
    """Check that at least ``required`` observations are available."""
    if n_obs < required:
        raise DataValidationError(f"insufficient history: {n_obs} {what}, need at least {required}")


def validate_unique_columns(columns: Sequence[str]) -> None:
    """Reject duplicate column names (duplicate assets)."""
    cols = list(columns)
    if len(set(cols)) != len(cols):
        raise DataValidationError(f"duplicate column names: {cols}")
