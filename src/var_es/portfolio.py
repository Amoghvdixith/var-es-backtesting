"""Fixed-weight, daily-rebalanced portfolio returns and the loss convention.

Conventions
-----------
* Returns are decimal simple returns (0.01 = 1%).
* ``portfolio_return_t = sum_i w_i * r_{i,t}`` -- exact for simple returns of a
  portfolio rebalanced to fixed weights at the start of each day.
* ``loss_t = -portfolio_return_t`` (positive = loss, negative = gain).
* ``loss_gbp_t = notional_gbp * loss_t`` for a constant reporting notional. This is a
  constant-notional risk comparison, not a compounded wealth simulation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .validation import DataValidationError, validate_weights


def portfolio_returns(
    asset_returns: pd.DataFrame, weights: Mapping[str, float], *, long_only: bool = True
) -> pd.Series:
    """Weighted sum of asset simple returns, with strict alignment and no reweighting.

    Raises if any asset return is missing: a missing asset is never handled by
    renormalising the remaining weights.
    """
    w = validate_weights(weights, list(asset_returns.columns), long_only=long_only)
    values = asset_returns.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise DataValidationError(
            "asset returns contain missing values; apply the complete-case policy before "
            "computing portfolio returns (no silent reweighting)"
        )
    return pd.Series(values @ w, index=asset_returns.index, name="portfolio_return")


def returns_to_losses(returns: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Loss convention: ``loss = -return``."""
    return -returns


def to_gbp(loss_decimal: np.ndarray | pd.Series | float, notional_gbp: float):
    """Translate decimal losses (or VaR/ES) into GBP on a constant notional."""
    return loss_decimal * float(notional_gbp)


def portfolio_moments(mean: np.ndarray, cov: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Portfolio mean ``w'mu`` and standard deviation ``sqrt(w' Sigma w)``."""
    var = float(w @ cov @ w)
    return float(w @ mean), float(np.sqrt(max(var, 0.0)))
