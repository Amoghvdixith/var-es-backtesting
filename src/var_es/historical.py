"""Historical simulation: empirical VaR/ES of the trailing estimation-window losses."""

from __future__ import annotations

import numpy as np

from .empirical import empirical_var_es


def historical_var_es(window_returns: np.ndarray, confidence_levels) -> dict[float, tuple[float, float]]:
    """Historical-simulation VaR/ES in decimal loss units.

    ``window_returns`` must contain only portfolio returns observed strictly before
    the forecast date; they are converted to losses and passed to the common
    empirical engine.
    """
    losses = -np.asarray(window_returns, dtype=float)
    return empirical_var_es(losses, confidence_levels)
