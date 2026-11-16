"""Expected Shortfall evaluation: FZ0 joint VaR/ES score and calibration diagnostics.

ES is *not* a quantile, so no exception-frequency test is applied to it. This module
provides (i) the FZ0 joint scoring rule for (VaR, ES) pairs and (ii) descriptive
calibration diagnostics. No formal ES hypothesis test is claimed.

FZ0 loss (Patton, Ziegel & Chen, 2019), lower-tail *return* convention
------------------------------------------------------------------------
With tail probability ``alpha = 1 - c``, return ``y``, return-quantile ``v`` and
return-ES ``e`` (both typically negative)::

    FZ0(y, v, e; alpha) = -1/(alpha e) * 1{y <= v} (v - y) + v/e + log(-e) - 1

It is a member of the Fissler-Ziegel (2016) class of strictly consistent scoring
functions for the pair (VaR, ES) and is defined only for ``e < 0``; we also require
``e <= v`` (ES at least as severe as VaR). Lower is better.

Conversion from this package's loss convention (``L = -y``, ``VaR_L = -v``,
``ES_L = -e``)::

    y = -L,  v = -VaR_L,  e = -ES_L,   1{y <= v} = 1{L >= VaR_L},  (v - y) = (L - VaR_L)

    FZ0 = 1/(alpha ES_L) * 1{L >= VaR_L} (L - VaR_L) + VaR_L/ES_L + log(ES_L) - 1

Domain in loss units: ``ES_L > 0`` and ``ES_L >= VaR_L``. Rows outside the domain are
marked ``invalid_domain`` and receive no score; the data are never altered to fit.

ES identification residual (loss convention)
--------------------------------------------
    Z_t = (1/alpha) * 1{L_t >= VaR_t} (L_t - VaR_t) - (ES_t - VaR_t)

For correctly specified forecasts of a continuous distribution, ``E[Z_t] = 0``
(because ``E[(L - VaR) 1{L >= VaR}] = alpha (ES - VaR)``). A positive mean indicates
realised tail losses larger than forecast ES on average (ES too low); negative
indicates ES too conservative. It is reported as a descriptive sample mean only.
"""

from __future__ import annotations

import numpy as np

from .validation import validate_confidence


def fz0_domain_valid(var: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Boolean mask: finite, ``ES_L > 0`` and ``ES_L >= VaR_L`` (loss units)."""
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & np.isfinite(e) & (e > 0) & (e >= v)


def fz0_score(losses, var, es, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """FZ0 score in loss convention. Returns ``(score, valid_mask)``; invalid rows are NaN."""
    alpha = 1.0 - validate_confidence(confidence)
    L = np.asarray(losses, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    L, v, e = np.broadcast_arrays(L, v, e)
    valid = fz0_domain_valid(v, e) & np.isfinite(L)
    score = np.full(L.shape, np.nan)
    Lv, vv, ev = L[valid], v[valid], e[valid]
    hit = (Lv >= vv).astype(float)
    score[valid] = hit * (Lv - vv) / (alpha * ev) + vv / ev + np.log(ev) - 1.0
    return score, valid


def fz0_score_return_convention(y, v, e, confidence: float) -> np.ndarray:
    """FZ0 in the published lower-tail return convention (for verification tests)."""
    alpha = 1.0 - validate_confidence(confidence)
    y, v, e = (np.asarray(a, dtype=float) for a in (y, v, e))
    if np.any(e >= 0):
        raise ValueError("FZ0 requires strictly negative return-ES")
    return -1.0 / (alpha * e) * (y <= v) * (v - y) + v / e + np.log(-e) - 1.0


def es_identification_residual(losses, var, es, confidence: float) -> np.ndarray:
    """Joint VaR/ES identification residual for ES (loss convention); mean ~ 0 if calibrated."""
    alpha = 1.0 - validate_confidence(confidence)
    L = np.asarray(losses, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    return (L >= v) * (L - v) / alpha - (e - v)


def var_identification_residual(losses, var, confidence: float) -> np.ndarray:
    """VaR identification residual ``1{L > VaR} - alpha``; mean ~ 0 if calibrated."""
    alpha = 1.0 - validate_confidence(confidence)
    return (np.asarray(losses, dtype=float) > np.asarray(var, dtype=float)).astype(float) - alpha


def tail_severity(losses, var, es) -> dict[str, float]:
    """Descriptive severity on exception days (``L > VaR``).

    Returns the number of exceptions, mean realised loss, mean forecast ES and VaR on
    those days, mean shortfall ``L - ES`` and the ratio of mean realised loss to mean
    forecast ES. NaN when there are no exceptions.
    """
    L = np.asarray(losses, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    hit = L > v
    n = int(hit.sum())
    if n == 0:
        return {"n_exceptions": 0, "mean_loss_on_exceptions": np.nan, "mean_es_on_exceptions": np.nan,
                "mean_var_on_exceptions": np.nan, "mean_loss_minus_es_on_exceptions": np.nan,
                "realized_to_forecast_es_ratio": np.nan, "max_loss_on_exceptions": np.nan}
    return {
        "n_exceptions": n,
        "mean_loss_on_exceptions": float(L[hit].mean()),
        "mean_es_on_exceptions": float(e[hit].mean()),
        "mean_var_on_exceptions": float(v[hit].mean()),
        "mean_loss_minus_es_on_exceptions": float((L[hit] - e[hit]).mean()),
        "realized_to_forecast_es_ratio": float(L[hit].mean() / e[hit].mean()),
        "max_loss_on_exceptions": float(L[hit].max()),
    }
