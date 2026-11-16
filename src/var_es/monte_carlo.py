"""Correlated multivariate-normal Monte Carlo VaR / ES.

Asset returns over the next day are simulated from ``N(mu_hat, Sigma_hat)`` with the
trailing-window sample mean and sample covariance (ddof=1). The fixed portfolio
weights are applied, simulated portfolio returns are converted to losses, and VaR/ES
are read off with the common empirical engine.

Interpretation: for a linear portfolio, ``w'X`` with ``X ~ N(mu, Sigma)`` is exactly
``N(w'mu, w'Sigma w)``, and ``w' Sigma_hat w`` equals the sample variance of the
historical portfolio returns. Monte Carlo therefore targets *the same distribution*
as the analytical normal model; differences are Monte Carlo sampling error (or
implementation error), not evidence of a better model.

The Gaussian model assigns positive probability to simple returns below -100%.
Simulated draws are *not* clipped; the number of such asset-level draws is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .empirical import empirical_var_es
from .parametric import normal_var_es
from .validation import DataValidationError

MODEL_CODES = {"historical": 1, "normal": 2, "student_t": 3, "monte_carlo": 4}
SYMMETRY_RTOL = 1e-10
PSD_RTOL = 1e-10


@dataclass
class CovarianceCheck:
    cov: np.ndarray
    min_eigenvalue: float
    repaired: bool
    factor_method: str  # "cholesky" or "eigh"


def estimate_mean_cov(asset_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean vector and sample covariance matrix (ddof=1)."""
    x = np.asarray(asset_window, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise DataValidationError("need a 2-D window with at least two observations")
    return x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)


def check_covariance(cov: np.ndarray) -> CovarianceCheck:
    """Validate symmetry and positive semidefiniteness.

    * Asymmetry beyond ``SYMMETRY_RTOL`` (relative to the largest entry) -> error.
    * Eigenvalues below ``-PSD_RTOL * max_eigenvalue`` -> error (genuinely invalid).
    * Tiny negative eigenvalues within that tolerance (floating-point noise) are
      repaired by clipping eigenvalues at zero and reconstructing ``V diag(l) V'``;
      ``repaired=True`` is recorded. No arbitrary diagonal loading is applied.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise DataValidationError("covariance must be square")
    if not np.all(np.isfinite(c)):
        raise DataValidationError("covariance contains non-finite values")
    scale = max(float(np.max(np.abs(c))), 1e-300)
    if np.max(np.abs(c - c.T)) > SYMMETRY_RTOL * scale:
        raise DataValidationError("covariance matrix is not symmetric")
    c = 0.5 * (c + c.T)
    eig, vec = np.linalg.eigh(c)
    lam_max = max(float(eig.max()), 0.0)
    lam_min = float(eig.min())
    if lam_min < -PSD_RTOL * max(lam_max, 1e-300):
        raise DataValidationError(f"covariance is not positive semidefinite (min eigenvalue {lam_min:.3e})")
    repaired = False
    if lam_min < 0:
        c = (vec * np.clip(eig, 0.0, None)) @ vec.T
        c = 0.5 * (c + c.T)
        repaired = True
    try:
        np.linalg.cholesky(c)
        method = "cholesky"
    except np.linalg.LinAlgError:
        method = "eigh"
    return CovarianceCheck(cov=c, min_eigenvalue=lam_min, repaired=repaired, factor_method=method)


def _factor(chk: CovarianceCheck) -> np.ndarray:
    if chk.factor_method == "cholesky":
        return np.linalg.cholesky(chk.cov)
    eig, vec = np.linalg.eigh(chk.cov)
    return vec * np.sqrt(np.clip(eig, 0.0, None))


def date_to_int(date: pd.Timestamp | str) -> int:
    """Stable integer ``YYYYMMDD`` for a date."""
    return int(pd.Timestamp(date).strftime("%Y%m%d"))


def derive_seed(master_seed: int, date: pd.Timestamp | str, model: str = "monte_carlo") -> np.random.SeedSequence:
    """Deterministic per-origin seed from master seed, date and model identifier.

    Uses ``numpy.random.SeedSequence`` entropy mixing; Python's process-randomised
    ``hash()`` is deliberately not used.
    """
    return np.random.SeedSequence([int(master_seed), date_to_int(date), MODEL_CODES[model]])


def simulate_asset_returns(mean: np.ndarray, chk: CovarianceCheck, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` correlated multivariate-normal asset-return scenarios."""
    z = rng.standard_normal((n, mean.size))
    return mean + z @ _factor(chk).T


@dataclass
class MonteCarloResult:
    var_es: dict[float, tuple[float, float]]
    n_scenarios: int
    n_asset_draws_below_minus_one: int
    cov_repaired: bool
    factor_method: str
    min_eigenvalue: float


def monte_carlo_var_es(
    asset_window: np.ndarray,
    weights: np.ndarray,
    confidence_levels,
    n_scenarios: int,
    rng: np.random.Generator,
) -> MonteCarloResult:
    """Fit mean/covariance on the window, simulate, and compute empirical VaR/ES of losses."""
    mean, cov = estimate_mean_cov(asset_window)
    chk = check_covariance(cov)
    sims = simulate_asset_returns(mean, chk, int(n_scenarios), rng)
    losses = -(sims @ np.asarray(weights, dtype=float))
    return MonteCarloResult(
        var_es=empirical_var_es(losses, confidence_levels),
        n_scenarios=int(n_scenarios),
        n_asset_draws_below_minus_one=int(np.sum(sims <= -1.0)),
        cov_repaired=chk.repaired,
        factor_method=chk.factor_method,
        min_eigenvalue=chk.min_eigenvalue,
    )


def convergence_experiment(
    asset_window: np.ndarray,
    weights: np.ndarray,
    confidence_levels,
    scenario_counts,
    seeds,
) -> pd.DataFrame:
    """MC VaR/ES at several scenario counts and seeds versus the analytical normal benchmark.

    The benchmark uses the same window: ``mu_p = w'mu_hat``, ``sigma_p = sqrt(w' Sigma_hat w)``.
    Each (seed, n) pair uses ``SeedSequence([seed, n])`` so runs are reproducible.
    """
    w = np.asarray(weights, dtype=float)
    mean, cov = estimate_mean_cov(asset_window)
    mu_p, sd_p = float(w @ mean), float(np.sqrt(w @ cov @ w))
    rows = []
    for n in scenario_counts:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(n)]))
            res = monte_carlo_var_es(asset_window, w, confidence_levels, int(n), rng)
            for c, (v, e) in res.var_es.items():
                av, ae = normal_var_es(mu_p, sd_p, c)
                rows.append(
                    {
                        "n_scenarios": int(n), "seed": int(seed), "confidence": c,
                        "mc_var": v, "mc_es": e, "analytic_var": av, "analytic_es": ae,
                        "var_error": v - av, "es_error": e - ae,
                        "var_rel_error": (v - av) / av, "es_rel_error": (e - ae) / ae,
                    }
                )
    return pd.DataFrame(rows)


def summarise_convergence(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (n, confidence) mean error, RMSE, SD across seeds, and mean absolute relative error."""
    g = raw.groupby(["n_scenarios", "confidence"])
    out = g.agg(
        n_seeds=("seed", "nunique"),
        analytic_var=("analytic_var", "first"),
        analytic_es=("analytic_es", "first"),
        mean_mc_var=("mc_var", "mean"),
        mean_mc_es=("mc_es", "mean"),
        var_mean_error=("var_error", "mean"),
        es_mean_error=("es_error", "mean"),
        var_sd_across_seeds=("mc_var", "std"),
        es_sd_across_seeds=("mc_es", "std"),
    ).reset_index()
    out["var_rmse"] = g["var_error"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).to_numpy()
    out["es_rmse"] = g["es_error"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).to_numpy()
    out["var_mean_abs_rel_error"] = g["var_rel_error"].apply(lambda s: float(np.mean(np.abs(s)))).to_numpy()
    out["es_mean_abs_rel_error"] = g["es_rel_error"].apply(lambda s: float(np.mean(np.abs(s)))).to_numpy()
    return out
