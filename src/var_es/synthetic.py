"""Reproducible synthetic five-asset equity returns.

Data-generating process (per retained day ``t = 0..n-1``)::

    z_t   ~ N(0, I_5)                                  (independent standard normals)
    W_t   ~ InvGamma(nu/2, nu/2) truncated to (0, U]   (shared across assets)
    s_t   = sqrt(W_t / kappa),  kappa = E[W | W <= U]  (so E[s_t^2] = 1)
    x_t   = mu + m_t * s_t * diag(sigma) * L * z_t     (latent daily LOG returns)
    r_t   = expm1(x_t)                                  (simple returns, > -1)

``L`` is the Cholesky factor of the correlation matrix ``C``; ``m_t`` is a predefined
volatility-regime multiplier. ``mu`` and ``sigma`` are parameters of the *latent
log-return* process (before the scale mixture and regimes), **not** the mean and
standard deviation of the simple returns that are saved.

The shared scale mixture induces heavy tails and tail dependence. Truncating ``W``
bounds the latent log returns' scale, so all moments of ``expm1(x)`` are finite --
unlike exponentiating untruncated Student-t draws, whose ``E[exp(x)]`` is infinite.
No returns are clipped. This is a stylised generator for testing risk models; it is
not a calibrated model of real equity markets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, stats

from .portfolio import portfolio_returns
from .validation import DataValidationError, validate_simple_returns


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncated_invgamma_kappa(nu: float, upper: float) -> tuple[float, float]:
    """Return ``(kappa, F(U))``: ``E[W | W <= U]`` and the truncation probability mass."""
    dist = stats.invgamma(a=nu / 2.0, scale=nu / 2.0)
    mass = float(dist.cdf(upper))
    num, _ = integrate.quad(lambda w: w * dist.pdf(w), 0.0, upper, limit=200)
    return float(num / mass), mass


def regime_schedule(regimes: list[dict[str, Any]], n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand predefined index ranges into per-day multipliers and labels.

    Ranges are inclusive and must cover ``0..n_obs-1`` exactly once.
    """
    mult = np.full(n_obs, np.nan)
    label = np.empty(n_obs, dtype=object)
    for reg in regimes:
        s, e = int(reg["start"]), int(reg["end"])
        if s < 0 or e >= n_obs or s > e:
            raise DataValidationError(f"invalid regime range {s}-{e} for n_obs={n_obs}")
        if not np.all(np.isnan(mult[s : e + 1])):
            raise DataValidationError(f"regime ranges overlap at {s}-{e}")
        mult[s : e + 1] = float(reg["multiplier"])
        label[s : e + 1] = str(reg["label"])
    if np.isnan(mult).any():
        raise DataValidationError("regime ranges do not cover all observations")
    return mult, label


def generate_synthetic_returns(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the synthetic dataset described in the module docstring.

    Returns a DataFrame with columns ``date``, the asset columns, ``is_stress`` and
    ``regime``, plus a metadata dict describing the generation.
    """
    n = int(cfg["n_obs"])
    assets = list(cfg["assets"])
    k = len(assets)
    mu = np.asarray(cfg["latent_mu"], dtype=float)
    sigma = np.asarray(cfg["latent_sigma"], dtype=float)
    corr = np.asarray(cfg["correlation"], dtype=float)
    if mu.shape != (k,) or sigma.shape != (k,) or corr.shape != (k, k):
        raise DataValidationError("parameter dimensions do not match number of assets")
    if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
        raise DataValidationError("correlation matrix must be symmetric with unit diagonal")
    chol = np.linalg.cholesky(corr)  # raises if not positive definite

    mix = cfg["mixing"]
    if mix["type"] != "truncated_inverse_gamma":
        raise DataValidationError(f"unsupported mixing type {mix['type']!r}")
    nu, upper = float(mix["nu"]), float(mix["upper_bound"])
    kappa, mass = truncated_invgamma_kappa(nu, upper)

    mult, label = regime_schedule(cfg["regimes"], n)

    rng = np.random.default_rng(int(cfg["seed"]))
    z = rng.standard_normal((n, k))
    u = rng.uniform(0.0, 1.0, size=n)
    w = stats.invgamma.ppf(u * mass, a=nu / 2.0, scale=nu / 2.0)  # inverse-CDF on truncated range
    s = np.sqrt(w / kappa)

    shocks = (z @ chol.T) * sigma
    latent = mu + (mult * s)[:, None] * shocks
    simple = np.expm1(latent)
    validate_simple_returns(simple, name="synthetic simple returns")

    dates = pd.bdate_range(start=cfg["start_date"], periods=n)
    df = pd.DataFrame(simple, columns=assets)
    df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    df["is_stress"] = label == "stress"
    df["regime"] = label

    stress_idx = np.flatnonzero(df["is_stress"].to_numpy())
    meta: dict[str, Any] = {
        "description": "Synthetic, offline, reproducible equity returns for testing VaR/ES models. "
        "Not real market data and not a calibrated model of real equity dynamics.",
        "seed": int(cfg["seed"]),
        "n_obs": n,
        "assets": assets,
        "date_convention": "Synthetic business days (Mon-Fri, no holidays) from pandas.bdate_range",
        "first_date": str(df["date"].iloc[0]),
        "last_date": str(df["date"].iloc[-1]),
        "return_units": "decimal simple returns (0.01 = 1%)",
        "process": "x_t = mu + m_t * s_t * diag(sigma) * chol(C) * z_t;  r_t = expm1(x_t)",
        "latent_mu_log_return": mu.tolist(),
        "latent_sigma_log_return": sigma.tolist(),
        "latent_parameters_note": "mu and sigma describe the latent Gaussian log-return innovations before the "
        "scale mixture and regime multipliers; they are not the mean/SD of the saved simple returns.",
        "correlation": corr.tolist(),
        "mixing": {
            "type": "truncated_inverse_gamma",
            "nu": nu,
            "upper_bound": upper,
            "truncation_probability_mass": mass,
            "kappa_E_W_given_W_le_U": kappa,
            "normalisation": "s_t = sqrt(W_t / kappa) so that E[s_t^2] = 1",
            "sampling": "inverse CDF of InvGamma(nu/2, scale=nu/2) evaluated at u * F(U), u ~ U(0,1)",
        },
        "regimes": cfg["regimes"],
        "stress_intervals_index_inclusive": _intervals(stress_idx),
        "clipping": "none",
        "random_draw_order": "rng = numpy.random.default_rng(seed); z = rng.standard_normal((n, k)); "
        "u = rng.uniform(0, 1, n)",
        "realised_simple_return_mean": dict(zip(assets, simple.mean(axis=0).round(8).tolist())),
        "realised_simple_return_sd": dict(zip(assets, simple.std(axis=0, ddof=1).round(8).tolist())),
        "min_simple_return": float(simple.min()),
        "max_simple_return": float(simple.max()),
    }
    return df, meta


def _intervals(idx: np.ndarray) -> list[list[int]]:
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def write_synthetic_dataset(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Generate and save asset returns, portfolio returns and generation metadata."""
    root = Path(project_root)
    syn = config["synthetic"]
    paths = config["data"]
    df, meta = generate_synthetic_returns(syn)

    asset_path = root / paths["asset_returns_path"]
    port_path = root / paths["portfolio_returns_path"]
    meta_path = root / paths["metadata_path"]
    asset_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(asset_path, index=False, float_format="%.17g")

    rets = df.set_index(pd.to_datetime(df["date"]))[syn["assets"]]
    port = portfolio_returns(rets, config["portfolio"]["weights"], long_only=config["portfolio"]["long_only"])
    port_df = pd.DataFrame(
        {
            "date": df["date"],
            "portfolio_return": port.to_numpy(),
            "portfolio_loss": -port.to_numpy(),
            "is_stress": df["is_stress"],
            "regime": df["regime"],
        }
    )
    port_df.to_csv(port_path, index=False, float_format="%.17g")

    meta["portfolio_weights"] = config["portfolio"]["weights"]
    meta["portfolio_definition"] = "portfolio_return_t = sum_i w_i * r_i,t (daily rebalanced to fixed weights)"
    meta["files"] = {
        "asset_returns": {"path": paths["asset_returns_path"], "sha256": file_sha256(asset_path)},
        "portfolio_returns": {"path": paths["portfolio_returns_path"], "sha256": file_sha256(port_path)},
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta
