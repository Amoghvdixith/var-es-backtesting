"""Strictly out-of-sample rolling one-day VaR / ES forecasts.

Timing (for forecast date with dataset position ``t``):

* the estimation window is positions ``t - window .. t - 1`` (observations through
  ``t - 1`` only);
* every model is fitted and every confidence level forecast from that window inside
  :func:`forecast_origin`, which never receives observation ``t``;
* the realised return at ``t`` is attached afterwards in :func:`attach_realised`.

Models are refitted at every origin. A model that fails at an origin produces rows
with ``forecast_status = "fit_failed"`` and missing VaR/ES -- no other method is
substituted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .historical import historical_var_es
from .monte_carlo import derive_seed, monte_carlo_var_es
from .parametric import fit_normal, fit_student_t, normal_var_es, student_t_var_es
from .validation import DataValidationError, validate_min_history

VALID_MODELS = ("historical", "normal", "student_t", "monte_carlo")


def forecast_origin(
    train_assets: np.ndarray,
    weights: np.ndarray,
    forecast_date: pd.Timestamp,
    confidence_levels: Sequence[float],
    models: Sequence[str],
    *,
    mc_master_seed: int,
    mc_scenarios: int,
    t_nu_upper: float = 1000.0,
    t_init_nu: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit every model on the training window and forecast VaR/ES for one date.

    ``train_assets`` must contain only observations strictly before ``forecast_date``.
    Returns ``(forecast_rows, diagnostic_rows)``.
    """
    port = train_assets @ weights
    rows: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []

    def add(model: str, status: str, results: dict[float, tuple[float, float]] | None) -> None:
        for c in confidence_levels:
            var, es = results[c] if results is not None else (np.nan, np.nan)
            rows.append({"model": model, "confidence": float(c), "var": var, "es": es, "forecast_status": status})

    for model in models:
        if model == "historical":
            add(model, "ok", historical_var_es(port, confidence_levels))
            diags.append({"model": model, "fit_status": "ok", "n_train": port.size})
        elif model == "normal":
            fit = fit_normal(port)
            res = {c: normal_var_es(fit.mu, fit.sigma, c) for c in confidence_levels}
            add(model, "ok", res)
            diags.append({"model": model, "fit_status": fit.status, "n_train": port.size, "mu": fit.mu, "sigma": fit.sigma})
        elif model == "student_t":
            fit = fit_student_t(port, nu_upper=t_nu_upper, init_nu=t_init_nu)
            if fit.usable:
                add(model, "ok", {c: student_t_var_es(fit.loc, fit.scale, fit.nu, c) for c in confidence_levels})
            else:
                add(model, "fit_failed", None)
            diags.append(
                {
                    "model": model, "fit_status": fit.status, "n_train": port.size,
                    "t_loc": fit.loc, "t_scale": fit.scale, "t_nu": fit.nu, "t_implied_sd": fit.implied_sd,
                    "t_converged": fit.converged, "t_at_lower_nu_bound": fit.at_lower_nu_bound,
                    "t_at_upper_nu_bound": fit.at_upper_nu_bound, "t_neg_loglik": fit.neg_loglik,
                    "t_n_iter": fit.n_iter, "message": fit.message,
                }
            )
        elif model == "monte_carlo":
            rng = np.random.default_rng(derive_seed(mc_master_seed, forecast_date, "monte_carlo"))
            try:
                mc = monte_carlo_var_es(train_assets, weights, confidence_levels, mc_scenarios, rng)
            except DataValidationError as exc:
                add(model, "fit_failed", None)
                diags.append({"model": model, "fit_status": "fit_failed", "n_train": port.size, "message": str(exc)})
                continue
            add(model, "ok", mc.var_es)
            diags.append(
                {
                    "model": model, "fit_status": "ok", "n_train": port.size,
                    "mc_scenarios": mc.n_scenarios, "mc_cov_repaired": mc.cov_repaired,
                    "mc_factor_method": mc.factor_method, "mc_min_eigenvalue": mc.min_eigenvalue,
                    "mc_asset_draws_below_minus_one": mc.n_asset_draws_below_minus_one,
                }
            )
        else:
            raise DataValidationError(f"unknown model {model!r}; valid: {VALID_MODELS}")
    return rows, diags


def rolling_forecasts(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    *,
    window: int,
    confidence_levels: Sequence[float],
    models: Sequence[str],
    mc_master_seed: int,
    mc_scenarios: int,
    t_nu_upper: float = 1000.0,
    t_init_nu: float = 5.0,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling forecasts for every position ``t = window .. n-1`` (no realised data attached).

    Output columns: ``obs_index, forecast_date, train_start, train_end, model,
    confidence, var, es, forecast_status``.
    """
    values = asset_returns.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(asset_returns.index)
    n = values.shape[0]
    validate_min_history(n, window + 1, what="observations (window + 1 required)")
    w = np.asarray(weights, dtype=float)

    f_rows: list[dict[str, Any]] = []
    d_rows: list[dict[str, Any]] = []
    for t in range(window, n):
        train = values[t - window : t]  # observations through t-1 only
        rows, diags = forecast_origin(
            train, w, dates[t], confidence_levels, models,
            mc_master_seed=mc_master_seed, mc_scenarios=mc_scenarios,
            t_nu_upper=t_nu_upper, t_init_nu=t_init_nu,
        )
        common = {
            "obs_index": t,
            "forecast_date": dates[t],
            "train_start": dates[t - window],
            "train_end": dates[t - 1],
        }
        f_rows.extend({**common, **r} for r in rows)
        d_rows.extend({**common, **d} for d in diags)
        if progress and (t - window) % 100 == 0:
            print(f"  forecast {t - window + 1}/{n - window}  {dates[t].date()}", flush=True)
    return pd.DataFrame(f_rows), pd.DataFrame(d_rows)


def attach_realised(
    forecasts: pd.DataFrame,
    portfolio_returns: pd.Series,
    *,
    notional_gbp: float,
    stress: pd.Series | None = None,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    """Join realised portfolio returns/losses (observed after the forecast) and GBP amounts."""
    out = forecasts.copy()
    realised = portfolio_returns.reindex(pd.DatetimeIndex(out["forecast_date"]))
    if realised.isna().any():
        raise DataValidationError("realised returns missing for some forecast dates")
    out["realized_return"] = realised.to_numpy()
    out["realized_loss"] = -out["realized_return"]
    out["var_gbp"] = out["var"] * notional_gbp
    out["es_gbp"] = out["es"] * notional_gbp
    out["realized_loss_gbp"] = out["realized_loss"] * notional_gbp
    idx = pd.DatetimeIndex(out["forecast_date"])
    out["is_stress"] = (stress.reindex(idx).fillna(False).astype(bool).to_numpy() if stress is not None else False)
    out["regime"] = regime.reindex(idx).to_numpy() if regime is not None else "unlabelled"
    return out


def is_valid(df: pd.DataFrame) -> pd.Series:
    """Valid forecast rows: status ok and finite VaR/ES."""
    return (df["forecast_status"] == "ok") & np.isfinite(df["var"]) & np.isfinite(df["es"])


def common_valid_dates(forecasts: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Dates on which every model has a valid forecast at every confidence level.

    Returns the common dates and a table of exclusions by model (rows that were not
    valid), so cross-model comparisons can report what was dropped.
    """
    ok = forecasts.assign(_ok=is_valid(forecasts))
    per_date = ok.groupby("forecast_date")["_ok"].all()
    common = pd.DatetimeIndex(per_date.index[per_date.to_numpy()])
    excl = (
        ok.loc[~ok["_ok"]]
        .groupby(["model", "confidence"])
        .size()
        .rename("n_invalid_rows")
        .reset_index()
    )
    return common, excl
