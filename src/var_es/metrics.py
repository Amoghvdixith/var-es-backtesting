"""Evaluation tables built from realised forecasts.

All cross-model tables use the *common valid sample*: forecast dates on which every
model has a valid forecast at every confidence level. Rows excluded from that sample
are counted in ``n_excluded_invalid``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtesting import (
    christoffersen_cc,
    christoffersen_ind,
    exceptions,
    kupiec_uc,
    max_consecutive_exceptions,
    quantile_loss,
    rolling_exception_rate,
    transition_counts,
)
from .es_evaluation import (
    es_identification_residual,
    fz0_domain_valid,
    fz0_score,
    tail_severity,
    var_identification_residual,
)
from .forecasting import common_valid_dates, is_valid
from .traffic_light import TL_CONFIDENCE, TL_WINDOW, rolling_traffic_light

FORECAST_COLUMNS = [
    "forecast_date", "train_start", "train_end", "obs_index", "model", "confidence",
    "realized_return", "realized_loss", "var", "es", "var_gbp", "es_gbp", "realized_loss_gbp",
    "exception", "quantile_loss", "fz0_score", "fz0_status", "es_id_residual",
    "is_stress", "regime", "forecast_status", "in_common_sample",
]


def evaluate_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Add exception flags, quantile loss, FZ0 score/status and ES identification residuals."""
    out = df.copy()
    valid = is_valid(out).to_numpy()
    out["exception"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["quantile_loss"] = np.nan
    out["fz0_score"] = np.nan
    out["fz0_status"] = "no_forecast"
    out["es_id_residual"] = np.nan
    for c, idx in out.groupby("confidence").groups.items():
        m = out.index.isin(idx) & valid
        L, v, e = (out.loc[m, k].to_numpy(dtype=float) for k in ("realized_loss", "var", "es"))
        out.loc[m, "exception"] = exceptions(L, v)
        out.loc[m, "quantile_loss"] = quantile_loss(L, v, c)
        score, ok = fz0_score(L, v, e, c)
        out.loc[m, "fz0_score"] = score
        out.loc[m, "fz0_status"] = np.where(ok, "ok", "invalid_domain")
        out.loc[m, "es_id_residual"] = es_identification_residual(L, v, e, c)
    common, _ = common_valid_dates(out)
    out["in_common_sample"] = out["forecast_date"].isin(common)
    return out[FORECAST_COLUMNS]


def _groups(ev: pd.DataFrame):
    cs = ev.loc[ev["in_common_sample"]]
    for (model, c), g in cs.groupby(["model", "confidence"], sort=False):
        yield model, float(c), g.sort_values("obs_index")


def _n_excluded(ev: pd.DataFrame, model: str, c: float) -> int:
    m = (ev["model"] == model) & (ev["confidence"] == c)
    return int((m & ~ev["in_common_sample"]).sum())


def summary_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Overall VaR/ES summary by model and confidence on the common valid sample."""
    rows = []
    for model, c, g in _groups(ev):
        L, v, e = g["realized_loss"].to_numpy(), g["var"].to_numpy(), g["es"].to_numpy()
        hits = g["exception"].to_numpy(dtype=bool)
        n, x = len(g), int(hits.sum())
        rows.append(
            {
                "model": model, "confidence": c, "n_forecasts": n,
                "n_excluded_invalid": _n_excluded(ev, model, c),
                "n_exceptions": x, "expected_exceptions": n * (1 - c),
                "observed_rate": x / n if n else np.nan, "nominal_rate": 1 - c,
                "mean_quantile_loss": float(g["quantile_loss"].mean()),
                "mean_var": float(v.mean()), "mean_es": float(e.mean()),
                "mean_var_gbp": float(g["var_gbp"].mean()), "mean_es_gbp": float(g["es_gbp"].mean()),
                "max_consecutive_exceptions": max_consecutive_exceptions(hits, g["obs_index"].to_numpy()),
                "mean_loss_beyond_var_on_exceptions": float((L[hits] - v[hits]).mean()) if x else np.nan,
                "n_es_below_var": int(np.sum(e < v)),
                "mean_fz0_score": float(g["fz0_score"].mean()),
                "n_fz0_invalid": int((g["fz0_status"] != "ok").sum()),
            }
        )
    return pd.DataFrame(rows)


def coverage_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Kupiec UC, Christoffersen IND and CC tests by model and confidence."""
    rows = []
    for model, c, g in _groups(ev):
        hits = g["exception"].to_numpy(dtype=bool)
        pos = g["obs_index"].to_numpy()
        uc = kupiec_uc(len(g), int(hits.sum()), c)
        tc = transition_counts(hits, pos)
        ind = christoffersen_ind(tc)
        cc = christoffersen_cc(uc, ind)
        rows.append(
            {"model": model, "confidence": c, "n_forecasts": len(g), "n_exceptions": int(hits.sum()),
             **tc, "n_transitions": sum(tc.values()),
             **uc.as_dict("uc"), **ind.as_dict("ind"), **cc.as_dict("cc")}
        )
    return pd.DataFrame(rows)


def es_table(ev: pd.DataFrame) -> pd.DataFrame:
    """FZ0 joint scores and descriptive ES calibration diagnostics."""
    rows = []
    for model, c, g in _groups(ev):
        L, v, e = g["realized_loss"].to_numpy(), g["var"].to_numpy(), g["es"].to_numpy()
        z = g["es_id_residual"].to_numpy(dtype=float)
        vz = var_identification_residual(L, v, c)
        ok = g["fz0_status"].to_numpy() == "ok"
        rows.append(
            {
                "model": model, "confidence": c, "n_forecasts": len(g),
                "n_es_ge_var": int(np.sum(e >= v)), "n_es_below_var": int(np.sum(e < v)),
                "n_fz0_valid": int(ok.sum()), "n_fz0_invalid_domain": int((~ok).sum()),
                "mean_fz0_score": float(np.nanmean(g["fz0_score"])) if ok.any() else np.nan,
                "mean_es_id_residual": float(z.mean()),
                "se_es_id_residual_descriptive": float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan,
                "mean_es_id_residual_gbp_per_1m": float(z.mean() * 1_000_000),
                "mean_var_id_residual": float(vz.mean()),
                **tail_severity(L, v, e),
            }
        )
    return pd.DataFrame(rows)


def stress_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Stress vs non-stress comparison (descriptive; no p-values for short periods)."""
    rows = []
    for model, c, g in _groups(ev):
        for label, sub in (("stress", g[g["is_stress"]]), ("non_stress", g[~g["is_stress"]])):
            n = len(sub)
            if n == 0:
                continue
            L, v, e = sub["realized_loss"].to_numpy(), sub["var"].to_numpy(), sub["es"].to_numpy()
            hits = sub["exception"].to_numpy(dtype=bool)
            sev = tail_severity(L, v, e)
            ok = sub["fz0_status"].to_numpy() == "ok"
            rows.append(
                {
                    "model": model, "confidence": c, "period": label, "n_forecasts": n,
                    "n_exceptions": int(hits.sum()), "expected_exceptions": n * (1 - c),
                    "observed_rate": hits.mean(), "nominal_rate": 1 - c,
                    "mean_var": float(v.mean()), "mean_es": float(e.mean()),
                    "mean_realized_loss": float(L.mean()), "sd_realized_loss": float(L.std(ddof=1)) if n > 1 else np.nan,
                    "mean_quantile_loss": float(sub["quantile_loss"].mean()),
                    "mean_fz0_score": float(np.nanmean(sub["fz0_score"])) if ok.any() else np.nan,
                    "n_fz0_invalid_domain": int((~ok).sum()),
                    "mean_es_id_residual": float(sub["es_id_residual"].mean()),
                    "mean_loss_on_exceptions": sev["mean_loss_on_exceptions"],
                    "mean_es_on_exceptions": sev["mean_es_on_exceptions"],
                    "realized_to_forecast_es_ratio": sev["realized_to_forecast_es_ratio"],
                }
            )
    return pd.DataFrame(rows)


def traffic_light_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Rolling 250-day traffic-light windows for 99% VaR, per model, over all forecast positions.

    Uses each model's own validity (not the common sample) so gaps are visible as
    ``unavailable`` windows rather than silently bridged.
    """
    out = []
    sub = ev[np.isclose(ev["confidence"], TL_CONFIDENCE)]
    for model, g in sub.groupby("model", sort=False):
        g = g.sort_values("obs_index")
        valid = is_valid(g).to_numpy()
        hits = g["exception"].fillna(False).to_numpy(dtype=bool)
        if len(g) < TL_WINDOW:
            continue
        tl = rolling_traffic_light(
            pd.DatetimeIndex(g["forecast_date"]), g["obs_index"].to_numpy(), hits, valid,
            confidence=TL_CONFIDENCE, window=TL_WINDOW,
        )
        tl.insert(0, "model", model)
        out.append(tl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["model", "window_start", "window_end", "n_valid", "n_exceptions", "zone"]
    )


def traffic_light_counts(tl: pd.DataFrame) -> pd.DataFrame:
    """Number of rolling windows per zone and model."""
    if tl.empty:
        return pd.DataFrame()
    t = tl.groupby(["model", "zone"]).size().unstack(fill_value=0)
    for z in ("green", "amber", "red", "unavailable"):
        if z not in t.columns:
            t[z] = 0
    return t[["green", "amber", "red", "unavailable"]].reset_index()


def rolling_rates(ev: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling exception rates by model/confidence on each model's own valid forecasts."""
    frames = []
    for (model, c), g in ev.groupby(["model", "confidence"], sort=False):
        g = g.sort_values("obs_index")
        valid = is_valid(g).to_numpy()
        pos = g["obs_index"].to_numpy()
        hits = g["exception"].fillna(False).to_numpy(dtype=bool)
        start, size = int(pos.min()), int(pos.max() - pos.min() + 1)
        if valid.any():
            rate = rolling_exception_rate(hits[valid], pos[valid], window, start=start, n_positions=size)
        else:
            rate = np.full(size, np.nan)
        grid = pd.DataFrame({"obs_index": np.arange(start, start + rate.size), "rolling_rate": rate})
        grid = grid.merge(g[["obs_index", "forecast_date"]], on="obs_index", how="left")
        grid["model"], grid["confidence"] = model, float(c)
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def fit_diagnostics_summary(diag: pd.DataFrame) -> dict[str, Any]:
    """Counts of fit statuses, Student-t boundary hits and MC covariance repairs."""
    out: dict[str, Any] = {"fit_status_counts": {}}
    for model, g in diag.groupby("model"):
        out["fit_status_counts"][model] = {str(k): int(v) for k, v in g["fit_status"].value_counts().items()}
    t = diag[diag["model"] == "student_t"]
    if not t.empty:
        out["student_t"] = {
            "n_upper_nu_bound": int(t["t_at_upper_nu_bound"].fillna(False).astype(bool).sum()),
            "n_lower_nu_bound": int(t["t_at_lower_nu_bound"].fillna(False).astype(bool).sum()),
            "nu_median": float(t["t_nu"].median()),
            "nu_min": float(t["t_nu"].min()),
            "nu_max": float(t["t_nu"].max()),
        }
    m = diag[diag["model"] == "monte_carlo"]
    if not m.empty and "mc_cov_repaired" in m:
        out["monte_carlo"] = {
            "n_cov_repaired": int(m["mc_cov_repaired"].fillna(False).astype(bool).sum()),
            "total_asset_draws_below_minus_one": int(m["mc_asset_draws_below_minus_one"].fillna(0).sum()),
        }
    return out


def es_ge_var_check(ev: pd.DataFrame) -> int:
    """Number of valid forecast rows with ES < VaR (should be zero for all models)."""
    v = is_valid(ev)
    return int((ev.loc[v, "es"] < ev.loc[v, "var"]).sum())


__all__ = [
    "evaluate_forecasts", "summary_table", "coverage_table", "es_table", "stress_table",
    "traffic_light_table", "traffic_light_counts", "rolling_rates", "fit_diagnostics_summary",
    "es_ge_var_check", "fz0_domain_valid",
]
