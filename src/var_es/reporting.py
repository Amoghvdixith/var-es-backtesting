"""Run provenance (metadata) and the generated Markdown research report.

The report is built exclusively from files saved in a results directory
(CSV tables and ``run_metadata.json``). It contains no hard-coded numerical findings:
every number and every data-dependent sentence is derived from those files.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_LABEL = {
    "historical": "Historical simulation",
    "normal": "Parametric normal",
    "student_t": "Parametric Student-t",
    "monte_carlo": "Monte Carlo (MVN)",
}
DEPENDENCIES = ("numpy", "pandas", "scipy", "matplotlib", "pytest")

EMPIRICAL_CONVENTION = (
    "Equal-weight empirical loss distribution. VaR_c = inverse empirical CDF at c = the ceil(n*c)-th "
    "smallest loss. ES_c = average of the worst (1-c) probability mass: with m = n(1-c) and j = floor(m), "
    "ES = [sum of the j largest losses + (m-j) * the (j+1)-th largest loss] / m. Exceptions are strict: "
    "realized_loss > VaR."
)
RETURN_LOSS_DEFINITIONS = {
    "returns": "decimal simple returns (0.01 = 1%)",
    "portfolio_return": "sum_i w_i * r_i,t with fixed weights (daily rebalanced)",
    "loss": "loss_t = -portfolio_return_t (positive = loss)",
    "gbp": "loss_gbp = notional_gbp * loss_t; constant-notional risk comparison, not a compounded wealth simulation",
    "var_es_units": "VaR and ES reported as decimal losses and in GBP on the fixed reporting notional",
}


# ------------------------------------------------------------------- provenance
def source_checksum(src_dir: Path) -> str:
    """SHA-256 over the names and bytes of ``src_dir/*.py`` (sorted)."""
    h = hashlib.sha256()
    for p in sorted(Path(src_dir).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment_info() -> dict[str, Any]:
    versions = {}
    for dep in DEPENDENCIES:
        try:
            versions[dep] = importlib_metadata.version(dep)
        except importlib_metadata.PackageNotFoundError:
            versions[dep] = None
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "dependency_versions": versions,
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas scalars and timestamps for JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def write_json(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


# ---------------------------------------------------------------------- report
def _fmt(x: Any, spec: str) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    try:
        if pd.isna(x):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if spec == "s":
        return str(x)
    if spec == "d":
        return f"{int(round(float(x))):,d}"
    return format(float(x), spec)


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    """Render ``df`` as a Markdown table. ``columns`` = [(column, header, format_spec)]."""
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], f) for c, _, f in columns) + " |")
    return "\n".join(lines)


def _label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out:
        out["model"] = out["model"].map(lambda m: MODEL_LABEL.get(m, m))
    if "confidence" in out:
        out["confidence"] = out["confidence"].map(lambda c: f"{c:.1%}")
    return out


SUMMARY_COLUMNS = [
    ("model", "Model", "s"), ("confidence", "Conf.", "s"), ("n_forecasts", "N", "d"),
    ("n_exceptions", "Exceptions", "d"), ("expected_exceptions", "Expected", ".1f"),
    ("observed_rate", "Obs. rate", ".4f"), ("nominal_rate", "Nominal", ".4f"),
    ("mean_var_gbp", "Mean VaR (GBP)", ",.0f"), ("mean_es_gbp", "Mean ES (GBP)", ",.0f"),
    ("mean_quantile_loss", "Mean quantile loss", ".3e"), ("max_consecutive_exceptions", "Max run", "d"),
]
COVERAGE_COLUMNS = [
    ("model", "Model", "s"), ("confidence", "Conf.", "s"), ("n_exceptions", "Exc.", "d"),
    ("uc_statistic", "LR_uc", ".3f"), ("uc_p_value", "p_uc", ".4f"),
    ("n11", "n11", "d"), ("ind_statistic", "LR_ind", ".3f"), ("ind_p_value", "p_ind", ".4f"),
    ("ind_status", "IND status", "s"), ("cc_statistic", "LR_cc", ".3f"), ("cc_p_value", "p_cc", ".4f"),
]
ES_COLUMNS = [
    ("model", "Model", "s"), ("confidence", "Conf.", "s"), ("n_fz0_valid", "FZ0 valid", "d"),
    ("n_fz0_invalid_domain", "FZ0 invalid", "d"), ("mean_fz0_score", "Mean FZ0", ".4f"),
    ("mean_es_id_residual", "Mean ES id. residual", ".2e"),
    ("se_es_id_residual_descriptive", "SE (descr.)", ".2e"),
    ("mean_loss_on_exceptions", "Mean loss on exc.", ".4f"), ("mean_es_on_exceptions", "Mean ES on exc.", ".4f"),
    ("realized_to_forecast_es_ratio", "Loss/ES ratio", ".3f"), ("n_es_below_var", "ES<VaR", "d"),
]
STRESS_COLUMNS = [
    ("model", "Model", "s"), ("confidence", "Conf.", "s"), ("period", "Period", "s"), ("n_forecasts", "N", "d"),
    ("n_exceptions", "Exc.", "d"), ("expected_exceptions", "Expected", ".1f"), ("observed_rate", "Obs. rate", ".4f"),
    ("mean_var", "Mean VaR", ".4f"), ("mean_es", "Mean ES", ".4f"), ("mean_quantile_loss", "Quantile loss", ".3e"),
    ("mean_fz0_score", "Mean FZ0", ".4f"), ("realized_to_forecast_es_ratio", "Loss/ES on exc.", ".3f"),
]
MC_COLUMNS = [
    ("n_scenarios", "Scenarios", "d"), ("confidence", "Conf.", "s"), ("n_seeds", "Seeds", "d"),
    ("analytic_var", "Analytic VaR", ".5f"), ("mean_mc_var", "Mean MC VaR", ".5f"),
    ("var_rmse", "VaR RMSE", ".2e"), ("var_mean_abs_rel_error", "VaR mean abs. rel. error", ".4f"),
    ("analytic_es", "Analytic ES", ".5f"), ("mean_mc_es", "Mean MC ES", ".5f"),
    ("es_rmse", "ES RMSE", ".2e"), ("es_mean_abs_rel_error", "ES mean abs. rel. error", ".4f"),
]

FIGURES = [
    ("portfolio_returns.png", "Daily portfolio returns with predefined stress periods shaded."),
    ("asset_correlations.png", "Full-sample asset correlations (descriptive)."),
    ("return_distribution.png", "Full-sample return distribution with normal and Student-t fits (descriptive)."),
    ("var_95_comparison.png", "95% VaR forecasts against realised losses; exceptions marked."),
    ("var_99_comparison.png", "99% VaR forecasts against realised losses; exceptions marked."),
    ("var_es_comparison.png", "97.5% VaR and ES forecasts against realised losses."),
    ("rolling_exception_rates.png", "Rolling exception rates against nominal rates."),
    ("stress_period_detail.png", "99% VaR around the predefined stress periods."),
    ("traffic_light_history.png", "Rolling 250-day traffic-light zones for 99% VaR (educational)."),
    ("mc_convergence.png", "Monte Carlo convergence towards the analytical normal benchmark."),
]


def _coverage_sentences(cov: pd.DataFrame, alpha: float) -> list[str]:
    out = []
    for name, col in (("Kupiec unconditional coverage", "uc_p_value"),
                      ("Christoffersen independence", "ind_p_value"),
                      ("Christoffersen conditional coverage", "cc_p_value")):
        rej = cov[cov[col] < alpha]
        undefined = cov[cov[col].isna()]
        tested = len(cov) - len(undefined)
        if rej.empty:
            s = f"- {name}: no rejection at the {alpha:.0%} level in {tested} defined test(s)."
        else:
            cases = ", ".join(f"{MODEL_LABEL.get(r.model, r.model)} at {r.confidence:.1%}" for r in rej.itertuples())
            s = f"- {name}: rejected at the {alpha:.0%} level in {len(rej)} of {tested} defined test(s): {cases}."
        if len(undefined):
            s += f" {len(undefined)} case(s) undefined/degenerate (see status columns; no p-value reported)."
        out.append(s)
    return out


def _fz0_sentences(es: pd.DataFrame) -> list[str]:
    out = []
    for c, g in es.groupby("confidence"):
        g = g.dropna(subset=["mean_fz0_score"]).sort_values("mean_fz0_score")
        if g.empty:
            continue
        order = " < ".join(f"{MODEL_LABEL.get(m, m)} ({s:.4f})" for m, s in zip(g["model"], g["mean_fz0_score"]))
        out.append(f"- {c:.1%}: {order}")
    return out


def build_report(results_dir: Path) -> str:
    """Assemble the research report Markdown from saved outputs in ``results_dir``."""
    rd = Path(results_dir)
    meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
    summ = pd.read_csv(rd / "summary.csv")
    cov = pd.read_csv(rd / "coverage_tests.csv")
    es = pd.read_csv(rd / "es_evaluation.csv")
    stress = pd.read_csv(rd / "stress_summary.csv")
    tl = pd.read_csv(rd / "traffic_light_windows.csv")
    mc_path = rd / "mc_convergence.csv"
    mc = pd.read_csv(mc_path) if mc_path.exists() else None
    cfg = meta["configuration"]
    alpha = float(cfg.get("backtest", {}).get("significance", 0.05))
    synthetic = cfg["data"]["source"] == "synthetic"
    ds = meta["dataset"]
    counts = meta["forecast_counts"]

    L: list[str] = []
    L.append(f"# Research report: {cfg['run_name']}")
    L.append("")
    L.append(f"*Generated from saved outputs in this folder. Backtest executed (UTC): {meta['execution_timestamp_utc']}. "
             f"Report generated (UTC): {utc_timestamp()}.*")
    L.append("")
    if synthetic:
        L.append("> **All results below are computed on synthetic data.** They describe model behaviour under a "
                 "known, stylised data-generating process and are not evidence about real equity markets.")
        L.append("")
    L.append("## 1. Overview")
    L.append("")
    L.append("Research question: how do historical simulation, parametric normal, parametric Student-t and "
             "correlated multivariate-normal Monte Carlo compare when forecasting one-day portfolio VaR and ES "
             "out of sample? VaR is assessed with formal coverage and independence tests; ES is assessed with the "
             "FZ0 joint VaR/ES score and descriptive calibration diagnostics (no formal ES hypothesis test).")
    L.append("")
    L.append("## 2. Configuration")
    L.append("")
    w = cfg["portfolio"]["weights"]
    L.append(f"- Data source: `{cfg['data']['source']}`; file `{ds['asset_returns_path']}` "
             f"(SHA-256 `{ds['asset_returns_sha256'][:16]}...`).")
    L.append(f"- Observations used: {ds['n_obs']:,} ({ds['first_date']} to {ds['last_date']}); "
             f"dates excluded by the complete-case policy: {ds['n_excluded_dates']}.")
    L.append("- Weights: " + ", ".join(f"{k} {v:.2f}" for k, v in w.items()) + "; long-only, daily rebalanced.")
    L.append(f"- Reporting notional: GBP {cfg['portfolio']['notional_gbp']:,.0f} (constant-notional comparison).")
    L.append(f"- Rolling estimation window: {cfg['forecast']['window']} observations; refit at every origin; "
             f"one-day horizon; confidence levels {', '.join(f'{c:.1%}' for c in cfg['forecast']['confidence_levels'])}.")
    L.append(f"- Monte Carlo: {cfg['monte_carlo']['n_scenarios']:,} scenarios per origin; master seed "
             f"{cfg['monte_carlo']['master_seed']}; per-origin seeds from SeedSequence(master, YYYYMMDD, model id).")
    L.append(f"- Forecast dates: {counts['n_forecast_dates']:,} ({counts['first_forecast_date']} to "
             f"{counts['last_forecast_date']}); common valid sample: {counts['n_common_valid_dates']:,} dates.")
    if ds.get("stress_intervals"):
        L.append("- Predefined stress intervals: " + "; ".join(f"{a} to {b}" for a, b in ds["stress_intervals"]) + ".")
    L.append("")
    L.append("## 3. Methods (summary)")
    L.append("")
    L.append("See `docs/methodology.md` for formulas. Historical simulation applies the empirical engine to the "
             "trailing window; the normal model uses the sample mean and SD (ddof=1); the Student-t model is a "
             "constrained MLE (nu in [2.01, upper bound]); Monte Carlo simulates correlated MVN asset returns from "
             "the window's mean vector and covariance. For a linear portfolio, MVN Monte Carlo and the analytical "
             "normal model target the same distribution, so their differences are sampling error.")
    L.append("")
    L.append("## 4. Results")
    L.append("")
    L.append("### 4.1 VaR summary (common valid sample)")
    L.append("")
    L.append(markdown_table(_label(summ), SUMMARY_COLUMNS))
    L.append("")
    L.append("### 4.2 VaR coverage and independence tests")
    L.append("")
    L.append(markdown_table(_label(cov), COVERAGE_COLUMNS))
    L.append("")
    L.append(f"Automated reading at the {alpha:.0%} significance level (asymptotic chi-square p-values):")
    L.append("")
    L.extend(_coverage_sentences(cov, alpha))
    L.append("")
    L.append("Non-rejection does not prove a model correct; at 99% the expected exception count is small, so the "
             "tests have low power. Fewer exceptions than expected is not automatically better.")
    L.append("")
    L.append("### 4.3 ES evaluation (joint score and descriptive diagnostics)")
    L.append("")
    L.append(markdown_table(_label(es), ES_COLUMNS))
    L.append("")
    L.append("Mean FZ0 score by confidence level, lowest (best) first. Differences have not been tested for "
             "statistical significance:")
    L.append("")
    L.extend(_fz0_sentences(es))
    L.append("")
    L.append("The ES identification residual has mean zero for correctly specified continuous forecasts; positive "
             "values mean realised tail losses exceeded forecast ES on average. The SE column is a descriptive "
             "i.i.d. standard error and is not used for a formal test.")
    L.append("")
    L.append("### 4.4 Stress versus non-stress periods")
    L.append("")
    L.append(markdown_table(_label(stress), STRESS_COLUMNS))
    L.append("")
    L.append("Stress periods are short; rates and scores there are descriptive and no p-values are reported.")
    L.append("")
    L.append("### 4.5 Basel-style traffic lights (99% VaR, educational)")
    L.append("")
    if not tl.empty:
        tlc = tl.groupby(["model", "zone"]).size().unstack(fill_value=0)
        for z in ("green", "amber", "red", "unavailable"):
            if z not in tlc:
                tlc[z] = 0
        tlc = tlc[["green", "amber", "red", "unavailable"]].reset_index()
        order = {m: i for i, m in enumerate(MODEL_LABEL)}
        tlc = tlc.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
        L.append(markdown_table(_label(tlc), [("model", "Model", "s"), ("green", "Green windows", "d"),
                                              ("amber", "Amber windows", "d"), ("red", "Red windows", "d"),
                                              ("unavailable", "Unavailable", "d")]))
    else:
        L.append("No complete 250-observation window was available.")
    L.append("")
    L.append("Zones count exceptions in overlapping 250-observation windows; they are not p-values, do not apply "
             "to ES or other confidence levels, and imply no regulatory approval or capital consequence.")
    L.append("")
    L.append("### 4.6 Fit diagnostics and failures")
    L.append("")
    fd = meta.get("fit_diagnostics", {})
    for model, st in fd.get("fit_status_counts", {}).items():
        L.append(f"- {MODEL_LABEL.get(model, model)}: " + ", ".join(f"{k} = {v:,}" for k, v in st.items()))
    if "student_t" in fd:
        t = fd["student_t"]
        L.append(f"- Student-t nu: median {t['nu_median']:.2f}, range {t['nu_min']:.2f} to {t['nu_max']:.2f}; "
                 f"upper-bound hits {t['n_upper_nu_bound']}, lower-bound hits {t['n_lower_nu_bound']}.")
    if "monte_carlo" in fd:
        m = fd["monte_carlo"]
        L.append(f"- Monte Carlo covariance repairs: {m['n_cov_repaired']}; simulated asset returns <= -100% "
                 f"(not clipped): {m['total_asset_draws_below_minus_one']}.")
    L.append(f"- Invalid forecast rows excluded from the common sample: {counts['n_invalid_forecast_rows']}.")
    L.append("")
    if mc is not None:
        L.append("### 4.7 Monte Carlo convergence")
        L.append("")
        conv = meta.get("mc_convergence", {})
        if conv:
            L.append(f"Origin: forecast date {conv['origin_date']} (training {conv['train_start']} to "
                     f"{conv['train_end']}); seeds: {len(conv['seeds'])}.")
            L.append("")
        L.append(markdown_table(_label(mc), MC_COLUMNS))
        L.append("")
    L.append("## 5. Figures")
    L.append("")
    for fname, cap in FIGURES:
        if (rd / "figures" / fname).exists():
            L.append(f"![{cap}](figures/{fname})")
            L.append("")
            L.append(f"*{cap}*")
            L.append("")
    L.append("## 6. Limitations")
    L.append("")
    for s in [
        "Synthetic data: conclusions concern this stylised process only and do not establish real-market superiority.",
        "Unconditional rolling-window models: none of the models has conditional volatility dynamics; GARCH/FHS is out of scope.",
        "Overlapping estimation windows and estimation error mean the standard asymptotic test distributions are approximations.",
        "Low test power at 99% and in the short stress intervals.",
        "ES is evaluated by a joint score and descriptive diagnostics only; no formal ES backtest is implemented.",
        "Traffic lights are an educational simplification; no regulatory capital or compliance claim is made.",
        "The normal and Monte Carlo models can assign probability to returns below -100%; draws are not clipped.",
    ]:
        L.append(f"- {s}")
    L.append("")
    return "\n".join(L)
