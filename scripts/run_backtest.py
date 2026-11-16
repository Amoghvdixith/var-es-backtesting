"""Run rolling out-of-sample VaR/ES forecasts and all backtests; write CSVs, metadata and figures.

Example (from the project root)::

    python scripts/run_backtest.py --config configs/synthetic_baseline.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import var_es  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from var_es import plotting  # noqa: E402
from var_es.data import load_config, load_returns_csv, stress_labels_from_intervals  # noqa: E402
from var_es.forecasting import attach_realised, common_valid_dates, is_valid, rolling_forecasts  # noqa: E402
from var_es.metrics import (  # noqa: E402
    coverage_table,
    es_ge_var_check,
    es_table,
    evaluate_forecasts,
    fit_diagnostics_summary,
    rolling_rates,
    stress_table,
    summary_table,
    traffic_light_table,
)
from var_es.portfolio import portfolio_returns  # noqa: E402
from var_es.reporting import (  # noqa: E402
    EMPIRICAL_CONVENTION,
    RETURN_LOSS_DEFINITIONS,
    environment_info,
    source_checksum,
    utc_timestamp,
    write_json,
)
from var_es.synthetic import file_sha256  # noqa: E402
from var_es.validation import DataValidationError, validate_min_history, validate_weights  # noqa: E402


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else root / p


def _intervals(dates: pd.DatetimeIndex, flag: np.ndarray) -> list[list[str]]:
    out, i = [], 0
    while i < flag.size:
        if flag[i]:
            j = i
            while j + 1 < flag.size and flag[j + 1]:
                j += 1
            out.append([dates[i].strftime("%Y-%m-%d"), dates[j].strftime("%Y-%m-%d")])
            i = j + 1
        else:
            i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rolling out-of-sample VaR/ES forecasts and backtests.")
    ap.add_argument("--config", required=True, help="Path to the JSON run configuration.")
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="Root directory for relative data/output paths in the config (default: project root).")
    ap.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    cfg = load_config(resolve(args.config, root))
    dcfg, pcfg, fcfg = cfg["data"], cfg["portfolio"], cfg["forecast"]
    out_dir = root / cfg["output_dir"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    say = (lambda *a: None) if args.quiet else print

    # ---- data ------------------------------------------------------------------
    asset_path = root / dcfg["asset_returns_path"]
    if not asset_path.exists():
        hint = " Run scripts/generate_data.py first." if dcfg["source"] == "synthetic" else ""
        raise SystemExit(f"Input file not found: {dcfg['asset_returns_path']}.{hint}")
    data = load_returns_csv(asset_path, date_column=dcfg.get("date_column", "date"),
                            input_mode=dcfg["input_mode"], units=dcfg["units"],
                            assets=list(pcfg["weights"].keys()))
    window = int(fcfg["window"])
    validate_min_history(len(data.returns), max(window + 1, int(dcfg.get("min_history", 0))))
    w = validate_weights(pcfg["weights"], data.assets, long_only=pcfg.get("long_only", True))
    port = portfolio_returns(data.returns, pcfg["weights"], long_only=pcfg.get("long_only", True))

    port_check = None
    if dcfg["source"] == "synthetic" and (root / dcfg["portfolio_returns_path"]).exists():
        saved = pd.read_csv(root / dcfg["portfolio_returns_path"], parse_dates=["date"]).set_index("date")
        diff = float(np.max(np.abs(saved["portfolio_return"].reindex(port.index) - port)))
        if not diff < 1e-12:
            raise DataValidationError(f"saved portfolio returns disagree with recomputation (max diff {diff})")
        port_check = diff

    if "is_stress" in data.metadata:
        stress = data.metadata["is_stress"].astype(str).str.lower().eq("true")
    else:
        stress = stress_labels_from_intervals(data.returns.index, cfg.get("stress_intervals", []))
    regime = data.metadata["regime"] if "regime" in data.metadata else None

    # ---- forecasts ---------------------------------------------------------------
    say(f"Forecasting {len(data.returns) - window} dates x {len(fcfg['models'])} models ...")
    t0 = time.perf_counter()
    fc, diag = rolling_forecasts(
        data.returns, w, window=window, confidence_levels=fcfg["confidence_levels"], models=fcfg["models"],
        mc_master_seed=int(cfg["monte_carlo"]["master_seed"]), mc_scenarios=int(cfg["monte_carlo"]["n_scenarios"]),
        t_nu_upper=float(cfg.get("student_t", {}).get("nu_upper", 1000.0)),
        t_init_nu=float(cfg.get("student_t", {}).get("init_nu", 5.0)), progress=not args.quiet,
    )
    runtime = time.perf_counter() - t0
    say(f"Forecasting finished in {runtime:.1f} s")
    realised = attach_realised(fc, port, notional_gbp=float(pcfg["notional_gbp"]), stress=stress, regime=regime)
    ev = evaluate_forecasts(realised)
    common, excl = common_valid_dates(ev)

    # ---- tables ------------------------------------------------------------------
    summ, cov, est, st = summary_table(ev), coverage_table(ev), es_table(ev), stress_table(ev)
    tl = traffic_light_table(ev)
    rates = rolling_rates(ev, int(cfg["backtest"]["rolling_window"]))

    fdate = lambda df, cols: df.assign(**{c: pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d") for c in cols})  # noqa: E731
    fdate(ev, ["forecast_date", "train_start", "train_end"]).to_csv(out_dir / "forecasts.csv", index=False,
                                                                     float_format="%.10g")
    summ.to_csv(out_dir / "summary.csv", index=False, float_format="%.10g")
    cov.to_csv(out_dir / "coverage_tests.csv", index=False, float_format="%.10g")
    est.to_csv(out_dir / "es_evaluation.csv", index=False, float_format="%.10g")
    st.to_csv(out_dir / "stress_summary.csv", index=False, float_format="%.10g")
    fdate(tl, ["window_start", "window_end"]).to_csv(out_dir / "traffic_light_windows.csv", index=False)
    fdate(diag, ["forecast_date", "train_start", "train_end"]).to_csv(out_dir / "fit_diagnostics.csv", index=False,
                                                                       float_format="%.10g")

    # ---- metadata ---------------------------------------------------------------
    n_invalid = int((~is_valid(ev)).sum())
    meta = {
        "run_name": cfg["run_name"],
        "execution_timestamp_utc": utc_timestamp(),
        "configuration": cfg,
        "seeds": {
            "synthetic_data_seed": cfg.get("synthetic", {}).get("seed"),
            "monte_carlo_master_seed": cfg["monte_carlo"]["master_seed"],
            "monte_carlo_seed_derivation": "numpy.random.SeedSequence([master_seed, YYYYMMDD of forecast date, model_id=4])",
            "mc_convergence_seeds": cfg.get("mc_convergence", {}).get("seeds"),
        },
        "source_checksum_sha256": source_checksum(PROJECT_ROOT / "src" / "var_es"),
        "dataset": {
            "source": dcfg["source"],
            "asset_returns_path": dcfg["asset_returns_path"],
            "asset_returns_sha256": file_sha256(asset_path),
            "input_mode": data.input_mode,
            "input_units": data.units,
            "assets": data.assets,
            "n_obs": len(data.returns),
            "first_date": data.returns.index[0].strftime("%Y-%m-%d"),
            "last_date": data.returns.index[-1].strftime("%Y-%m-%d"),
            "n_excluded_dates": len(data.excluded_dates),
            "excluded_dates": data.excluded_dates,
            "missing_data_policy": "complete-case exclusion; no forward-fill; no reweighting",
            "stress_intervals": _intervals(data.returns.index, stress.to_numpy(dtype=bool)),
            "saved_vs_recomputed_portfolio_max_abs_diff": port_check,
        },
        "definitions": RETURN_LOSS_DEFINITIONS,
        "empirical_var_es_convention": EMPIRICAL_CONVENTION,
        "exception_definition": "realized_loss > VaR (strict)",
        "independence_adjacency_policy": "transitions counted only between consecutive dataset positions that both have valid forecasts",
        "environment": environment_info(),
        "forecast_runtime_seconds": round(runtime, 2),
        "forecast_counts": {
            "n_forecast_dates": int(ev["forecast_date"].nunique()),
            "first_forecast_date": str(ev["forecast_date"].min().date()),
            "last_forecast_date": str(ev["forecast_date"].max().date()),
            "n_forecast_rows": len(ev),
            "rows_by_model": {str(k): int(v) for k, v in ev["model"].value_counts().items()},
            "n_invalid_forecast_rows": n_invalid,
            "invalid_rows_by_model_confidence": excl.to_dict(orient="records"),
            "n_common_valid_dates": len(common),
            "n_valid_rows_with_es_below_var": es_ge_var_check(ev),
        },
        "fit_diagnostics": fit_diagnostics_summary(diag),
    }
    meta_path = out_dir / "run_metadata.json"
    if meta_path.exists():  # keep a previously recorded MC convergence section
        import json
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if "mc_convergence" in old:
            meta["mc_convergence"] = old["mc_convergence"]
    write_json(meta, meta_path)

    # ---- figures ----------------------------------------------------------------
    if not args.no_figures:
        figs = out_dir / "figures"
        synthetic = dcfg["source"] == "synthetic"
        pdf = pd.DataFrame({"date": port.index, "portfolio_return": port.to_numpy(),
                            "is_stress": stress.reindex(port.index).to_numpy(dtype=bool)})
        plotting.plot_portfolio_returns(pdf, ev["forecast_date"].min(), figs / "portfolio_returns.png", synthetic)
        plotting.plot_asset_correlations(data.returns, figs / "asset_correlations.png", synthetic)
        plotting.plot_return_distribution(pdf, figs / "return_distribution.png", synthetic)
        levels = [float(c) for c in fcfg["confidence_levels"]]
        if 0.95 in levels:
            plotting.plot_var_comparison(ev, 0.95, figs / "var_95_comparison.png", synthetic)
        if 0.99 in levels:
            plotting.plot_var_comparison(ev, 0.99, figs / "var_99_comparison.png", synthetic)
            plotting.plot_stress_detail(ev, 0.99, figs / "stress_period_detail.png", synthetic=synthetic)
            plotting.plot_traffic_light_history(tl, figs / "traffic_light_history.png", synthetic)
        if 0.975 in levels:
            plotting.plot_var_es_comparison(ev, 0.975, figs / "var_es_comparison.png", synthetic)
        plotting.plot_rolling_exception_rates(rates, ev, int(cfg["backtest"]["rolling_window"]),
                                              figs / "rolling_exception_rates.png", synthetic)

    say(f"Wrote results to {cfg['output_dir']}")
    say(summ[["model", "confidence", "n_forecasts", "n_exceptions", "expected_exceptions"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
