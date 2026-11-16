"""Monte Carlo convergence experiment at one predefined forecast origin.

Compares MVN Monte Carlo VaR/ES for several scenario counts and fixed seeds with the
analytical normal VaR/ES implied by the same estimation window.

Example (from the project root)::

    python scripts/run_mc_convergence.py --config configs/synthetic_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import var_es  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from var_es import plotting  # noqa: E402
from var_es.data import load_config, load_returns_csv  # noqa: E402
from var_es.monte_carlo import convergence_experiment, summarise_convergence  # noqa: E402
from var_es.reporting import utc_timestamp, write_json  # noqa: E402
from var_es.validation import validate_weights  # noqa: E402


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else root / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Monte Carlo VaR/ES convergence versus the analytical normal benchmark.")
    ap.add_argument("--config", required=True, help="Path to the JSON run configuration.")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="Root for relative paths (default: project root).")
    ap.add_argument("--no-figures", action="store_true", help="Skip the convergence figure.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    cfg = load_config(resolve(args.config, root))
    conv = cfg["mc_convergence"]
    window = int(cfg["forecast"]["window"])
    data = load_returns_csv(root / cfg["data"]["asset_returns_path"], input_mode=cfg["data"]["input_mode"],
                            units=cfg["data"]["units"], assets=list(cfg["portfolio"]["weights"].keys()))
    if conv.get("origin", "first_forecast") != "first_forecast":
        raise SystemExit("only origin='first_forecast' is supported (predefined before results)")
    t = window  # first out-of-sample forecast position
    train = data.returns.iloc[t - window : t]
    w = validate_weights(cfg["portfolio"]["weights"], data.assets)

    raw = convergence_experiment(train.to_numpy(), w, conv["confidence_levels"], conv["scenario_counts"], conv["seeds"])
    summ = summarise_convergence(raw)
    out_dir = root / cfg["output_dir"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / "mc_convergence.csv", index=False, float_format="%.10g")
    raw.to_csv(out_dir / "mc_convergence_raw.csv", index=False, float_format="%.10g")

    meta_path = out_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["mc_convergence"] = {
        "executed_utc": utc_timestamp(),
        "origin_position": t,
        "origin_date": data.returns.index[t].strftime("%Y-%m-%d"),
        "train_start": train.index[0].strftime("%Y-%m-%d"),
        "train_end": train.index[-1].strftime("%Y-%m-%d"),
        "scenario_counts": conv["scenario_counts"],
        "seeds": conv["seeds"],
        "seed_derivation": "numpy.random.SeedSequence([seed, n_scenarios])",
        "benchmark": "analytical normal with mu_p = w'mu_hat, sigma_p = sqrt(w' Sigma_hat w) from the same window",
    }
    write_json(meta, meta_path)
    if not args.no_figures:
        plotting.plot_mc_convergence(raw, summ, out_dir / "figures" / "mc_convergence.png",
                                     cfg["data"]["source"] == "synthetic")
    print(summ[["n_scenarios", "confidence", "analytic_var", "var_rmse", "analytic_es", "es_rmse"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
