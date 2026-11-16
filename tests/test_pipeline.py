"""Small end-to-end smoke test of the command-line workflow (tiny configuration)."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from var_es.metrics import FORECAST_COLUMNS
from var_es.reporting import SUMMARY_COLUMNS, _label, markdown_table
from var_es.synthetic import file_sha256


@pytest.fixture(scope="module")
def pipeline_root(tmp_path_factory, tiny_config, project_root):
    root = tmp_path_factory.mktemp("pipeline") / "root with spaces"
    (root / "configs").mkdir(parents=True)
    cfg_path = root / "configs" / "tiny.json"
    cfg_path.write_text(json.dumps(tiny_config), encoding="utf-8")

    def run(script, *args):
        cmd = [sys.executable, str(project_root / "scripts" / script), *args]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=root, timeout=600)
        assert res.returncode == 0, res.stderr
        return res

    run("generate_data.py", "--config", str(cfg_path), "--root", str(root))
    sha1 = file_sha256(root / "data" / "asset_returns.csv")
    run("generate_data.py", "--config", str(cfg_path), "--root", str(root))
    assert file_sha256(root / "data" / "asset_returns.csv") == sha1  # deterministic data
    run("run_backtest.py", "--config", str(cfg_path), "--root", str(root), "--quiet")
    run("run_mc_convergence.py", "--config", str(cfg_path), "--root", str(root))
    run("generate_report.py", "--results-dir", str(root / "results" / "pytest_tiny"))
    return root, root / "results" / "pytest_tiny"


def test_pipeline_outputs_and_schemas(pipeline_root):
    root, rd = pipeline_root
    for f in ["forecasts.csv", "summary.csv", "coverage_tests.csv", "es_evaluation.csv", "stress_summary.csv",
              "traffic_light_windows.csv", "fit_diagnostics.csv", "mc_convergence.csv", "run_metadata.json",
              "research_report.md"]:
        assert (rd / f).exists(), f
    for f in ["portfolio_returns.png", "var_99_comparison.png", "traffic_light_history.png", "mc_convergence.png"]:
        assert (rd / "figures" / f).stat().st_size > 1000

    fc = pd.read_csv(rd / "forecasts.csv")
    assert list(fc.columns) == FORECAST_COLUMNS
    assert len(fc) == (360 - 100) * 4 * 3
    assert (pd.to_datetime(fc["train_end"]) < pd.to_datetime(fc["forecast_date"])).all()
    ok = fc["forecast_status"] == "ok"
    assert (fc.loc[ok, "es"] >= fc.loc[ok, "var"]).all()

    summ = pd.read_csv(rd / "summary.csv")
    assert set(summ["model"]) == {"historical", "normal", "student_t", "monte_carlo"}
    # summary exception counts agree with forecasts.csv
    common = fc[fc["in_common_sample"]]
    for r in summ.itertuples():
        g = common[(common["model"] == r.model) & (common["confidence"] == r.confidence)]
        assert int(g["exception"].astype(bool).sum()) == r.n_exceptions

    cov = pd.read_csv(rd / "coverage_tests.csv")
    p = cov["uc_p_value"].dropna()
    assert ((p >= 0) & (p <= 1)).all()

    meta_text = (rd / "run_metadata.json").read_text(encoding="utf-8")
    assert str(root) not in meta_text and str(root).replace("\\", "\\\\") not in meta_text
    meta = json.loads(meta_text)
    for key in ("configuration", "seeds", "source_checksum_sha256", "dataset", "environment",
                "execution_timestamp_utc", "forecast_counts", "mc_convergence"):
        assert key in meta


def test_report_values_match_saved_results(pipeline_root):
    _, rd = pipeline_root
    report = (rd / "research_report.md").read_text(encoding="utf-8")
    summ = pd.read_csv(rd / "summary.csv")
    for i in range(len(summ)):
        row_md = markdown_table(_label(summ.iloc[[i]]), SUMMARY_COLUMNS).splitlines()[-1]
        assert row_md in report
    assert "synthetic data" in report.lower()
    assert "](figures/var_99_comparison.png)" in report


def test_user_price_data_pipeline(tmp_path, tiny_config, project_root):
    """User-supplied adjusted prices with an explicit input mode and date-based stress intervals."""
    import numpy as np

    root = tmp_path / "user root"
    (root / "data" / "user").mkdir(parents=True)
    rng = np.random.default_rng(5)
    n = 300
    rets = 0.01 * rng.standard_normal((n, 3))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    dates = pd.bdate_range("2019-01-01", periods=n)
    df = pd.DataFrame(prices, columns=["AAA", "BBB", "CCC"])
    df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    df.loc[150, "BBB"] = np.nan  # one missing price -> two excluded return dates
    df.to_csv(root / "data" / "user" / "prices.csv", index=False)
    cfg = json.loads((project_root / "configs" / "user_data_example.json").read_text(encoding="utf-8"))
    cfg["data"].update(asset_returns_path="data/user/prices.csv", input_mode="prices", min_history=200)
    cfg["portfolio"]["weights"] = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    cfg["forecast"]["window"] = 100
    cfg["monte_carlo"]["n_scenarios"] = 1000
    cfg["mc_convergence"].update(scenario_counts=[500, 1000], seeds=[1, 2])
    cfg["stress_intervals"] = [{"start_date": "2019-09-02", "end_date": "2019-09-30"}]
    cfg["output_dir"] = "results/user_data"
    cfg_path = root / "user.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    for script, args in [
        ("run_backtest.py", ["--config", str(cfg_path), "--root", str(root), "--quiet", "--no-figures"]),
        ("run_mc_convergence.py", ["--config", str(cfg_path), "--root", str(root), "--no-figures"]),
        ("generate_report.py", ["--results-dir", str(root / "results" / "user_data")]),
    ]:
        res = subprocess.run([sys.executable, str(project_root / "scripts" / script), *args],
                             capture_output=True, text=True, cwd=root, timeout=600)
        assert res.returncode == 0, res.stderr
    meta = json.loads((root / "results" / "user_data" / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset"]["input_mode"] == "prices"
    assert meta["dataset"]["n_excluded_dates"] == 2
    assert meta["dataset"]["n_obs"] == n - 1 - 2
    assert meta["dataset"]["stress_intervals"]
    report = (root / "results" / "user_data" / "research_report.md").read_text(encoding="utf-8")
    assert "synthetic data.**" not in report
