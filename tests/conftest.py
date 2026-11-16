"""Shared fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def baseline_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "synthetic_baseline.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tiny_config(baseline_config) -> dict:
    """Very small synthetic configuration for fast pipeline tests (not the baseline)."""
    cfg = copy.deepcopy(baseline_config)
    cfg["run_name"] = "pytest_tiny"
    s = cfg["synthetic"]
    s["n_obs"] = 360
    s["regimes"] = [
        {"label": "normal", "start": 0, "end": 179, "multiplier": 1.0},
        {"label": "stress", "start": 180, "end": 209, "multiplier": 2.5},
        {"label": "normal", "start": 210, "end": 359, "multiplier": 1.0},
    ]
    cfg["forecast"]["window"] = 100
    cfg["monte_carlo"]["n_scenarios"] = 2000
    cfg["mc_convergence"]["scenario_counts"] = [500, 2000]
    cfg["mc_convergence"]["seeds"] = [1, 2]
    cfg["data"]["asset_returns_path"] = "data/asset_returns.csv"
    cfg["data"]["portfolio_returns_path"] = "data/portfolio_returns.csv"
    cfg["data"]["metadata_path"] = "data/generation_metadata.json"
    cfg["output_dir"] = "results/pytest_tiny"
    return cfg
