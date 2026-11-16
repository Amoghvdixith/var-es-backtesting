"""Out-of-sample timing, no look-ahead, failure handling and common-sample alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_es.forecasting import attach_realised, common_valid_dates, is_valid, rolling_forecasts
from var_es.metrics import coverage_table, evaluate_forecasts
from var_es.portfolio import portfolio_returns

ASSETS = ["A", "B", "C"]
WEIGHTS = {"A": 0.5, "B": 0.3, "C": 0.2}
W = np.array([0.5, 0.3, 0.2])
LEVELS = [0.95, 0.99]
MODELS = ["historical", "normal", "student_t", "monte_carlo"]


def _data(n=140, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_t(5, size=(n, 3)) + 0.0002
    return pd.DataFrame(x, columns=ASSETS, index=pd.bdate_range("2021-01-04", periods=n))


def _run(df, window=60):
    return rolling_forecasts(df, W, window=window, confidence_levels=LEVELS, models=MODELS,
                             mc_master_seed=99, mc_scenarios=2000)


def test_training_window_strictly_before_forecast_date():
    df = _data()
    fc, _ = _run(df)
    assert (pd.to_datetime(fc["train_end"]) < pd.to_datetime(fc["forecast_date"])).all()
    dates = df.index
    for _, r in fc.drop_duplicates("obs_index").iterrows():
        t = r["obs_index"]
        assert r["train_start"] == dates[t - 60]
        assert r["train_end"] == dates[t - 1]
        assert r["forecast_date"] == dates[t]
    assert fc["obs_index"].min() == 60 and fc["obs_index"].max() == len(df) - 1
    assert len(fc) == (len(df) - 60) * len(MODELS) * len(LEVELS)


def test_changing_future_does_not_change_earlier_forecasts():
    df = _data()
    base, _ = _run(df)
    k = 100
    shocked = df.copy()
    shocked.iloc[k:] = shocked.iloc[k:] * 5 - 0.03  # alter observation k (the realised day) and everything after
    new, _ = _run(shocked)
    keep = base["obs_index"] <= k  # forecasts for dates <= k use data through k-1 only
    cols = ["var", "es", "forecast_status"]
    pd.testing.assert_frame_equal(base.loc[keep, cols].reset_index(drop=True), new.loc[keep, cols].reset_index(drop=True))
    # and a later forecast does change (sanity check that the test can detect differences)
    assert not np.allclose(base.loc[base["obs_index"] == k + 1, "var"], new.loc[new["obs_index"] == k + 1, "var"])


def test_realised_loss_attached_after_forecast():
    df = _data()
    fc, _ = _run(df)
    port = portfolio_returns(df, WEIGHTS)
    ev = attach_realised(fc, port, notional_gbp=1_000_000)
    r = ev.iloc[0]
    assert r["realized_return"] == pytest.approx(port.loc[r["forecast_date"]])
    assert r["realized_loss"] == pytest.approx(-port.loc[r["forecast_date"]])
    assert r["var_gbp"] == pytest.approx(r["var"] * 1_000_000)


def test_fit_failure_recorded_not_substituted():
    df = _data(n=90)
    df.iloc[:60] = 0.001  # constant first window: t fit must fail at the first origin
    fc, diag = _run(df)
    first = fc[fc["obs_index"] == 60]
    t_rows = first[first["model"] == "student_t"]
    assert (t_rows["forecast_status"] == "fit_failed").all()
    assert t_rows["var"].isna().all() and t_rows["es"].isna().all()
    n_rows = first[first["model"] == "normal"]
    assert (n_rows["forecast_status"] == "ok").all()
    assert n_rows["var"].to_numpy() == pytest.approx([-0.001, -0.001])  # zero volatility: VaR = -mu
    assert diag.loc[(diag["obs_index"] == 60) & (diag["model"] == "normal"), "fit_status"].item() == "zero_volatility"


def test_common_sample_excludes_failed_dates_and_no_artificial_transitions():
    df = _data(n=100)
    fc, _ = _run(df)
    port = portfolio_returns(df, WEIGHTS)
    ev = attach_realised(fc, port, notional_gbp=1e6)
    bad_pos = 75
    m = (ev["obs_index"] == bad_pos) & (ev["model"] == "student_t")
    ev.loc[m, ["var", "es"]] = np.nan
    ev.loc[m, "forecast_status"] = "fit_failed"
    common, excl = common_valid_dates(ev)
    assert ev.loc[ev["obs_index"] == bad_pos, "forecast_date"].iloc[0] not in common
    assert len(common) == ev["forecast_date"].nunique() - 1
    assert excl["n_invalid_rows"].sum() == len(LEVELS)
    assert (~is_valid(ev)).sum() == len(LEVELS)
    evd = evaluate_forecasts(ev)
    cov = coverage_table(evd)
    n_common = len(common)
    # one gap in the common sample -> one fewer transition than (n - 1)
    assert (cov["n_transitions"] == n_common - 2).all()
    assert (cov["n_forecasts"] == n_common).all()


def test_mc_forecasts_reproducible():
    df = _data(n=80)
    a, _ = _run(df)
    b, _ = _run(df)
    pd.testing.assert_frame_equal(a, b)
