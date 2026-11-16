"""Data loading, schema validation, missing-data policy and synthetic generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_es.data import load_returns_csv, percent_to_decimal, prices_to_simple_returns
from var_es.synthetic import generate_synthetic_returns, regime_schedule, truncated_invgamma_kappa
from var_es.validation import DataValidationError, validate_dates


def _write(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_returns_valid(tmp_path):
    p = _write(tmp_path, "date,A,B\n2020-01-01,0.01,-0.02\n2020-01-02,0.0,0.03\n")
    d = load_returns_csv(p)
    assert d.assets == ["A", "B"]
    assert d.returns.loc["2020-01-01", "A"] == pytest.approx(0.01)
    assert d.excluded_dates == []


def test_path_with_spaces(tmp_path):
    folder = tmp_path / "folder with spaces"
    folder.mkdir()
    p = _write(folder, "date,A\n2020-01-01,0.01\n", name="my returns.csv")
    assert len(load_returns_csv(p).returns) == 1


def test_duplicate_dates_rejected(tmp_path):
    p = _write(tmp_path, "date,A\n2020-01-01,0.01\n2020-01-01,0.02\n")
    with pytest.raises(DataValidationError, match="unique"):
        load_returns_csv(p)


def test_unsorted_dates_rejected():
    with pytest.raises(DataValidationError, match="increasing"):
        validate_dates(pd.Series(["2020-01-02", "2020-01-01"]))


def test_duplicate_asset_columns_rejected(tmp_path):
    p = _write(tmp_path, "date,A,A\n2020-01-01,0.01,0.02\n")
    with pytest.raises(DataValidationError, match="duplicate"):
        load_returns_csv(p)


def test_returns_le_minus_one_rejected(tmp_path):
    p = _write(tmp_path, "date,A\n2020-01-01,-1.0\n")
    with pytest.raises(DataValidationError, match="-1"):
        load_returns_csv(p)


def test_infinite_rejected(tmp_path):
    p = _write(tmp_path, "date,A\n2020-01-01,inf\n")
    with pytest.raises(DataValidationError):
        load_returns_csv(p)


def test_invalid_input_mode_rejected(tmp_path):
    p = _write(tmp_path, "date,A\n2020-01-01,0.01\n")
    with pytest.raises(DataValidationError, match="input_mode"):
        load_returns_csv(p, input_mode="auto")


def test_missing_values_complete_case_no_ffill(tmp_path):
    p = _write(tmp_path, "date,A,B\n2020-01-01,0.01,0.02\n2020-01-02,,0.03\n2020-01-03,0.04,0.05\n")
    d = load_returns_csv(p)
    assert d.excluded_dates == ["2020-01-02"]
    assert len(d.returns) == 2
    # the remaining value on 2020-01-03 is untouched (not filled from earlier rows)
    assert d.returns.loc["2020-01-03", "A"] == pytest.approx(0.04)


def test_percent_to_decimal_explicit(tmp_path):
    assert percent_to_decimal(np.array([1.0]))[0] == pytest.approx(0.01)
    p = _write(tmp_path, "date,A\n2020-01-01,1.5\n")
    assert load_returns_csv(p, units="percent").returns.iloc[0, 0] == pytest.approx(0.015)
    assert load_returns_csv(p, units="decimal").returns.iloc[0, 0] == pytest.approx(1.5)


def test_prices_to_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.date_range("2020-01-01", periods=3))
    r = prices_to_simple_returns(prices)
    assert r["A"].tolist() == pytest.approx([0.10, -0.10])


def test_prices_missing_creates_no_multiday_return(tmp_path):
    p = _write(tmp_path, "date,A\n2020-01-01,100\n2020-01-02,\n2020-01-03,121\n2020-01-06,133.1\n")
    d = load_returns_csv(p, input_mode="prices")
    assert d.excluded_dates == ["2020-01-02", "2020-01-03"]
    assert d.returns["A"].tolist() == pytest.approx([0.10])


def test_metadata_columns_kept_out_of_assets(tmp_path):
    p = _write(tmp_path, "date,A,is_stress,regime\n2020-01-01,0.01,False,normal\n")
    d = load_returns_csv(p)
    assert d.assets == ["A"]
    assert list(d.metadata.columns) == ["is_stress", "regime"]


def test_kappa_normalisation():
    kappa, mass = truncated_invgamma_kappa(4.0, 25.0)
    assert 0.99 < mass < 1.0
    assert 1.0 < kappa < 2.0  # untruncated mean is nu/(nu-2) = 2; truncation lowers it


def test_regime_schedule_validation():
    with pytest.raises(DataValidationError, match="cover"):
        regime_schedule([{"label": "a", "start": 0, "end": 3, "multiplier": 1}], 5)
    with pytest.raises(DataValidationError, match="overlap"):
        regime_schedule(
            [
                {"label": "a", "start": 0, "end": 3, "multiplier": 1},
                {"label": "b", "start": 3, "end": 4, "multiplier": 1},
            ],
            5,
        )


def test_synthetic_deterministic_and_schema(baseline_config):
    cfg = dict(baseline_config["synthetic"])
    df1, meta = generate_synthetic_returns(cfg)
    df2, _ = generate_synthetic_returns(cfg)
    pd.testing.assert_frame_equal(df1, df2)
    assert list(df1.columns) == ["date", *cfg["assets"], "is_stress", "regime"]
    assert len(df1) == 2000
    vals = df1[cfg["assets"]].to_numpy()
    assert np.all(vals > -1) and np.all(np.isfinite(vals))
    stress = np.flatnonzero(df1["is_stress"].to_numpy())
    assert stress.tolist() == list(range(900, 1000)) + list(range(1500, 1600))
    assert meta["stress_intervals_index_inclusive"] == [[900, 999], [1500, 1599]]
    assert pd.to_datetime(df1["date"]).is_monotonic_increasing
    c = np.corrcoef(vals, rowvar=False)
    assert np.all(c[np.triu_indices(5, 1)] > 0.2)  # moderate positive cross-correlation
