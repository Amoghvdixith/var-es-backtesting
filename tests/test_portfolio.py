"""Portfolio return construction and loss convention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_es.portfolio import portfolio_moments, portfolio_returns, returns_to_losses, to_gbp
from var_es.validation import DataValidationError, validate_weights


def _rets():
    return pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.00]}, index=pd.date_range("2020-01-01", periods=2))


def test_hand_calculated_portfolio_return():
    p = portfolio_returns(_rets(), {"A": 0.6, "B": 0.4})
    assert p.tolist() == pytest.approx([0.6 * 0.01 + 0.4 * 0.03, 0.6 * -0.02])


def test_weight_alignment_by_name_not_position():
    p1 = portfolio_returns(_rets(), {"A": 0.6, "B": 0.4})
    p2 = portfolio_returns(_rets(), {"B": 0.4, "A": 0.6})
    assert p1.tolist() == pytest.approx(p2.tolist())


@pytest.mark.parametrize(
    "w,msg",
    [
        ({"A": 0.6}, "align"),
        ({"A": 0.6, "B": 0.4, "C": 0.0}, "align"),
        ({"A": 0.6, "B": 0.5}, "sum"),
        ({"A": float("nan"), "B": 0.4}, "finite"),
        ({"A": 1.2, "B": -0.2}, "negative"),
    ],
)
def test_invalid_weights(w, msg):
    with pytest.raises(DataValidationError, match=msg):
        validate_weights(w, ["A", "B"])


def test_short_weights_allowed_when_not_long_only():
    w = validate_weights({"A": 1.2, "B": -0.2}, ["A", "B"], long_only=False)
    assert w.tolist() == pytest.approx([1.2, -0.2])


def test_no_silent_reweighting_with_missing():
    r = _rets()
    r.iloc[0, 1] = np.nan
    with pytest.raises(DataValidationError, match="reweighting"):
        portfolio_returns(r, {"A": 0.6, "B": 0.4})


def test_loss_and_gbp_conventions():
    assert returns_to_losses(np.array([0.01, -0.02])).tolist() == pytest.approx([-0.01, 0.02])
    assert to_gbp(0.0123, 1_000_000) == pytest.approx(12_300.0)


def test_decimal_not_percent():
    # a 1% return stored as 0.01 is a GBP 10,000 gain (negative loss) on GBP 1m
    assert to_gbp(returns_to_losses(0.01), 1_000_000) == pytest.approx(-10_000.0)


def test_portfolio_moments_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3)) * [0.01, 0.02, 0.015]
    w = np.array([0.5, 0.3, 0.2])
    mu, sd = portfolio_moments(x.mean(0), np.cov(x, rowvar=False, ddof=1), w)
    assert mu == pytest.approx((x @ w).mean())
    assert sd == pytest.approx((x @ w).std(ddof=1))
