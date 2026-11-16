"""Empirical VaR/ES convention."""

from __future__ import annotations

import numpy as np
import pytest

from var_es.empirical import empirical_es, empirical_var, empirical_var_es
from var_es.historical import historical_var_es


def test_integer_tail_size():
    x = np.arange(1, 101, dtype=float)  # 1..100
    # c=0.95: k = ceil(95) = 95 -> VaR = 95; ES = mean of top 5 = 98
    assert empirical_var(x, 0.95) == 95.0
    assert empirical_es(x, 0.95) == pytest.approx(98.0)
    assert empirical_var(x, 0.99) == 99.0
    assert empirical_es(x, 0.99) == pytest.approx(100.0)


def test_fractional_tail_mass():
    x = np.arange(1, 11, dtype=float)  # n=10
    # c=0.975 -> m = 0.25 of the largest observation
    assert empirical_var(x, 0.975) == 10.0
    assert empirical_es(x, 0.975) == pytest.approx(10.0)
    # c = 0.85 -> m = 1.5: ES = (10 + 0.5*9)/1.5
    assert empirical_var(x, 0.85) == 9.0  # ceil(8.5) = 9
    assert empirical_es(x, 0.85) == pytest.approx((10 + 0.5 * 9) / 1.5)


def test_fractional_with_n750():
    x = np.arange(1, 751, dtype=float)  # n = 750, c=0.99 -> m = 7.5
    var, es = empirical_var_es(x, [0.99])[0.99]
    assert var == 743.0  # ceil(742.5)
    expected = (sum(range(744, 751)) + 0.5 * 743) / 7.5
    assert es == pytest.approx(expected)


def test_ties_where_naive_mean_above_var_fails():
    x = np.array([0, 0, 0, 0, 0, 0, 0, 5, 5, 5], dtype=float)
    # c = 0.8: VaR = x_(8) = 5; ES = mean of top 2 observations = 5
    assert empirical_var(x, 0.8) == 5.0
    assert empirical_es(x, 0.8) == pytest.approx(5.0)
    # naive "mean of losses strictly greater than VaR" would be an empty set here
    assert not np.any(x > 5.0)
    # c = 0.75 with ties: m = 2.5 -> ES = (5 + 5 + 0.5*5)/2.5 = 5
    assert empirical_es(x, 0.75) == pytest.approx(5.0)
    # c = 0.65: m = 3.5 -> (5+5+5+0.5*0)/3.5
    assert empirical_es(x, 0.65) == pytest.approx(15 / 3.5)


def test_all_equal():
    x = np.full(37, 0.02)
    for c in (0.95, 0.975, 0.99):
        assert empirical_var(x, c) == pytest.approx(0.02)
        assert empirical_es(x, c) == pytest.approx(0.02)


def test_small_arrays():
    assert empirical_var([3.0], 0.99) == 3.0
    assert empirical_es([3.0], 0.99) == 3.0
    x = [1.0, 2.0]
    assert empirical_var(x, 0.95) == 2.0
    assert empirical_es(x, 0.95) == pytest.approx(2.0)
    assert empirical_var(x, 0.4) == 1.0  # ceil(0.8) = 1
    assert empirical_es(x, 0.4) == pytest.approx((2 + 0.2 * 1) / 1.2)  # m = 1.2


def test_order_invariance_and_es_ge_var():
    rng = np.random.default_rng(3)
    for n in (7, 50, 333, 750):
        x = rng.standard_t(3, size=n)
        for c in (0.9, 0.95, 0.975, 0.99):
            v, e = empirical_var_es(x, [c])[c]
            assert e >= v - 1e-15
            assert empirical_var(rng.permutation(x), c) == v


def test_negative_var_not_clipped():
    x = np.array([-0.05, -0.04, -0.03, -0.02])  # all gains
    assert empirical_var(x, 0.95) < 0
    assert empirical_es(x, 0.95) < 0


def test_historical_uses_losses():
    r = np.arange(-50, 50) / 1000.0
    out = historical_var_es(r, [0.95])
    assert out[0.95][0] == pytest.approx(empirical_var(-r, 0.95))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        empirical_var([], 0.95)
    with pytest.raises(ValueError):
        empirical_var([1.0], 1.0)
    with pytest.raises(ValueError):
        empirical_es([1.0, np.nan], 0.95)
