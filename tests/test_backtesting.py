"""VaR backtests: likelihood ratios, transitions, degenerate cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from var_es.backtesting import (
    christoffersen_cc,
    christoffersen_ind,
    exceptions,
    kupiec_uc,
    max_consecutive_exceptions,
    quantile_loss,
    rolling_exception_rate,
    transition_counts,
)


def test_strict_exceptions():
    assert exceptions(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])).tolist() == [False, False, True]


def test_kupiec_hand_calculation():
    n, x, p = 250, 5, 0.01
    pi = x / n
    l0 = (n - x) * math.log(1 - p) + x * math.log(p)
    l1 = (n - x) * math.log(1 - pi) + x * math.log(pi)
    lr = -2 * (l0 - l1)
    res = kupiec_uc(n, x, 0.99)
    assert res.statistic == pytest.approx(lr, rel=1e-12)
    assert res.p_value == pytest.approx(stats.chi2.sf(lr, 1), rel=1e-12)
    assert res.status == "ok" and res.df == 1


def test_kupiec_zero_and_all_exceptions():
    z = kupiec_uc(250, 0, 0.99)
    assert z.statistic == pytest.approx(-2 * 250 * math.log(0.99))
    a = kupiec_uc(20, 20, 0.95)
    assert a.statistic == pytest.approx(-2 * 20 * math.log(0.05))
    for r in (z, a):
        assert 0.0 <= r.p_value <= 1.0


def test_kupiec_exact_nominal_gives_zero():
    r = kupiec_uc(1000, 50, 0.95)
    assert r.statistic == pytest.approx(0.0, abs=1e-9)
    assert r.p_value == pytest.approx(1.0)


def test_kupiec_no_observations():
    r = kupiec_uc(0, 0, 0.99)
    assert r.status == "no_observations" and math.isnan(r.p_value)


def test_known_transition_table():
    hits = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
    assert transition_counts(hits) == {"n00": 2, "n01": 2, "n10": 2, "n11": 1}


def test_gaps_do_not_create_transitions():
    hits = np.array([1, 1, 0, 1, 1], dtype=bool)
    pos = np.array([0, 1, 2, 5, 6])
    tc = transition_counts(hits, pos)
    assert tc == {"n00": 0, "n01": 0, "n10": 1, "n11": 2}
    assert sum(tc.values()) == 3
    assert max_consecutive_exceptions(hits, pos) == 2
    assert max_consecutive_exceptions(np.array([1, 1, 1], dtype=bool), np.array([0, 1, 3])) == 2


def test_christoffersen_hand_calculation():
    tc = {"n00": 200, "n01": 8, "n10": 7, "n11": 3}
    n00, n01, n10, n11 = 200, 8, 7, 3
    pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    l0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    l1 = n00 * math.log(1 - pi01) + n01 * math.log(pi01) + n10 * math.log(1 - pi11) + n11 * math.log(pi11)
    lr = -2 * (l0 - l1)
    r = christoffersen_ind(tc)
    assert r.statistic == pytest.approx(lr, rel=1e-12)
    assert r.p_value == pytest.approx(stats.chi2.sf(lr, 1))
    uc = kupiec_uc(219, 11, 0.95)
    cc = christoffersen_cc(uc, r)
    assert cc.statistic == pytest.approx(uc.statistic + lr)
    assert cc.p_value == pytest.approx(stats.chi2.sf(uc.statistic + lr, 2))
    assert cc.df == 2


def test_christoffersen_n11_zero_is_defined():
    r = christoffersen_ind({"n00": 240, "n01": 4, "n10": 4, "n11": 0})
    assert r.status == "ok" and 0 <= r.p_value <= 1


@pytest.mark.parametrize(
    "tc,status",
    [
        ({"n00": 0, "n01": 0, "n10": 0, "n11": 0}, "no_transitions"),
        ({"n00": 249, "n01": 0, "n10": 0, "n11": 0}, "degenerate_no_transitions_from_exception_state"),
        ({"n00": 0, "n01": 0, "n10": 0, "n11": 30}, "degenerate_no_transitions_from_non_exception_state"),
    ],
)
def test_christoffersen_degenerate(tc, status):
    r = christoffersen_ind(tc)
    assert r.status == status
    assert math.isnan(r.p_value) and math.isnan(r.statistic)
    cc = christoffersen_cc(kupiec_uc(250, 0, 0.99), r)
    assert cc.status.startswith("undefined") and math.isnan(cc.p_value)


def test_quantile_loss_values():
    q = quantile_loss(np.array([3.0, 1.0, 2.0]), np.array([2.0, 2.0, 2.0]), 0.9)
    # L > q: (0 - 0.9)(2 - 3) = 0.9 ; L < q: (1 - 0.9)(2 - 1) = 0.1 ; L == q: 0
    assert q.tolist() == pytest.approx([0.9, 0.1, 0.0])
    rng = np.random.default_rng(0)
    assert np.all(quantile_loss(rng.normal(size=100), rng.normal(size=100), 0.99) >= 0)


def test_quantile_loss_minimised_near_true_quantile():
    rng = np.random.default_rng(1)
    L = rng.standard_normal(100_000)
    qs = np.linspace(1.5, 3.0, 31)
    scores = [quantile_loss(L, np.full_like(L, q), 0.99).mean() for q in qs]
    assert abs(qs[int(np.argmin(scores))] - stats.norm.ppf(0.99)) < 0.11


def test_rolling_rate_unavailable_across_gap():
    pos = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    hits = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    r = rolling_exception_rate(hits, pos, window=3)
    assert r.size == 9
    assert r[2] == pytest.approx(1 / 3)
    assert np.isnan(r[4]) and np.isnan(r[5]) and np.isnan(r[6])  # windows containing position 4
    assert r[7] == pytest.approx(1 / 3)
    assert np.all((r[~np.isnan(r)] >= 0) & (r[~np.isnan(r)] <= 1))
