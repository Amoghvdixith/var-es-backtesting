"""FZ0 joint score, domain checks and ES identification residuals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from var_es.es_evaluation import (
    es_identification_residual,
    fz0_score,
    fz0_score_return_convention,
    tail_severity,
)
from var_es.parametric import normal_var_es


def test_fz0_hand_example_exception_day():
    L, v, e, c = 0.03, 0.02, 0.025, 0.975
    a = 1 - c
    expected = (L - v) / (a * e) + v / e + math.log(e) - 1
    s, ok = fz0_score(np.array([L]), np.array([v]), np.array([e]), c)
    assert ok.all()
    assert s[0] == pytest.approx(expected, rel=1e-12)
    assert s[0] == pytest.approx(16.0 + 0.8 + math.log(0.025) - 1.0, rel=1e-12)


def test_fz0_hand_example_no_exception():
    s, _ = fz0_score(np.array([0.01]), np.array([0.02]), np.array([0.025]), 0.975)
    assert s[0] == pytest.approx(0.8 + math.log(0.025) - 1.0, rel=1e-12)


def test_loss_convention_matches_published_return_convention():
    rng = np.random.default_rng(0)
    L = rng.normal(0, 0.01, 500)
    v = np.abs(rng.normal(0.02, 0.003, 500))
    e = v + np.abs(rng.normal(0.005, 0.001, 500))
    for c in (0.95, 0.975, 0.99):
        s, ok = fz0_score(L, v, e, c)
        assert ok.all()
        assert s == pytest.approx(fz0_score_return_convention(-L, -v, -e, c), rel=1e-12)


@pytest.mark.parametrize(
    "v,e",
    [(0.02, 0.0), (0.02, -0.01), (0.03, 0.02), (np.nan, 0.02), (0.01, np.inf)],
)
def test_fz0_domain_invalid_marked_not_altered(v, e):
    s, ok = fz0_score(np.array([0.01]), np.array([v]), np.array([e]), 0.99)
    assert not ok[0]
    assert np.isnan(s[0])


def test_fz0_negative_var_positive_es_allowed():
    s, ok = fz0_score(np.array([0.0]), np.array([-0.001]), np.array([0.002]), 0.95)
    assert ok[0] and np.isfinite(s[0])


def test_return_convention_rejects_nonnegative_es():
    with pytest.raises(ValueError):
        fz0_score_return_convention(np.array([0.0]), np.array([-0.01]), np.array([0.0]), 0.99)


def test_fz0_true_forecast_scores_best_in_simulation():
    rng = np.random.default_rng(2024)
    mu, sd = 0.0003, 0.012
    L = -(mu + sd * rng.standard_normal(400_000))
    c = 0.975
    v0, e0 = normal_var_es(mu, sd, c)
    true_score = np.mean(fz0_score(L, v0, e0, c)[0])
    for k in (0.7, 0.85, 1.2, 1.5):
        alt = np.mean(fz0_score(L, v0 * k, e0 * k, c)[0])
        assert true_score < alt
    # wrong ES with correct VaR is also penalised
    assert true_score < np.mean(fz0_score(L, v0, e0 * 1.3, c)[0])
    assert true_score < np.mean(fz0_score(L, v0, max(v0, e0 * 0.85), c)[0])


def test_identification_residual_signs_and_zero_mean():
    rng = np.random.default_rng(7)
    L = rng.standard_normal(1_000_000)
    c = 0.975
    v, e = normal_var_es(0.0, 1.0, c)
    z = es_identification_residual(L, v, e, c)
    assert abs(z.mean()) < 0.03  # broad tolerance; sampling SD of mean ~ 0.006
    # ES forecast too low -> positive mean; too high -> negative mean
    assert es_identification_residual(L, v, e * 0.8, c).mean() > 0.3
    assert es_identification_residual(L, v, e * 1.2, c).mean() < -0.3


def test_identification_residual_hand_values():
    z = es_identification_residual(np.array([0.05, 0.01]), np.array([0.02, 0.02]), np.array([0.03, 0.03]), 0.99)
    assert z.tolist() == pytest.approx([(0.05 - 0.02) / 0.01 - 0.01, -0.01])


def test_es_is_not_a_quantile_exceedance_frequency():
    """P(L > ES) is not 1 - c, so an ES 'exception rate' test would be wrong."""
    c = 0.975
    v, e = normal_var_es(0.0, 1.0, c)
    p_exceed_es = stats.norm.sf(e)
    assert p_exceed_es < 0.5 * (1 - c)  # ~0.0097 vs nominal 0.025


def test_tail_severity():
    sev = tail_severity(np.array([0.05, 0.01, 0.03]), np.full(3, 0.02), np.full(3, 0.025))
    assert sev["n_exceptions"] == 2
    assert sev["mean_loss_on_exceptions"] == pytest.approx(0.04)
    assert sev["realized_to_forecast_es_ratio"] == pytest.approx(0.04 / 0.025)
    none = tail_severity(np.array([0.0]), np.array([0.02]), np.array([0.025]))
    assert none["n_exceptions"] == 0 and math.isnan(none["mean_loss_on_exceptions"])
