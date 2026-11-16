"""Correlated multivariate-normal Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from var_es.monte_carlo import (
    check_covariance,
    convergence_experiment,
    derive_seed,
    estimate_mean_cov,
    monte_carlo_var_es,
    simulate_asset_returns,
    summarise_convergence,
)
from var_es.parametric import normal_var_es
from var_es.validation import DataValidationError

W = np.array([0.30, 0.25, 0.20, 0.15, 0.10])


def _window(n=750, seed=5):
    rng = np.random.default_rng(seed)
    corr = np.full((5, 5), 0.5) + 0.5 * np.eye(5)
    sd = np.array([0.010, 0.012, 0.011, 0.009, 0.013])
    cov = corr * np.outer(sd, sd)
    return rng.multivariate_normal(np.full(5, 2e-4), cov, size=n)


def test_fixed_seed_reproducible():
    x = _window()
    a = monte_carlo_var_es(x, W, [0.99], 5000, np.random.default_rng(derive_seed(7, "2020-03-02")))
    b = monte_carlo_var_es(x, W, [0.99], 5000, np.random.default_rng(derive_seed(7, "2020-03-02")))
    assert a.var_es == b.var_es


def test_seed_depends_on_date_and_master_and_is_stable():
    s1 = derive_seed(7, "2020-03-02").generate_state(4)
    s2 = derive_seed(7, "2020-03-02").generate_state(4)
    s3 = derive_seed(7, "2020-03-03").generate_state(4)
    s4 = derive_seed(8, "2020-03-02").generate_state(4)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert not np.array_equal(s1, s4)


def test_correlation_respected():
    x = _window()
    mean, cov = estimate_mean_cov(x)
    sims = simulate_asset_returns(mean, check_covariance(cov), 200_000, np.random.default_rng(1))
    target = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.allclose(np.corrcoef(sims, rowvar=False), target, atol=0.01)


def test_portfolio_covariance_identity():
    x = _window()
    mean, cov = estimate_mean_cov(x)
    assert W @ cov @ W == pytest.approx(np.var(x @ W, ddof=1), rel=1e-12)
    assert W @ mean == pytest.approx(np.mean(x @ W), rel=1e-12)


def test_mc_approaches_analytical_normal():
    x = _window()
    p = x @ W
    res = monte_carlo_var_es(x, W, [0.95, 0.99], 400_000, np.random.default_rng(123))
    for c in (0.95, 0.99):
        av, ae = normal_var_es(p.mean(), p.std(ddof=1), c)
        v, e = res.var_es[c]
        # deterministic seed; ~0.5% sampling SE at 400k draws, so 3% is a broad tolerance
        assert v == pytest.approx(av, rel=0.03)
        assert e == pytest.approx(ae, rel=0.03)


def test_invalid_covariance_rejected():
    with pytest.raises(DataValidationError, match="symmetric"):
        check_covariance(np.array([[1.0, 0.5], [0.4, 1.0]]))
    with pytest.raises(DataValidationError, match="semidefinite"):
        check_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(DataValidationError, match="non-finite"):
        check_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_tiny_negative_eigenvalue_repair_recorded():
    v = np.array([1.0, 2.0, 3.0])
    cov = np.outer(v, v)  # rank one; eigenvalues may be +/- 1e-15 noise
    cov = cov - 1e-14 * np.eye(3)  # tiny genuine negative eigenvalue within tolerance
    chk = check_covariance(cov)
    assert chk.repaired
    assert np.linalg.eigvalsh(chk.cov).min() >= -1e-12
    assert np.abs(chk.cov - np.outer(v, v)).max() < 1e-12


def test_pd_covariance_not_repaired():
    chk = check_covariance(np.cov(_window(), rowvar=False))
    assert not chk.repaired
    assert chk.factor_method == "cholesky"


def test_draws_below_minus_one_counted_not_clipped():
    x = np.random.default_rng(0).standard_normal((200, 2)) * 0.8
    res = monte_carlo_var_es(x, np.array([0.5, 0.5]), [0.99], 10_000, np.random.default_rng(0))
    assert res.n_asset_draws_below_minus_one > 0


def test_convergence_experiment_shape_and_errors_shrink():
    x = _window()
    raw = convergence_experiment(x, W, [0.99], [1000, 100_000], seeds=range(10))
    assert len(raw) == 2 * 10
    summ = summarise_convergence(raw)
    rmse = summ.set_index("n_scenarios")["var_rmse"]
    assert rmse[100_000] < rmse[1000]
