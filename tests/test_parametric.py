"""Parametric normal and Student-t VaR/ES."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from var_es.parametric import (
    fit_normal,
    fit_student_t,
    normal_var_es,
    student_t_sd,
    student_t_var_es,
)


def test_normal_known_values():
    # standard normal, c = 0.975: VaR = 1.959964, ES = phi(1.959964)/0.025 = 2.337803
    v, e = normal_var_es(0.0, 1.0, 0.975)
    assert v == pytest.approx(1.959963985, rel=1e-8)
    assert e == pytest.approx(2.337802792, rel=1e-8)
    # c = 0.99: VaR = 2.326348, ES = 2.665214
    v, e = normal_var_es(0.0, 1.0, 0.99)
    assert v == pytest.approx(2.326347874, rel=1e-8)
    assert e == pytest.approx(2.665214220, rel=1e-8)


def test_normal_location_scale():
    v, e = normal_var_es(0.001, 0.02, 0.95)
    assert v == pytest.approx(-0.001 + 0.02 * 1.644853627, rel=1e-8)
    assert e == pytest.approx(-0.001 + 0.02 * 2.062712808, rel=1e-8)


def test_normal_es_by_integration():
    mu, s, c = 0.0005, 0.013, 0.99
    v, e = normal_var_es(mu, s, c)
    num, _ = integrate.quad(lambda l: l * stats.norm.pdf(l, loc=-mu, scale=s), v, np.inf)
    assert e == pytest.approx(num / (1 - c), rel=1e-7)


def test_zero_volatility():
    assert normal_var_es(0.001, 0.0, 0.99) == (-0.001, -0.001)
    fit = fit_normal(np.full(10, 0.002))
    assert fit.status == "zero_volatility"
    assert fit.sigma == 0.0


def test_fit_normal_ddof1():
    r = np.array([0.01, -0.02, 0.03, 0.0])
    fit = fit_normal(r)
    assert fit.sigma == pytest.approx(np.std(r, ddof=1))


@pytest.mark.parametrize("nu", [2.5, 4.0, 8.0, 50.0])
@pytest.mark.parametrize("c", [0.95, 0.975, 0.99])
def test_student_t_es_by_integration(nu, c):
    loc, scale = 0.0004, 0.009
    v, e = student_t_var_es(loc, scale, nu, c)
    z = stats.t.ppf(c, nu)
    assert v == pytest.approx(-loc + scale * z, rel=1e-10)
    tail, _ = integrate.quad(lambda t: (-loc + scale * t) * stats.t.pdf(t, nu), z, np.inf, limit=200)
    assert e == pytest.approx(tail / (1 - c), rel=1e-6)
    # alternative check: ES as the average of VaR_u over u in (c, 1)
    avg, _ = integrate.quad(lambda u: -loc + scale * stats.t.ppf(u, nu), c, 1.0, limit=200)
    assert e == pytest.approx(avg / (1 - c), rel=1e-5)
    assert e > v


def test_student_t_scale_not_sd():
    assert student_t_sd(1.0, 5.0) == pytest.approx(np.sqrt(5 / 3))
    assert stats.t.std(5.0, scale=0.01) == pytest.approx(student_t_sd(0.01, 5.0))
    assert student_t_sd(1.0, 2.0) == float("inf")


def test_student_t_fit_recovers_parameters():
    rng = np.random.default_rng(11)
    r = 0.0005 + 0.01 * rng.standard_t(5.0, size=5000)
    fit = fit_student_t(r)
    assert fit.status == "converged"
    assert 3.5 < fit.nu < 8.0
    assert fit.scale == pytest.approx(0.01, rel=0.1)
    assert fit.loc == pytest.approx(0.0005, abs=0.0005)
    assert fit.implied_sd == pytest.approx(fit.scale * np.sqrt(fit.nu / (fit.nu - 2)))


def test_student_t_fit_deterministic():
    rng = np.random.default_rng(12)
    r = 0.01 * rng.standard_t(4.0, size=750)
    a, b = fit_student_t(r), fit_student_t(r)
    assert (a.loc, a.scale, a.nu) == (b.loc, b.scale, b.nu)


def test_student_t_boundary_flagged():
    rng = np.random.default_rng(13)
    r = 0.01 * rng.standard_normal(5000)
    fit = fit_student_t(r, nu_upper=10.0)
    assert fit.at_upper_nu_bound
    assert fit.status == "converged_at_boundary"
    assert fit.nu == pytest.approx(10.0, rel=1e-2)


@pytest.mark.parametrize("bad", [np.full(100, 0.01), np.array([0.01, np.nan, 0.02, 0.0]), np.array([0.01])])
def test_student_t_fit_failure_reported_not_replaced(bad):
    fit = fit_student_t(bad)
    assert fit.status == "fit_failed"
    assert not fit.usable
    assert fit.message


def test_invalid_parameters():
    with pytest.raises(ValueError):
        student_t_var_es(0.0, 0.0, 5.0, 0.99)
    with pytest.raises(ValueError):
        student_t_var_es(0.0, 1.0, 1.0, 0.99)
    with pytest.raises(ValueError):
        normal_var_es(0.0, -1.0, 0.99)
