"""Parametric normal and Student-t VaR / ES for one-day portfolio losses.

Both models describe a *fitted unconditional one-period return distribution* for the
trailing estimation window. They are not conditional-volatility (e.g. GARCH)
forecasts; changing volatility is captured only through the rolling re-estimation.

Loss convention: ``L = -R``. For confidence ``c``:

Normal (``R ~ N(mu, sigma^2)``, ``z_c = Phi^{-1}(c)``)::

    VaR_c = -mu + sigma * z_c
    ES_c  = -mu + sigma * phi(z_c) / (1 - c)

Student-t (``R = loc + scale * T``, ``T ~ t_nu``, ``z_c = t_nu^{-1}(c)``)::

    VaR_c = -loc + scale * z_c
    ES_c  = -loc + scale * (nu + z_c^2) / (nu - 1) * f_nu(z_c) / (1 - c)

``scale`` is *not* the standard deviation: ``sd = scale * sqrt(nu / (nu - 2))`` for
``nu > 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from .validation import validate_confidence

NU_LOWER = 2.01  # nu must exceed 2 for a finite variance; enforced via this bound
BOUNDARY_RTOL = 1e-3


# --------------------------------------------------------------------------- normal
def normal_var_es(mu: float, sigma: float, confidence: float) -> tuple[float, float]:
    """Normal VaR and ES in loss units. ``sigma == 0`` gives ``VaR = ES = -mu``."""
    c = validate_confidence(confidence)
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError(f"sigma must be finite and non-negative; got {sigma}")
    if sigma == 0.0:
        return -mu, -mu
    z = stats.norm.ppf(c)
    var = -mu + sigma * z
    es = -mu + sigma * stats.norm.pdf(z) / (1.0 - c)
    return float(var), float(es)


@dataclass
class NormalFit:
    mu: float
    sigma: float
    status: str  # "ok" or "zero_volatility"


def fit_normal(returns: np.ndarray) -> NormalFit:
    """Sample mean and sample standard deviation (ddof=1)."""
    r = np.asarray(returns, dtype=float)
    mu = float(r.mean())
    # exact zero for constant samples (floating-point std can return ~1e-19 instead)
    sigma = 0.0 if np.ptp(r) == 0 else float(r.std(ddof=1))
    return NormalFit(mu=mu, sigma=sigma, status="zero_volatility" if sigma == 0.0 else "ok")


# ------------------------------------------------------------------------ student-t
def student_t_var_es(loc: float, scale: float, nu: float, confidence: float) -> tuple[float, float]:
    """Student-t VaR and ES in loss units (requires ``nu > 1`` for ES, ``scale > 0``)."""
    c = validate_confidence(confidence)
    if not scale > 0:
        raise ValueError(f"scale must be positive; got {scale}")
    if not nu > 1:
        raise ValueError(f"ES requires nu > 1; got {nu}")
    z = stats.t.ppf(c, nu)
    var = -loc + scale * z
    es = -loc + scale * (nu + z * z) / (nu - 1.0) * stats.t.pdf(z, nu) / (1.0 - c)
    return float(var), float(es)


def student_t_sd(scale: float, nu: float) -> float:
    """Standard deviation of a scaled Student-t: ``scale * sqrt(nu / (nu - 2))``."""
    if nu <= 2:
        return float("inf")
    return float(scale * np.sqrt(nu / (nu - 2.0)))


@dataclass
class StudentTFit:
    loc: float
    scale: float
    nu: float
    status: str  # "converged", "converged_at_boundary", "fit_failed"
    converged: bool
    at_lower_nu_bound: bool = False
    at_upper_nu_bound: bool = False
    neg_loglik: float = float("nan")
    n_iter: int = 0
    message: str = ""
    implied_sd: float = field(default=float("nan"))

    @property
    def usable(self) -> bool:
        return self.status != "fit_failed"


def fit_student_t(returns: np.ndarray, *, nu_upper: float = 1000.0, init_nu: float = 5.0) -> StudentTFit:
    """Constrained maximum-likelihood Student-t fit.

    Parameterisation: ``theta = (loc_s, log scale_s, log(nu - 2))`` on data
    standardised by the sample mean and SD (for conditioning only; estimates are
    mapped back). ``nu`` is bounded to ``[NU_LOWER, nu_upper]`` with L-BFGS-B, so the
    estimates are *constrained* MLEs; hitting a bound is flagged. Initialisation is
    deterministic: median, MAD-based scale, ``nu = init_nu``.

    A failed optimisation returns ``status="fit_failed"`` -- callers must not
    substitute another model.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 3 or not np.all(np.isfinite(r)):
        return StudentTFit(np.nan, np.nan, np.nan, "fit_failed", False, message="insufficient or non-finite data")
    m0, s0 = float(r.mean()), float(r.std(ddof=1))
    if np.ptp(r) == 0 or not s0 > 0:
        return StudentTFit(np.nan, np.nan, np.nan, "fit_failed", False, message="zero sample variance")
    y = (r - m0) / s0

    mad = float(np.median(np.abs(y - np.median(y)))) * 1.4826
    scale_init = mad if mad > 0 else np.sqrt((init_nu - 2.0) / init_nu)
    x0 = np.array([float(np.median(y)), np.log(scale_init), np.log(init_nu - 2.0)])
    a_lo, a_hi = np.log(NU_LOWER - 2.0), np.log(nu_upper - 2.0)

    def nll(theta: np.ndarray) -> float:
        loc, log_s, a = theta
        val = -np.sum(stats.t.logpdf(y, 2.0 + np.exp(a), loc=loc, scale=np.exp(log_s)))
        return float(val) if np.isfinite(val) else 1e300

    try:
        res = optimize.minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=[(None, None), (np.log(1e-6), np.log(1e3)), (a_lo, a_hi)],
            options={"maxiter": 500},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return StudentTFit(np.nan, np.nan, np.nan, "fit_failed", False, message=f"optimizer error: {exc}")

    loc_s, log_s, a = res.x
    nu = 2.0 + float(np.exp(a))
    loc = m0 + s0 * float(loc_s)
    scale = s0 * float(np.exp(log_s))
    finite = np.all(np.isfinite([loc, scale, nu, res.fun])) and res.fun < 1e299
    if not (res.success and finite):
        return StudentTFit(
            loc, scale, nu, "fit_failed", False, neg_loglik=float(res.fun), n_iter=int(res.nit),
            message=str(res.message),
        )
    at_lo = a <= a_lo + BOUNDARY_RTOL
    at_hi = a >= a_hi - BOUNDARY_RTOL
    status = "converged_at_boundary" if (at_lo or at_hi) else "converged"
    # log-likelihood on original scale = standardised nll + n log s0
    nll_orig = float(res.fun) + r.size * np.log(s0)
    return StudentTFit(
        loc=loc, scale=scale, nu=nu, status=status, converged=True,
        at_lower_nu_bound=bool(at_lo), at_upper_nu_bound=bool(at_hi),
        neg_loglik=nll_orig, n_iter=int(res.nit), message=str(res.message),
        implied_sd=student_t_sd(scale, nu),
    )
