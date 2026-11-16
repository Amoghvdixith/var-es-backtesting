"""VaR backtesting: exceptions, quantile loss, Kupiec and Christoffersen tests.

Exception (hit) convention: ``I_t = 1{L_t > VaR_t}`` (strict).

Adjacency policy: independence statistics use transitions ``I_{t-1} -> I_t`` only
between *genuinely consecutive* evaluation observations, i.e. dataset positions
``k`` and ``k + 1`` that both have a valid forecast. Observations separated by a
missing forecast are never treated as adjacent.

Likelihoods use ``scipy.special.xlogy`` so that ``0 * log(0) = 0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy

from .validation import validate_confidence


def exceptions(losses: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Strict VaR exceptions ``loss > VaR``."""
    return np.asarray(losses, dtype=float) > np.asarray(var, dtype=float)


def quantile_loss(losses: np.ndarray, var: np.ndarray, confidence: float) -> np.ndarray:
    """Pinball (tick) loss for the ``c``-quantile of the loss distribution.

    ``S(q, L) = (1{L <= q} - c) (q - L)``; non-negative, zero only when ``L = q``.
    This is a consistent scoring function for the ``c``-quantile.
    """
    c = validate_confidence(confidence)
    L = np.asarray(losses, dtype=float)
    q = np.asarray(var, dtype=float)
    return ((L <= q).astype(float) - c) * (q - L)


@dataclass
class TestResult:
    statistic: float
    p_value: float
    df: int
    status: str  # "ok" or an explicit reason why the test is undefined

    def as_dict(self, prefix: str) -> dict[str, float | int | str]:
        return {f"{prefix}_{k}": v for k, v in asdict(self).items()}


def _bern_loglik(n0: float, n1: float, p: float) -> float:
    return float(xlogy(n0, 1.0 - p) + xlogy(n1, p))


def kupiec_uc(n_obs: int, n_exc: int, confidence: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    ``LR_uc = -2 [ l(p) - l(pi_hat) ]`` with ``p = 1 - c`` and ``pi_hat = x / n``;
    asymptotically ``chi^2(1)``. Defined for ``x = 0`` and ``x = n`` via ``0 log 0 = 0``.
    """
    c = validate_confidence(confidence)
    p = 1.0 - c
    if n_obs <= 0:
        return TestResult(np.nan, np.nan, 1, "no_observations")
    if not 0 <= n_exc <= n_obs:
        raise ValueError("n_exc must lie in [0, n_obs]")
    pi = n_exc / n_obs
    lr = -2.0 * (_bern_loglik(n_obs - n_exc, n_exc, p) - _bern_loglik(n_obs - n_exc, n_exc, pi))
    lr = max(lr, 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)), 1, "ok")


def transition_counts(hits: np.ndarray, positions: np.ndarray | None = None) -> dict[str, int]:
    """Count ``n00, n01, n10, n11`` over genuinely consecutive observations.

    ``positions`` are integer dataset positions; a transition is counted only when
    ``positions[i+1] == positions[i] + 1``. If omitted, all neighbours are adjacent.
    """
    h = np.asarray(hits, dtype=bool)
    pos = np.arange(h.size) if positions is None else np.asarray(positions, dtype=int)
    if pos.size != h.size:
        raise ValueError("hits and positions must have the same length")
    if np.any(np.diff(pos) <= 0):
        raise ValueError("positions must be strictly increasing")
    adj = np.diff(pos) == 1
    prev, curr = h[:-1][adj], h[1:][adj]
    return {
        "n00": int(np.sum(~prev & ~curr)),
        "n01": int(np.sum(~prev & curr)),
        "n10": int(np.sum(prev & ~curr)),
        "n11": int(np.sum(prev & curr)),
    }


def christoffersen_ind(counts: dict[str, int]) -> TestResult:
    """Christoffersen (1998) first-order Markov independence LR test, ``chi^2(1)``.

    ``pi01 = n01/(n00+n01)``, ``pi11 = n11/(n10+n11)``, ``pi = (n01+n11)/N``.
    Degenerate cases are returned with an explicit status and no p-value:
    no transitions at all, no transitions from a no-exception day, or no
    transitions from an exception day (so ``pi11`` is not identified).
    """
    n00, n01, n10, n11 = (counts[k] for k in ("n00", "n01", "n10", "n11"))
    total = n00 + n01 + n10 + n11
    if total == 0:
        return TestResult(np.nan, np.nan, 1, "no_transitions")
    if n00 + n01 == 0:
        return TestResult(np.nan, np.nan, 1, "degenerate_no_transitions_from_non_exception_state")
    if n10 + n11 == 0:
        return TestResult(np.nan, np.nan, 1, "degenerate_no_transitions_from_exception_state")
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / total
    l0 = _bern_loglik(n00 + n10, n01 + n11, pi)
    l1 = _bern_loglik(n00, n01, pi01) + _bern_loglik(n10, n11, pi11)
    lr = max(-2.0 * (l0 - l1), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)), 1, "ok")


def christoffersen_cc(uc: TestResult, ind: TestResult) -> TestResult:
    """Conditional coverage ``LR_cc = LR_uc + LR_ind``, ``chi^2(2)``.

    Uses the common practical decomposition; ``LR_uc`` is computed on all
    evaluation observations while ``LR_ind`` uses consecutive transitions only.
    """
    if uc.status != "ok" or ind.status != "ok":
        return TestResult(np.nan, np.nan, 2, f"undefined_component(uc={uc.status},ind={ind.status})")
    lr = uc.statistic + ind.statistic
    return TestResult(lr, float(stats.chi2.sf(lr, 2)), 2, "ok")


def max_consecutive_exceptions(hits: np.ndarray, positions: np.ndarray | None = None) -> int:
    """Longest run of exceptions on genuinely consecutive positions."""
    h = np.asarray(hits, dtype=bool)
    pos = np.arange(h.size) if positions is None else np.asarray(positions, dtype=int)
    best = run = 0
    for i in range(h.size):
        if h[i]:
            run = run + 1 if (i > 0 and h[i - 1] and pos[i] == pos[i - 1] + 1) else 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def rolling_exception_rate(
    hits: np.ndarray,
    positions: np.ndarray,
    window: int,
    start: int | None = None,
    n_positions: int | None = None,
) -> np.ndarray:
    """Exception rate over trailing ``window`` consecutive positions, on a full position grid.

    ``hits``/``positions`` describe valid evaluation observations only. The output is
    indexed by ``position - start`` (default ``start = positions.min()``) over
    ``n_positions`` grid points; entries are NaN where the trailing window contains
    any position without a valid observation (gaps are never bridged).
    """
    pos = np.asarray(positions, dtype=int)
    start = int(pos.min()) if start is None else int(start)
    size = (int(pos.max()) - start + 1) if n_positions is None else int(n_positions)
    grid_hit = np.zeros(size)
    grid_ok = np.zeros(size)
    grid_hit[pos - start] = np.asarray(hits, dtype=float)
    grid_ok[pos - start] = 1.0
    out = np.full(size, np.nan)
    if size < window:
        return out
    ch = np.concatenate([[0.0], np.cumsum(grid_hit)])
    co = np.concatenate([[0.0], np.cumsum(grid_ok)])
    ends = np.arange(window - 1, size)
    cnt_ok = co[ends + 1] - co[ends + 1 - window]
    rate = (ch[ends + 1] - ch[ends + 1 - window]) / window
    out[ends] = np.where(cnt_ok == window, rate, np.nan)
    return out
