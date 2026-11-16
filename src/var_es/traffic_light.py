"""Educational Basel-style traffic-light classification for 99% one-day VaR.

Simplified use of the Basel Committee's 1996 backtesting framework: count exceptions
of 99% one-day VaR over 250 consecutive daily observations and classify

* green: 0-4 exceptions
* amber: 5-9 exceptions
* red:   10 or more exceptions

These fixed boundaries are tied to c = 0.99 and a 250-observation window, so other
confidence levels or window lengths are rejected. Zones are not p-values, are not
applied to ES, and no capital multiplier or regulatory conclusion is derived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TL_CONFIDENCE = 0.99
TL_WINDOW = 250


def traffic_light_zone(n_exceptions: int) -> str:
    """Zone for an exception count in a 250-day window at 99%."""
    if n_exceptions < 0:
        raise ValueError("exception count must be non-negative")
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "amber"
    return "red"


def _check(confidence: float, window: int) -> None:
    if abs(float(confidence) - TL_CONFIDENCE) > 1e-12:
        raise ValueError(f"traffic-light zones are defined here only for 99% VaR; got {confidence}")
    if int(window) != TL_WINDOW:
        raise ValueError(f"traffic-light zones require a {TL_WINDOW}-observation window; got {window}")


def rolling_traffic_light(
    dates: pd.DatetimeIndex,
    positions: np.ndarray,
    hits: np.ndarray,
    valid: np.ndarray,
    *,
    confidence: float = TL_CONFIDENCE,
    window: int = TL_WINDOW,
) -> pd.DataFrame:
    """Rolling traffic-light windows ending at each position with enough history.

    ``positions`` are consecutive dataset positions (one per forecast date); a window
    is classified only when all ``window`` positions in it have a valid forecast.
    Otherwise the zone is ``"unavailable"`` -- gaps are never bridged.
    """
    _check(confidence, window)
    pos = np.asarray(positions, dtype=int)
    if np.any(np.diff(pos) != 1):
        raise ValueError("pass one row per consecutive position (mark missing forecasts via `valid`)")
    h = np.asarray(hits, dtype=bool) & np.asarray(valid, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    rows = []
    for end in range(window - 1, pos.size):
        sl = slice(end - window + 1, end + 1)
        n_valid = int(ok[sl].sum())
        if n_valid == window:
            n_exc = int(h[sl].sum())
            zone = traffic_light_zone(n_exc)
        else:
            n_exc, zone = np.nan, "unavailable"
        rows.append(
            {
                "window_start": dates[end - window + 1],
                "window_end": dates[end],
                "n_valid": n_valid,
                "n_exceptions": n_exc,
                "zone": zone,
            }
        )
    return pd.DataFrame(rows)
