"""Educational Basel-style traffic-light classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_es.traffic_light import rolling_traffic_light, traffic_light_zone


@pytest.mark.parametrize("n,zone", [(0, "green"), (4, "green"), (5, "amber"), (9, "amber"), (10, "red"), (25, "red")])
def test_zone_boundaries(n, zone):
    assert traffic_light_zone(n) == zone


def test_negative_count_rejected():
    with pytest.raises(ValueError):
        traffic_light_zone(-1)


def _inputs(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return dates, np.arange(n), np.zeros(n, dtype=bool), np.ones(n, dtype=bool)


@pytest.mark.parametrize("c", [0.95, 0.975])
def test_wrong_confidence_rejected(c):
    d, p, h, v = _inputs()
    with pytest.raises(ValueError, match="99%"):
        rolling_traffic_light(d, p, h, v, confidence=c)


@pytest.mark.parametrize("w", [249, 251, 500])
def test_wrong_window_rejected(w):
    d, p, h, v = _inputs()
    with pytest.raises(ValueError, match="250"):
        rolling_traffic_light(d, p, h, v, window=w)


def test_rolling_counts_and_zones():
    d, p, h, v = _inputs(270)
    h[[10, 20, 30, 40, 50]] = True  # five exceptions in the first window
    tl = rolling_traffic_light(d, p, h, v)
    assert len(tl) == 270 - 250 + 1
    assert tl["n_exceptions"].iloc[0] == 5 and tl["zone"].iloc[0] == "amber"
    assert tl["window_start"].iloc[0] == d[0] and tl["window_end"].iloc[0] == d[249]
    # window starting at position 11 drops the exception at 10 -> 4 -> green
    assert tl["n_exceptions"].iloc[11] == 4 and tl["zone"].iloc[11] == "green"


def test_missing_forecast_makes_window_unavailable():
    d, p, h, v = _inputs(260)
    v[5] = False  # a missing forecast at position 5
    tl = rolling_traffic_light(d, p, h, v)
    assert (tl["zone"].iloc[:6] == "unavailable").all()  # windows covering position 5
    assert tl["n_exceptions"].iloc[:6].isna().all()
    assert (tl["zone"].iloc[6:] == "green").all()


def test_non_consecutive_positions_rejected():
    d, p, h, v = _inputs(260)
    p = p.copy()
    p[100:] += 1
    with pytest.raises(ValueError, match="consecutive"):
        rolling_traffic_light(d, p, h, v)
