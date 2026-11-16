"""Loading configuration and return data.

Input modes are explicit: ``input_mode="returns"`` expects decimal (or explicitly
declared percentage) simple returns; ``input_mode="prices"`` expects adjusted
prices and converts them with ``P_t / P_{t-1} - 1``. The loader never guesses which
of the two a file contains.

Missing-data policy: complete-case. Any date on which at least one asset return is
missing is excluded for every asset; nothing is forward-filled, interpolated or
reweighted. The excluded dates are returned so they can be reported.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .validation import (
    DataValidationError,
    validate_dates,
    validate_simple_returns,
    validate_unique_columns,
)

METADATA_COLUMNS = ("is_stress", "regime")
VALID_INPUT_MODES = ("returns", "prices")
VALID_UNITS = ("decimal", "percent")


@dataclass
class ReturnsData:
    """Validated asset simple returns plus optional per-date metadata."""

    returns: pd.DataFrame  # index: DatetimeIndex named "date"; columns: assets; decimal units
    metadata: pd.DataFrame  # same index; columns such as is_stress, regime
    excluded_dates: list[str] = field(default_factory=list)
    input_mode: str = "returns"
    units: str = "decimal"

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON configuration file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def percent_to_decimal(values: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Convert percentage returns (1.0 means 1%) to decimal returns (0.01)."""
    return values / 100.0


def prices_to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns ``P_t / P_{t-1} - 1``.

    A missing price makes both the return ending on that date and the return
    starting on it missing; no multi-day return is silently created. The first
    row (no previous price) is dropped.
    """
    p = prices.astype(float)
    if (p.dropna() <= 0).any().any():
        raise DataValidationError("prices must be strictly positive")
    rets = p / p.shift(1) - 1.0
    return rets.iloc[1:]


def load_returns_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    input_mode: str = "returns",
    units: str = "decimal",
    assets: list[str] | None = None,
) -> ReturnsData:
    """Load and validate a CSV of returns or prices.

    Parameters
    ----------
    path:
        CSV with a date column and one column per asset. Columns named in
        :data:`METADATA_COLUMNS` are kept separately and never treated as assets.
    input_mode:
        ``"returns"`` or ``"prices"`` (required, never inferred).
    units:
        For returns only: ``"decimal"`` (0.01 = 1%) or ``"percent"`` (1.0 = 1%).
    assets:
        Optional explicit asset list; defaults to all non-date, non-metadata columns.
    """
    path = Path(path)
    if input_mode not in VALID_INPUT_MODES:
        raise DataValidationError(f"input_mode must be one of {VALID_INPUT_MODES}; got {input_mode!r}")
    if units not in VALID_UNITS:
        raise DataValidationError(f"units must be one of {VALID_UNITS}; got {units!r}")
    if input_mode == "prices" and units != "decimal":
        raise DataValidationError("units applies to returns only; use units='decimal' with prices")

    header = _read_header(path)
    validate_unique_columns(header)
    if date_column not in header:
        raise DataValidationError(f"date column {date_column!r} not found in {header}")

    df = pd.read_csv(path)
    dates = validate_dates(df[date_column])
    df.index = dates
    df.index.name = "date"
    df = df.drop(columns=[date_column])

    meta_cols = [c for c in df.columns if c in METADATA_COLUMNS]
    if assets is None:
        assets = [c for c in df.columns if c not in METADATA_COLUMNS]
    else:
        unknown = [a for a in assets if a not in df.columns]
        if unknown:
            raise DataValidationError(f"assets not found in file: {unknown}")
    if not assets:
        raise DataValidationError("no asset columns found")

    values = df[assets].apply(pd.to_numeric, errors="coerce").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise DataValidationError("asset values contain infinite entries")
    metadata = df[meta_cols].copy()

    if input_mode == "prices":
        values = prices_to_simple_returns(values)
        metadata = metadata.loc[values.index]
    elif units == "percent":
        values = percent_to_decimal(values)

    incomplete = values.isna().any(axis=1)
    excluded = [d.strftime("%Y-%m-%d") for d in values.index[incomplete]]
    values = values.loc[~incomplete]
    metadata = metadata.loc[values.index]

    validate_simple_returns(values)
    return ReturnsData(
        returns=values, metadata=metadata, excluded_dates=excluded, input_mode=input_mode, units=units
    )


def stress_labels_from_intervals(
    index: pd.DatetimeIndex, intervals: list[dict[str, str]]
) -> pd.Series:
    """Build a boolean stress flag from user-specified date intervals (inclusive)."""
    flag = pd.Series(False, index=index, name="is_stress")
    for iv in intervals:
        start, end = pd.Timestamp(iv["start_date"]), pd.Timestamp(iv["end_date"])
        flag.loc[(index >= start) & (index <= end)] = True
    return flag
