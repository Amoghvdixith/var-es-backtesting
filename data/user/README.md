# User-supplied data

Put your own CSV here, for example `my_returns.csv`. Then point a copy of
`configs/user_data_example.json` at it.

## Format
```
date,AAA,BBB,CCC
2020-01-02,0.0041,-0.0103,0.0022
2020-01-03,-0.0067,0.0015,-0.0040
...
```

* **Dates:** one column (default name `date`, set with `data.date_column`). Dates must be
  unique, strictly increasing and parseable as dates.
* **Assets:** one column per asset. The asset names must match the keys of
  `portfolio.weights` exactly, and duplicate names are rejected.
* **`data.input_mode`** is required and never inferred:
  * `"returns"`: simple returns `P_t / P_{t-1} - 1`. Set `data.units` to `"decimal"`
    (0.01 = 1%) or `"percent"` (1.0 = 1%).
  * `"prices"`: adjusted prices. Returns are computed as `P_t / P_{t-1} - 1`, and the
    first date is dropped.
* **Optional columns:** `is_stress` and `regime` are treated as metadata. Alternatively,
  give `stress_intervals` in the config as `[{"start_date": ..., "end_date": ...}]`.

## Validation and missing data
* Every value must be finite, and every simple return must be greater than −1.
* Weights must be finite, align by name, and sum to 1. With `long_only: true`, they must
  also be non-negative.
* **Complete-case policy.** A date with any missing asset value is excluded for all
  assets. Nothing is forward-filled, interpolated or reweighted. The excluded dates are
  listed in `run_metadata.json`.
  * In price mode, a missing price removes both returns that touch it, so no multi-day
    return is created.
* At least `forecast.window + 1` observations are required, and also `data.min_history`
  if it is set.

## Run
```
python scripts/run_backtest.py --config configs/my_data.json
python scripts/generate_report.py --results-dir results/user_data
```
`run_mc_convergence.py` also works on user data, because it reads the same
`mc_convergence` section of the config.

Results from synthetic data must not be presented as evidence about real markets.
Results from your own data are only as reliable as that data.
