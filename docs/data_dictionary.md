# Data dictionary

All return, loss, VaR and ES columns are **decimal** (0.01 = 1%) unless the name ends in
`_gbp`. Dates are ISO `YYYY-MM-DD`. Missing numeric values are written as empty fields
(read as `NaN`).

## Inputs

### `data/synthetic/asset_returns.csv`
Key: `date` (unique, strictly increasing synthetic business days, Mon–Fri, no holidays).

| Column | Type | Units | Description |
|---|---|---|---|
| `date` | date | – | Synthetic business day |
| `Equity_A` … `Equity_E` | float | decimal simple return | One-day simple return, always > −1 |
| `is_stress` | bool | – | True inside the predefined stress intervals (indices 900–999, 1500–1599) |
| `regime` | string | – | `normal`, `calm`, `elevated` or `stress` (predefined schedule) |

`is_stress` and `regime` are metadata. The loader keeps them out of the asset-return matrix.

### `data/synthetic/portfolio_returns.csv`
| Column | Type | Units | Description |
|---|---|---|---|
| `date` | date | – | Key |
| `portfolio_return` | float | decimal | `sum_i w_i r_i,t` with baseline weights |
| `portfolio_loss` | float | decimal | `-portfolio_return` |
| `is_stress`, `regime` | bool, string | – | As above |

`run_backtest.py` recomputes portfolio returns from the asset file. It checks that they
agree with this file to within `1e-12`.

### `data/synthetic/generation_metadata.json`
This file records the following:
* the seed and the data-generating process;
* the latent `mu` and `sigma` (log-return parameters);
* the correlation matrix;
* the mixing distribution, with its truncation mass and normalising constant `kappa`;
* the regime schedule and stress intervals;
* the random-draw order;
* realised simple-return means and SDs;
* the min and max return;
* SHA-256 checksums of both CSVs.

#### Correlation matrix of the latent innovations (baseline)

|  | A | B | C | D | E |
|---|---|---|---|---|---|
| A | 1.00 | 0.55 | 0.45 | 0.40 | 0.35 |
| B | 0.55 | 1.00 | 0.50 | 0.45 | 0.40 |
| C | 0.45 | 0.50 | 1.00 | 0.50 | 0.40 |
| D | 0.40 | 0.45 | 0.50 | 1.00 | 0.45 |
| E | 0.35 | 0.40 | 0.40 | 0.45 | 1.00 |

The matrix is positive definite; `numpy.linalg.cholesky` succeeds on it at generation time.

#### Regime schedule (baseline, inclusive retained-observation indices)

| Indices | Regime | Volatility multiplier |
|---|---|---|
| 0–299 | normal | 1.0 |
| 300–699 | calm | 0.8 |
| 700–899 | normal | 1.0 |
| 900–999 | **stress** | 2.5 |
| 1000–1199 | elevated | 1.4 |
| 1200–1499 | normal | 1.0 |
| 1500–1599 | **stress** | 2.5 |
| 1600–1999 | normal | 1.0 |

### User-supplied CSV (`data/user/`)
* One `date` column (the name is configurable through `date_column`).
* One column per asset, either decimal simple returns (`input_mode: "returns"`) or
  adjusted prices (`input_mode: "prices"`).

See [../data/user/README.md](../data/user/README.md).

## Outputs (`results/<run>/`)

### `forecasts.csv` (one row per forecast date × model × confidence)
| Column | Type | Units | Description |
|---|---|---|---|
| `forecast_date` | date | – | Date being forecast (position `t`) |
| `train_start`, `train_end` | date | – | First and last date of the estimation window (`t-750`, `t-1`) |
| `obs_index` | int | – | Dataset position `t` (used for adjacency) |
| `model` | string | – | `historical`, `normal`, `student_t`, `monte_carlo` |
| `confidence` | float | – | 0.95, 0.975, 0.99 |
| `realized_return` | float | decimal | Portfolio return on `forecast_date` |
| `realized_loss` | float | decimal | `-realized_return` |
| `var`, `es` | float | decimal loss | Forecast VaR and ES (missing if the forecast failed) |
| `var_gbp`, `es_gbp`, `realized_loss_gbp` | float | GBP | × 1,000,000 notional |
| `exception` | bool | – | `realized_loss > var` (missing if no forecast) |
| `quantile_loss` | float | decimal | Pinball loss for the VaR |
| `fz0_score` | float | – | FZ0 joint score (missing if invalid) |
| `fz0_status` | string | – | `ok`, `invalid_domain`, `no_forecast` |
| `es_id_residual` | float | decimal | ES identification residual |
| `is_stress`, `regime` | bool, string | – | Stress label of the forecast date |
| `forecast_status` | string | – | `ok` or `fit_failed` |
| `in_common_sample` | bool | – | Date valid for every model and confidence level |

Key: (`forecast_date`, `model`, `confidence`).

### `summary.csv` (model × confidence, common valid sample)
Columns:
* counts: `n_forecasts`, `n_excluded_invalid`, `n_exceptions`, `expected_exceptions`;
* rates: `observed_rate`, `nominal_rate`;
* averages: `mean_quantile_loss`, `mean_var`, `mean_es`, `mean_var_gbp`, `mean_es_gbp`;
* exception structure: `max_consecutive_exceptions`, `mean_loss_beyond_var_on_exceptions`;
* ES checks: `n_es_below_var`, `mean_fz0_score`, `n_fz0_invalid`.

### `coverage_tests.csv`
* Transition counts: `n00`, `n01`, `n10`, `n11`, `n_transitions`.
* For each test prefix `uc_` (Kupiec), `ind_` (Christoffersen independence) and `cc_`
  (conditional coverage): `statistic`, `p_value`, `df`, `status`.

A status other than `ok` means the p-value is deliberately missing.

### `es_evaluation.csv`
* ES vs VaR checks: `n_es_ge_var`, `n_es_below_var`.
* FZ0: `n_fz0_valid`, `n_fz0_invalid_domain`, `mean_fz0_score`.
* Identification residuals: `mean_es_id_residual`, `se_es_id_residual_descriptive`,
  `mean_es_id_residual_gbp_per_1m`, `mean_var_id_residual`.
* Severity on exception days: `n_exceptions`, `mean_loss_on_exceptions`,
  `mean_es_on_exceptions`, `mean_var_on_exceptions`, `mean_loss_minus_es_on_exceptions`,
  `realized_to_forecast_es_ratio`, `max_loss_on_exceptions`.

### `stress_summary.csv`
One row per model × confidence × `period` (`stress` / `non_stress`), covering:
* counts and rates: N, exceptions, expected, observed/nominal rate;
* averages: mean VaR/ES, mean and SD of realised loss, quantile loss, mean FZ0;
* other: invalid FZ0 count, mean ES residual, severity on exception days.

### `traffic_light_windows.csv`
| Column | Description |
|---|---|
| `model` | Model |
| `window_start`, `window_end` | First and last forecast date of the 250-day window |
| `n_valid` | Valid forecasts in the window (must be 250 for a zone) |
| `n_exceptions` | Exceptions (missing if unavailable) |
| `zone` | `green`, `amber`, `red`, `unavailable` |

### `fit_diagnostics.csv` (one row per forecast date × model)
* Common columns: `obs_index`, `forecast_date`, `train_start`, `train_end`, `model`,
  `fit_status`, `n_train`.
* Normal: `mu`, `sigma`.
* Student-t: `t_loc`, `t_scale`, `t_nu`, `t_implied_sd`, `t_converged`,
  `t_at_lower_nu_bound`, `t_at_upper_nu_bound`, `t_neg_loglik`, `t_n_iter`, `message`.
* Monte Carlo: `mc_scenarios`, `mc_cov_repaired`, `mc_factor_method`,
  `mc_min_eigenvalue`, `mc_asset_draws_below_minus_one`.

Columns that do not apply to a model are empty.

### `mc_convergence.csv` (scenario count × confidence)
* Setup: `n_seeds`, `analytic_var`, `analytic_es`.
* Means and errors: `mean_mc_var`, `mean_mc_es`, `var_mean_error`, `es_mean_error`.
* Dispersion: `var_sd_across_seeds`, `es_sd_across_seeds`, `var_rmse`, `es_rmse`,
  `var_mean_abs_rel_error`, `es_mean_abs_rel_error`.

`mc_convergence_raw.csv` has one row per (N, seed, confidence).

### `run_metadata.json`
This file records the following:
* the full configuration and seeds;
* a SHA-256 of `src/var_es/*.py`;
* dataset identity: relative path, SHA-256, dates, exclusions, stress intervals;
* the input mode and units;
* return/loss definitions and the empirical VaR/ES convention;
* the exception and adjacency policies;
* Python and dependency versions;
* the UTC execution timestamp and runtime;
* forecast counts, invalid rows, fit-status counts and the MC convergence set-up.

It contains no absolute paths.

### `test_results.xml`
JUnit XML from the final `pytest` run.
