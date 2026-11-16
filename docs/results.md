# Results: synthetic baseline

All numbers below are copied from the saved outputs in
[`results/synthetic_baseline/`](../results/synthetic_baseline/). The auto-generated
[research report](../results/synthetic_baseline/research_report.md) has the complete
tables. **The data are synthetic.** The findings describe how these models behave on a
known, stylised process with predefined volatility regimes. They are not evidence about
real equity markets.

## Configuration actually run
* `configs/synthetic_baseline.json`, seed 42: 2,000 observations of five assets,
  2015-01-02 to 2022-09-01 (synthetic business days). No dates were excluded.
* Weights 0.30 / 0.25 / 0.20 / 0.15 / 0.10, long-only, daily rebalanced. Reporting
  notional GBP 1,000,000 (constant notional).
* Rolling 750-observation window, refit daily, one-day horizon, 95% / 97.5% / 99%.
* 1,250 out-of-sample forecast dates (2017-11-17 to 2022-09-01) for each of 4 models ×
  3 confidence levels, giving 15,000 forecast rows.
* Monte Carlo used 20,000 scenarios per origin with master seed 20240917.
* The forecasting step took about 15 s on the development machine (it varies by machine).
  A rerun reproduced `forecasts.csv` and `summary.csv` byte for byte.
* Stress intervals were fixed before modelling: 2018-06-15 to 2018-11-01 and 2020-10-02
  to 2021-02-18 (indices 900–999 and 1500–1599). Together they give 200 forecast dates.

## Failures and exclusions
* Every model produced a valid forecast at every origin, so the common sample is all
  1,250 dates.
* The Student-t fit reported `converged` at all 1,250 origins, with no ν-bound hits.
  Fitted ν had median 2.53 and range 2.15–5.30.
* Monte Carlo needed no covariance repairs and drew no simulated asset returns at or
  below −100%.
* ES was never below VaR, every row was inside the FZ0 domain, and no VaR forecast was
  negative.

## VaR calibration (common sample, N = 1,250)

| Model | Conf. | Exceptions | Expected | Kupiec p | Christoffersen IND p | CC p |
|---|---|---|---|---|---|---|
| Historical simulation | 95% | 82 | 62.5 | 0.0155 | 0.7796 | 0.0514 |
| Historical simulation | 97.5% | 43 | 31.25 | 0.0438 | 0.6736 | 0.1200 |
| Historical simulation | 99% | 19 | 12.5 | 0.0861 | 0.0312 | 0.0225 |
| Parametric normal | 95% | 52 | 62.5 | 0.1610 | 0.2400 | 0.1877 |
| Parametric normal | 97.5% | 36 | 31.25 | 0.4006 | 0.3851 | 0.4817 |
| Parametric normal | 99% | 24 | 12.5 | 0.0037 | 0.0833 | 0.0033 |
| Parametric Student-t | 95% | 74 | 62.5 | 0.1465 | 0.4343 | 0.2567 |
| Parametric Student-t | 97.5% | 35 | 31.25 | 0.5049 | 0.3498 | 0.5171 |
| Parametric Student-t | 99% | 18 | 12.5 | 0.1424 | 0.0247 | 0.0274 |
| Monte Carlo (MVN) | 95% | 52 | 62.5 | 0.1610 | 0.2400 | 0.1877 |
| Monte Carlo (MVN) | 97.5% | 37 | 31.25 | 0.3112 | 0.4221 | 0.4339 |
| Monte Carlo (MVN) | 99% | 24 | 12.5 | 0.0037 | 0.0833 | 0.0033 |

What these results show:
* **Normal and MVN Monte Carlo** give almost identical results. This is expected: for a
  linear portfolio they target the same distribution. The one difference is 36 vs 37
  exceptions at 97.5%, and it comes from Monte Carlo sampling noise.
  * At 99%, both have about twice the nominal number of exceptions. Kupiec rejects them
    (p = 0.0037), and so does conditional coverage (p = 0.0033).
  * At 95%, both have fewer exceptions than nominal (52 vs 62.5), without rejection. The
    normal shape is too wide in the body of this mixture distribution and too thin in the
    far tail.
* **Historical simulation** has more exceptions than nominal at every level. Kupiec
  rejects at 95% and 97.5%. At 99%, independence rejects (p = 0.031) and so does
  conditional coverage (p = 0.023).
* **Student-t** is not rejected by Kupiec at any level. At 99%, independence rejects
  (p = 0.025) and so does conditional coverage (p = 0.027), because its exceptions bunch
  in the stress windows.
* Every model fails the 5% conditional-coverage test at 99%. In all cases the exceptions
  are concentrated in the two predefined stress windows, which a rolling unconditional
  model reacts to only slowly.
* The 99% tests have low power, since only 12.5 exceptions are expected. Non-rejection
  is not evidence of correctness.

## ES: joint score and calibration diagnostics

| Model | Conf. | Mean FZ0 (lower better) | Mean ES id. residual | Loss/ES on exception days |
|---|---|---|---|---|
| Historical simulation | 95% | −3.4717 | 0.0034 | 1.006 |
| Historical simulation | 97.5% | −3.1429 | 0.0052 | 1.055 |
| Historical simulation | 99% | −2.6990 | 0.0104 | 1.147 |
| Parametric normal | 95% | −3.4526 | 0.0065 | 1.417 |
| Parametric normal | 97.5% | −3.0761 | 0.0134 | 1.472 |
| Parametric normal | 99% | −2.3349 | 0.0319 | 1.550 |
| Parametric Student-t | 95% | −3.4895 | 0.0011 | 0.987 |
| Parametric Student-t | 97.5% | −3.1685 | 0.0017 | 1.060 |
| Parametric Student-t | 99% | −2.6974 | 0.0057 | 1.078 |
| Monte Carlo (MVN) | 95% | −3.4516 | 0.0065 | 1.419 |
| Monte Carlo (MVN) | 97.5% | −3.0750 | 0.0134 | 1.461 |
| Monte Carlo (MVN) | 99% | −2.3256 | 0.0321 | 1.555 |

* **Mean FZ0.** Student-t has the lowest score at 95% and 97.5%. At 99%, historical
  simulation (−2.6990) and Student-t (−2.6974) are essentially tied. Normal and Monte
  Carlo have clearly higher (worse) scores at every level. These are point comparisons
  only: no Diebold–Mariano or similar test of score differences was run.
* **ES identification residual.** The mean is positive for every model, meaning realised
  tail losses exceeded forecast ES on average.
  * Normal and Monte Carlo are largest: the 99% residual is about 3 times its
    descriptive standard error (0.0319 vs 0.0102).
  * Student-t at 97.5% and 99% is within about one descriptive standard error of zero.
  * This is descriptive. **No formal ES hypothesis test is claimed.**
* **Tail severity.** On 99% exception days, realised losses averaged about 1.55 times
  the forecast ES for normal and Monte Carlo, against 1.08–1.15 for Student-t and
  historical simulation. The thin normal tail understates how bad the bad days are.

## Stress versus non-stress (200 vs 1,050 forecast dates)
* Each stress window has 100 days, so the stress sample is 200 dates. Descriptive only;
  no p-values.

| Model | 99% exceptions, stress (exp. 2.0) | 99% exceptions, non-stress (exp. 10.5) | Mean FZ0 99% stress | Mean FZ0 99% non-stress |
|---|---|---|---|---|
| Historical simulation | 14 | 5 | 0.2232 | −3.2556 |
| Parametric normal | 16 | 8 | 3.3572 | −3.4191 |
| Parametric Student-t | 14 | 4 | 0.1189 | −3.2338 |
| Monte Carlo (MVN) | 16 | 8 | 3.3885 | −3.4140 |

* **Stress periods.** All models record 7–8 times the nominal 99% exception rate. None
  of these unconditional rolling models adapts quickly to a 2.5× volatility shock.
* **Non-stress periods.** Exception counts fall below nominal for every model,
  especially Student-t (4 vs 10.5) and historical simulation (5 vs 10.5). The windows
  still contain stress observations, which keeps VaR high after the shock has passed.
* **Overall.** Exceptions are too many in stress and too few afterwards. This is the
  classic sign of models without conditional volatility, so an overall exception count
  close to nominal can hide poor conditional calibration.

## Traffic lights (99% VaR, 250-day rolling windows, educational)
1,001 complete windows per model; no window was unavailable.

| Model | Green | Amber | Red |
|---|---|---|---|
| Historical simulation | 765 | 85 | 151 |
| Parametric normal | 563 | 287 | 151 |
| Parametric Student-t | 765 | 85 | 151 |
| Monte Carlo (MVN) | 563 | 287 | 151 |

* **Red windows.** Every model has 151 red windows: exactly the windows that contain the
  whole first stress interval (window ends 2018-11-01 to 2019-05-30).
* **Amber around the second stress interval.** Normal and Monte Carlo stay amber for
  most windows covering the second stress interval. Historical simulation and Student-t
  return to green sooner (see `traffic_light_history.png`).
* **Overlap.** The windows overlap heavily, so these counts are not independent
  observations. Zones are not p-values and carry no regulatory meaning.

## Monte Carlo sampling error
* Origin: 2017-11-17, with training data from 2015-01-02 to 2017-11-16. 20 seeds per
  scenario count.
* Analytical normal benchmark: 99% VaR = 0.01750 and ES = 0.02007 (decimal loss).

| Scenarios | 99% VaR RMSE | 99% VaR mean abs. rel. error | 99% ES RMSE | 99% ES mean abs. rel. error |
|---|---|---|---|---|
| 1,000 | 9.67e-04 | 4.31% | 1.05e-03 | 4.15% |
| 5,000 | 5.12e-04 | 2.41% | 6.29e-04 | 2.49% |
| 20,000 | 1.78e-04 | 0.76% | 2.35e-04 | 0.97% |
| 100,000 | 1.15e-04 | 0.52% | 1.49e-04 | 0.55% |

* **Overall behaviour.** Errors shrink with N roughly in line with 1/√N, although
  20 seeds give a noisy RMSE estimate. The 20,000 → 100,000 step falls by less than
  √5 would suggest.
* **What 20,000 scenarios means.** At the baseline 20,000 scenarios, the per-origin Monte
  Carlo error is under 1% of VaR and ES at this origin. That error is small relative to
  the model differences above, but it does produce visible noise. An example is 36 vs 37
  exceptions at 97.5% for normal vs Monte Carlo.

## Uncertainty and what is *not* claimed
* There is a single synthetic path. Repeating with other seeds would change every count.
* There are no confidence intervals on FZ0 differences.
* The asymptotic test p-values ignore parameter-estimation error and the overlap between
  consecutive estimation windows.
* None of these results shows that one method is superior in general or on real markets.
  They show how each method responds to a heavy-tailed, regime-switching process, given
  a rolling unconditional estimator.
