# Validation

## Final automated test run
* Command: `python -m pytest -q --junitxml=results/synthetic_baseline/test_results.xml`.
* Environment: CPython 3.14.4 on Windows, using the project `.venv` (versions in
  `requirements-lock.txt`).
* **Outcome: 127 tests, 127 passed, 0 failed, 0 errors, 0 skipped** (about 26 s).
* The JUnit XML is saved as
  [`results/synthetic_baseline/test_results.xml`](../results/synthetic_baseline/test_results.xml).
  Its `hostname` attribute was replaced with `redacted` so the machine name is not
  published. Nothing else in the file was altered.

| Test file | Tests | What is checked |
|---|---|---|
| `test_data.py` | 16 | Valid load; paths with spaces; duplicate/unsorted dates; duplicate asset columns; returns ≤ −1; non-finite values; explicit input mode; complete-case exclusion without forward-fill; explicit percent→decimal; price→return conversion; a missing price creating no multi-day return; metadata columns kept out of assets; truncated mixing normalisation; regime-schedule validation; deterministic synthetic schema with stress indices 900–999 and 1500–1599 |
| `test_portfolio.py` | 12 | Hand-calculated portfolio return; alignment by name; invalid weights (missing, extra, sum ≠ 1, NaN, negative); short weights only when allowed; no reweighting around missing data; loss and GBP conventions; decimal vs percent; `w'Σw` identity |
| `test_empirical.py` | 10 | Integer and fractional tail mass (n = 10, 100, 750); ties, including a case where "mean above VaR" is empty; all-equal samples; n = 1 and n = 2; ES ≥ VaR and order invariance; negative VaR not clipped; historical simulation uses losses; invalid inputs |
| `test_parametric.py` | 25 | Normal VaR/ES against known constants and numerical integration; zero volatility; ddof = 1; Student-t ES against two integrals (12 ν × c cases); scale vs SD; parameter recovery; deterministic fit; boundary flag; fit failures reported, not replaced; invalid parameters |
| `test_monte_carlo.py` | 10 | Fixed-seed reproducibility; seeds stable across calls and dependent on date and master seed; correlation reproduced (±0.01 at 200k draws); portfolio covariance identity; MC → analytical normal (±3% at 400k draws, fixed seed); invalid covariances rejected (asymmetric, indefinite, non-finite); tiny-negative-eigenvalue repair recorded; no repair for positive-definite input; draws ≤ −100% counted, not clipped; convergence RMSE falls with N |
| `test_forecasting.py` | 6 | Training window ends before the forecast date and spans exactly 750/60 positions; altering observation *k* and later leaves forecasts ≤ *k* unchanged (and later ones do change); realised loss attached afterwards; failed t fit recorded without substitution; common-sample exclusion; missing forecasts create no artificial transitions; MC reproducibility |
| `test_backtesting.py` | 15 | Strict exceptions; Kupiec vs hand log-likelihood; zero and all exceptions; LR = 0 at the nominal rate; no-observation status; known transition table; gaps; Christoffersen IND/CC vs hand calculation; n11 = 0; three degenerate cases with no p-value; pinball values and minimiser; rolling rate missing across gaps; p-values in [0, 1] |
| `test_es_evaluation.py` | 15 | FZ0 hand examples (exception and non-exception day); loss convention equals the published return convention; domain violations marked invalid without altering data; negative VaR with positive ES allowed; return-convention guard; simulation: the true (VaR, ES) scores best; identification residual zero-mean and signs; hand residual values; P(L > ES) ≠ 1−c; tail severity |
| `test_traffic_light.py` | 15 | Zone boundaries 0/4/5/9/10/25; negative count; confidence ≠ 99% rejected; window ≠ 250 rejected; rolling counts and zone change at the 4/5 boundary; missing-forecast windows unavailable; non-consecutive positions rejected |
| `test_pipeline.py` | 3 | Full CLI workflow on a tiny synthetic config in a path containing spaces: deterministic data (identical SHA-256 on regeneration), all output files, `forecasts.csv` schema, ES ≥ VaR, summary exception counts recomputed from `forecasts.csv`, p-value bounds, no absolute path in metadata, report rows equal to `summary.csv`; plus a user adjusted-price pipeline with a missing price (2 excluded dates) and date-based stress intervals |

## Independent numerical checks
* **Normal VaR/ES** are compared with standard constants: at 97.5%, z = 1.959964 and
  ES/σ = 2.337803; at 99%, z = 2.326348 and ES/σ = 2.665214. They are also compared with
  `scipy.integrate.quad` of the loss density (relative tolerance 1e-7).
* **Student-t ES** is checked against two independent integrals (tail expectation and
  quantile average), with relative tolerance 1e-6 and 1e-5.
* **Empirical VaR/ES** are compared with hand-computed order statistics and fractional
  weights, including n = 750 at 99% (m = 7.5).
* **Kupiec and Christoffersen** are recomputed directly with `math.log` from the counts
  (relative tolerance 1e-12).
* **FZ0** is compared with a hand calculation and with the return-convention formula as
  printed in Patton, Ziegel & Chen (2019, eq. 6, checked against the arXiv text).
* **Monte Carlo** is compared with the analytical normal model at a fixed seed, and the
  convergence experiment in `results/` shows the error shrinking.
* **Portfolio returns.** `run_backtest.py` recomputes them from the asset file and checks
  them against the saved `portfolio_returns.csv` (maximum absolute difference < 1e-12).
* **Reproducibility.** Running the baseline a second time produced byte-identical
  `forecasts.csv` and `summary.csv`.

## Tolerances and flakiness
* Every stochastic test uses a fixed seed. It is deterministic on a given platform and
  NumPy version.
* Tolerances are set to several sampling standard errors. For example, the
  identification-residual mean tolerance is 0.03, against a sampling SD of about 0.006.
* No test depends on wall-clock time or network access.

## Limitations of the validation
* The tests check that the code implements the stated formulas. They cannot show that the
  models are appropriate for real data.
* Statistical tests use asymptotic chi-square distributions. No size or power simulation
  was run for this sample length.
* The FZ0 formula was checked against the arXiv preprint, not the paywalled journal text.
* Results were produced on one platform (Windows, Python 3.14.4). Other BLAS or NumPy
  builds could change Monte Carlo draws or optimiser paths in the last decimal places.
* Figures were inspected visually. There is no automated image test beyond checking that
  the files exist and are non-empty.
