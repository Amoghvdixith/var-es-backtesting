# Methodology

This document states every convention and formula implemented in `src/var_es/`. Where a
formula comes from the literature, the source is listed in [references.md](references.md).

## 1. Units, returns and losses

| Quantity | Definition | Units |
|---|---|---|
| Asset return `r_i,t` | simple return `P_t / P_{t-1} - 1` | decimal (0.01 = 1%) |
| Portfolio return | `r_p,t = sum_i w_i r_i,t` | decimal |
| Loss | `L_t = -r_p,t` (positive = loss, negative = gain) | decimal |
| Monetary loss | `L_gbp,t = N * L_t`, `N = GBP 1,000,000` | GBP |
| VaR, ES | upper-tail statistics of `L` | decimal and GBP |

* **Daily rebalancing.** The portfolio is rebalanced to fixed target weights at the start
  of each day. For simple returns, `sum_i w_i r_i,t` is then the exact one-day portfolio
  simple return. Log returns are not summed across assets anywhere in the core.
* **Constant notional.** GBP figures are the decimal loss times a fixed reporting
  notional. This is a **constant-notional risk comparison**, so every day's risk is
  expressed on the same GBP 1m base. It is **not** a compounded wealth simulation, and
  P&L is not accumulated through time.
* Negative VaR/ES (a forecast gain) is allowed and never clipped to zero. None occurred
  in the baseline run, where the minimum VaR was 1.17% of notional.
* Percentages are converted only through an explicit `units: "percent"` setting. Prices
  are converted only through `input_mode: "prices"`. Nothing is inferred.

### Portfolio validation
`validation.validate_weights` checks the following:
* the weight names match the asset columns exactly, with no missing, extra or duplicate names;
* the weights are finite and sum to 1 within `1e-9`;
* no weight is negative when `long_only` is set.

Weights are never renormalised. `portfolio.portfolio_returns` raises an error if any
asset return is missing, so a missing asset can never be "handled" by silent reweighting.

## 2. Empirical VaR and ES (common engine, `empirical.py`)

Let `x_(1) <= ... <= x_(n)` be the sorted losses, `c` the confidence level and
`p = 1 - c` the tail probability. Each observation has weight `1/n`.

* **VaR** is the inverse empirical CDF at `c`:
  `VaR_c = inf{x : F_n(x) >= c} = x_(k)`, `k = ceil(n c)`.
  A relative guard of `1e-9` stops floating-point error from moving `k`. For example,
  `n = 100, c = 0.95` must give `k = 95`.
* **ES** is the average of the worst `p` probability mass. That mass equals
  `m = n p` observations' worth. With `j = floor(m)`:

  `ES_c = [ x_(n) + ... + x_(n-j+1) + (m - j) * x_(n-j) ] / m`

  The boundary observation receives fractional weight `m - j`. This is the
  Acerbi–Tasche (2002) / Rockafellar–Uryasev (2002) tail mean applied to the empirical
  distribution. It satisfies `ES_c >= VaR_c` exactly, and it remains correct when there
  are ties at the boundary.
* **Why not "mean of losses above VaR"?** With ties at VaR, or when `n p` is not an
  integer, the set `{x > VaR}` may contain fewer than `n p` observations, or none at all.
  Its mean then averages the wrong amount of probability mass. `tests/test_empirical.py`
  includes a case where that set is empty.
* **Finite-sample effects.** With `n = 750`, the 99% tail holds `m = 7.5` observations'
  worth of mass. VaR is a single order statistic (`x_(743)`), and ES is driven by about
  eight observations. Both move in discrete jumps as the window rolls, which is visible as
  the step pattern in the historical-simulation charts. Their sampling error is
  substantial.
* **Exceptions** are strict: `I_t = 1{L_t > VaR_t}`.

## 3. Historical simulation (`historical.py`)
The 750 portfolio returns before the forecast date are converted to losses and passed
to the common empirical engine.

## 4. Parametric normal (`parametric.py`)
The model fits the window sample mean `mu` and the sample SD `sigma` (`ddof=1`). With
`z_c = Phi^{-1}(c)`:

    VaR_c = -mu + sigma z_c
    ES_c  = -mu + sigma phi(z_c) / (1 - c)

* If `sigma = 0` (a constant window, detected exactly), then `VaR = ES = -mu` and the fit
  status is `zero_volatility`.
* This is a fitted **unconditional** one-period return distribution for the window. It is
  not a conditional-volatility forecast. Changing volatility enters only through the
  rolling re-estimation.
* Check: for `c = 0.975`, `ES = 2.3378 sigma - mu`, which the tests also confirm by
  numerical integration.

## 5. Parametric Student-t (`parametric.py`)
The model is `R = loc + scale * T`, `T ~ t_nu`. Note that `scale` is **not** the standard
deviation:

    sd = scale * sqrt(nu / (nu - 2)),   nu > 2

With `z_c = t_nu^{-1}(c)`:

    VaR_c = -loc + scale z_c
    ES_c  = -loc + scale * (nu + z_c^2)/(nu - 1) * f_nu(z_c) / (1 - c)

**Estimation.** Constrained maximum likelihood using `scipy.optimize.minimize`
(L-BFGS-B) and `scipy.stats.t.logpdf`:
* Parameters are `(loc, log scale, log(nu - 2))`.
* For numerical conditioning, the data are standardised by the window mean and SD. The
  estimates are mapped back afterwards.
* `nu` is bounded to `[2.01, nu_upper]`, with `nu_upper = 1000` in the baseline. The
  estimates are therefore **constrained** MLEs, and any fit at a bound is flagged
  `converged_at_boundary`.
* Initialisation is deterministic: median, MAD-based scale and `nu = 5`.

The model records `nu`, `loc`, `scale`, the implied SD, convergence, the iteration count,
the negative log-likelihood and any boundary hits (`fit_diagnostics.csv`).

**Failures.** A failed optimisation produces `forecast_status = fit_failed` with VaR/ES
left missing. No other model is substituted.

**Verification.** The ES formula is checked against `scipy.integrate.quad` in two ways:
* the tail expectation `∫ l f(l) dl / (1-c)`;
* the quantile average `∫_c^1 VaR_u du / (1-c)`.

These checks cover `nu ∈ {2.5, 4, 8, 50}` and `c ∈ {0.95, 0.975, 0.99}`.

## 6. Correlated Monte Carlo (`monte_carlo.py`)
1. Estimate the asset mean vector and the sample covariance (`ddof=1`) on the 750-day window.
2. Validate the covariance:
   * symmetry within `1e-10` relative;
   * eigenvalues `>= -1e-10 * lambda_max`, otherwise it is rejected as invalid.

   Tiny negative eigenvalues within that tolerance would be repaired by clipping the
   eigenvalues at 0 and recording `mc_cov_repaired`; no diagonal loading is used. No
   repairs occurred in the baseline.
3. Draw 20,000 scenarios `mu + L z` (Cholesky factor `L`, or an eigen-factor if the
   matrix is only semidefinite), apply the fixed weights, convert to losses and use the
   empirical engine.
4. **Seeds.** Each origin uses `np.random.SeedSequence([master_seed, YYYYMMDD, model_id])`
   with `model_id = 4`. This is stable across processes; Python's randomised `hash()` is
   not used.

**Equivalence with the normal model.** For a linear portfolio,
`w'X ~ N(w'mu, w'Sigma w)` when `X ~ N(mu, Sigma)`. Also, `w' Sigma_hat w` equals the
sample variance of the window's portfolio returns, and the tests verify this identity.
MVN Monte Carlo therefore targets **exactly the same distribution** as the analytical
normal model. Their differences are Monte Carlo sampling error, and they are not
evidence that simulation is a better model.

**Returns below −100%.** The Gaussian model gives positive probability to simple returns
at or below −100%. Simulated draws are not clipped. Their count is recorded instead, and
it was 0 in the baseline.

**Convergence experiment** (`scripts/run_mc_convergence.py`):
* Origin: the predefined first forecast date.
* Scenario counts N ∈ {1,000; 5,000; 20,000; 100,000}, each with 20 fixed seeds using
  `SeedSequence([seed, N])`.
* Results are compared with the analytical normal VaR/ES from the same window. Reported
  error measures are the mean error, RMSE, standard deviation across seeds and mean
  absolute relative error.

## 7. Out-of-sample forecasting (`forecasting.py`)
* For the forecast at dataset position `t`, the window is positions `t-750 … t-1`, and
  `train_start`/`train_end` are saved.
* `forecast_origin` receives only that window. It fits all models and forecasts all
  confidence levels from the same fit.
* The realised return at `t` is attached afterwards (`attach_realised`).
* Every model is refitted at every origin.
* The baseline has 2,000 observations, so there are 1,250 forecast dates
  (positions 750–1999).
* **No look-ahead** is tested directly. Altering observation `k` and every later
  observation leaves every forecast for positions `<= k` unchanged. That includes the
  forecast for `k` itself, whose realised loss was altered.
* **Common sample.** Cross-model tables use the dates on which every model has a valid
  forecast at every confidence level. Excluded rows are counted (`n_excluded_invalid`).
  In the baseline, all 1,250 dates were valid.

## 8. VaR backtests (`backtesting.py`)
With `n` forecasts, `x` exceptions and `p = 1-c`:

* **Quantile (pinball) loss:** `S(q, L) = (1{L <= q} - c)(q - L) >= 0`. It is a
  consistent scoring function for the `c`-quantile; lower is better.
* **Kupiec (1995) unconditional coverage (POF):**
  `LR_uc = -2[ (n-x) ln(1-p) + x ln p - (n-x) ln(1-pi) - x ln pi ]`, with `pi = x/n`.
  It is `chi^2(1)` asymptotically. This is the likelihood-ratio test, not an exact
  binomial test.
* **Christoffersen (1998) independence:** count first-order transitions `n_ij`
  (state `i` → state `j`) and set
  `pi01 = n01/(n00+n01)`, `pi11 = n11/(n10+n11)`, `pi = (n01+n11)/N`.

  `LR_ind = -2[ ln L(pi) - ln L(pi01, pi11) ]` is `chi^2(1)`.
* **Conditional coverage:** `LR_cc = LR_uc + LR_ind`, which is `chi^2(2)`. This is the
  common practical decomposition: `LR_uc` uses all `n` observations, while `LR_ind` uses
  the consecutive transitions.
* `0 * ln 0 = 0` is handled with `scipy.special.xlogy`.
* **Degenerate cases** return an explicit status and no p-value:
  * no transitions at all;
  * no transitions out of a non-exception day;
  * no transitions out of an exception day, so `pi11` is unidentified, which includes
    zero exceptions.

  In each of these cases, CC is `undefined_component(...)`.
* **Adjacency policy.** A transition is counted only between dataset positions `k` and
  `k+1` when both have valid forecasts. Missing forecasts never create artificial
  transitions, and maximum exception runs follow the same rule.
* Also reported: expected exceptions, observed vs nominal rate, mean VaR, maximum
  consecutive exceptions, mean `L - VaR` on exception days, and rolling 250-forecast
  exception rates (missing if the trailing window has a gap).

**Caveats.**
* At 99% with 1,250 days, about 12.5 exceptions are expected, so the tests have low power.
* The asymptotic distributions ignore estimation error and the overlap between
  consecutive windows.
* Non-rejection does not establish correctness.
* Fewer exceptions than nominal can indicate over-conservative forecasts. It is not
  automatically "better".

## 9. ES evaluation (`es_evaluation.py`)
ES is not a quantile. `P(L > ES_c) ≠ 1 - c`; for a normal distribution at 97.5% it is
about 0.0097. So no exception-frequency test is applied to ES.

**This version gives VaR formal coverage and independence tests. ES receives a joint
scoring rule and calibration diagnostics, not a formal hypothesis test.**

### 9.1 FZ0 joint VaR/ES score
Fissler & Ziegel (2016) showed that the pair (VaR, ES) is jointly elicitable. Patton,
Ziegel & Chen (2019, eq. 6) define the FZ0 member of that class, in their lower-tail
**return** convention: `alpha` small, `v` the return `alpha`-quantile, `e` the return
ES (`E[Y | Y <= v]`).

    FZ0(Y, v, e; alpha) = -1/(alpha e) * 1{Y <= v} (v - Y) + v/e + log(-e) - 1

**Domain.**
* `log(-e)` requires `e < 0`.
* Consistency also requires `e <= v`, per the paper's footnote 1.
* The paper's degree-zero homogeneity result additionally assumes VaR and ES are both
  negative in its return convention.

**Conversion to this package's loss convention.** Set `alpha = 1-c`, `y = -L`,
`v = -VaR`, `e = -ES`. Then `1{y <= v} = 1{L >= VaR}` and `v - y = L - VaR`, which gives:

    FZ0 = 1/(alpha ES) * 1{L >= VaR} (L - VaR) + VaR/ES + log(ES) - 1

The domain becomes `ES > 0` and `ES >= VaR`. Rows outside it are marked
`fz0_status = invalid_domain` with no score, and the data are never modified to fit. In
the baseline, every row was inside the domain and every VaR was positive. Lower mean
FZ0 is better.

**Verification.**
* The loss-convention implementation equals the published return-convention formula on
  random inputs.
* Two hand-worked examples are checked. For instance, `L = 0.03`, `VaR = 0.02`,
  `ES = 0.025`, `c = 0.975` gives `16 + 0.8 + ln 0.025 - 1`.
* A simulation confirms that the true normal (VaR, ES) has a lower mean score than
  scaled-wrong alternatives, and than alternatives with correct VaR but wrong ES.

### 9.2 Calibration diagnostics
* **ES identification residual** (loss convention):
  `Z_t = (1/alpha) 1{L_t >= VaR_t} (L_t - VaR_t) - (ES_t - VaR_t)`.
  * `E[(L - VaR) 1{L >= VaR}] = alpha (ES - VaR)` for the true pair of a continuous
    distribution, so `E[Z_t] = 0` under correct specification.
  * Positive mean: realised tail losses exceed forecast ES on average (ES too low).
  * Negative mean: ES is conservative.
  * The sample mean and an i.i.d. standard error are reported **descriptively**. No test
    is performed.
* VaR identification residual `1{L > VaR} - alpha`.
* Tail severity on exception days: mean realised loss, mean forecast ES and VaR, mean
  `L - ES`, the ratio of mean realised loss to mean ES, and the maximum loss.
* `ES >= VaR` is checked for every forecast and reported (`n_es_below_var`).

VaR and ES errors are never compared as if they targeted the same statistic.

## 10. Basel-style traffic lights (`traffic_light.py`)
This is an educational simplification of the BCBS (1996) backtesting framework. Over
**250 consecutive** daily comparisons of **99% one-day VaR**:
* green: 0–4 exceptions;
* amber: 5–9;
* red: 10 or more.

Rules:
* Other confidence levels and window lengths are rejected.
* A window containing any missing forecast is `unavailable`. Gaps are never bridged.
* Zones are not p-values and are not applied to ES.
* No capital multiplier, plus factor or regulatory conclusion is computed or implied.

## 11. Stress evaluation
The stress intervals are fixed in the configuration before any modelling: observation
indices 900–999 and 1500–1599, both inclusive, with a 2.5× volatility multiplier.

`stress_summary.csv` compares stress and non-stress forecast dates on the following
measures:
* N, exceptions vs expected and the exception rate;
* mean VaR and ES;
* quantile loss and mean FZ0;
* the ES identification residual and the loss/ES ratio on exception days.

No p-values are reported for the 200-day stress sample.

## 12. Synthetic data-generating process (`synthetic.py`)
See [data_dictionary.md](data_dictionary.md) and `data/synthetic/generation_metadata.json`.
In summary:
* Correlated Gaussian latent log-return innovations (Cholesky of the documented
  correlation matrix).
* A shared truncated inverse-gamma scale mixture: `nu = 4`, `W <= 25`, normalised so
  `E[s^2] = 1`.
* Predefined regime multipliers.
* `r = expm1(x)`.

`mu` and `sigma` parameterise the latent log returns, not the saved simple returns.
Truncating the mixing variable keeps every moment of `expm1(x)` finite. Untruncated
Student-t log returns would give `E[exp(x)] = ∞`. No returns are clipped.
