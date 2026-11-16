# Learning guide

This is a short conceptual companion to the code. The formulas are in
[methodology.md](methodology.md).

## 1. VaR versus ES
* **VaR at 99%** is the loss level exceeded on about 1% of days. It is a quantile. It
  says nothing about *how large* losses are once that level is exceeded.
* **ES at 99%** is the average loss over the worst 1% of outcomes. It looks inside the
  tail, so ES ≥ VaR always.
* Two portfolios can share the same VaR and have very different ES. One might lose a
  little more than VaR in bad cases; the other might occasionally lose ten times more.
* ES is coherent: it is sub-additive, so diversification never increases it. VaR is not
  coherent in general (Acerbi & Tasche, 2002).

## 2. Historical vs parametric vs simulation
| Method | What it assumes | Typical behaviour |
|---|---|---|
| Historical simulation | The next day looks like a random draw from the last 750 days | No distributional assumption. The tail estimate rests on a handful of observations, and it jumps when a large loss enters or leaves the window. |
| Parametric normal | Returns are normal with the window's mean and SD | Smooth and cheap. It understates extreme quantiles when returns are fat-tailed, and ES/VaR is fixed at about 1.15 at 99%. |
| Parametric Student-t | Returns follow a scaled t with fitted tail parameter ν | Fat tails. ES/VaR grows as ν falls, and small ν makes the estimates sensitive to the fit. |
| MVN Monte Carlo | Assets are jointly normal with the window's mean and covariance | For a *linear* portfolio this is the normal model computed by simulation, plus sampling error. Simulation pays off for nonlinear books (options) or non-normal joint models, which are out of scope here. |

## 3. Student-t tails
* The t density decays like a power law (`|x|^-(ν+1)`), not like `exp(-x²/2)`.
* Smaller ν means fatter tails. For ν ≤ 2 the variance is infinite, which is why the fit
  constrains ν > 2.
* `scale` is not the standard deviation: `sd = scale·sqrt(ν/(ν−2))`. With ν = 3, the
  SD is 1.73 × scale.
* Each window in this project mixes calm and stressed days. A single *unconditional* t
  fit absorbs that volatility clustering as very heavy tails: the baseline median ν was
  about 2.5. That is a symptom of model misspecification (constant volatility), not a
  deep property of the data.

## 4. Correlation and diversification
* Portfolio variance is `w'Σw`. With positive but imperfect correlation (about 0.35–0.55
  here), portfolio volatility is below the weighted average of asset volatilities.
* In the synthetic data, a **shared** scale factor hits all assets on the same day.
  Extreme days are therefore extreme for every asset together (tail dependence).
* A Gaussian copula with the same correlations would understate that joint tail. This
  is one reason the MVN models struggle at 99%.

## 5. Forecasting versus in-sample fitting
* Fitting on the whole dataset and then "testing" on the same data guarantees flattering
  results.
* Here, every forecast for day *t* uses only days *t−750 … t−1*. The tests verify that
  changing day *t* or later cannot change the forecast for *t*.
* The full-sample fits in `return_distribution.png` are descriptive only.

## 6. Exceptions and clustering
* An *exception* is a day when the realised loss exceeds the VaR forecast.
* A well-calibrated 99% VaR has about 1% exceptions (Kupiec test), and they should not
  arrive in bunches (Christoffersen test).
* Clustering means the model reacts too slowly when volatility rises, which is exactly
  what rolling unconditional models do in the stress windows.

## 7. Why ES is not tested as another quantile
* ES is not a quantile, so "how often was the loss above ES?" has no target frequency of
  1−c. Under a normal distribution at 97.5%, `P(L > ES)` is about 1%, not 2.5%.
* ES on its own is not elicitable (see Fissler & Ziegel, 2016, and the references therein). The
  pair (VaR, ES) *is* jointly elicitable, so a joint scoring function such as **FZ0** can
  rank forecasting methods.
* The **identification residual** has mean zero when VaR and ES are both right. It is
  used here as a descriptive calibration check, not a formal test.

## 8. Traffic-light limitations
* The fixed 0–4 / 5–9 / ≥10 cut-offs are calibrated for 99% VaR over exactly 250 days.
* They are not transferable to 95% VaR, to other window lengths, or to ES.
* A green zone is not proof of a good model: with 250 days the test has low power.
* Overlapping rolling windows are highly dependent, so counting "red windows" is not
  counting independent events.
* This project shows zones for education only. It computes no capital multiplier and
  makes no compliance claim.

## 9. Interview self-check questions
1. Why is ES ≥ VaR at the same confidence level, and when are they equal?
2. You have 750 observations. Which order statistic is the empirical 99% VaR here, and how much tail mass does the ES average?
3. Why is "mean of losses strictly above VaR" a flawed empirical ES estimator?
4. Derive the normal ES `μ + σφ(z)/(1−c)` for losses. What is ES/VaR at 99% when μ = 0?
5. Student-t scale is 1%, ν = 4. What is the standard deviation?
6. Why should MVN Monte Carlo and parametric normal VaR agree for a linear portfolio? What would make them differ?
7. How would you choose the number of Monte Carlo scenarios? What does 1/√N imply?
8. State the Kupiec LR statistic. What happens with zero exceptions?
9. What does the Christoffersen independence test detect that Kupiec cannot?
10. Why is low power a bigger problem at 99% than at 95%?
11. A model has 3 exceptions where 12.5 were expected. Is it a good model?
12. Why can't you backtest ES by counting exceedances of ES?
13. What does a positive mean ES identification residual tell you?
14. Why is FZ0 defined only for negative ES in the return convention?
15. What is wrong with fitting the model on the full sample and calling it a backtest?
16. Why do the traffic-light thresholds not apply to 97.5% VaR?
17. Why did the fitted Student-t ν come out close to 2 in this project, and what would a GARCH filter change?
18. How would you handle a missing forecast in an independence test?
