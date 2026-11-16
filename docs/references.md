# References

Bibliographic details were checked against publisher, BIS, RePEc/EconPapers or arXiv
records in November 2026. Where a detail could not be confirmed from a primary record,
that is stated.

## VaR backtesting

1. **Kupiec, P. H. (1995).** *Techniques for Verifying the Accuracy of Risk Measurement
   Models.* Journal of Derivatives, 3(2), 73–84 (Winter 1995).
   * The journal issue and pages are as listed in library catalogues (e.g. EconBiz). I
     did not verify a DOI from the publisher.
   * The working-paper version is Federal Reserve Board, Finance and Economics
     Discussion Series 95-24 (1995):
     <https://econpapers.repec.org/RePEc:fip:fedgfe:95-24>.
   * Used for the proportion-of-failures (unconditional coverage) likelihood-ratio test.
2. **Christoffersen, P. F. (1998).** *Evaluating Interval Forecasts.* International
   Economic Review, 39(4), 841–862. doi:10.2307/2527341.
   * Record: <https://ideas.repec.org/a/ier/iecrev/v39y1998i4p841-62.html>.
   * Used for the first-order Markov independence and conditional-coverage
     likelihood-ratio tests.
3. **Basel Committee on Banking Supervision (1996).** *Supervisory framework for the use
   of "backtesting" in conjunction with the internal models approach to market risk
   capital requirements.* Bank for International Settlements, January 1996 (status on the
   BIS site: superseded). <https://www.bis.org/publ/bcbs22.htm>.
   * Source of the 250-day, 99% green/amber/red exception-count zones.
   * This project uses only the zone boundaries, for education. It does not implement
     the plus factors or any capital consequence.

## Expected Shortfall definitions

4. **Acerbi, C., & Tasche, D. (2002).** *On the coherence of expected shortfall.* Journal
   of Banking & Finance, 26(7), 1487–1503.
   doi:10.1016/S0378-4266(02)00283-2. Preprint: <https://arxiv.org/abs/cond-mat/0104295>.
   * Motivates the tail-mass definition of ES. Under that definition, the boundary atom
     gets fractional weight for discontinuous (e.g. empirical) distributions.
5. **Rockafellar, R. T., & Uryasev, S. (2002).** *Conditional value-at-risk for general
   loss distributions.* Journal of Banking & Finance, 26(7), 1443–1471.
   * The general-distribution CVaR/ES definition, which is consistent with reference 4.
   * Cited from standard bibliographic knowledge; its page range was not re-checked
     against the publisher during this project.

## Joint VaR/ES scoring

6. **Fissler, T., & Ziegel, J. F. (2016).** *Higher order elicitability and Osband's
   principle.* Annals of Statistics, 44(4), 1680–1707. doi:10.1214/16-AOS1439.
   Preprint: <https://arxiv.org/abs/1503.08123>.
   * Shows that (VaR, ES) is jointly elicitable and characterises the strictly consistent
     scoring functions.
   * The authors later published a correction to Proposition 3.4 and Theorem 5.2(ii);
     this project does not rely on those specific statements.
7. **Patton, A. J., Ziegel, J. F., & Chen, R. (2019).** *Dynamic semiparametric models
   for expected shortfall (and Value-at-Risk).* Journal of Econometrics, 211(2), 388–413.
   doi:10.1016/j.jeconom.2018.10.008. Preprint: <https://arxiv.org/abs/1707.05108>.
   * Source of the FZ0 loss, eq. (6) of the preprint. It is
     `-(1/(alpha e)) 1{Y<=v}(v-Y) + v/e + log(-e) - 1`, in the lower-tail return
     convention.
   * The domain follows footnote 1 of the preprint (`e <= v`) and the requirement
     `e < 0`.
   * The implementation was checked against the arXiv text, which may differ in detail
     from the published version.

## Further reading (not used for formulas)

* Nolde, N., & Ziegel, J. F. (2017). *Elicitability and backtesting: Perspectives for
  banking regulation.* Annals of Applied Statistics, 11(4), 1833–1874. This is background
  on why ES is not elicitable on its own and how that affects backtesting. It is cited by
  Patton, Ziegel & Chen; the details were not re-checked against the publisher here.
