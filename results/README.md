# Results

`synthetic_baseline/` holds the outputs of the research baseline
(`configs/synthetic_baseline.json`). Every file in it comes from actual execution of the
commands below. No result was typed in by hand.

```
python scripts/generate_data.py --config configs/synthetic_baseline.json
python scripts/run_backtest.py --config configs/synthetic_baseline.json
python scripts/run_mc_convergence.py --config configs/synthetic_baseline.json
python scripts/generate_report.py --results-dir results/synthetic_baseline
python -m pytest -q --junitxml=results/synthetic_baseline/test_results.xml
```

| File | Produced by | Content |
|---|---|---|
| `forecasts.csv` | run_backtest | Every forecast with realised loss, exception flag, quantile loss, FZ0, ES residual, status |
| `summary.csv` | run_backtest | VaR/ES summary by model and confidence |
| `coverage_tests.csv` | run_backtest | Kupiec, Christoffersen IND and CC tests |
| `es_evaluation.csv` | run_backtest | FZ0 scores, ES calibration diagnostics, tail severity |
| `stress_summary.csv` | run_backtest | Stress vs non-stress comparison |
| `traffic_light_windows.csv` | run_backtest | Rolling 250-day 99% traffic-light zones (educational) |
| `fit_diagnostics.csv` | run_backtest | Per-origin fit details (normal, Student-t, Monte Carlo) |
| `mc_convergence.csv`, `mc_convergence_raw.csv` | run_mc_convergence | MC error versus the analytical normal benchmark |
| `run_metadata.json` | run_backtest + run_mc_convergence | Configuration, seeds, checksums, environment, counts, failures |
| `test_results.xml` | pytest | Final test results (JUnit XML) |
| `research_report.md` | generate_report | Report built only from the files above |
| `figures/*.png` | run_backtest, run_mc_convergence | Static charts |

Column definitions are in [../docs/data_dictionary.md](../docs/data_dictionary.md).

`smoke_test/` (if present) comes from the small smoke configuration. It is a local
pipeline check, not a research result.
