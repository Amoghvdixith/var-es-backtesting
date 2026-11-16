# Data

## `synthetic/`
These files are generated offline by `python scripts/generate_data.py --config configs/synthetic_baseline.json`
with seed 42:

* `asset_returns.csv`: date, five decimal simple returns (`Equity_A` … `Equity_E`), `is_stress` and `regime`.
* `portfolio_returns.csv`: derived fixed-weight portfolio return and loss.
* `generation_metadata.json`: the data-generating process, its parameters and SHA-256 checksums.

**Provenance.** These are synthetic numbers produced by `src/var_es/synthetic.py`. They
are not real market data. They come from a stylised generator: correlated Gaussian latent
log returns, a shared truncated inverse-gamma scale mixture, and predefined volatility
regimes, converted with `expm1`. The generator is meant for testing risk models and makes
no claim about real equity dynamics. Regenerating with the same configuration reproduces
the same bytes.

See [../docs/data_dictionary.md](../docs/data_dictionary.md) for the schemas.

## `user/`
This folder is for optional user-supplied data. No file ships with the project. See
[user/README.md](user/README.md).

## `smoke_test/` (if present)
Produced by the smoke configuration `configs/smoke_test.json`. These files are a local
check only and are not part of the research deliverables.
