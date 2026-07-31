# resid

`resid` is a small library for daily cross-sectional return residualization.
It includes a CLI for local
[FinanceData/marcap](https://github.com/FinanceData/marcap) KRX data but does
not download data or prescribe a universe, factor model, or residualizer.

`run_pipeline` requires a data source, universe builder, return calculator,
factor model, and residualizer. CSV writing is a separate operation.

## Setup

Clone marcap next to this repository so its data is under `../marcap/data/`,
then install Python 3.14 and the locked dependencies:

```bash
git clone https://github.com/FinanceData/marcap.git ../marcap
mise install
uv sync
```

Mise also supplies Ruff and ty.

## CLI

This invocation selects a fixed first-day top-200 universe, a one-year output
window, size and momentum factors, daily OLS, and CSV output:

```bash
uv run resid \
  --data-dir ../marcap/data \
  --output-dir output \
  --end-date 2025-12-31 \
  --output-years 1 \
  --universe-size 200 \
  --momentum-lookback-days 252 \
  --momentum-skip-days 21 \
  --momentum-min-periods 126 \
  --winsor-quantile 0.01
```

Omit `--end-date` to use the latest local observation. See all arguments with
`uv run resid --help`.

## Outputs

The CSV files follow standard factor-model notation:

| Notation | File | Contents | Schema |
|---|---|---|---|
| ε | `epsilon.csv` | Specific returns | `(date, ticker, specific_return)` |
| r | `r.csv` | Raw returns used by the regressions | `(date, ticker, return)` |
| X | `X.csv` | Normalized factor exposures | `(date, ticker, <one column per factor>)` |
| f | `f.csv` | Estimated daily factor returns | `(date, <one column per factor>)` |

`epsilon.csv` is the primary output; r, X, and f are supporting artifacts. The
included model exports `INTERCEPT`, `SIZE`, and `MOMENTUM` columns in X and f,
making `r = Xf + ε` directly reproducible.

Ticker values are zero-padded strings. Preserve them when reading CSV:

```python
import pandas as pd

epsilon = pd.read_csv("output/epsilon.csv", dtype={"ticker": "string"})
```

For another format, write the DataFrames on `ResidualizationResult` directly.

## Python API

Every pipeline component is supplied explicitly:

```python
from pathlib import Path

import resid

source = resid.MarcapDataSource(Path("../marcap/data"))
base_factor_model = resid.CharacteristicFactorModel(
    (
        resid.size_factor(name="SIZE"),
        resid.momentum_factor(
            lookback_days=252,
            skip_days=21,
            min_periods=126,
            name="MOMENTUM",
        ),
    )
)
factor_model = resid.RecursiveMarketBetaModel(
    base=base_factor_model,
    lookback_days=252,
    min_periods=126,
    decay=0.97,
)

result = resid.run_pipeline(
    window=resid.analysis_window(source, years=1, end=None),
    source=source,
    universe_builder=resid.FixedTopMarketCapUniverse(size=200),
    return_calculator=resid.PercentageReturns(),
    factor_model=factor_model,
    residualizer=resid.OLSResidualizer(winsor_quantile=0.01),
    validations=(
        resid.RegressionCoverageValidation(minimum_date_coverage=0.95),
        resid.FiniteOutputValidation(),
        resid.ReturnReconstructionValidation(absolute_tolerance=1e-12),
    ),
)
resid.CsvArtifactWriter().write(result, Path("output"))
```

`RecursiveMarketBetaModel` initializes beta from lagged-cap-weighted market
returns. Each date uses the prior beta, fits `MARKET_BETA`, then updates beta
for the next date. Use `base_factor_model` directly to omit it.

Failed validations raise `RegressionValidationError`; passing results are in
`result.validation_results`. Custom checks implement `RegressionValidation`.

A custom `Factor(name, build, history_business_days)` receives aligned
date-by-ticker return and market-cap matrices. `history_business_days` controls
loading padding, not an exact number of exchange observations. Alternative
regression methods implement the `Residualizer` protocol.

## Stage contracts

| Output | Contract |
|---|---|
| Market data | DataFrame indexed by `(date, ticker)` with `return_percent` and `market_cap` |
| Universe | Boolean Series indexed by `(date, ticker)` |
| Returns | Series named `return`, indexed by `(date, ticker)` |
| Exposures | DataFrame indexed by `(date, ticker)`, one column per factor |
| Result | `ResidualizationResult` containing ε, r, X, f, diagnostics, membership, and validation results |

These boundaries allow another market feed, point-in-time universe, factor
model, or residualizer without restructuring the other stages.

## Notebook

The executed [usage notebook](examples/usage.ipynb) checks `r = Xf + ε`, reports
daily cross-sectional R², plots factor returns, runs the stages individually,
and demonstrates a custom factor model. Its outputs are saved for online
preview.

```bash
uv run jupyter lab examples/usage.ipynb
```

## Checks

```bash
mise run test
mise run typecheck
mise run lint
mise run format
```
