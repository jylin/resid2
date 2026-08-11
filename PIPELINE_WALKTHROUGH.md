# Residualization pipeline walkthrough

This document describes how `resid` moves from raw marcap observations to
daily residual returns. The pipeline is a sequence of aligned tables rather
than one large table that is mutated in place.

## Overall flow

```text
raw Parquet files
      |
      v
market_data -------> returns
      |                  |
      +---------+--------+
                v
        factor exposures
                |
                +-------> regression weights
                |
                v
        one cross-section per date
                |
                v
        factor returns and residuals
                |
                v
        accumulated, flat Parquet artifacts
```

The main orchestration is in `src/resid/pipeline.py:129`,
`run_pipeline`.

## Table conventions

Internally, most tables use a pandas `MultiIndex`:

```text
index levels: date, ticker
```

For example, `market_data` has two index keys and two value columns:

```text
date       ticker  return_percent  market_cap
2025-01-02 A               1.20       100000
           B              -0.40        80000
2025-01-03 A               0.50       102000
           B               1.10        81000
```

The date is normalized to a daily session in `src/resid/data.py:98-110`.
The final Parquet artifacts are flattened and contain explicit `date` and
`ticker` columns.

## 1. Build the universe

The CLI chooses either `FixedTopMarketCapUniverse` or
`DailyTopMarketCapUniverse` in `src/resid/cli.py:264-267`.

The daily builder (`src/resid/data.py:183-232`) produces a membership Series:

```text
date       ticker  in_universe
2025-01-02 A       True
           B       True
           C       True
2025-01-03 A       True
           C       True
           D       True
```

For daily selection, the members on date `t` are selected using the previous
trading session's market caps. The schedule is calculated up front, but the
membership can change every day.

The pipeline then takes the union of all tickers appearing in the schedule.
That union is used to determine which historical market data to load. It does
not mean every ticker is used in every day's regression.

## 2. Load market data

`run_pipeline` calculates a history start date and calls
`MarcapDataSource.load` in `src/resid/pipeline.py:158-168`.

The load includes:

- the complete analysis window;
- enough prior history for the longest factor lookback;
- every ticker that appears in the universe at any point in the window.

The result is an indexed DataFrame:

```text
index: date, ticker
values: return_percent, market_cap

date       ticker  return_percent  market_cap
2025-01-02 A               1.20       100000
           B              -0.40        80000
2025-01-03 A               0.50       102000
```

This is why `market_data` can have substantially more rows than
`analysis_dates * universe_size`: it includes factor history and the union of
all daily members.

## 3. Calculate returns

`PercentageReturns.calculate` in `src/resid/returns.py:15` creates a Series on
the same `(date, ticker)` index:

```text
date       ticker  return
2025-01-02 A        0.0120
           B       -0.0040
2025-01-03 A        0.0050
```

The calculation is simply:

```text
return = return_percent / 100
```

## 4. Prepare factor exposures

`CharacteristicFactorModel._exposures` in `src/resid/factors.py:70-96`
temporarily reshapes the indexed tables into wide date-by-ticker matrices:

```text
returns.unstack("ticker")

             A       B       C
date
2025-01-02  ...     ...     ...
2025-01-03  ...     ...     ...
```

Market caps are reshaped and aligned with `reindex_like`. Each factor builder
then returns a date-by-ticker matrix. Those matrices are flattened back to a
`(date, ticker)` DataFrame:

```text
date       ticker  SIZE   HML   MOM
2025-01-02 A       1.10  -0.30  0.42
           B       0.85   0.10 -0.15
2025-01-03 A       1.12  -0.20  0.38
```

The base factors are:

- `SIZE`: lagged log market cap;
- `HML`: long-term reversal from older returns;
- `MOM`: trailing momentum excluding the most recent observations.

If MKT is enabled, the recursive market-beta model adds the current beta to
the daily exposure table when that date is processed. It is not part of the
static base exposure table shown above.

## 5. Calculate regression weights

The weight model creates another Series aligned on `(date, ticker)`:

```text
date       ticker  regression_weight
2025-01-02 A              316.2
           B              282.8
2025-01-03 A              319.4
```

For `sqrt-cap-wls`, this is the square root of the previous session's positive
market cap. For OLS, the values are all `1.0`.

## 6. Build one daily cross-section

`_residualize` in `src/resid/pipeline.py:308-340` loops over the universe dates.
For each date it performs three aligned lookups:

```python
tickers = universe_members(universe, date)
day_returns = returns.xs(date, level="date").reindex(tickers)
day_exposures = factors.exposures(date, tickers)
day_weights = regression_weights.xs(date, level="date").reindex(tickers)
```

Conceptually, the result is one joined daily table:

```text
ticker  return   MKT   SIZE   HML   MOM   regression_weight
A       0.0120   0.82  1.10  -0.30  0.42              316.2
B      -0.0040   1.15  0.85   0.10 -0.15              282.8
C       0.0070   0.67  0.92   0.40  0.30              250.0
```

The built-in residualizers use an optimized NumPy path, so they do not
necessarily materialize this combined DataFrame on every date. The logical
operation is still the same: align returns, exposures, and weights by ticker,
drop unusable rows, and fit the cross-section.

The fit produces:

```text
factor_returns
----------------
INTERCEPT    0.0002
MKT          0.0110
SIZE         0.0040
HML         -0.0020
MOM          0.0060
```

and per-ticker fitted and specific returns:

```text
ticker  fitted_return  specific_return
A            0.0089          0.0031
B           -0.0023         -0.0017
C            0.0056          0.0014
```

With the sequential residualizers, factors are removed in the configured
order. The factor exposures are orthogonalized internally so the final
specific return is orthogonal to the modeled factor span.

### Recursive market beta timing

When MKT is enabled:

1. The beta state before date `t` becomes the MKT exposure for date `t`.
2. The cross-sectional regression is fit using that exposure.
3. After the observed return and market proxy for `t` are available, the beta
   state is updated.
4. The updated state is used on date `t + 1`.

This logic is in `_PreparedRecursiveMarketBeta.exposures` and
`_PreparedRecursiveMarketBeta.update` in `src/resid/market_beta.py:215-270`.

The market return itself is calculated from previous-session market-cap weights
in `_weighted_market_returns` at `src/resid/market_beta.py:273-296`. Market cap
therefore defines the market proxy; beta estimates each stock's sensitivity to
that proxy.

## 7. Accumulate results

`_Artifacts.add` in `src/resid/pipeline.py:255-291` converts each daily fit into
flat rows and appends them to per-output lists. At the end, those lists are
concatenated into the `ResidualizationResult`.

The logical result tables are:

| Table | Grain | Columns |
|---|---|---|
| `r` | date and ticker | `date`, `ticker`, `return` |
| `epsilon` | date and ticker | `date`, `ticker`, `specific_return` |
| `X` | date and ticker | `date`, `ticker`, `INTERCEPT`, one column per factor |
| `f` | date | `date`, `INTERCEPT`, one column per factor |
| `regression_weights` | date and ticker | `date`, `ticker`, `regression_weight` |
| `diagnostics` | date | `date`, `n_observations`, `rank`, `r_squared`, `residual_mean` |

Example final tables:

```text
r.parquet
date        ticker  return
2025-01-02  A       0.0120
2025-01-02  B      -0.0040

epsilon.parquet
date        ticker  specific_return
2025-01-02  A       0.0031
2025-01-02  B      -0.0017

X.parquet
date        ticker  INTERCEPT  MKT   SIZE   HML   MOM
2025-01-02  A       1.0        0.82  1.10  -0.30  0.42
2025-01-02  B       1.0        1.15  0.85   0.10 -0.15

f.parquet
date        INTERCEPT  MKT    SIZE    HML    MOM
2025-01-02  0.0002     0.011  0.004  -0.002  0.006
```

The Parquet writer in `src/resid/artifacts.py:30-59` currently persists
`epsilon`, `r`, `X`, and `f`. Weights, diagnostics, and the universe are
available in the in-memory `ResidualizationResult` and are used for validation
and diagnosis.

## Representative dimensions

For a one-year daily-top-1,000 run, a representative progress report looked
like this:

```text
universe:
  rows: 244,000
  dates: 244
  tickers: 1,337

market_data:
  rows: 1,794,681
  keys: 1,521 dates x 1,337 tickers

base exposures:
  rows: 2,033,577
  keys: 1,521 dates x 1,337 tickers

fitted output:
  rows: 208,494
  keys: 244 dates x 1,125 fitted tickers
```

The key-domain product is not necessarily equal to the row count: some
date/ticker pairs are absent because a name did not trade, lacked factor
history, or was excluded by the regression's finite-data filters.

## Validation and output

After all dates are fitted, the pipeline runs coverage, finite-value,
reconstruction, residual-orthogonality, and sequential-orthogonality checks in
`src/resid/validation.py`. A failed validation raises
`RegressionValidationError` before the CLI writes the final artifacts.

