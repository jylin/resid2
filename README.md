# resid

`resid` is a small library for daily cross-sectional return residualization.
It includes a CLI for local
[FinanceData/marcap](https://github.com/FinanceData/marcap) KRX data and does
not download or supplement that dataset.

## Report

- [Overview](https://jylin.github.io/resid2/)
- [Walkthrough](https://jylin.github.io/resid2/walkthrough.html)
- [Design](https://jylin.github.io/resid2/design.html)

## Setup

Install [mise](https://mise.jdx.dev/getting-started), then:

```bash
git clone https://github.com/FinanceData/marcap.git ../marcap
mise install
uv sync
```

Python 3.14 and the dependencies are pinned in `mise.toml` and `uv.lock`.

## CLI

```bash
uv run resid \
  --data-dir ../marcap/data \
  --output-dir output/full \
  --universe-method daily \
  --output-years 10 \
  --universe-size 1000 \
  --momentum-lookback-days 189 \
  --momentum-skip-days 63 \
  --momentum-min-periods 126 \
  --value-lookback-days 1008 \
  --value-skip-days 252 \
  --value-min-periods 756 \
  --market-beta \
  --market-beta-lookback-days 252 \
  --market-beta-min-periods 126 \
  --market-beta-decay 0.97 \
  --regression-method ols \
  --winsor-quantile 0.01 \
  --minimum-date-coverage 0.99 \
  --validation-tolerance 1e-12
```

The CLI selects a daily point-in-time top-1,000 universe and gives every name
equal regression influence. It removes `MKT`, `SMB`, `HML`, then `MOM`; X and f
preserve that order. Every exposure is centered, and all but `MKT` are scaled to
unit variance, so the `MKT` slope is a return per unit of beta while the others
are per standardized unit. Use `--regression-method sqrt-cap-wls` for the
cap-weighted alternative. See the walkthrough for the universe, attribution, and
diagnostic evidence.

### Outputs

| Notation | File | Schema |
|---|---|---|
| ε | `epsilon.csv` | `(date, ticker, specific_return)` |
| r | `r.csv` | `(date, ticker, return)` |
| X | `X.csv` | `(date, ticker, <one column per factor>)` |
| f | `f.csv` | `(date, <one column per factor>)` |

`epsilon.csv` is the primary output. The other files support reconstruction
and diagnosis.
