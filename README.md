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
uv run resid ../marcap/data --config resid.toml
```

The positional data path is required. Other CLI values override matching TOML
fields; for example, `--universe-size 500` or `--regression-method sqrt-cap-wls`.
Copy `resid.toml` for another experiment. See `uv run resid --help` for all
overrides. Artifacts are written as typed, compressed Parquet files.

### Outputs

| Notation | File | Schema |
|---|---|---|
| ε | `epsilon.parquet` | `(date, ticker, specific_return)` |
| r | `r.parquet` | `(date, ticker, return)` |
| X | `X.parquet` | `(date, ticker, <one column per factor>)` |
| f | `f.parquet` | `(date, <one column per factor>)` |

`epsilon` is the primary output. The other files support reconstruction and
diagnosis. Sequential fits expose a weighted orthogonal basis, so `X` is the
exact design used for reconstruction.
