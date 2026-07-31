"""Command-line entry point."""

import argparse
from pathlib import Path

from resid.artifacts import CsvArtifactWriter
from resid.data import (
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    analysis_window,
)
from resid.factors import CharacteristicFactorModel, momentum_factor, size_factor
from resid.pipeline import run_pipeline
from resid.regression import OLSResidualizer
from resid.returns import PercentageReturns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residualize returns with a configurable marcap model"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../marcap/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD output endpoint")
    parser.add_argument("--universe-size", type=int, default=200)
    parser.add_argument("--output-years", type=int, default=1)
    parser.add_argument("--momentum-lookback-days", type=int, default=252)
    parser.add_argument("--momentum-skip-days", type=int, default=21)
    parser.add_argument("--momentum-min-periods", type=int, default=126)
    parser.add_argument("--winsor-quantile", type=float, default=0.01)
    args = parser.parse_args()

    source = MarcapDataSource(args.data_dir)
    window = analysis_window(source, years=args.output_years, end=args.end_date)
    factor_model = CharacteristicFactorModel(
        (
            size_factor(name="SIZE"),
            momentum_factor(
                lookback_days=args.momentum_lookback_days,
                skip_days=args.momentum_skip_days,
                min_periods=args.momentum_min_periods,
                name="MOMENTUM",
            ),
        )
    )
    result = run_pipeline(
        window=window,
        source=source,
        universe_builder=FixedTopMarketCapUniverse(size=args.universe_size),
        return_calculator=PercentageReturns(),
        factor_model=factor_model,
        residualizer=OLSResidualizer(winsor_quantile=args.winsor_quantile),
    )
    manifest = CsvArtifactWriter().write(result, args.output_dir)
    dates = result.factor_returns["date"]
    universe_sizes = result.universe.groupby(level="date").sum()
    print(
        f"{len(dates)} regressions from {dates.min().date()} to "
        f"{dates.max().date()} with {int(universe_sizes.min())}-"
        f"{int(universe_sizes.max())} names/day"
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
