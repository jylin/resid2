"""Command-line entry point."""

import argparse
from pathlib import Path

from resid.artifacts import CsvArtifactWriter
from resid.data import (
    DailyTopMarketCapUniverse,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    analysis_window,
)
from resid.factors import (
    CharacteristicFactorModel,
    FactorModel,
    long_term_reversal_factor,
    momentum_factor,
    size_factor,
)
from resid.market_beta import RecursiveMarketBetaModel
from resid.pipeline import run_pipeline
from resid.regression import (
    SequentialOLSResidualizer,
    SequentialWLSResidualizer,
)
from resid.returns import PercentageReturns
from resid.validation import (
    FiniteOutputValidation,
    RegressionCoverageValidation,
    ReturnReconstructionValidation,
    SequentialOrthogonalityValidation,
)
from resid.weights import EqualRegressionWeights, SquareRootMarketCapWeights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Residualize returns with a configurable marcap model"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../marcap/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD output endpoint")
    parser.add_argument("--universe-method", choices=("fixed", "daily"), required=True)
    parser.add_argument("--universe-size", type=int, default=1000)
    parser.add_argument("--output-years", type=int, default=1)
    parser.add_argument("--momentum-lookback-days", type=int, default=189)
    parser.add_argument("--momentum-skip-days", type=int, default=63)
    parser.add_argument("--momentum-min-periods", type=int, default=126)
    parser.add_argument("--value-lookback-days", type=int, default=1008)
    parser.add_argument("--value-skip-days", type=int, default=252)
    parser.add_argument("--value-min-periods", type=int, default=756)
    parser.add_argument("--market-beta", action="store_true")
    parser.add_argument("--market-beta-lookback-days", type=int, default=252)
    parser.add_argument("--market-beta-min-periods", type=int, default=126)
    parser.add_argument("--market-beta-decay", type=float, default=0.97)
    parser.add_argument(
        "--regression-method",
        choices=("ols", "sqrt-cap-wls"),
        required=True,
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.01)
    parser.add_argument("--minimum-date-coverage", type=float, default=0.95)
    parser.add_argument("--validation-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    source = MarcapDataSource(args.data_dir)
    window = analysis_window(source, years=args.output_years, end=args.end_date)
    base_factor_model = CharacteristicFactorModel(
        (
            size_factor(name="SMB", small_minus_big=True),
            long_term_reversal_factor(
                lookback_days=args.value_lookback_days,
                skip_days=args.value_skip_days,
                min_periods=args.value_min_periods,
                name="HML",
            ),
            momentum_factor(
                lookback_days=args.momentum_lookback_days,
                skip_days=args.momentum_skip_days,
                min_periods=args.momentum_min_periods,
                name="MOM",
            ),
        )
    )
    factor_model: FactorModel = base_factor_model
    if args.market_beta:
        factor_model = RecursiveMarketBetaModel(
            base=base_factor_model,
            lookback_days=args.market_beta_lookback_days,
            min_periods=args.market_beta_min_periods,
            decay=args.market_beta_decay,
            name="MKT",
        )
    universe_builder = (
        DailyTopMarketCapUniverse(size=args.universe_size)
        if args.universe_method == "daily"
        else FixedTopMarketCapUniverse(size=args.universe_size)
    )
    factor_order = (
        ("MKT", "SMB", "HML", "MOM") if args.market_beta else ("SMB", "HML", "MOM")
    )
    unscaled_factors = ("MKT",) if args.market_beta else ()
    if args.regression_method == "sqrt-cap-wls":
        residualizer = SequentialWLSResidualizer(
            factor_order=factor_order,
            winsor_quantile=args.winsor_quantile,
            unscaled_factors=unscaled_factors,
        )
        regression_weight_model = SquareRootMarketCapWeights()
    else:
        residualizer = SequentialOLSResidualizer(
            factor_order=factor_order,
            winsor_quantile=args.winsor_quantile,
            unscaled_factors=unscaled_factors,
        )
        regression_weight_model = EqualRegressionWeights()
    result = run_pipeline(
        window=window,
        source=source,
        universe_builder=universe_builder,
        return_calculator=PercentageReturns(),
        factor_model=factor_model,
        residualizer=residualizer,
        regression_weight_model=regression_weight_model,
        validations=(
            RegressionCoverageValidation(
                minimum_date_coverage=args.minimum_date_coverage
            ),
            FiniteOutputValidation(),
            ReturnReconstructionValidation(
                absolute_tolerance=args.validation_tolerance
            ),
            SequentialOrthogonalityValidation(
                absolute_tolerance=args.validation_tolerance
            ),
        ),
    )
    manifest = CsvArtifactWriter().write(result, args.output_dir)
    dates = result.factor_returns["date"]
    observations = result.diagnostics["n_observations"]
    print(
        f"{len(dates)} regressions from {dates.min().date()} to "
        f"{dates.max().date()} with {int(observations.min())}-"
        f"{int(observations.max())} usable names/day "
        f"({args.universe_size} target)"
    )
    print(
        "validations: "
        + ", ".join(
            f"{validation.name}={'pass' if validation.passed else 'fail'}"
            for validation in result.validation_results
        )
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
