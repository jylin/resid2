"""Command-line entry point."""

import argparse
import tomllib
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from resid.artifacts import ParquetArtifactWriter
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
from resid.progress import TerminalProgress, reported_stage
from resid.regression import SequentialOLSResidualizer, SequentialWLSResidualizer
from resid.returns import PercentageReturns
from resid.validation import (
    FiniteOutputValidation,
    RegressionCoverageValidation,
    ResidualOrthogonalityValidation,
    ReturnReconstructionValidation,
    SequentialOrthogonalityValidation,
)
from resid.weights import (
    EqualRegressionWeights,
    RegressionWeightModel,
    SquareRootMarketCapWeights,
)


class _ConfigModel(BaseModel):
    """Reject misspelled TOML keys instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(_ConfigModel):
    years: int = Field(gt=0)
    end_date: str | None = None


class OutputConfig(_ConfigModel):
    directory: Path


class UniverseConfig(_ConfigModel):
    method: Literal["fixed", "daily"]
    size: int = Field(gt=0)


class FactorWindowConfig(_ConfigModel):
    lookback_days: int = Field(gt=0)
    skip_days: int = Field(gt=0)
    min_periods: int = Field(gt=0)

    @model_validator(mode="after")
    def periods_fit_window(self) -> FactorWindowConfig:
        if self.min_periods > self.lookback_days:
            raise ValueError("min_periods cannot exceed lookback_days")
        return self


class MarketBetaConfig(_ConfigModel):
    enabled: bool
    lookback_days: int = Field(gt=0)
    min_periods: int = Field(gt=0)
    # Must match RecursiveMarketBetaModel.decay: 1.0 never forgets, so the model
    # itself rejects it, and accepting it here would defer that to a raw traceback.
    decay: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def periods_fit_window(self) -> MarketBetaConfig:
        if self.min_periods > self.lookback_days:
            raise ValueError("min_periods cannot exceed lookback_days")
        return self


class FactorsConfig(_ConfigModel):
    order: tuple[str, ...]
    momentum: FactorWindowConfig
    value: FactorWindowConfig
    market_beta: MarketBetaConfig

    @model_validator(mode="after")
    def order_matches_factors(self) -> FactorsConfig:
        expected = {"SIZE", "HML", "MOM"}
        if self.market_beta.enabled:
            expected.add("MKT")
        if len(self.order) != len(set(self.order)) or set(self.order) != expected:
            raise ValueError(f"order must contain exactly {sorted(expected)}")
        return self


class RegressionConfig(_ConfigModel):
    """Both methods remove factors sequentially in `factors.order`.

    `ols` weights every name equally; `sqrt-cap-wls` weights by the square root of
    prior-session market cap. Neither is a single multivariate fit — for that, call
    `run_pipeline` directly with `OLSResidualizer`.
    """

    method: Literal["ols", "sqrt-cap-wls"]
    winsor_quantile: float = Field(ge=0, lt=0.5)


class ValidationConfig(_ConfigModel):
    minimum_date_coverage: float = Field(gt=0, le=1)
    tolerance: float = Field(gt=0)


class CliConfig(_ConfigModel):
    """Complete, explicit run configuration loaded from TOML."""

    run: RunConfig
    output: OutputConfig
    universe: UniverseConfig
    factors: FactorsConfig
    regression: RegressionConfig
    validation: ValidationConfig


def load_config(path: Path) -> CliConfig:
    """Load and validate a hierarchical TOML run configuration."""

    with path.open("rb") as stream:
        return CliConfig.model_validate(tomllib.load(stream))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Residualize returns")
    parser.add_argument("data_dir", type=Path, help="marcap data directory")
    parser.add_argument("--config", type=Path, required=True, help="TOML defaults")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--end-date")
    parser.add_argument("--years", type=int)
    parser.add_argument("--universe-method", choices=("fixed", "daily"))
    parser.add_argument("--universe-size", type=int)
    parser.add_argument("--factor-order", nargs="+")
    parser.add_argument("--momentum-lookback-days", type=int)
    parser.add_argument("--momentum-skip-days", type=int)
    parser.add_argument("--momentum-min-periods", type=int)
    parser.add_argument("--value-lookback-days", type=int)
    parser.add_argument("--value-skip-days", type=int)
    parser.add_argument("--value-min-periods", type=int)
    parser.add_argument(
        "--market-beta",
        dest="market_beta_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--market-beta-lookback-days", type=int)
    parser.add_argument("--market-beta-min-periods", type=int)
    parser.add_argument("--market-beta-decay", type=float)
    parser.add_argument(
        "--regression-method",
        choices=("ols", "sqrt-cap-wls"),
        help="sequential removal weighting: equal (ols) or sqrt market cap",
    )
    parser.add_argument("--winsor-quantile", type=float)
    parser.add_argument("--minimum-date-coverage", type=float)
    parser.add_argument("--validation-tolerance", type=float)
    return parser


def apply_cli_overrides(config: CliConfig, args: argparse.Namespace) -> CliConfig:
    """Apply non-empty CLI values over a validated TOML configuration."""

    values = config.model_dump(mode="python")
    overrides = (
        ("run", "years", args.years),
        ("run", "end_date", args.end_date),
        ("output", "directory", args.output_dir),
        ("universe", "method", args.universe_method),
        ("universe", "size", args.universe_size),
        ("regression", "method", args.regression_method),
        ("regression", "winsor_quantile", args.winsor_quantile),
        ("validation", "minimum_date_coverage", args.minimum_date_coverage),
        ("validation", "tolerance", args.validation_tolerance),
    )
    for section, name, value in overrides:
        if value is not None:
            values[section][name] = value
    if args.factor_order is not None:
        values["factors"]["order"] = tuple(args.factor_order)

    factor_overrides = (
        ("momentum", "lookback_days", args.momentum_lookback_days),
        ("momentum", "skip_days", args.momentum_skip_days),
        ("momentum", "min_periods", args.momentum_min_periods),
        ("value", "lookback_days", args.value_lookback_days),
        ("value", "skip_days", args.value_skip_days),
        ("value", "min_periods", args.value_min_periods),
        ("market_beta", "enabled", args.market_beta_enabled),
        ("market_beta", "lookback_days", args.market_beta_lookback_days),
        ("market_beta", "min_periods", args.market_beta_min_periods),
        ("market_beta", "decay", args.market_beta_decay),
    )
    for section, name, value in factor_overrides:
        if value is not None:
            values["factors"][section][name] = value
    return CliConfig.model_validate(values)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_started = perf_counter()
    progress = TerminalProgress(total_stages=7)
    try:
        config = apply_cli_overrides(load_config(args.config), args)
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        parser.error(f"invalid config or override: {exc}")

    source = MarcapDataSource(args.data_dir)
    window = analysis_window(source, years=config.run.years, end=config.run.end_date)
    base_factor_model = CharacteristicFactorModel(
        (
            size_factor(name="SIZE"),
            long_term_reversal_factor(
                lookback_days=config.factors.value.lookback_days,
                skip_days=config.factors.value.skip_days,
                min_periods=config.factors.value.min_periods,
                name="HML",
            ),
            momentum_factor(
                lookback_days=config.factors.momentum.lookback_days,
                skip_days=config.factors.momentum.skip_days,
                min_periods=config.factors.momentum.min_periods,
                name="MOM",
            ),
        )
    )
    factor_model: FactorModel = base_factor_model
    if config.factors.market_beta.enabled:
        factor_model = RecursiveMarketBetaModel(
            base=base_factor_model,
            lookback_days=config.factors.market_beta.lookback_days,
            min_periods=config.factors.market_beta.min_periods,
            decay=config.factors.market_beta.decay,
            name="MKT",
        )
    universe_builder = (
        DailyTopMarketCapUniverse(size=config.universe.size)
        if config.universe.method == "daily"
        else FixedTopMarketCapUniverse(size=config.universe.size)
    )
    factor_order = config.factors.order
    unscaled_factors = ("MKT",) if config.factors.market_beta.enabled else ()
    residualizer: SequentialOLSResidualizer | SequentialWLSResidualizer
    if config.regression.method == "sqrt-cap-wls":
        residualizer = SequentialWLSResidualizer(
            factor_order=factor_order,
            winsor_quantile=config.regression.winsor_quantile,
            unscaled_factors=unscaled_factors,
        )
        regression_weight_model: RegressionWeightModel = SquareRootMarketCapWeights()
    else:
        residualizer = SequentialOLSResidualizer(
            factor_order=factor_order,
            winsor_quantile=config.regression.winsor_quantile,
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
                minimum_date_coverage=config.validation.minimum_date_coverage
            ),
            FiniteOutputValidation(),
            ReturnReconstructionValidation(
                absolute_tolerance=config.validation.tolerance
            ),
            ResidualOrthogonalityValidation(
                absolute_tolerance=config.validation.tolerance
            ),
            SequentialOrthogonalityValidation(
                absolute_tolerance=config.validation.tolerance
            ),
        ),
        progress=progress,
    )
    with reported_stage(
        progress,
        "Write Parquet artifacts",
        total=4,
        unit="files",
    ) as stage:
        manifest = ParquetArtifactWriter().write(
            result,
            config.output.directory,
            progress=stage,
        )
    elapsed = perf_counter() - run_started
    dates = result.factor_returns["date"]
    observations = result.diagnostics["n_observations"]
    fitted_rows = len(result.specific_returns)
    output_bytes = sum(
        Path(str(path)).stat().st_size for path in manifest["path"].tolist()
    )
    regression_rate = len(dates) / elapsed if elapsed > 0 else 0.0
    row_rate = fitted_rows / elapsed if elapsed > 0 else 0.0
    output_rate = output_bytes / elapsed / 1024**2 if elapsed > 0 else 0.0
    print("run summary")
    print(
        f"  regressions: {len(dates):,} ({dates.min().date()} to {dates.max().date()})"
    )
    print(
        f"  usable names/day: {int(observations.min()):,}-"
        f"{int(observations.max()):,} (target: {config.universe.size:,})"
    )
    print("  validations:")
    for validation in result.validation_results:
        status = "pass" if validation.passed else "fail"
        print(f"    {validation.name}: {status}")
    print("  artifacts:")
    for artifact, rows, path in manifest.itertuples(index=False, name=None):
        print(f"    {artifact}: {int(rows):,} rows -> {path}")
    print("  timing:")
    print(f"    total: {elapsed:.2f}s")
    print(f"    regressions: {len(dates):,} ({regression_rate:.1f}/s)")
    print(f"    fitted rows: {fitted_rows:,} ({row_rate:,.0f}/s)")
    print(f"    output: {_format_bytes(output_bytes)} ({output_rate:.1f} MiB/s)")


if __name__ == "__main__":
    main()
