"""Explicit orchestration across the pipeline boundaries."""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from resid.data import (
    AnalysisWindow,
    MarketDataSource,
    UniverseBuilder,
    history_start,
    universe_dates,
    universe_members,
)
from resid.factors import FactorModel, PreparedFactorModel
from resid.progress import NullProgress, ProgressReporter, StageProgress, reported_stage
from resid.regression import CrossSectionFit, ResidualizationResult, Residualizer
from resid.returns import ReturnCalculator
from resid.validation import RegressionValidation, RegressionValidationError
from resid.weights import RegressionWeightModel


def _table_key_fields(table: pd.DataFrame | pd.Series) -> tuple[tuple[str, int], ...]:
    fields: list[tuple[str, int]] = []
    index = table.index
    if isinstance(index, pd.MultiIndex):
        for level, name in enumerate(index.names):
            if name is not None:
                fields.append(
                    (
                        str(name),
                        int(index.get_level_values(level).nunique()),
                    )
                )
    elif index.name is not None:
        fields.append((str(index.name), int(index.nunique())))

    if isinstance(table, pd.DataFrame):
        known = {name for name, _ in fields}
        for name in ("date", "ticker"):
            if name in table.columns and name not in known:
                fields.append((name, int(table[name].nunique())))
    return tuple(fields)


def _table_preview(name: str, table: pd.DataFrame | pd.Series) -> str:
    key_fields = _table_key_fields(table)
    key_names = {key for key, _ in key_fields}
    if isinstance(table, pd.DataFrame):
        values = [
            str(column) for column in table.columns if str(column) not in key_names
        ]
    else:
        values = [str(table.name)] if table.name is not None else ["value"]

    value_text = ", ".join(values)
    if len(value_text) > 72:
        value_text = f"{value_text[:69]}..."
    lines = [f"{name}: {len(table):,} rows"]
    if key_fields:
        dimensions = " x ".join(
            f"{count:,} {key_name}s"
            if key_name in {"date", "ticker"}
            else f"{count:,} {key_name}"
            for key_name, count in key_fields
        )
        lines.append(f"  keys: {dimensions}")
    if value_text:
        lines.append(f"  values: {value_text}")
    return "\n".join(lines)


def _tables_preview(*tables: tuple[str, pd.DataFrame | pd.Series]) -> str:
    return "\n".join(
        ("tables:", *(f"  {_table_preview(name, table)}" for name, table in tables))
    )


def _universe_preview(universe: pd.Series) -> str:
    index = universe.index
    dates = index.get_level_values("date").nunique()
    tickers = index.get_level_values("ticker").nunique()
    active = int(np.count_nonzero(universe.to_numpy(dtype="bool")))
    return "\n".join(
        (
            "universe:",
            f"  rows: {len(universe):,}",
            f"  dates: {dates:,}",
            f"  tickers: {tickers:,}",
            f"  active: {active:,}",
        )
    )


def _prepared_preview(
    prepared: PreparedFactorModel,
    market_data: pd.DataFrame,
    returns: pd.Series,
) -> str:
    base = getattr(prepared, "values", None)
    if base is None:
        base = getattr(getattr(prepared, "base", None), "values", None)
    tables = [("market_data", market_data), ("returns", returns)]
    if isinstance(base, pd.DataFrame):
        tables.append(("exposures", base))
    market_returns = getattr(prepared, "market_returns", None)
    if isinstance(market_returns, pd.Series):
        tables.append(("market_returns", market_returns))
    return _tables_preview(*tables)


def _result_preview(result: ResidualizationResult) -> str:
    return "\n".join(
        (
            _universe_preview(result.universe),
            _tables_preview(
                ("epsilon", result.specific_returns),
                ("r", result.returns),
                ("X", result.exposures),
                ("f", result.factor_returns),
                ("diagnostics", result.diagnostics),
                ("weights", result.regression_weights),
            ),
        )
    )


def run_pipeline(
    *,
    window: AnalysisWindow,
    source: MarketDataSource,
    universe_builder: UniverseBuilder,
    return_calculator: ReturnCalculator,
    factor_model: FactorModel,
    residualizer: Residualizer,
    regression_weight_model: RegressionWeightModel,
    validations: tuple[RegressionValidation, ...] = (),
    progress: ProgressReporter | None = None,
) -> ResidualizationResult:
    """Run the caller-specified stages without writing artifacts."""

    reporter = progress if progress is not None else NullProgress()
    universe_steps = getattr(universe_builder, "progress_steps", None)
    with reported_stage(
        reporter,
        "Build universe",
        total=universe_steps,
        unit="steps",
    ) as universe_stage:
        build_with_progress = getattr(universe_builder, "build_with_progress", None)
        if build_with_progress is None:
            universe = universe_builder.build(source, window)
        else:
            universe = build_with_progress(source, window, universe_stage)
        universe_stage.update(universe_steps or 0)
        universe_stage.summary(_universe_preview(universe))
    tickers = universe.index.get_level_values("ticker").unique()
    start = history_start(
        window.start,
        max(
            factor_model.history_business_days,
            regression_weight_model.history_business_days,
        ),
    )
    with reported_stage(reporter, "Load market data") as stage:
        market_data = source.load(start, window.end, tickers)
        stage.summary(_tables_preview(("market_data", market_data)))
    returns = return_calculator.calculate(market_data)
    factor_steps = getattr(factor_model, "progress_steps", None)
    with reported_stage(
        reporter,
        "Prepare factors",
        total=factor_steps,
        unit="steps",
    ) as factor_stage:
        prepare_with_progress = getattr(factor_model, "prepare_with_progress", None)
        if prepare_with_progress is None:
            prepared = factor_model.prepare(market_data, returns, universe)
        else:
            prepared = prepare_with_progress(
                market_data, returns, universe, factor_stage
            )
        factor_stage.update(factor_steps or 0)
        factor_stage.summary(_prepared_preview(prepared, market_data, returns))
    with reported_stage(reporter, "Calculate regression weights") as stage:
        regression_weights = regression_weight_model.calculate(market_data)
        stage.summary(
            _tables_preview(
                ("market_data", market_data),
                ("regression_weights", regression_weights),
            )
        )
    dates = universe_dates(universe)
    with reported_stage(
        reporter,
        "Residualize returns",
        total=len(dates),
        unit="dates",
    ) as stage:
        result = _residualize(
            returns,
            prepared,
            residualizer,
            regression_weights,
            universe,
            dates,
            stage,
        )
        stage.update(len(dates))
        stage.summary(_result_preview(result))

    with reported_stage(
        reporter,
        "Validate outputs",
        total=len(validations),
        unit="checks",
    ) as stage:
        validation_results_list: list = []
        for position, validation in enumerate(validations, start=1):
            validation_results_list.append(validation.validate(result))
            stage.update(position)
        validation_results = tuple(validation_results_list)
        stage.summary(
            "\n".join(
                (
                    "checks:",
                    *(
                        f"  {item.name}: {'pass' if item.passed else 'fail'}"
                        for item in validation_results
                    ),
                    _result_preview(result),
                )
            )
        )
    result = replace(result, validation_results=validation_results)
    failures = tuple(item for item in validation_results if not item.passed)
    if failures:
        raise RegressionValidationError(failures)
    return result


@dataclass(slots=True, eq=False)
class _Artifacts:
    """Accumulate one row group per completed daily fit."""

    model_columns: tuple[str, ...] | None = None
    exposures: list[pd.DataFrame] = field(default_factory=list)
    returns: list[pd.DataFrame] = field(default_factory=list)
    specific_returns: list[pd.DataFrame] = field(default_factory=list)
    regression_weights: list[pd.DataFrame] = field(default_factory=list)
    factor_returns: list[dict[str, object]] = field(default_factory=list)
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    def add(self, date: pd.Timestamp, fit: CrossSectionFit) -> None:
        columns = tuple(str(name) for name in fit.factor_returns.index)
        if self.model_columns is None:
            self.model_columns = columns
        elif self.model_columns != columns:
            raise ValueError("residualizer changed model columns between dates")

        date_column = np.repeat(date.to_datetime64(), len(fit.returns))
        ticker_column = fit.returns.index.astype("string")
        keys = {"date": date_column, "ticker": ticker_column}

        exposures = fit.exposures.copy()
        exposures.insert(0, "INTERCEPT", 1.0)
        exposures.insert(0, "ticker", ticker_column)
        exposures.insert(0, "date", date_column)
        self.exposures.append(exposures.reset_index(drop=True))
        self.returns.append(pd.DataFrame({**keys, "return": fit.returns.to_numpy()}))
        self.specific_returns.append(
            pd.DataFrame({**keys, "specific_return": fit.specific_returns.to_numpy()})
        )
        self.regression_weights.append(
            pd.DataFrame(
                {**keys, "regression_weight": fit.regression_weights.to_numpy()}
            )
        )
        self.factor_returns.append({"date": date, **fit.factor_returns.to_dict()})
        self.diagnostics.append(
            {
                "date": date,
                "n_observations": len(fit.returns),
                "rank": fit.rank,
                "r_squared": fit.r_squared,
                "residual_mean": float(
                    np.average(fit.specific_returns, weights=fit.regression_weights)
                ),
            }
        )

    def build(self, universe: pd.Series) -> ResidualizationResult:
        if self.model_columns is None:
            raise ValueError("no dates had enough valid observations to fit")
        return ResidualizationResult(
            specific_returns=pd.concat(self.specific_returns, ignore_index=True),
            returns=pd.concat(self.returns, ignore_index=True),
            exposures=pd.concat(self.exposures, ignore_index=True),
            factor_returns=pd.DataFrame(self.factor_returns),
            diagnostics=pd.DataFrame(self.diagnostics),
            regression_weights=pd.concat(self.regression_weights, ignore_index=True),
            universe=universe,
            model_columns=self.model_columns,
        )


def _residualize(
    returns: pd.Series,
    factors: PreparedFactorModel,
    residualizer: Residualizer,
    regression_weights: pd.Series,
    universe: pd.Series,
    dates: pd.DatetimeIndex,
    progress: StageProgress,
) -> ResidualizationResult:
    artifacts = _Artifacts()
    for position, date in enumerate(dates, start=1):
        tickers = universe_members(universe, date)
        day_returns = returns.xs(date, level="date").reindex(tickers)
        day_exposures = factors.exposures(date, tickers)
        day_weights = regression_weights.xs(date, level="date").reindex(tickers)
        fit_numpy = getattr(residualizer, "fit_numpy", None)
        if fit_numpy is None:
            fit = residualizer.fit(day_returns, day_exposures, day_weights)
        else:
            # The sequential built-ins can filter and fit directly on arrays,
            # avoiding a temporary concatenated DataFrame for every date. Keep
            # the protocol fallback above for caller-supplied residualizers.
            fit = fit_numpy(
                returns=day_returns.to_numpy(dtype="float64"),
                exposures=day_exposures,
                regression_weights=day_weights.to_numpy(dtype="float64"),
                index=day_returns.index,
            )
        if fit is None:
            progress.update(position)
            continue
        artifacts.add(date, fit)
        # Preserve the public two-argument PreparedFactorModel API while giving
        # stateful factors access to finite returns excluded by another factor's
        # missing history.
        factors.update(date, replace(fit, observed_returns=day_returns))
        progress.update(position)
    return artifacts.build(universe)
