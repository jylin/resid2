"""Explicit orchestration across the pipeline boundaries."""

from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd

from resid.data import AnalysisWindow, MarketDataSource, UniverseBuilder
from resid.factors import FactorModel, PreparedFactorModel
from resid.regression import CrossSectionFit, ResidualizationResult, Residualizer
from resid.returns import ReturnCalculator
from resid.validation import RegressionValidation, RegressionValidationError
from resid.weights import RegressionWeightModel


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
) -> ResidualizationResult:
    """Run the caller-specified stages without writing artifacts."""

    universe = universe_builder.build(source, window)
    tickers = universe.index.get_level_values("ticker").unique()
    history_days = max(
        factor_model.history_business_days,
        regression_weight_model.history_business_days,
    )
    history_start = window.start - pd.offsets.BDay(history_days)
    market_data = source.load(history_start, window.end, tickers)
    returns = return_calculator.calculate(market_data)
    prepared = factor_model.prepare(market_data, returns, universe)
    regression_weights = regression_weight_model.calculate(market_data)
    result = _residualize(returns, prepared, residualizer, regression_weights, universe)

    validation_results = tuple(
        validation.validate(result) for validation in validations
    )
    result = replace(result, validation_results=validation_results)
    failures = tuple(item for item in validation_results if not item.passed)
    if failures:
        raise RegressionValidationError(failures)
    return result


def _residualize(
    returns: pd.Series,
    factors: PreparedFactorModel,
    residualizer: Residualizer,
    regression_weights: pd.Series,
    universe: pd.Series,
) -> ResidualizationResult:
    model_columns: tuple[str, ...] | None = None
    exposure_rows: list[pd.DataFrame] = []
    return_rows: list[pd.DataFrame] = []
    specific_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    factor_return_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    dates = universe.loc[universe].index.get_level_values("date").unique().sort_values()

    for date in dates:
        date_value = cast(pd.Timestamp, pd.Timestamp(date))
        membership = universe.xs(date_value, level="date")
        tickers = membership.index[membership.to_numpy(dtype="bool")]
        day_returns = returns.xs(date_value, level="date").reindex(tickers)
        fit = residualizer.fit(
            day_returns,
            factors.exposures(date_value, tickers),
            regression_weights.xs(date_value, level="date").reindex(tickers),
        )
        if fit is None:
            continue

        fit_columns = tuple(str(name) for name in fit.factor_returns.index)
        if model_columns is None:
            model_columns = fit_columns
        elif model_columns != fit_columns:
            raise ValueError("residualizer changed model columns between dates")

        (
            exposure_frame,
            return_frame,
            specific_frame,
            weight_frame,
            factor_return_row,
            diagnostic_row,
        ) = _fit_outputs(date_value, fit)
        exposure_rows.append(exposure_frame)
        return_rows.append(return_frame)
        specific_rows.append(specific_frame)
        weight_rows.append(weight_frame)
        factor_return_rows.append(factor_return_row)
        diagnostic_rows.append(diagnostic_row)
        factors.update(date_value, fit)

    if not factor_return_rows:
        raise ValueError("no dates had enough valid observations to fit")
    if model_columns is None:
        raise RuntimeError("regression results are missing model columns")
    return ResidualizationResult(
        specific_returns=pd.concat(specific_rows, ignore_index=True),
        returns=pd.concat(return_rows, ignore_index=True),
        exposures=pd.concat(exposure_rows, ignore_index=True),
        factor_returns=pd.DataFrame(factor_return_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
        regression_weights=pd.concat(weight_rows, ignore_index=True),
        universe=universe,
        model_columns=model_columns,
    )


def _fit_outputs(
    date: pd.Timestamp,
    fit: CrossSectionFit,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
]:
    """Convert one fit into the tabular artifacts accumulated by the pipeline."""

    date_column = np.repeat(date.to_datetime64(), len(fit.returns))
    ticker_column = fit.returns.index.astype("string")
    used_exposures = fit.exposures.copy()
    used_exposures.insert(0, "INTERCEPT", 1.0)
    used_exposures.insert(0, "ticker", ticker_column)
    used_exposures.insert(0, "date", date_column)
    common = {"date": date_column, "ticker": ticker_column}
    return (
        used_exposures.reset_index(drop=True),
        pd.DataFrame({**common, "return": fit.returns.to_numpy()}),
        pd.DataFrame({**common, "specific_return": fit.specific_returns.to_numpy()}),
        pd.DataFrame(
            {
                **common,
                "regression_weight": fit.regression_weights.to_numpy(),
            }
        ),
        {"date": date, **fit.factor_returns.to_dict()},
        {
            "date": date,
            "n_observations": len(fit.returns),
            "rank": fit.rank,
            "r_squared": fit.r_squared,
            "residual_mean": float(
                np.average(
                    fit.specific_returns,
                    weights=fit.regression_weights,
                )
            ),
        },
    )
