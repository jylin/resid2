"""Explicit orchestration across the pipeline boundaries."""

from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd

from resid.data import AnalysisWindow, MarketDataSource, UniverseBuilder
from resid.factors import FactorModel, PreparedFactorModel
from resid.regression import ResidualizationResult, Residualizer
from resid.returns import ReturnCalculator
from resid.validation import RegressionValidation, RegressionValidationError


def run_pipeline(
    *,
    window: AnalysisWindow,
    source: MarketDataSource,
    universe_builder: UniverseBuilder,
    return_calculator: ReturnCalculator,
    factor_model: FactorModel,
    residualizer: Residualizer,
    validations: tuple[RegressionValidation, ...] = (),
) -> ResidualizationResult:
    """Run the caller-specified stages without writing artifacts."""

    universe = universe_builder.build(source, window)
    tickers = universe.index.get_level_values("ticker").unique()
    history_start = window.start - pd.offsets.BDay(factor_model.history_business_days)
    market_data = source.load(history_start, window.end, tickers)
    returns = return_calculator.calculate(market_data)
    prepared = factor_model.prepare(market_data, returns, universe)
    result = _residualize(returns, prepared, residualizer, universe)

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
    universe: pd.Series,
) -> ResidualizationResult:
    model_columns = ("INTERCEPT", *factors.names)
    exposure_rows: list[pd.DataFrame] = []
    return_rows: list[pd.DataFrame] = []
    specific_rows: list[pd.DataFrame] = []
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
        )
        if fit is None:
            continue

        date_column = np.repeat(date_value.to_datetime64(), len(fit.returns))
        ticker_column = fit.returns.index.astype("string")
        used_exposures = fit.exposures.copy()
        used_exposures.insert(0, "INTERCEPT", 1.0)
        used_exposures.insert(0, "ticker", ticker_column)
        used_exposures.insert(0, "date", date_column)
        exposure_rows.append(used_exposures.reset_index(drop=True))
        return_rows.append(
            pd.DataFrame(
                {
                    "date": date_column,
                    "ticker": ticker_column,
                    "return": fit.returns.to_numpy(),
                }
            )
        )
        specific_rows.append(
            pd.DataFrame(
                {
                    "date": date_column,
                    "ticker": ticker_column,
                    "specific_return": fit.specific_returns.to_numpy(),
                }
            )
        )
        factor_return_rows.append({"date": date_value, **fit.factor_returns.to_dict()})
        diagnostic_rows.append(
            {
                "date": date_value,
                "n_observations": len(fit.returns),
                "rank": fit.rank,
                "r_squared": fit.r_squared,
                "residual_mean": float(fit.specific_returns.mean()),
            }
        )
        factors.update(date_value, fit)

    if not factor_return_rows:
        raise ValueError("no dates had enough valid observations to fit")
    return ResidualizationResult(
        specific_returns=pd.concat(specific_rows, ignore_index=True),
        returns=pd.concat(return_rows, ignore_index=True),
        exposures=pd.concat(exposure_rows, ignore_index=True),
        factor_returns=pd.DataFrame(factor_return_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
        universe=universe,
        model_columns=model_columns,
    )
