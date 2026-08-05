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
    start = history_start(
        window.start,
        max(
            factor_model.history_business_days,
            regression_weight_model.history_business_days,
        ),
    )
    market_data = source.load(start, window.end, tickers)
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
) -> ResidualizationResult:
    artifacts = _Artifacts()
    for date in universe_dates(universe):
        tickers = universe_members(universe, date)
        day_returns = returns.xs(date, level="date").reindex(tickers)
        day_exposures = factors.exposures(date, tickers)
        day_weights = regression_weights.xs(date, level="date").reindex(tickers)
        fit = residualizer.fit(
            day_returns,
            day_exposures,
            day_weights,
        )
        if fit is None:
            continue
        artifacts.add(date, fit)
        # Preserve the public two-argument PreparedFactorModel API while giving
        # stateful factors access to finite returns excluded by another factor's
        # missing history.
        factors.update(date, replace(fit, observed_returns=day_returns))
    return artifacts.build(universe)
