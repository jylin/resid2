"""Sequential market-beta factor model."""

from dataclasses import dataclass as plain_dataclass

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, SkipValidation
from pydantic.dataclasses import dataclass

from resid.factors import FactorModel, PreparedFactorModel
from resid.regression import CrossSectionFit


@dataclass(
    frozen=True,
    slots=True,
    config=ConfigDict(arbitrary_types_allowed=True),
)
class RecursiveMarketBetaModel:
    """Add market beta using prior state, then update it after each daily fit.

    Exposures are raw betas on the same scale as the bootstrap and the
    `initial_beta` prior, and `update` inverts the regression's normalization so
    the recursion stays on that scale. Naming this factor in the residualizer's
    `unscaled_factors` keeps its reported slope in per-unit-beta return terms.
    """

    base: SkipValidation[FactorModel]
    lookback_days: int = Field(gt=0)
    min_periods: int = Field(gt=0)
    decay: float = Field(gt=0, lt=1)
    name: str = "MARKET_BETA"
    initial_beta: float = 1.0
    prior_variance: float = Field(default=1e-4, gt=0)
    lower_bound: float = -5.0
    upper_bound: float = 5.0

    def __post_init__(self) -> None:
        if self.min_periods > self.lookback_days:
            raise ValueError("min_periods must not exceed lookback_days")
        if not self.name:
            raise ValueError("market beta name must not be empty")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be below upper_bound")

    @property
    def history_business_days(self) -> int:
        return max(self.base.history_business_days, self.lookback_days)

    def prepare(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> PreparedFactorModel:
        base = self.base.prepare(market_data, returns, universe)
        if self.name in base.names:
            raise ValueError(f"duplicate factor name: {self.name}")
        numerators, denominators = self._initial_state(market_data, returns, universe)
        return _PreparedRecursiveMarketBeta(
            base=base,
            name=self.name,
            initial_beta=self.initial_beta,
            decay=self.decay,
            prior_variance=self.prior_variance,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            numerators=numerators,
            denominators=denominators,
        )

    def _initial_state(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> tuple[dict[str, float], dict[str, float]]:
        analysis_start = pd.Timestamp(universe.index.get_level_values("date").min())
        weights = market_data["market_cap"].groupby(level="ticker").shift(1)
        history = pd.concat(
            [returns.rename("asset_return"), weights.rename("weight")], axis=1
        )
        dates = history.index.get_level_values("date")
        history = history.loc[dates < analysis_start].dropna()
        history = history.loc[history["weight"] > 0]
        initial_members = universe.xs(analysis_start, level="date")
        initial_tickers = initial_members.index[initial_members.to_numpy(dtype="bool")]
        market_history = history.loc[
            history.index.get_level_values("ticker").isin(initial_tickers)
        ]
        weighted_returns = market_history["asset_return"] * market_history["weight"]
        market_returns = (
            weighted_returns.groupby(level="date").sum()
            / market_history["weight"].groupby(level="date").sum()
        )
        history["market_return"] = history.index.get_level_values("date").map(
            market_returns
        )

        tickers = returns.index.get_level_values("ticker").unique()
        initial_betas = {str(ticker): self.initial_beta for ticker in tickers}
        for ticker, observations in history.groupby(level="ticker", sort=False):
            observations = observations.tail(self.lookback_days).dropna()
            if len(observations) < self.min_periods:
                continue
            market = observations["market_return"].to_numpy(dtype="float64")
            asset = observations["asset_return"].to_numpy(dtype="float64")
            market = market - market.mean()
            asset = asset - asset.mean()
            variance = float(np.dot(market, market))
            if variance <= 0:
                continue
            beta = float(np.dot(market, asset) / variance)
            initial_betas[str(ticker)] = float(
                np.clip(beta, self.lower_bound, self.upper_bound)
            )

        denominators = {ticker: self.prior_variance for ticker in initial_betas}
        numerators = {
            ticker: beta * self.prior_variance for ticker, beta in initial_betas.items()
        }
        return numerators, denominators


@plain_dataclass(slots=True)
class _PreparedRecursiveMarketBeta:
    base: PreparedFactorModel
    name: str
    initial_beta: float
    decay: float
    prior_variance: float
    lower_bound: float
    upper_bound: float
    numerators: dict[str, float]
    denominators: dict[str, float]

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.base.names, self.name)

    def exposures(self, date: pd.Timestamp, tickers: pd.Index) -> pd.DataFrame:
        exposures = self.base.exposures(date, tickers).copy()
        exposures[self.name] = [self._beta(str(ticker)) for ticker in tickers]
        return exposures

    def update(self, date: pd.Timestamp, fit: CrossSectionFit) -> None:
        self.base.update(date, fit)
        factor_return = float(fit.factor_returns[self.name])
        center = float(fit.exposure_centers[self.name])
        scale = float(fit.exposure_scales[self.name])
        if not np.isfinite(factor_return) or not np.isfinite(center):
            return
        if not np.isfinite(scale) or scale <= 0:
            return

        # The regression normalizes this column to x = (beta - center) / scale, so
        # its slope prices a normalized unit, not a unit of beta. Undo that affine
        # map before updating state, otherwise the recursion drifts onto whatever
        # scale the cross-section happened to have and stops being comparable to
        # the bootstrap beta or to the explicit prior.
        beta_return = factor_return / scale
        factor_fitted = fit.exposures[self.name] * factor_return
        target_returns = (
            fit.returns - (fit.fitted_returns - factor_fitted) + center * beta_return
        )
        for ticker, target_return in target_returns.items():
            if not np.isfinite(target_return):
                continue
            key = str(ticker)
            denominator = self.denominators.get(key, self.prior_variance)
            numerator = self.numerators.get(
                key, self.initial_beta * self.prior_variance
            )
            denominator = self.decay * denominator + beta_return**2
            numerator = self.decay * numerator + beta_return * float(target_return)
            # Write the bounded beta back rather than clipping on read. Keeping the
            # raw ratio in state lets it wander far outside the bounds and recover
            # slowly, which pins twice as much of the exposure column to the rail.
            beta = float(
                np.clip(
                    numerator / denominator,
                    self.lower_bound,
                    self.upper_bound,
                )
            )
            self.denominators[key] = denominator
            self.numerators[key] = beta * denominator

    def _beta(self, ticker: str) -> float:
        denominator = self.denominators.get(ticker, self.prior_variance)
        numerator = self.numerators.get(ticker, self.initial_beta * self.prior_variance)
        return numerator / denominator
