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
        market_returns = self._market_returns(market_data, returns, universe)
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
            market_returns=market_returns,
        )

    @staticmethod
    def _market_returns(
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> pd.Series:
        """Build the point-in-time market return used by the beta update."""

        return _weighted_market_returns(
            returns,
            market_data["market_cap"],
            index=universe.index[universe],
        )

    def _initial_state(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> tuple[dict[str, float], dict[str, float]]:
        analysis_start = pd.Timestamp(universe.index.get_level_values("date").min())
        initial_members = universe.xs(analysis_start, level="date")
        initial_tickers = initial_members.index[initial_members.to_numpy(dtype="bool")]
        initial_index = returns.index[
            returns.index.get_level_values("ticker").isin(initial_tickers)
        ]
        market_returns = _weighted_market_returns(
            returns.reindex(initial_index),
            market_data["market_cap"].reindex(initial_index),
        )
        market_returns = market_returns.loc[market_returns.index < analysis_start]

        weights = _lagged_market_caps(market_data["market_cap"])
        history = pd.concat(
            [returns.rename("asset_return"), weights.rename("weight")], axis=1
        )
        dates = history.index.get_level_values("date")
        history = history.loc[dates < analysis_start]
        history["market_return"] = history.index.get_level_values("date").map(
            market_returns
        )
        history = history.loc[
            history["weight"].gt(0)
            & np.isfinite(history["weight"])
            & np.isfinite(history["asset_return"])
        ].dropna(subset=["market_return"])

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
    market_returns: pd.Series

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
        market_return = float(self.market_returns.get(pd.Timestamp(date), np.nan))
        center = float(fit.exposure_centers[self.name])
        scale = float(fit.exposure_scales[self.name])
        if not np.isfinite(factor_return) or not np.isfinite(market_return):
            return
        if not np.isfinite(center):
            return
        if not np.isfinite(scale) or scale <= 0:
            return

        # The regression normalizes this column to x = (beta - center) / scale.
        # Remove the fitted x * factor_return contribution, then restore the
        # centered level with the observed market return. The observed return is
        # the EWLS shock; the fitted cross-sectional premium is not a time-series
        # market return and would feed the estimator back into its own state.
        factor_fitted = fit.exposures[self.name] * factor_return
        target_returns = (
            fit.returns - (fit.fitted_returns - factor_fitted) + center * market_return
        )
        for ticker, target_return in target_returns.items():
            if not np.isfinite(target_return):
                continue
            key = str(ticker)
            denominator = self.denominators.get(key, self.prior_variance)
            numerator = self.numerators.get(
                key, self.initial_beta * self.prior_variance
            )
            denominator = self.decay * denominator + market_return**2
            numerator = self.decay * numerator + market_return * float(target_return)
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


def _lagged_market_caps(market_caps: pd.Series) -> pd.Series:
    return market_caps.groupby(level="ticker").shift(1)


def _weighted_market_returns(
    returns: pd.Series,
    market_caps: pd.Series,
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    lagged_caps = _lagged_market_caps(market_caps)
    observations = pd.concat(
        [returns.rename("asset_return"), lagged_caps.rename("weight")], axis=1
    )
    if index is not None:
        observations = observations.reindex(index)
    observations = observations.loc[
        observations["weight"].gt(0)
        & np.isfinite(observations["weight"])
        & np.isfinite(observations["asset_return"])
    ]
    weighted = observations["asset_return"] * observations["weight"]
    return (
        weighted.groupby(level="date").sum()
        / observations["weight"].groupby(level="date").sum()
    ).rename("market_return")
