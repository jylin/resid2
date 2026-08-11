"""Sequential market-beta factor model."""

from dataclasses import dataclass as plain_dataclass

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, SkipValidation
from pydantic.dataclasses import dataclass

from resid.data import (
    previous_session_values,
    universe_dates,
    universe_index,
    universe_members,
)
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
    `initial_beta` prior, and `update` restores the level the regression centered
    away so the recursion stays on that scale. Naming this factor in the
    residualizer's `unscaled_factors` keeps its reported slope in per-unit-beta
    return terms; the recursion itself is invariant to that choice, because a
    rescaled exposure earns a reciprocally rescaled factor return.

    `prior_variance` is a weight in units of squared market return, so it is
    directly comparable to the `market_return ** 2` a single session contributes.
    With a 1% market-return scale, the default is about one session's worth of
    prior information. The decay controls how quickly that bootstrap fades.
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
        tickers, initial_betas = self._initial_state(market_data, returns, universe)
        market_returns = self._market_returns(market_data, returns, universe)
        return _PreparedRecursiveMarketBeta(
            base=base,
            name=self.name,
            initial_beta=self.initial_beta,
            decay=self.decay,
            prior_variance=self.prior_variance,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            ticker_index=tickers,
            numerators=initial_betas * self.prior_variance,
            denominators=np.full(len(tickers), self.prior_variance, dtype="float64"),
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
            index=universe_index(universe),
        )

    def _initial_state(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> tuple[pd.Index, np.ndarray]:
        analysis_start = universe_dates(universe)[0]
        initial_tickers = universe_members(universe, analysis_start)
        initial_index = returns.index[
            returns.index.get_level_values("ticker").isin(initial_tickers)
        ]
        market_returns = _weighted_market_returns(
            returns.reindex(initial_index),
            market_data["market_cap"].reindex(initial_index),
        )
        market_returns = market_returns.loc[market_returns.index < analysis_start]

        history = pd.concat(
            [
                returns.rename("asset_return"),
                previous_session_values(market_data["market_cap"]).rename("weight"),
            ],
            axis=1,
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
        positions = {str(ticker): position for position, ticker in enumerate(tickers)}
        initial_betas = np.full(len(tickers), self.initial_beta, dtype="float64")
        for ticker, observations in history.groupby(level="ticker", sort=False):
            observations = observations.tail(self.lookback_days)
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
            position = positions.get(str(ticker))
            if position is not None:
                initial_betas[position] = np.clip(
                    beta, self.lower_bound, self.upper_bound
                )

        return tickers, initial_betas


@plain_dataclass(slots=True, eq=False)
class _PreparedRecursiveMarketBeta:
    base: PreparedFactorModel
    name: str
    initial_beta: float
    decay: float
    prior_variance: float
    lower_bound: float
    upper_bound: float
    ticker_index: pd.Index
    numerators: np.ndarray
    denominators: np.ndarray
    market_returns: pd.Series

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.base.names, self.name)

    def exposures(self, date: pd.Timestamp, tickers: pd.Index) -> pd.DataFrame:
        exposures = self.base.exposures(date, tickers).copy()
        positions = self.ticker_index.get_indexer(tickers)
        betas = np.full(len(tickers), self.initial_beta, dtype="float64")
        known = positions >= 0
        betas[known] = (
            self.numerators[positions[known]] / self.denominators[positions[known]]
        )
        exposures[self.name] = betas
        return exposures

    def update(self, date: pd.Timestamp, fit: CrossSectionFit) -> None:
        self.base.update(date, fit)
        market_return = float(self.market_returns.get(pd.Timestamp(date), np.nan))
        center = float(fit.exposure_centers[self.name])
        intercept = float(fit.factor_returns.get("INTERCEPT", np.nan))
        if not np.isfinite(intercept) or not np.isfinite(market_return):
            return
        if not np.isfinite(center):
            return

        # Estimate the ordinary time-series market beta.  The old target removed
        # every other fitted cross-sectional factor, but those factor returns can
        # themselves absorb market co-movement when their estimated exposures are
        # noisy.  That systemic feedback biased beta against correlated styles.
        # Strip only the fitted intercept and restore the beta level centered out
        # of the exposure.  The pipeline attaches all observed returns so names
        # excluded by a long-window style (such as a new IPO missing HML) still
        # receive a state update.
        observed_returns = (
            fit.observed_returns if fit.observed_returns is not None else fit.returns
        )
        target_returns = observed_returns - intercept + center * market_return
        shock = market_return**2
        target_values = target_returns.to_numpy(dtype="float64", na_value=np.nan)
        positions = self.ticker_index.get_indexer(target_returns.index)
        valid = (positions >= 0) & np.isfinite(target_values)
        if not valid.any():
            return

        positions = positions[valid]
        target_values = target_values[valid]
        denominator = self.decay * self.denominators[positions] + shock
        numerator = (
            self.decay * self.numerators[positions] + market_return * target_values
        )
        # Write the bounded beta back rather than clipping on read. Keeping the
        # raw ratio in state lets it wander far outside the bounds and recover
        # slowly, which pins twice as much of the exposure column to the rail.
        beta = np.clip(
            numerator / denominator,
            self.lower_bound,
            self.upper_bound,
        )
        self.denominators[positions] = denominator
        self.numerators[positions] = beta * denominator


def _weighted_market_returns(
    returns: pd.Series,
    market_caps: pd.Series,
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    observations = pd.concat(
        [
            returns.rename("asset_return"),
            previous_session_values(market_caps).rename("weight"),
        ],
        axis=1,
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
