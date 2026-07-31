"""Factor-model boundaries and characteristic factors."""

from collections.abc import Callable
from dataclasses import dataclass as plain_dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from resid.regression import CrossSectionFit

FactorBuilder = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class Factor:
    name: str
    build: FactorBuilder
    history_business_days: int = Field(ge=0)


class PreparedFactorModel(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...

    def exposures(self, date: pd.Timestamp, tickers: pd.Index) -> pd.DataFrame: ...

    def update(self, date: pd.Timestamp, fit: CrossSectionFit) -> None: ...


class FactorModel(Protocol):
    @property
    def history_business_days(self) -> int: ...

    def prepare(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> PreparedFactorModel: ...


@plain_dataclass(frozen=True, slots=True)
class CharacteristicFactorModel:
    """Build aligned exposures from independent characteristic functions."""

    factors: tuple[Factor, ...]

    def __post_init__(self) -> None:
        names = [factor.name for factor in self.factors]
        if not names:
            raise ValueError("factor model requires at least one factor")
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique")

    @property
    def history_business_days(self) -> int:
        return max(factor.history_business_days for factor in self.factors)

    def exposures(self, market_data: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        wide_returns = returns.unstack("ticker")
        market_caps = (
            market_data["market_cap"].unstack("ticker").reindex_like(wide_returns)
        )
        matrices = {
            factor.name: factor.build(wide_returns, market_caps)
            for factor in self.factors
        }
        index = pd.MultiIndex.from_product(
            [wide_returns.index, wide_returns.columns], names=["date", "ticker"]
        )
        return pd.DataFrame(
            {name: matrix.to_numpy().ravel() for name, matrix in matrices.items()},
            index=index,
        )

    def prepare(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
    ) -> PreparedFactorModel:
        del universe
        return _StaticFactorModel(self.exposures(market_data, returns))


@plain_dataclass(frozen=True, slots=True)
class _StaticFactorModel:
    values: pd.DataFrame

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.values.columns)

    def exposures(self, date: pd.Timestamp, tickers: pd.Index) -> pd.DataFrame:
        return self.values.xs(date, level="date").reindex(tickers)

    def update(self, date: pd.Timestamp, fit: CrossSectionFit) -> None:
        del date, fit


def size_factor(name: str) -> Factor:
    """Log of market capitalization lagged by one trading day."""

    def build(_: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        lagged = market_caps.shift(1).where(lambda values: values > 0)
        return pd.DataFrame(
            np.log(lagged.to_numpy()),
            index=lagged.index,
            columns=lagged.columns,
        )

    return Factor(name, build, history_business_days=1)


def momentum_factor(
    lookback_days: int,
    skip_days: int,
    min_periods: int,
    name: str,
) -> Factor:
    """Compounded trailing return, excluding the most recent observations."""

    if lookback_days < 1 or skip_days < 1:
        raise ValueError("momentum lookback and skip must be positive")
    if not 1 <= min_periods <= lookback_days:
        raise ValueError("min_periods must be between 1 and lookback_days")

    def build(returns: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        safe_returns = returns.where(returns > -1.0)
        log_returns = pd.DataFrame(
            np.log1p(safe_returns.to_numpy()),
            index=returns.index,
            columns=returns.columns,
        )
        trailing = (
            log_returns.shift(skip_days)
            .rolling(lookback_days, min_periods=min_periods)
            .sum()
        )
        return pd.DataFrame(
            np.expm1(trailing.to_numpy()),
            index=trailing.index,
            columns=trailing.columns,
        )

    return Factor(
        name,
        build,
        history_business_days=lookback_days + skip_days,
    )
