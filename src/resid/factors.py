"""Factor-model boundaries and characteristic factors."""

from collections.abc import Callable
from dataclasses import dataclass as plain_dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from resid.progress import StageProgress
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

    @property
    def progress_steps(self) -> int:
        return len(self.factors)

    def exposures(self, market_data: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        return self._exposures(market_data, returns)

    def _exposures(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        progress: StageProgress | None = None,
    ) -> pd.DataFrame:
        wide_returns = returns.unstack("ticker")
        market_caps = (
            market_data["market_cap"].unstack("ticker").reindex_like(wide_returns)
        )
        index = pd.MultiIndex.from_product(
            [wide_returns.index, wide_returns.columns], names=["date", "ticker"]
        )
        # Flattening assumes each builder's output is laid out exactly like
        # wide_returns, so align rather than trust it: a builder that reindexes
        # would otherwise scramble exposures across dates without any error.
        matrices: dict[str, pd.DataFrame] = {}
        for position, factor in enumerate(self.factors, start=1):
            matrices[factor.name] = factor.build(wide_returns, market_caps).reindex(
                index=wide_returns.index, columns=wide_returns.columns
            )
            if progress is not None:
                progress.update(position, factor.name)
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

    def prepare_with_progress(
        self,
        market_data: pd.DataFrame,
        returns: pd.Series,
        universe: pd.Series,
        progress: StageProgress,
    ) -> PreparedFactorModel:
        del universe
        return _StaticFactorModel(self._exposures(market_data, returns, progress))


@plain_dataclass(frozen=True, slots=True, eq=False)
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
    """Lagged log market cap; larger names have higher exposure."""

    def build(_: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        lagged = market_caps.shift(1).where(lambda values: values > 0)
        values = np.log(lagged.to_numpy())
        return pd.DataFrame(values, index=lagged.index, columns=lagged.columns)

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
        # Missing observations represent non-trading days and may be skipped by
        # the rolling minimum.  A return at or below -100%, however, is not a
        # missing observation: allowing the rolling sum to skip it would make a
        # collapsed name look as though the crash never happened.
        valid_returns = returns.gt(-1.0) & np.isfinite(returns)
        safe_returns = returns.where(valid_returns)
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
        crash_windows = (
            returns.le(-1.0)
            .astype("int8")
            .shift(skip_days)
            .rolling(lookback_days, min_periods=1)
            .sum()
        )
        trailing = trailing.mask(crash_windows.gt(0))
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


def long_term_reversal_factor(
    lookback_days: int,
    skip_days: int,
    min_periods: int,
    name: str,
) -> Factor:
    """Negative trailing return, excluding the recent trend window."""

    if lookback_days < 1 or skip_days < 1:
        raise ValueError("reversal lookback and skip must be positive")
    if not 1 <= min_periods <= lookback_days:
        raise ValueError("min_periods must be between 1 and lookback_days")

    trailing_return = momentum_factor(
        lookback_days=lookback_days,
        skip_days=skip_days,
        min_periods=min_periods,
        name=name,
    )

    def build(returns: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
        return -trailing_return.build(returns, market_caps)

    return Factor(name, build, trailing_return.history_business_days)
