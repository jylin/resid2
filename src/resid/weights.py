"""Regression-weight boundaries and built-in weighting schemes."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class RegressionWeightModel(Protocol):
    @property
    def history_business_days(self) -> int: ...

    def calculate(self, market_data: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True, slots=True)
class EqualRegressionWeights:
    """Give every available observation equal regression influence."""

    history_business_days: int = 0

    def calculate(self, market_data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=market_data.index, name="regression_weight")


@dataclass(frozen=True, slots=True)
class SquareRootMarketCapWeights:
    """Use the square root of prior-session market cap as regression weight."""

    history_business_days: int = 1

    def calculate(self, market_data: pd.DataFrame) -> pd.Series:
        market_caps = market_data["market_cap"].unstack("ticker")
        lagged = market_caps.shift(1).where(lambda values: values > 0)
        values = np.sqrt(lagged.to_numpy(dtype="float64"))
        index = pd.MultiIndex.from_product(
            [lagged.index, lagged.columns], names=["date", "ticker"]
        )
        return pd.Series(values.ravel(), index=index, name="regression_weight").reindex(
            market_data.index
        )
