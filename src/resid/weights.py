"""Regression-weight boundaries and built-in weighting schemes."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from resid.data import previous_session_values


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
        lagged = previous_session_values(market_data["market_cap"])
        positive = lagged.where(lagged > 0)
        return pd.Series(
            np.sqrt(positive.to_numpy(dtype="float64")),
            index=positive.index,
            name="regression_weight",
        )
