"""Return-calculation boundary."""

from typing import Protocol

import pandas as pd


class ReturnCalculator(Protocol):
    def calculate(self, market_data: pd.DataFrame) -> pd.Series: ...


class PercentageReturns:
    """Convert percentage return observations to decimal returns."""

    def calculate(self, market_data: pd.DataFrame) -> pd.Series:
        return (market_data["return_percent"] / 100.0).rename("return")
