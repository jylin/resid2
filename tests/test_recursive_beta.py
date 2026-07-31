from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from resid import (
    AnalysisWindow,
    CharacteristicFactorModel,
    Factor,
    FixedTopMarketCapUniverse,
    OLSResidualizer,
    PercentageReturns,
    RecursiveMarketBetaModel,
    ResidualizationResult,
    run_pipeline,
)


@dataclass
class FrameSource:
    frame: pd.DataFrame

    def latest_date(self, end: str | None = None) -> pd.Timestamp:
        dates = self.frame.index.get_level_values("date")
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        return cast(pd.Timestamp, pd.Timestamp(dates.max()))

    def trading_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        dates = self.frame.index.get_level_values("date")
        return pd.DatetimeIndex(dates[(dates >= start) & (dates <= end)].unique())

    def load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: pd.Index | None = None,
    ) -> pd.DataFrame:
        dates = self.frame.index.get_level_values("date")
        selected = self.frame.loc[(dates >= start) & (dates <= end)]
        if tickers is not None:
            selected = selected.loc[
                selected.index.get_level_values("ticker").isin(tickers)
            ]
        return selected


def run_market_beta(
    *, perturb_first_analysis_date: bool
) -> tuple[pd.DatetimeIndex, ResidualizationResult]:
    rng = np.random.default_rng(41)
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    tickers = [f"A{i:03d}" for i in range(30)]
    true_betas = np.linspace(0.4, 1.6, len(tickers))
    market_returns = rng.normal(0.0, 0.015, len(dates))
    values = market_returns[:, None] * true_betas[None, :]
    values += rng.normal(0.0, 0.002, values.shape)
    if perturb_first_analysis_date:
        values[25, 0] += 0.50

    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    market_caps = np.tile(np.geomspace(1e8, 1e10, len(tickers)), len(dates))
    frame = pd.DataFrame(
        {
            "return_percent": values.reshape(-1) * 100,
            "market_cap": market_caps,
        },
        index=index,
    )

    def reversal(returns: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        return -returns.shift(1)

    factor_model = RecursiveMarketBetaModel(
        base=CharacteristicFactorModel(
            (Factor("REVERSAL", reversal, history_business_days=1),)
        ),
        lookback_days=20,
        min_periods=15,
        decay=0.9,
    )
    result = run_pipeline(
        window=AnalysisWindow(dates[25], dates[-1]),
        source=FrameSource(frame),
        universe_builder=FixedTopMarketCapUniverse(size=len(tickers)),
        return_calculator=PercentageReturns(),
        factor_model=factor_model,
        residualizer=OLSResidualizer(winsor_quantile=0),
    )
    return dates, result


def test_market_beta_uses_prior_state_then_updates_after_fit() -> None:
    dates, baseline = run_market_beta(perturb_first_analysis_date=False)
    _, perturbed = run_market_beta(perturb_first_analysis_date=True)

    def betas(result: ResidualizationResult, date: pd.Timestamp) -> pd.Series:
        rows = result.exposures.loc[result.exposures["date"] == date]
        return rows.set_index("ticker")["MARKET_BETA"]

    np.testing.assert_allclose(betas(baseline, dates[25]), betas(perturbed, dates[25]))
    assert not np.allclose(betas(baseline, dates[26]), betas(perturbed, dates[26]))
    assert "MARKET_BETA" in baseline.factor_returns
