from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pytest

from resid import (
    AnalysisWindow,
    CharacteristicFactorModel,
    CrossSectionFit,
    EqualRegressionWeights,
    Factor,
    FixedTopMarketCapUniverse,
    PercentageReturns,
    RecursiveMarketBetaModel,
    ResidualizationResult,
    SequentialOLSResidualizer,
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
        residualizer=SequentialOLSResidualizer(
            factor_order=("MARKET_BETA", "REVERSAL"), winsor_quantile=0
        ),
        regression_weight_model=EqualRegressionWeights(),
    )
    return dates, result


def drive_recursive_beta(*, unscaled: bool) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Advance beta state by hand so the raw state itself can be inspected."""

    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    tickers = [f"A{i:03d}" for i in range(24)]
    true_betas = np.linspace(0.4, 1.6, len(tickers))
    market_returns = rng.normal(0.0, 0.012, len(dates))
    values = market_returns[:, None] * true_betas[None, :]
    values += rng.normal(0.0, 0.001, values.shape)

    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    frame = pd.DataFrame(
        {
            "return_percent": values.reshape(-1) * 100,
            "market_cap": np.tile(np.geomspace(1e8, 1e10, len(tickers)), len(dates)),
        },
        index=index,
    )
    analysis_dates = dates[40:]
    universe = pd.Series(
        True,
        index=pd.MultiIndex.from_product(
            [analysis_dates, tickers], names=["date", "ticker"]
        ),
        name="in_universe",
    )
    model = RecursiveMarketBetaModel(
        base=CharacteristicFactorModel(
            (
                Factor(
                    "REVERSAL",
                    lambda returns, _: -returns.shift(1),
                    history_business_days=1,
                ),
            )
        ),
        lookback_days=30,
        min_periods=20,
        decay=0.9,
    )
    residualizer = SequentialOLSResidualizer(
        factor_order=("MARKET_BETA", "REVERSAL"),
        winsor_quantile=0,
        unscaled_factors=("MARKET_BETA",) if unscaled else (),
    )
    returns = PercentageReturns().calculate(frame)
    prepared = model.prepare(frame, returns, universe)
    ticker_index = pd.Index(tickers, name="ticker")
    bootstrap = prepared.exposures(analysis_dates[0], ticker_index)[
        "MARKET_BETA"
    ].copy()
    for date in analysis_dates:
        day_returns = returns.xs(date, level="date").reindex(ticker_index)
        fit = residualizer.fit(
            day_returns,
            prepared.exposures(date, ticker_index),
            pd.Series(1.0, index=ticker_index),
        )
        assert fit is not None
        prepared.update(date, fit)
    final = prepared.exposures(analysis_dates[-1], ticker_index)["MARKET_BETA"]
    return bootstrap, final, true_betas


def test_recursive_beta_state_is_invariant_to_exposure_scaling() -> None:
    _, unscaled_final, _ = drive_recursive_beta(unscaled=True)
    _, scaled_final, _ = drive_recursive_beta(unscaled=False)

    np.testing.assert_allclose(unscaled_final, scaled_final, atol=1e-12)


def test_recursive_beta_state_stays_on_the_bootstrap_scale() -> None:
    bootstrap, final, true_betas = drive_recursive_beta(unscaled=True)

    # The recursion must keep estimating a beta, not the standardized exposure
    # the regression happens to price, so its level tracks the bootstrap instead
    # of collapsing onto a mean-zero unit-variance cross-section.
    assert final.mean() == pytest.approx(bootstrap.mean(), abs=0.05)
    assert final.min() > 0.0
    assert np.corrcoef(final.to_numpy(), true_betas)[0, 1] > 0.95


def test_market_beta_uses_prior_state_then_updates_after_fit() -> None:
    dates, baseline = run_market_beta(perturb_first_analysis_date=False)
    _, perturbed = run_market_beta(perturb_first_analysis_date=True)

    def betas(result: ResidualizationResult, date: pd.Timestamp) -> pd.Series:
        rows = result.exposures.loc[result.exposures["date"] == date]
        return rows.set_index("ticker")["MARKET_BETA"]

    np.testing.assert_allclose(betas(baseline, dates[25]), betas(perturbed, dates[25]))
    assert not np.allclose(betas(baseline, dates[26]), betas(perturbed, dates[26]))
    assert "MARKET_BETA" in baseline.factor_returns


def test_initial_market_proxy_ignores_future_universe_members() -> None:
    dates = pd.date_range("2025-01-01", periods=25, freq="B")
    tickers = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    values = np.tile([0.01, 0.02, -0.03], (len(dates), 1))
    market_data = pd.DataFrame(
        {
            "return_percent": values.reshape(-1) * 100,
            "market_cap": np.tile([3e9, 2e9, 1e9], len(dates)),
        },
        index=index,
    )
    analysis_dates = dates[-5:]
    universe_index = pd.MultiIndex.from_tuples(
        [
            (day, ticker)
            for day in analysis_dates
            for ticker in (["A", "B"] if day == analysis_dates[0] else tickers)
        ],
        names=["date", "ticker"],
    )
    universe = pd.Series(True, index=universe_index, name="in_universe")

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        model = RecursiveMarketBetaModel(
            base=CharacteristicFactorModel(
                (
                    Factor(
                        "REVERSAL",
                        lambda returns, _: -returns.shift(1),
                        history_business_days=1,
                    ),
                )
            ),
            lookback_days=15,
            min_periods=10,
            decay=0.9,
        )
        returns = PercentageReturns().calculate(frame)
        prepared = model.prepare(frame, returns, universe)
        return prepared.exposures(analysis_dates[0], pd.Index(["A", "B"]))

    baseline = prepare(market_data)
    perturbed_data = market_data.copy()
    perturbed_data.loc[(dates[:-5], "C"), "return_percent"] = np.linspace(
        -50, 50, len(dates) - 5
    )
    perturbed = prepare(perturbed_data)

    pd.testing.assert_series_equal(baseline["MARKET_BETA"], perturbed["MARKET_BETA"])


def test_recursive_beta_update_uses_observed_market_return() -> None:
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    tickers = pd.Index(["A", "B", "C"], name="ticker")
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    market_data = pd.DataFrame(
        {
            "return_percent": np.full(len(index), 2.0),
            "market_cap": np.tile([1e8, 2e8, 3e8], len(dates)),
        },
        index=index,
    )
    universe = pd.Series(
        True,
        index=pd.MultiIndex.from_product(
            [[dates[-1]], tickers], names=["date", "ticker"]
        ),
        name="in_universe",
    )
    model = RecursiveMarketBetaModel(
        base=CharacteristicFactorModel(
            (Factor("STYLE", lambda returns, _: returns * 0 + 1, 0),)
        ),
        lookback_days=2,
        min_periods=2,
        decay=0.97,
        name="MKT",
    )
    returns = PercentageReturns().calculate(market_data)
    prepared = model.prepare(market_data, returns, universe)
    exposures = pd.DataFrame({"MKT": [0.2, 0.4, 0.6]}, index=tickers)
    fit_returns = pd.Series([0.01, 0.02, 0.03], index=tickers, name="return")
    factor_returns = pd.Series({"INTERCEPT": 0.0, "MKT": -0.5}, name="factor_return")
    fitted_returns = exposures["MKT"] * factor_returns["MKT"]
    fit = CrossSectionFit(
        returns=fit_returns,
        exposures=exposures,
        regression_weights=pd.Series(1.0, index=tickers),
        factor_returns=factor_returns,
        fitted_returns=fitted_returns,
        specific_returns=fit_returns - fitted_returns,
        rank=2,
        r_squared=0.0,
        exposure_centers=pd.Series({"MKT": 1.0}),
        exposure_scales=pd.Series({"MKT": 1.0}),
    )

    prior_variance = model.prior_variance
    expected = (
        model.decay * model.initial_beta * prior_variance
        + 0.02 * (fit_returns["A"] + 0.02)
    ) / (model.decay * prior_variance + 0.02**2)
    prepared.update(dates[-1], fit)

    actual = prepared.exposures(dates[-1], tickers).loc["A", "MKT"]
    assert actual == pytest.approx(expected)
    assert actual > 0
