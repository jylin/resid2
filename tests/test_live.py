from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from resid import (
    AnalysisWindow,
    CharacteristicFactorModel,
    Factor,
    FixedTopMarketCapUniverse,
    IncrementalRegression,
    JsonlEventLog,
    LiveRegressionRunner,
    OLSResidualizer,
    PercentageReturns,
    PeriodClosed,
    PeriodKey,
    PeriodOpened,
    RecursiveMarketBetaModel,
    ReturnObserved,
    SequentialOLSResidualizer,
    SequentialWLSResidualizer,
    SquareRootMarketCapWeights,
    historical_events,
    history_start,
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

    def trading_dates(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DatetimeIndex:
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


@pytest.mark.parametrize(
    "residualizer",
    (
        OLSResidualizer(winsor_quantile=0.01),
        SequentialOLSResidualizer(
            factor_order=("SIZE", "MOMENTUM"), winsor_quantile=0.01
        ),
        SequentialWLSResidualizer(
            factor_order=("SIZE", "MOMENTUM"), winsor_quantile=0.01
        ),
    ),
)
def test_incremental_ols_reconciles_to_canonical_fit(
    residualizer: (
        OLSResidualizer | SequentialOLSResidualizer | SequentialWLSResidualizer
    ),
) -> None:
    tickers = pd.Index([f"A{i}" for i in range(10)], name="ticker")
    exposures = pd.DataFrame(
        {
            "SIZE": np.linspace(-2, 2, len(tickers)),
            "MOMENTUM": np.sin(np.linspace(0, 2 * np.pi, len(tickers))),
        },
        index=tickers,
    )
    returns = pd.Series(
        np.linspace(-0.03, 0.04, len(tickers)),
        index=tickers,
        name="return",
    )
    regression_weights = pd.Series(np.geomspace(1, 10, len(tickers)), index=tickers)
    incremental = IncrementalRegression(
        residualizer, exposures, regression_weights, returns
    )

    provisional = incremental.update("A4", 0.075)
    updated_returns = returns.copy()
    updated_returns.loc["A4"] = 0.075
    canonical = residualizer.fit(updated_returns, exposures, regression_weights)

    assert provisional is not None and canonical is not None
    np.testing.assert_allclose(
        provisional.factor_returns,
        canonical.factor_returns,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        provisional.specific_returns,
        canonical.specific_returns,
        atol=1e-12,
    )
    final = incremental.finalize()
    assert final is not None
    pd.testing.assert_series_equal(final.factor_returns, canonical.factor_returns)
    pd.testing.assert_series_equal(final.specific_returns, canonical.specific_returns)


def test_jsonl_event_log_is_durable_ordered_and_idempotent(tmp_path: Path) -> None:
    key = PeriodKey(
        market="KRX",
        interval="1m",
        as_of=datetime(2025, 1, 2, 9, 1, tzinfo=UTC),
    )
    events = (
        PeriodOpened(
            event_id="open",
            source_sequence=0,
            key=key,
            known_at=key.as_of,
            tickers=("A", "B"),
            regression_weights=(1.0, 1.0),
        ),
        ReturnObserved(
            event_id="return-a",
            source_sequence=1,
            key=key,
            effective_at=key.as_of,
            known_at=key.as_of,
            ticker="A",
            return_value=0.01,
        ),
        PeriodClosed(
            event_id="close",
            source_sequence=2,
            key=key,
            known_at=key.as_of,
        ),
    )
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path)

    first = log.append_many(events)
    duplicate = log.append(events[1])

    assert [item.offset for item in first] == [1, 2, 3]
    assert duplicate.offset == 2
    assert log.latest_offset == 3
    reopened = JsonlEventLog(path)
    assert [item.event for item in reopened.read()] == list(events)
    assert [item.offset for item in reopened.read(after_offset=1)] == [2, 3]

    conflicting = events[1].model_copy(update={"return_value": 0.02})
    with pytest.raises(ValueError, match="different content"):
        reopened.append(conflicting)


def test_logged_live_replay_matches_vectorized_recursive_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    tickers = [f"A{i:03d}" for i in range(20)]
    betas = np.linspace(0.5, 1.5, len(tickers))
    market_returns = rng.normal(0, 0.012, len(dates))
    values = market_returns[:, None] * betas[None, :]
    values += rng.normal(0, 0.002, values.shape)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    frame = pd.DataFrame(
        {
            "return_percent": values.reshape(-1) * 100,
            "market_cap": np.tile(np.geomspace(1e8, 1e10, len(tickers)), len(dates)),
        },
        index=index,
    )
    source = FrameSource(frame)
    window = AnalysisWindow(dates[25], dates[-1])

    def factor_model() -> RecursiveMarketBetaModel:
        return RecursiveMarketBetaModel(
            base=CharacteristicFactorModel(
                (
                    Factor(
                        "REVERSAL",
                        lambda returns, _: -returns.shift(1),
                        history_business_days=1,
                    ),
                )
            ),
            lookback_days=10,
            min_periods=8,
            decay=0.9,
        )

    residualizer = SequentialWLSResidualizer(
        factor_order=("MARKET_BETA", "REVERSAL"), winsor_quantile=0
    )
    batch = run_pipeline(
        window=window,
        source=source,
        universe_builder=FixedTopMarketCapUniverse(size=len(tickers)),
        return_calculator=PercentageReturns(),
        factor_model=factor_model(),
        residualizer=residualizer,
        regression_weight_model=SquareRootMarketCapWeights(),
    )
    assert batch.model_columns == ("INTERCEPT", "MARKET_BETA", "REVERSAL")

    universe = FixedTopMarketCapUniverse(size=len(tickers)).build(source, window)
    model = factor_model()
    # Must match run_pipeline's own history span, or the hand-built model below
    # bootstraps beta from a different slice of history than the batch run did.
    market_data = source.load(
        history_start(window.start, model.history_business_days), window.end
    )
    returns = PercentageReturns().calculate(market_data)
    regression_weights = SquareRootMarketCapWeights().calculate(market_data)
    prepared = model.prepare(market_data, returns, universe)
    log = JsonlEventLog(tmp_path / "replay.jsonl")
    logged = log.append_many(
        historical_events(
            universe=universe,
            returns=returns,
            regression_weights=regression_weights,
            market="KRX",
            interval="1d",
        )
    )
    runner = LiveRegressionRunner(
        model_version="test-model",
        market="KRX",
        interval="1d",
        factors=prepared,
        residualizer=residualizer,
    )
    final = [result for result in runner.replay(logged) if result.status == "final"]

    assert len(final) == len(batch.factor_returns)
    for live_result in final:
        date = pd.Timestamp(live_result.key.as_of)
        batch_factors = batch.factor_returns.set_index("date").loc[date]
        np.testing.assert_allclose(
            live_result.fit.factor_returns,
            batch_factors.loc[live_result.fit.factor_returns.index],
            atol=1e-12,
        )
        batch_specific = (
            batch.specific_returns.loc[batch.specific_returns["date"] == date]
            .set_index("ticker")["specific_return"]
            .reindex(live_result.fit.specific_returns.index)
        )
        np.testing.assert_allclose(
            live_result.fit.specific_returns,
            batch_specific,
            atol=1e-12,
        )

    replay_model = factor_model()
    replay_prepared = replay_model.prepare(market_data, returns, universe)
    replay_runner = LiveRegressionRunner(
        model_version="test-model",
        market="KRX",
        interval="1d",
        factors=replay_prepared,
        residualizer=residualizer,
    )
    replayed = [
        result
        for result in replay_runner.replay(JsonlEventLog(log.path).read())
        if result.status == "final"
    ]
    for original, replay in zip(final, replayed, strict=True):
        pd.testing.assert_series_equal(
            original.fit.factor_returns,
            replay.fit.factor_returns,
        )
        pd.testing.assert_series_equal(
            original.fit.specific_returns,
            replay.fit.specific_returns,
        )
