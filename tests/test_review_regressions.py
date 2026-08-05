"""Regressions for defects found by review, one test per fixed behavior."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from resid import (
    AnalysisWindow,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    SequentialWLSResidualizer,
    SquareRootMarketCapWeights,
    history_start,
    previous_session_values,
)
from resid.cli import MarketBetaConfig


def timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def write_marcap(directory: Path, frame: pd.DataFrame) -> MarcapDataSource:
    directory.mkdir(parents=True, exist_ok=True)
    for year, year_frame in frame.groupby(frame["Date"].dt.year):
        year_frame.to_parquet(directory / f"marcap-{year}.parquet", index=False)
    return MarcapDataSource(directory)


def test_trading_dates_are_normalized_like_loaded_dates(tmp_path: Path) -> None:
    """Universe indexes are built from trading_dates then joined against load()."""

    sessions = pd.date_range("2025-01-02 09:00", periods=4, freq="B")
    all_sessions = pd.DatetimeIndex([timestamp("2025-01-01 09:00"), *sessions])
    frame = pd.DataFrame(
        {
            "Date": np.repeat(all_sessions, 2),
            "Code": ["000001", "000002"] * len(all_sessions),
            "ChangesRatio": 1.0,
            "Marcap": [2e9, 1e9] * len(all_sessions),
        }
    )
    source = write_marcap(tmp_path / "data", frame)

    dates = source.trading_dates(timestamp("2025-01-01"), timestamp("2025-01-31"))
    loaded = source.load(timestamp("2025-01-01"), timestamp("2025-01-31"))

    assert all(value == cast(pd.Timestamp, value).normalize() for value in dates)
    assert set(dates) == set(loaded.index.get_level_values("date").unique())
    # A normalized end date must still find that day's intraday observations.
    assert source.latest_date("2025-01-02") == timestamp("2025-01-02")
    assert source.latest_date() == sessions[-1].normalize()
    universe = FixedTopMarketCapUniverse(size=1).build(
        source, AnalysisWindow(sessions[0].normalize(), sessions[-1].normalize())
    )
    # Would raise KeyError if the two date representations disagreed.
    assert universe.xs(sessions[0].normalize(), level="date").index.tolist() == [
        "000001"
    ]


def test_fixed_universe_rejects_an_empty_window(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-02")],
            "Code": ["000001"],
            "ChangesRatio": [1.0],
            "Marcap": [1e9],
        }
    )
    source = write_marcap(tmp_path / "data", frame)

    with pytest.raises(ValueError, match="no trading dates"):
        FixedTopMarketCapUniverse(size=1).build(
            source,
            AnalysisWindow(timestamp("2030-01-01"), timestamp("2030-12-31")),
        )


def test_market_beta_decay_bound_matches_the_model() -> None:
    """decay=1.0 never forgets; the model rejects it, so the config must too."""

    with pytest.raises(ValidationError):
        MarketBetaConfig(enabled=True, lookback_days=10, min_periods=5, decay=1.0)
    assert (
        MarketBetaConfig(
            enabled=True, lookback_days=10, min_periods=5, decay=0.97
        ).decay
        == 0.97
    )


def test_degenerate_exempt_column_is_not_fitted() -> None:
    """A constant column stays collinear with the intercept even when unscaled."""

    tickers = pd.Index([f"T{i}" for i in range(20)], name="ticker")
    returns = pd.Series(np.linspace(-0.02, 0.02, len(tickers)), index=tickers)
    exposures = pd.DataFrame(
        {
            "BETA": np.ones(len(tickers)),
            "SIZE": np.square(np.linspace(1, 20, len(tickers))),
        },
        index=tickers,
    )
    residualizer = SequentialWLSResidualizer(
        factor_order=("BETA", "SIZE"), winsor_quantile=0, unscaled_factors=("BETA",)
    )

    assert residualizer.fit(returns, exposures, pd.Series(1.0, index=tickers)) is None

    exposures["BETA"] = np.linspace(0.8, 1.2, len(tickers))
    fit = residualizer.fit(returns, exposures, pd.Series(1.0, index=tickers))
    assert fit is not None
    assert fit.exposure_scales["BETA"] == 1.0


def test_history_start_covers_the_requested_sessions() -> None:
    """Holidays make sessions scarcer than business days, so BDay alone is short."""

    business_days = pd.date_range("2015-01-01", "2025-12-31", freq="B")
    holidays = business_days[:: 261 // 15]
    sessions = business_days.difference(holidays)
    start = timestamp("2025-01-02")

    for requested in (252, 1008, 1260):
        available = sessions[
            (sessions >= history_start(start, requested)) & (sessions < start)
        ]
        assert len(available) >= requested
        naive = sessions[
            (sessions >= start - pd.offsets.BDay(requested)) & (sessions < start)
        ]
        assert len(naive) < requested


def test_previous_session_lag_does_not_carry_stale_values() -> None:
    """A ticker that did not trade yesterday has no prior-session market cap."""

    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    index = pd.MultiIndex.from_tuples(
        [
            (dates[0], "A"),
            (dates[0], "B"),
            (dates[1], "A"),
            (dates[2], "A"),
            (dates[2], "B"),
        ],
        names=["date", "ticker"],
    )
    market_caps = pd.Series([100.0, 400.0, 900.0, 1600.0, 2500.0], index=index)

    lagged = previous_session_values(market_caps)

    assert lagged.loc[(dates[1], "A")] == 100.0
    # B skipped a session: a per-observation shift would wrongly report 400.
    assert np.isnan(lagged.loc[(dates[2], "B")])
    weights = SquareRootMarketCapWeights().calculate(
        pd.DataFrame({"market_cap": market_caps})
    )
    assert weights.loc[(dates[1], "A")] == 10.0
    assert np.isnan(weights.loc[(dates[2], "B")])
