"""Canonical market-data source and universe boundaries."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

MARCAP_COLUMNS = ["Date", "Code", "ChangesRatio", "Marcap"]


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    start: pd.Timestamp
    end: pd.Timestamp


class MarketDataSource(Protocol):
    """Source of canonical date/ticker market observations."""

    def latest_date(self, end: str | None = None) -> pd.Timestamp: ...

    def trading_dates(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex: ...

    def load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: pd.Index | None = None,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class MarcapDataSource:
    """Adapt local marcap Parquet files to the canonical market-data schema."""

    data_dir: Path

    def latest_date(self, end: str | None = None) -> pd.Timestamp:
        requested = _timestamp(end) if end else None
        for path in reversed(self._paths()):
            if requested is not None and _year(path) > requested.year:
                continue
            date_filter = ds.field("Date") < _next_day(requested) if requested else None
            dates = ds.dataset(path, format="parquet").to_table(
                columns=["Date"], filter=date_filter
            )["Date"]
            if len(dates):
                return _timestamp(max(dates.to_pylist()))
        raise ValueError("no marcap observations found on or before end_date")

    def trading_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        paths = self._paths_in_range(start, end)
        if not paths:
            return pd.DatetimeIndex([], dtype="datetime64[ns]", name="date")
        dates = ds.dataset(paths, format="parquet").to_table(
            columns=["Date"], filter=_within_days(start, end)
        )["Date"]
        # Normalize to match load(), whose index these dates are joined against.
        return (
            pd.DatetimeIndex(
                [
                    cast(pd.Timestamp, pd.Timestamp(value)).normalize()
                    for value in dates.to_pylist()
                ]
            )
            .unique()
            .sort_values()
        )

    def load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: pd.Index | None = None,
    ) -> pd.DataFrame:
        paths = self._paths_in_range(start, end)
        if not paths:
            return _empty_market_data()
        data_filter = _within_days(start, end)
        if tickers is not None:
            data_filter &= ds.field("Code").isin(tickers.astype(str).tolist())
        frame = (
            ds.dataset(paths, format="parquet")
            .to_table(columns=MARCAP_COLUMNS, filter=data_filter)
            .to_pandas()
            .rename(
                columns={
                    "Date": "date",
                    "Code": "ticker",
                    "ChangesRatio": "return_percent",
                    "Marcap": "market_cap",
                }
            )
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = frame["ticker"].astype("string").str.zfill(6)
        return frame.set_index(["date", "ticker"]).sort_index()

    def _paths_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
        return [path for path in self._paths() if start.year <= _year(path) <= end.year]

    def _paths(self) -> list[Path]:
        paths = sorted(self.data_dir.expanduser().resolve().glob("marcap-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no marcap-*.parquet files found in {self.data_dir}"
            )
        return paths


class UniverseBuilder(Protocol):
    def build(self, source: MarketDataSource, window: AnalysisWindow) -> pd.Series: ...


@pydantic_dataclass(frozen=True, slots=True)
class FixedTopMarketCapUniverse:
    """Select one pre-window universe and hold it through the window."""

    size: int = Field(gt=0)

    def build(self, source: MarketDataSource, window: AnalysisWindow) -> pd.Series:
        dates = source.trading_dates(window.start, window.end)
        if not len(dates):
            raise ValueError("analysis window contains no trading dates")
        try:
            snapshot_date = source.latest_date(
                (dates[0] - pd.Timedelta(days=1)).date().isoformat()
            )
        except ValueError as exc:
            raise ValueError(
                "fixed universe requires a market-cap snapshot before the first "
                "analysis date"
            ) from exc
        snapshot = source.load(snapshot_date, snapshot_date).reset_index()
        tickers = (
            snapshot.loc[snapshot["market_cap"].gt(0)]
            .sort_values(
                ["market_cap", "ticker"],
                ascending=[False, True],
            )
            .head(self.size)["ticker"]
            .astype("string")
            .to_numpy()
        )
        if not len(tickers):
            raise ValueError("fixed universe snapshot contains no positive market caps")
        index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        return pd.Series(True, index=index, name="in_universe")


@pydantic_dataclass(frozen=True, slots=True)
class DailyTopMarketCapUniverse:
    """Select each day's universe using the previous trading day's market cap."""

    size: int = Field(gt=0)

    def build(self, source: MarketDataSource, window: AnalysisWindow) -> pd.Series:
        dates = source.trading_dates(window.start, window.end)
        if not len(dates):
            raise ValueError("analysis window contains no trading dates")
        previous_date = source.latest_date(
            (dates[0] - pd.Timedelta(days=1)).date().isoformat()
        )
        snapshot_dates = pd.DatetimeIndex([previous_date, *dates[:-1]])
        snapshot_end = snapshot_dates[-1]
        snapshots = (
            source.load(previous_date, snapshot_end)["market_cap"]
            .rename("market_cap")
            .reset_index()
        )
        effective_dates = dict(zip(snapshot_dates, dates, strict=True))
        snapshots["date"] = snapshots["date"].map(effective_dates)
        members = (
            snapshots.dropna(subset=["date", "market_cap"])
            .sort_values(
                ["date", "market_cap", "ticker"],
                ascending=[True, False, True],
            )
            .groupby("date", sort=True)
            .head(self.size)
        )
        index = pd.MultiIndex.from_frame(members[["date", "ticker"]])
        return pd.Series(True, index=index, name="in_universe").sort_index()


def analysis_window(
    source: MarketDataSource,
    years: int,
    end: str | None,
) -> AnalysisWindow:
    end_date = source.latest_date(end)
    return AnalysisWindow(end_date - pd.DateOffset(years=years), end_date)


def universe_index(universe: pd.Series) -> pd.MultiIndex:
    """The (date, ticker) pairs that are actually members."""

    return cast(pd.MultiIndex, universe.index[universe.to_numpy(dtype="bool")])


def universe_dates(universe: pd.Series) -> pd.DatetimeIndex:
    """Sorted dates on which the universe has at least one member."""

    dates = universe_index(universe).get_level_values("date").unique()
    return cast(pd.DatetimeIndex, dates.sort_values())


def universe_members(universe: pd.Series, date: pd.Timestamp) -> pd.Index:
    """Tickers that are members on one date."""

    membership = universe.xs(date, level="date")
    return membership.index[membership.to_numpy(dtype="bool")]


def previous_session_values(values: pd.Series) -> pd.Series:
    """Shift each ticker's observations forward to the next trading session.

    The lag is one *session* of the loaded calendar, not one observation, so a
    ticker that did not trade on the previous session yields NaN rather than a
    stale value carried over from whenever it last traded.
    """

    lagged = values.unstack("ticker").shift(1)
    index = pd.MultiIndex.from_product(
        [lagged.index, lagged.columns], names=["date", "ticker"]
    )
    return pd.Series(lagged.to_numpy().ravel(), index=index, name=values.name).reindex(
        values.index
    )


def history_start(start: pd.Timestamp, business_days: int) -> pd.Timestamp:
    """Calendar date far enough back to contain `business_days` trading sessions.

    Exchange holidays make trading sessions scarcer than business days, so
    counting back by BDay alone leaves a rolling window short at the start of the
    analysis period. Add proportional slack, generous enough for the ~15 annual
    KRX holidays, so a configured lookback of N sessions really sees N sessions.
    """

    if business_days <= 0:
        return start
    return cast(
        pd.Timestamp,
        start
        - pd.offsets.BDay(business_days)
        - pd.Timedelta(days=int(business_days * 0.1) + 7),
    )


def _year(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _next_day(value: pd.Timestamp) -> np.datetime64:
    return (value.normalize() + pd.Timedelta(days=1)).to_datetime64()


def _within_days(start: pd.Timestamp, end: pd.Timestamp) -> ds.Expression:
    """Match every observation on the calendar days from `start` to `end`.

    `load` normalizes the Date column it returns, so callers legitimately pass
    normalized dates back in. Comparing those against a stored intraday timestamp
    would exclude the very rows the caller asked for, so bound the filter by whole
    days rather than by the instants given.
    """

    return (ds.field("Date") >= start.normalize().to_datetime64()) & (
        ds.field("Date") < _next_day(end)
    )


def _empty_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "ticker": pd.Series(dtype="string"),
            "return_percent": pd.Series(dtype="float64"),
            "market_cap": pd.Series(dtype="float64"),
        }
    ).set_index(["date", "ticker"])


def _timestamp(
    value: str | date | datetime | np.datetime64 | pd.Timestamp,
) -> pd.Timestamp:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(timestamp):
        raise ValueError(f"invalid date: {value}")
    return timestamp.normalize()
