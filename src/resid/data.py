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
            date_filter = (
                ds.field("Date") <= requested.to_datetime64() if requested else None
            )
            dates = ds.dataset(path, format="parquet").to_table(
                columns=["Date"], filter=date_filter
            )["Date"]
            if len(dates):
                return _timestamp(str(max(dates.to_pylist())))
        raise ValueError("no marcap observations found on or before end_date")

    def trading_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        date_filter = (ds.field("Date") >= start.to_datetime64()) & (
            ds.field("Date") <= end.to_datetime64()
        )
        dates = self._dataset(start, end).to_table(
            columns=["Date"], filter=date_filter
        )["Date"]
        return pd.DatetimeIndex(dates.to_pandas().unique()).sort_values()

    def load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: pd.Index | None = None,
    ) -> pd.DataFrame:
        data_filter = (ds.field("Date") >= start.to_datetime64()) & (
            ds.field("Date") <= end.to_datetime64()
        )
        if tickers is not None:
            data_filter &= ds.field("Code").isin(tickers.astype(str).tolist())
        frame = (
            self._dataset(start, end)
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

    def _dataset(self, start: pd.Timestamp, end: pd.Timestamp) -> ds.Dataset:
        paths = [
            path for path in self._paths() if start.year <= _year(path) <= end.year
        ]
        return ds.dataset(paths, format="parquet")

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
    """Select one first-day universe and hold it through the window."""

    size: int = Field(gt=0)

    def build(self, source: MarketDataSource, window: AnalysisWindow) -> pd.Series:
        dates = source.trading_dates(window.start, window.end)
        snapshot = source.load(dates[0], dates[0])
        tickers = (
            snapshot["market_cap"].nlargest(self.size).index.get_level_values("ticker")
        )
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


def _year(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _timestamp(
    value: str | date | datetime | np.datetime64 | pd.Timestamp,
) -> pd.Timestamp:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value))
    if pd.isna(timestamp):
        raise ValueError(f"invalid date: {value}")
    return timestamp.normalize()
