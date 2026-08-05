"""Replayable live-period orchestration over the shared factor and OLS core."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Literal, cast

import numpy as np
import pandas as pd

from resid.data import universe_dates, universe_members
from resid.events import (
    EventLog,
    LiveEvent,
    LoggedEvent,
    PeriodClosed,
    PeriodKey,
    PeriodOpened,
    ReturnObserved,
)
from resid.factors import PreparedFactorModel
from resid.regression import (
    CrossSectionFit,
    IncrementalRegression,
    IncrementalResidualizer,
)


@dataclass(frozen=True, slots=True, eq=False)
class LivePeriodResult:
    """One provisional or authoritative live cross-sectional result."""

    input_offset: int
    model_version: str
    key: PeriodKey
    revision: int
    status: Literal["provisional", "final"]
    fit: CrossSectionFit


@dataclass(slots=True, eq=False)
class _ActivePeriod:
    key: PeriodKey
    regression: IncrementalRegression
    revision: int = 0


@dataclass(slots=True, eq=False)
class LiveRegressionRunner:
    """Consume logged return events and advance factor state only at close."""

    model_version: str
    market: str
    interval: str
    factors: PreparedFactorModel
    residualizer: IncrementalResidualizer
    _active: _ActivePeriod | None = field(init=False, default=None, repr=False)
    _last_offset: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.model_version or not self.market or not self.interval:
            raise ValueError("model_version, market, and interval are required")

    @property
    def last_offset(self) -> int:
        return self._last_offset

    @property
    def active_key(self) -> PeriodKey | None:
        return self._active.key if self._active is not None else None

    def capture(
        self,
        log: EventLog,
        event: LiveEvent,
    ) -> LivePeriodResult | None:
        """Durably append an event before applying it to live state."""

        return self.apply(log.append(event))

    def apply(self, logged: LoggedEvent) -> LivePeriodResult | None:
        """Apply one event in durable offset order."""

        if logged.offset <= self._last_offset:
            return None
        event = logged.event
        self._check_key(event.key)

        if isinstance(event, PeriodOpened):
            self._open(event)
            result = None
        elif isinstance(event, ReturnObserved):
            result = self._observe(logged.offset, event)
        elif isinstance(event, PeriodClosed):
            result = self._close(logged.offset, event)
        else:
            raise TypeError(f"unsupported event kind: {event.kind}")
        self._last_offset = logged.offset
        return result

    def replay(self, events: Iterable[LoggedEvent]) -> tuple[LivePeriodResult, ...]:
        """Apply a recorded stream exactly as originally ingested."""

        results = []
        for logged in events:
            result = self.apply(logged)
            if result is not None:
                results.append(result)
        return tuple(results)

    def _check_key(self, key: PeriodKey) -> None:
        if key.market != self.market or key.interval != self.interval:
            raise ValueError(
                f"event belongs to {key.market}/{key.interval}, expected "
                f"{self.market}/{self.interval}"
            )

    def _open(self, event: PeriodOpened) -> None:
        if self._active is not None:
            raise ValueError(f"period is already active: {self._active.key}")
        tickers = pd.Index(event.tickers, name="ticker", dtype="string")
        date = cast(pd.Timestamp, pd.Timestamp(event.key.as_of))
        exposures = self.factors.exposures(date, tickers)
        if tuple(str(name) for name in exposures.columns) != self.factors.names:
            raise ValueError("prepared factor names changed when the period opened")
        regression_weights = pd.Series(
            event.regression_weights,
            index=tickers,
            name="regression_weight",
        )
        self._active = _ActivePeriod(
            key=event.key,
            regression=IncrementalRegression(
                self.residualizer,
                exposures,
                regression_weights,
            ),
        )

    def _observe(
        self,
        offset: int,
        event: ReturnObserved,
    ) -> LivePeriodResult | None:
        active = self._require_active(event.key)
        fit = active.regression.update(event.ticker, event.return_value)
        if fit is None:
            return None
        active.revision += 1
        return LivePeriodResult(
            input_offset=offset,
            model_version=self.model_version,
            key=event.key,
            revision=active.revision,
            status="provisional",
            fit=fit,
        )

    def _close(
        self,
        offset: int,
        event: PeriodClosed,
    ) -> LivePeriodResult:
        active = self._require_active(event.key)
        fit = active.regression.finalize()
        if fit is None:
            raise ValueError("period does not contain enough valid observations to fit")
        active.revision += 1
        date = cast(pd.Timestamp, pd.Timestamp(event.key.as_of))
        self.factors.update(
            date,
            replace(fit, observed_returns=active.regression.returns),
        )
        result = LivePeriodResult(
            input_offset=offset,
            model_version=self.model_version,
            key=event.key,
            revision=active.revision,
            status="final",
            fit=fit,
        )
        self._active = None
        return result

    def _require_active(self, key: PeriodKey) -> _ActivePeriod:
        if self._active is None:
            raise ValueError("no period is active")
        if self._active.key != key:
            raise ValueError(f"active period is {self._active.key}, received {key}")
        return self._active


def historical_events(
    *,
    universe: pd.Series,
    returns: pd.Series,
    regression_weights: pd.Series,
    market: str,
    interval: str,
) -> Iterator[LiveEvent]:
    """Expose finalized historical returns through the canonical live event schema."""

    sequence = 0
    for timestamp in universe_dates(universe):
        as_of = timestamp.to_pydatetime()
        key = PeriodKey(market=market, interval=interval, as_of=as_of)
        tickers = universe_members(universe, timestamp).astype(str)
        day_weights = regression_weights.xs(timestamp, level="date").reindex(tickers)
        eligible = np.isfinite(day_weights) & day_weights.gt(0)
        tickers = tickers[eligible.to_numpy()]
        day_weights = day_weights.loc[tickers]
        prefix = f"{market}:{interval}:{timestamp.isoformat()}"
        yield PeriodOpened(
            event_id=f"{prefix}:open",
            source_sequence=sequence,
            key=key,
            known_at=as_of,
            tickers=tuple(tickers),
            regression_weights=tuple(float(value) for value in day_weights),
        )
        sequence += 1

        day_returns = returns.xs(timestamp, level="date").reindex(tickers)
        for ticker, value in day_returns.items():
            if not np.isfinite(value):
                continue
            yield ReturnObserved(
                event_id=f"{prefix}:return:{ticker}",
                source_sequence=sequence,
                key=key,
                effective_at=as_of,
                known_at=as_of,
                ticker=str(ticker),
                return_value=float(value),
            )
            sequence += 1
        yield PeriodClosed(
            event_id=f"{prefix}:close",
            source_sequence=sequence,
            key=key,
            known_at=as_of,
        )
        sequence += 1
