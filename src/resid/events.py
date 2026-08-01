"""Canonical live events and a durable local replay log."""

import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class PeriodKey(BaseModel):
    """Identity of one market-local residualization cross-section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    as_of: datetime


class _Event(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    event_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    key: PeriodKey
    known_at: datetime


PositiveRegressionWeight = Annotated[float, Field(gt=0)]


class PeriodOpened(_Event):
    """Freeze membership and regression weights before accepting returns."""

    kind: Literal["period_opened"] = "period_opened"
    tickers: tuple[str, ...] = Field(min_length=1)
    regression_weights: tuple[PositiveRegressionWeight, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tickers(self) -> PeriodOpened:
        if len(self.tickers) != len(set(self.tickers)):
            raise ValueError("period tickers must be unique")
        if any(not ticker for ticker in self.tickers):
            raise ValueError("period tickers must not be empty")
        if len(self.regression_weights) != len(self.tickers):
            raise ValueError("period weights must align one-for-one with tickers")
        return self


class ReturnObserved(_Event):
    """Replace one ticker's latest same-horizon return within a period."""

    kind: Literal["return_observed"] = "return_observed"
    ticker: str = Field(min_length=1)
    effective_at: datetime
    return_value: float
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def known_after_effective(self) -> ReturnObserved:
        effective_is_aware = self.effective_at.tzinfo is not None
        known_is_aware = self.known_at.tzinfo is not None
        if effective_is_aware != known_is_aware:
            raise ValueError(
                "effective_at and known_at must use the same timezone form"
            )
        if self.known_at < self.effective_at:
            raise ValueError("known_at must not precede effective_at")
        return self


class PeriodClosed(_Event):
    """Declare the period complete enough to publish and advance state."""

    kind: Literal["period_closed"] = "period_closed"


LiveEvent = Annotated[
    PeriodOpened | ReturnObserved | PeriodClosed,
    Field(discriminator="kind"),
]
_EVENT_ADAPTER = TypeAdapter(LiveEvent)


class LoggedEvent(BaseModel):
    """An event with its durable ingestion order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    offset: int = Field(gt=0)
    event: LiveEvent


class EventLog(Protocol):
    @property
    def latest_offset(self) -> int: ...

    def append(self, event: LiveEvent) -> LoggedEvent: ...

    def append_many(self, events: Iterable[LiveEvent]) -> tuple[LoggedEvent, ...]: ...

    def read(self, after_offset: int = 0) -> Iterator[LoggedEvent]: ...


class JsonlEventLog:
    """Single-process append-only JSONL log for local capture and replay."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._records: list[LoggedEvent] = []
        self._by_id: dict[str, tuple[LoggedEvent, str]] = {}
        self._load()

    @property
    def latest_offset(self) -> int:
        return self._records[-1].offset if self._records else 0

    def append(self, event: LiveEvent) -> LoggedEvent:
        return self.append_many((event,))[0]

    def append_many(self, events: Iterable[LiveEvent]) -> tuple[LoggedEvent, ...]:
        incoming = tuple(events)
        if not incoming:
            return ()

        with self._lock:
            results: list[LoggedEvent] = []
            pending: list[tuple[LoggedEvent, str]] = []
            known = dict(self._by_id)
            next_offset = self.latest_offset + 1
            for event in incoming:
                serialized = _serialize_event(event)
                existing = known.get(event.event_id)
                if existing is not None:
                    if existing[1] != serialized:
                        raise ValueError(
                            f"event_id already has different content: {event.event_id}"
                        )
                    results.append(existing[0])
                    continue
                logged = LoggedEvent(offset=next_offset, event=event)
                next_offset += 1
                pending.append((logged, serialized))
                known[event.event_id] = (logged, serialized)
                results.append(logged)

            if pending:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    for logged, _ in pending:
                        record = {
                            "offset": logged.offset,
                            "event": logged.event.model_dump(mode="json"),
                        }
                        stream.write(
                            json.dumps(record, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                for logged, serialized in pending:
                    self._records.append(logged)
                    self._by_id[logged.event.event_id] = (logged, serialized)
            return tuple(results)

    def read(self, after_offset: int = 0) -> Iterator[LoggedEvent]:
        if after_offset < 0:
            raise ValueError("after_offset must not be negative")
        with self._lock:
            records = tuple(
                record for record in self._records if record.offset > after_offset
            )
        return iter(records)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for expected_offset, line in enumerate(stream, start=1):
                record = json.loads(line)
                offset = int(record["offset"])
                if offset != expected_offset:
                    raise ValueError(
                        f"event log offset {offset} is not contiguous at "
                        f"{expected_offset}"
                    )
                event = _EVENT_ADAPTER.validate_python(record["event"])
                serialized = _serialize_event(event)
                if event.event_id in self._by_id:
                    raise ValueError(f"duplicate event_id in log: {event.event_id}")
                logged = LoggedEvent(offset=offset, event=event)
                self._records.append(logged)
                self._by_id[event.event_id] = (logged, serialized)


def _serialize_event(event: LiveEvent) -> str:
    return json.dumps(
        event.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
