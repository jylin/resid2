"""Optional progress reporting for long-running residualization runs."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, TextIO


class StageProgress(Protocol):
    """Progress updates for one named pipeline stage."""

    def update(self, completed: int, detail: str | None = None) -> None: ...

    def summary(self, text: str | None = None) -> None: ...


class ProgressReporter(Protocol):
    """Reporter used by the pipeline without coupling it to a terminal."""

    def start_stage(
        self,
        name: str,
        total: int | None = None,
        unit: str = "items",
    ) -> StageProgress: ...

    def finish_stage(
        self,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None: ...


@contextmanager
def reported_stage(
    reporter: ProgressReporter,
    name: str,
    total: int | None = None,
    unit: str = "items",
) -> Iterator[StageProgress]:
    """Run a stage and mark it failed if its body raises."""

    stage = reporter.start_stage(name, total=total, unit=unit)
    try:
        yield stage
    except BaseException as exc:
        reporter.finish_stage(success=False, message=type(exc).__name__)
        raise
    else:
        reporter.finish_stage()


class _NullStage:
    def update(self, completed: int, detail: str | None = None) -> None:
        del completed, detail

    def summary(self, text: str | None = None) -> None:
        del text


class NullProgress:
    """No-op reporter used by library callers that do not request progress."""

    def start_stage(
        self,
        name: str,
        total: int | None = None,
        unit: str = "items",
    ) -> StageProgress:
        del name, total, unit
        return _NullStage()

    def finish_stage(
        self,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        del success, message


@dataclass(slots=True)
class _StageState:
    number: int
    name: str
    total: int | None
    unit: str
    started: float
    completed: int = 0
    detail: str | None = None
    summary: str | None = None
    last_rendered: float = 0.0
    last_logged: int = -1


@dataclass(slots=True)
class _TerminalStage:
    owner: TerminalProgress

    def update(self, completed: int, detail: str | None = None) -> None:
        self.owner._update(completed, detail)

    def summary(self, text: str | None = None) -> None:
        self.owner._summary(text)


@dataclass(slots=True)
class TerminalProgress:
    """Render colored, throttled stage progress to a terminal stream."""

    total_stages: int
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    min_interval: float = 0.1
    use_color: bool | None = None
    bar_width: int = 24
    _stage_number: int = field(default=0, init=False)
    _current: _StageState | None = field(default=None, init=False)
    _interactive: bool = field(default=False, init=False)
    _colors: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.total_stages < 1:
            raise ValueError("total_stages must be positive")
        if self.bar_width < 1:
            raise ValueError("bar_width must be positive")
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._interactive = is_tty
        self._colors = (
            is_tty and "NO_COLOR" not in os.environ
            if self.use_color is None
            else self.use_color
        )

    def start_stage(
        self,
        name: str,
        total: int | None = None,
        unit: str = "items",
    ) -> StageProgress:
        if self._current is not None:
            raise RuntimeError(
                "cannot start a progress stage before finishing the current stage"
            )
        self._stage_number += 1
        self._current = _StageState(
            number=self._stage_number,
            name=name,
            total=total,
            unit=unit,
            started=perf_counter(),
        )
        self._render(force=True)
        return _TerminalStage(self)

    def finish_stage(
        self,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        state = self._current
        if state is None:
            raise RuntimeError("cannot finish progress without an active stage")
        if success and state.total is not None:
            state.completed = state.total
        elapsed = perf_counter() - state.started
        line = self._format_line(state, include_detail=False)
        status = (
            f"done in {elapsed:.2f}s" if success else f"FAILED after {elapsed:.2f}s"
        )
        if success and state.completed > 0 and elapsed > 0:
            status = f"{status} | {state.completed / elapsed:,.1f} {state.unit}/s"
        if message:
            status = f"{status} | {message}"
        color = "\033[32m" if success else "\033[31m"
        if self._interactive or state.total is not None:
            self._write_line(line, final=True)
        self._write_report(f"  status: {self._paint(status, color)}")
        if state.summary:
            heading = self._paint("details:", "\033[36m\033[1m")
            self._write_report(f"  {heading}")
            for summary_line in state.summary.splitlines():
                self._write_report(f"    {summary_line}" if summary_line else "")
        self._write_report("")
        self._current = None

    def _update(self, completed: int, detail: str | None) -> None:
        state = self._current
        if state is None:
            raise RuntimeError("cannot update progress without an active stage")
        if completed < 0:
            raise ValueError("completed progress cannot be negative")
        if state.total is not None:
            completed = min(completed, state.total)
        state.completed = completed
        state.detail = detail
        self._render()

    def _summary(self, text: str | None) -> None:
        state = self._current
        if state is None:
            raise RuntimeError("cannot summarize progress without an active stage")
        state.summary = text

    def _render(self, force: bool = False) -> None:
        state = self._current
        if state is None:
            return
        now = perf_counter()
        complete = state.total is not None and state.completed >= state.total
        if not force and not complete and now - state.last_rendered < self.min_interval:
            return
        if not self._interactive and not force:
            threshold = max(1, (state.total or 1) // 10)
            if state.total is None or state.completed - state.last_logged < threshold:
                return
        state.last_rendered = now
        state.last_logged = state.completed
        self._write_line(self._format_line(state))

    def _format_line(self, state: _StageState, *, include_detail: bool = True) -> str:
        prefix = self._paint(
            f"[{state.number}/{self.total_stages}]",
            "\033[36m\033[1m",
        )
        name = self._paint(state.name, "\033[1m")
        if state.total is None:
            progress = self._paint("[working]", "\033[33m")
            line = f"{prefix} {name} {progress}"
        else:
            ratio = 1.0 if state.total == 0 else state.completed / state.total
            filled = round(self.bar_width * ratio)
            bar = "#" * filled + "-" * (self.bar_width - filled)
            progress = self._paint(f"[{bar}]", "\033[32m")
            percent = self._paint(f"{ratio:>4.0%}", "\033[33m")
            line = (
                f"{prefix} {name} {progress} {percent} "
                f"{state.completed:,}/{state.total:,} {state.unit}"
            )
        if include_detail and state.detail:
            line = f"{line} · {state.detail}"
        return line

    def _write_line(self, line: str, *, final: bool = False) -> None:
        if self._interactive:
            self.stream.write(f"\r\033[2K{line}")
            if final:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()

    def _write_report(self, line: str) -> None:
        self.stream.write(f"{line}\n")
        self.stream.flush()

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}\033[0m" if self._colors else text
