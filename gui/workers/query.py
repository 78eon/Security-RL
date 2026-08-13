"""Runs blocking work off the UI thread.

Qt's rule is absolute: only the main thread may touch widgets. Violating it
produces intermittent crashes that are near-impossible to reproduce, so the
structure prevents it rather than relying on discipline -- a worker receives a
callable and emits its plain-data result by signal. No worker ever holds a
widget reference.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str, str)  # message, detail


class Task(QRunnable):
    """Runs ``fn`` on the thread pool and reports back by signal."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _Signals()

    def run(self) -> None:  # executed on a pool thread
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
            detail = getattr(exc, "detail", "") or traceback.format_exc(limit=3)
            self.signals.failed.emit(message, detail)
        else:
            self.signals.finished.emit(result)


def run_async(
    fn: Callable[[], Any],
    on_done: Callable[[Any], None],
    on_error: Callable[[str, str], None] | None = None,
) -> Task:
    """Schedule ``fn``; deliver its result to ``on_done`` on the UI thread."""
    task = Task(fn)
    task.signals.finished.connect(on_done)
    if on_error is not None:
        task.signals.failed.connect(on_error)
    QThreadPool.globalInstance().start(task)
    return task
