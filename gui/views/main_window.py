"""Stable import location for the supplied research-console window."""

from __future__ import annotations

from gui.backend import ApplicationBackend, BackendPort
from gui.views.research_console import MainWindow as ResearchMainWindow


class MainWindow(ResearchMainWindow):
    def __init__(self, backend: BackendPort | None = None) -> None:
        super().__init__(backend=backend or ApplicationBackend())
