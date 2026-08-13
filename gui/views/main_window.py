"""The shell: title, 186px nav rail, stacked views, status bar.

Layout is Qt layout managers, not QSS -- QHBoxLayout for rail + QStackedWidget,
exactly as the visual direction specifies.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data.repository import ConnectionSettings, Repository
from gui.views.replay_view import ReplayView
from gui.views.results_view import ResultsView
from gui.views.runs_view import RunsView
from gui.views.train_view import TrainView
from gui.widgets.common import StatusDot

VIEWS = ("Runs", "Results", "Replay", "Train")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = ConnectionSettings.from_env()
        self.repository = Repository(self.settings)

        self.setWindowTitle("RLRedTeam Analyst")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 680)

        root = QWidget()
        root.setObjectName("surface")
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(self._build_rail())

        self.stack = QStackedWidget()
        self.runs_view = RunsView(self.repository)
        self.results_view = ResultsView(self.repository)
        self.replay_view = ReplayView(self.repository)
        self.train_view = TrainView(self.repository)
        for view in (self.runs_view, self.results_view, self.replay_view, self.train_view):
            self.stack.addWidget(view)
        row.addWidget(self.stack, 1)

        self.setCentralWidget(root)
        self._build_status_bar()

        # Cross-view navigation: acting on a run should land you in the right
        # place rather than making you re-find it.
        self.runs_view.open_in_results.connect(self._open_results)
        self.runs_view.open_in_replay.connect(self._open_replay)
        self.runs_view.go_to_train.connect(lambda: self._select(3))
        self.train_view.run_completed.connect(self.runs_view.refresh)

        self._select(0)

    # -- chrome ----------------------------------------------------------

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("navRail")
        rail.setFixedWidth(theme.RAIL_WIDTH)

        column = QVBoxLayout(rail)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        title = QLabel("RLRedTeam Analyst")
        title.setObjectName("appTitle")
        column.addWidget(title)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for index, name in enumerate(VIEWS):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, i=index: self._select(i))
            self._nav_group.addButton(button, index)
            column.addWidget(button)

        column.addStretch(1)

        footer = QLabel(
            f"{self.settings.user}@\n{self.settings.host}:{self.settings.port}"
        )
        footer.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:'{theme.FONT_MONO}'; "
            f"font-size:{theme.SIZE_LABEL}px; padding:10px 14px; "
            f"border-top:1px solid {theme.BORDER}; background:{theme.PANEL};"
        )
        footer.setWordWrap(True)
        column.addWidget(footer)
        return rail

    def _build_status_bar(self) -> None:
        bar = self.statusBar()

        self._db_dot = StatusDot(theme.TEXT_MUTED)
        self._db_label = QLabel(f"postgres {self.settings.port}")
        self._db_label.setStyleSheet(
            f"color:{theme.TEXT_TERTIARY}; font-family:'{theme.FONT_MONO}'; "
            f"font-size:{theme.SIZE_LABEL}px;"
        )
        holder = QWidget()
        strip = QHBoxLayout(holder)
        strip.setContentsMargins(theme.SP_M, 0, theme.SP_M, 0)
        strip.setSpacing(theme.SP_S)
        strip.addWidget(self._db_dot)
        strip.addWidget(self._db_label)
        bar.addPermanentWidget(holder)

        self.runs_view.connection_state.connect(self._set_db_state)

    def _set_db_state(self, ok: bool, note: str) -> None:
        colour = theme.OK if ok else theme.ERROR
        self._db_dot._colour.setNamedColor(colour)
        self._db_dot.update()
        self._db_label.setText(note)
        self._db_label.setStyleSheet(
            f"color:{colour if not ok else theme.TEXT_TERTIARY}; "
            f"font-family:'{theme.FONT_MONO}'; font-size:{theme.SIZE_LABEL}px;"
        )

    # -- navigation ------------------------------------------------------

    def _select(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        button = self._nav_group.button(index)
        if button is not None:
            button.setChecked(True)
        widget = self.stack.currentWidget()
        if hasattr(widget, "on_shown"):
            widget.on_shown()

    def _open_results(self, run_names: list[str]) -> None:
        self.results_view.load_runs(run_names)
        self._select(1)

    def _open_replay(self, run_name: str) -> None:
        self.replay_view.load_run(run_name)
        self._select(2)

    # -- shutdown --------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Never leave an orphaned training process behind."""
        if self.train_view.is_running():
            answer = QMessageBox.question(
                self,
                "Training is still running",
                "A training run is active. Closing the app will kill the "
                "process group and the run will not finish.\n\nClose anyway?",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Close:
                event.ignore()
                return
            self.train_view.stop()
        event.accept()
