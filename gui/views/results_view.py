"""Results — the charts.

Native reward is the default; shaped is a deliberate act.

The two arms optimise different objectives, so their shaped returns are measured
with different rulers and the shaped arm wins trivially on its own scale.
Plotting shaped across arms is therefore a research error, and the UI makes it
one you have to choose on purpose.
"""

from __future__ import annotations

import statistics

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data.models import EpisodeRow, RunSummary
from gui.data.repository import Repository
from gui.widgets.common import Banner, MetricTile, label, section_label
from gui.workers.query import run_async

pg.setConfigOption("background", theme.SURFACE)
pg.setConfigOption("foreground", theme.TEXT_TERTIARY)
pg.setConfigOption("antialias", True)


class ResultsView(QWidget):
    def __init__(self, repository: Repository) -> None:
        super().__init__()
        self.repository = repository
        self._runs: list[RunSummary] = []
        self._episodes: list[EpisodeRow] = []
        self._scale = "native"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SP_L, theme.SP_L, theme.SP_L, theme.SP_L)
        outer.setSpacing(theme.SP_M)

        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("Results", "viewTitle"))
        self.subtitle = label(
            "native reward is the default; shaped is a deliberate act", "viewSubtitle"
        )
        head.addWidget(self.subtitle)
        outer.addLayout(head)

        outer.addLayout(self._build_toolbar())

        self.banner_slot = QVBoxLayout()
        outer.addLayout(self.banner_slot)

        self.tiles_row = QHBoxLayout()
        self.tiles_row.setSpacing(theme.SP_M)
        self.tiles = {
            "return": MetricTile("Mean native return"),
            "success": MetricTile("Success rate"),
            "steps": MetricTile("Steps to goal"),
            "cvss": MetricTile("Mean CVSS exploited"),
            "episodes": MetricTile("Episodes"),
        }
        for tile in self.tiles.values():
            self.tiles_row.addWidget(tile)
        outer.addLayout(self.tiles_row)

        charts = QHBoxLayout()
        charts.setSpacing(theme.SP_M)
        self.return_plot = self._make_plot("Return vs episode", "return")
        self.steps_plot = self._make_plot("Steps to goal — lower is better", "steps")
        charts.addWidget(self.return_plot, 1)
        charts.addWidget(self.steps_plot, 1)
        outer.addLayout(charts, 1)

        outer.addWidget(section_label("Per-seed final return"))
        outer.addWidget(self._build_seed_table(), 1)

    # -- construction ----------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.SP_S)

        self.runs_label = label("No runs loaded", "panelHeading")
        row.addWidget(self.runs_label)
        row.addStretch(1)

        row.addWidget(section_label("Reward scale"))
        segment = QWidget()
        segment.setObjectName("segment")
        seg_row = QHBoxLayout(segment)
        seg_row.setContentsMargins(0, 0, 0, 0)
        seg_row.setSpacing(0)

        self.scale_group = QButtonGroup(self)
        self.scale_group.setExclusive(True)
        self.native_button = QPushButton("Native")
        self.native_button.setCheckable(True)
        self.native_button.setChecked(True)
        self.shaped_button = QPushButton("Shaped")
        self.shaped_button.setCheckable(True)
        self.shaped_button.setProperty("warnState", True)
        for button in (self.native_button, self.shaped_button):
            self.scale_group.addButton(button)
            seg_row.addWidget(button)
        self.native_button.clicked.connect(lambda: self._set_scale("native"))
        self.shaped_button.clicked.connect(self._request_shaped)
        row.addWidget(segment)

        self.scale_note = label("the comparable one", "sectionLabel")
        row.addWidget(self.scale_note)
        return row

    def _make_plot(self, title: str, kind: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setMinimumHeight(190)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.getAxis("bottom").setLabel("episode")
        plot.setMenuEnabled(False)
        plot.addLegend(offset=(-10, 10), labelTextColor=theme.TEXT_SECONDARY)
        plot._kind = kind
        return plot

    def _build_seed_table(self) -> QWidget:
        self.seed_table = QTableWidget(0, 5)
        self.seed_table.setHorizontalHeaderLabels(
            ["Seed", "Run", "Reward", "Final native", "Success"]
        )
        self.seed_table.verticalHeader().setVisible(False)
        self.seed_table.verticalHeader().setDefaultSectionSize(theme.ROW_HEIGHT_DENSE)
        self.seed_table.setShowGrid(False)
        self.seed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.seed_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        head = self.seed_table.horizontalHeader()
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in (0, 2, 3, 4):
            head.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.seed_table.setMaximumHeight(200)
        return self.seed_table

    # -- scale toggle ----------------------------------------------------

    def _request_shaped(self) -> None:
        """Switching to shaped requires confirmation, by design."""
        answer = QMessageBox.warning(
            self,
            "Shaped reward is not comparable across arms",
            "The shaped and sparse agents optimise different objectives, so "
            "their shaped returns are on different scales. A chart of shaped "
            "reward across both arms is not a valid comparison and should not "
            "appear in the dissertation.\n\nShow it anyway?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._set_scale("shaped")
        else:
            self.native_button.setChecked(True)

    def _set_scale(self, scale: str) -> None:
        self._scale = scale
        if scale == "native":
            self.scale_note.setText("the comparable one")
            self.scale_note.setStyleSheet("")
        else:
            self.scale_note.setText("NOT COMPARABLE ACROSS ARMS")
            self.scale_note.setStyleSheet(
                f"color:{theme.WARN_TEXT}; font-family:'{theme.FONT_MONO}'; "
                f"font-size:{theme.SIZE_LABEL}px;"
            )
        self._redraw()

    # -- loading ---------------------------------------------------------

    def on_shown(self) -> None:
        if not self._runs:
            self._show_banner(
                Banner(
                    "No runs selected",
                    "Choose one or two runs in the Runs view and open them here. "
                    "Two runs are only charted together when their provenance "
                    "hashes match.",
                    kind="info",
                )
            )

    def load_runs(self, names: list[str]) -> None:
        self._clear_banners()
        run_async(
            lambda: self._fetch(names), self._on_loaded, self._on_error
        )

    def _fetch(self, names: list[str]):
        runs = [r for r in self.repository.list_runs() if r.name in names]
        episodes = self.repository.episodes([r.experiment_id for r in runs])
        return runs, episodes

    def _on_loaded(self, payload) -> None:
        self._runs, self._episodes = payload
        self.runs_label.setText(
            " vs ".join(r.name for r in self._runs) if self._runs else "No runs loaded"
        )
        if not self._episodes:
            self._show_banner(
                Banner(
                    "These runs have no episodes",
                    "The experiment row exists but no episode completed. The run "
                    "may have failed at startup.",
                    kind="warn",
                )
            )
        self._redraw()

    def _on_error(self, message: str, detail: str) -> None:
        self._show_banner(Banner(message, "", kind="error", detail=detail))

    # -- drawing ---------------------------------------------------------

    def _series_colour(self, index: int, reward_mode: str) -> str:
        # Colour follows the arm, not the row order, so a filter that changes
        # which runs are shown never repaints the survivors.
        if reward_mode == "shaped":
            return theme.ARM_2
        if reward_mode in ("sparse", "native"):
            return theme.ARM_1
        return theme.ARM_1 if index == 0 else theme.ARM_2

    def _redraw(self) -> None:
        for plot in (self.return_plot, self.steps_plot):
            plot.clear()

        by_run: dict[str, list[EpisodeRow]] = {}
        for row in self._episodes:
            by_run.setdefault(row.run_name, []).append(row)

        for index, (name, rows) in enumerate(sorted(by_run.items())):
            rows.sort(key=lambda r: r.episode_idx)
            colour = self._series_colour(index, rows[0].reward_mode)
            pen = pg.mkPen(colour, width=2)
            xs = [r.episode_idx for r in rows]

            ys = [
                r.native_reward if self._scale == "native" else r.total_reward
                for r in rows
            ]
            self.return_plot.plot(xs, ys, pen=pen, name=name)

            successes = [(r.episode_idx, r.length) for r in rows if r.goal_reached]
            if successes:
                self.steps_plot.plot(
                    [p[0] for p in successes], [p[1] for p in successes],
                    pen=pen, name=name,
                )

        self.return_plot.setTitle(
            f"{'Native' if self._scale == 'native' else 'Shaped'} return vs episode"
        )
        self._update_tiles(by_run)
        self._update_seed_table(by_run)

    def _update_tiles(self, by_run: dict[str, list[EpisodeRow]]) -> None:
        rows = [r for group in by_run.values() for r in group]
        if not rows:
            for tile in self.tiles.values():
                tile.set_value("—")
            return

        natives = [r.native_reward for r in rows]
        successes = [r for r in rows if r.goal_reached]
        cvss = [r.mean_cvss_exploited for r in rows if r.mean_cvss_exploited is not None]

        self.tiles["return"].set_value(
            theme.fmt_num(statistics.mean(natives), 0), "mean across loaded runs"
        )
        self.tiles["success"].set_value(
            f"{len(successes) / len(rows):.2f}", f"n = {len(rows)} episodes"
        )
        self.tiles["steps"].set_value(
            theme.fmt_num(statistics.mean([r.length for r in successes]), 1)
            if successes else "—",
            "successful episodes only",
        )
        self.tiles["cvss"].set_value(
            theme.fmt_num(statistics.mean(cvss), 2) if cvss else "—",
            "severity of exploited hosts",
        )
        self.tiles["episodes"].set_value(
            theme.fmt_num(len(rows)), f"{len(by_run)} run(s)"
        )

    def _update_seed_table(self, by_run: dict[str, list[EpisodeRow]]) -> None:
        self.seed_table.setRowCount(0)
        for name, rows in sorted(by_run.items()):
            rows.sort(key=lambda r: r.episode_idx)
            last = rows[-1]
            successes = sum(1 for r in rows if r.goal_reached)
            row_idx = self.seed_table.rowCount()
            self.seed_table.insertRow(row_idx)
            values = [
                (str(last.seed), True),
                (name, True),
                (last.reward_mode, False),
                (theme.fmt_num(last.native_reward, 1), True),
                (f"{successes / len(rows):.2f}", True),
            ]
            for col, (text, is_mono) in enumerate(values):
                item = QTableWidgetItem(text)
                if is_mono:
                    from PySide6.QtGui import QFont

                    item.setFont(QFont(theme.FONT_MONO, int(theme.SIZE_MONO * 0.75)))
                if col in (0, 3, 4):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.seed_table.setItem(row_idx, col, item)

    # -- banners ---------------------------------------------------------

    def _show_banner(self, banner: QLabel | Banner) -> None:
        self._clear_banners()
        self.banner_slot.addWidget(banner)

    def _clear_banners(self) -> None:
        while self.banner_slot.count():
            item = self.banner_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
