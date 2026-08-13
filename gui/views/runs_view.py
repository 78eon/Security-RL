"""Runs — the browser.

Comparability is the headline, not the hash. Two runs may only be charted
against each other when their topology and CVE-catalogue fingerprints match, and
the whole point of this screen is that a researcher can see that at a glance
instead of diffing 64-character strings by eye.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data import runs as runs_folder
from gui.data.models import Comparison, RunSummary
from gui.data.repository import Repository
from gui.widgets.common import Banner, hline, label, section_label, status_chip
from gui.workers.query import run_async

COLUMNS = ["Run", "Reward", "Seed", "Episodes", "Created", "Provenance", "Status"]


class RunsView(QWidget):
    open_in_results = Signal(list)
    open_in_replay = Signal(str)
    go_to_train = Signal()
    connection_state = Signal(bool, str)

    def __init__(self, repository: Repository) -> None:
        super().__init__()
        self.repository = repository
        self._runs: list[RunSummary] = []
        self._loaded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SP_L, theme.SP_L, theme.SP_L, theme.SP_L)
        outer.setSpacing(theme.SP_M)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(label("Runs", "viewTitle"))
        header.addWidget(
            label("comparability is the headline, not the hash", "viewSubtitle")
        )
        outer.addLayout(header)

        outer.addLayout(self._build_toolbar())

        self.banner_slot = QVBoxLayout()
        self.banner_slot.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.banner_slot)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.addWidget(self._build_table())
        self.splitter.addWidget(self._build_detail())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setSizes([900, theme.DETAIL_WIDTH])
        outer.addWidget(self.splitter, 1)

        self.footer = label("", "sectionLabel")
        outer.addWidget(self.footer)

    # -- construction ----------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.SP_S)

        self.count_label = label("Experiments", "panelHeading")
        row.addWidget(self.count_label)

        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("Filter runs…")
        self.filter_box.setFixedWidth(220)
        self.filter_box.textChanged.connect(self._apply_filter)
        row.addWidget(self.filter_box)

        self.reward_filter = QComboBox()
        self.reward_filter.addItems(["Reward: all", "shaped", "sparse", "native"])
        self.reward_filter.currentIndexChanged.connect(self._apply_filter)
        row.addWidget(self.reward_filter)

        row.addStretch(1)

        self.compare_button = QPushButton("Compare selected (0)")
        self.compare_button.setProperty("primary", True)
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._compare_selected)
        row.addWidget(self.compare_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        return row

    def _build_table(self) -> QWidget:
        holder = QWidget()
        holder.setObjectName("surface")
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(theme.SP_S)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(theme.ROW_HEIGHT)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection)

        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setMinimumSectionSize(64)
        self.table.setWordWrap(False)
        for i in range(1, len(COLUMNS) - 1):
            head.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        # Status holds a cell widget, which reports no size hint, so
        # ResizeToContents would collapse it and clip the word.
        head.setSectionResizeMode(
            len(COLUMNS) - 1, QHeaderView.ResizeMode.Fixed
        )
        self.table.setColumnWidth(len(COLUMNS) - 1, 116)

        column.addWidget(self.table, 1)
        return holder

    def _build_detail(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setMinimumWidth(280)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Without this a 64-character hash forces a horizontal scrollbar and
        # runs off the panel instead of wrapping onto a second line.
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.detail_body = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_body)
        self.detail_layout.setContentsMargins(
            theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD
        )
        self.detail_layout.setSpacing(theme.SP_M)
        self.detail_layout.addStretch(1)
        scroll.setWidget(self.detail_body)

        wrapper = QVBoxLayout(panel)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)
        self._show_detail_placeholder()
        return panel

    # -- loading ---------------------------------------------------------

    def on_shown(self) -> None:
        if not self._loaded:
            self.refresh()

    def refresh(self) -> None:
        self._loaded = True
        self.footer.setText("loading…")
        run_async(self.repository.list_runs, self._on_runs, self._on_error)

    def _on_runs(self, runs: list[RunSummary]) -> None:
        self._runs = runs
        self._clear_banners()
        self.connection_state.emit(True, f"postgres {self.repository.settings.port}")

        if not runs:
            self._show_empty_state()
        self._apply_filter()

    def _on_error(self, message: str, detail: str) -> None:
        self._runs = []
        self.table.setRowCount(0)
        self._clear_banners()
        self.connection_state.emit(False, "postgres unreachable")
        self.footer.setText("")

        banner = Banner(
            message,
            "The container may not be running. The app stays usable — runs/ "
            "folder artefacts still load.",
            kind="error",
            detail=detail.strip().splitlines()[0] if detail.strip() else "",
        )
        retry = QPushButton("Retry")
        retry.clicked.connect(self.refresh)
        banner.add_action(retry)

        folders = QPushButton("Read from runs/ only")
        folders.clicked.connect(self._load_from_folders)
        banner.add_action(folders)
        self.banner_slot.addWidget(banner)

    def _show_empty_state(self) -> None:
        banner = Banner(
            "No runs in the database yet",
            "Training writes to experiments when launched with --postgres. "
            "Start one from the Train view, or point the app at a different "
            "database.",
            kind="info",
        )
        button = QPushButton("Go to Train")
        button.setProperty("primary", True)
        button.clicked.connect(self.go_to_train.emit)
        banner.add_action(button)
        self.banner_slot.addWidget(banner)

    def _load_from_folders(self) -> None:
        """Degraded mode: list what is on disk when the database is gone."""
        names = runs_folder.list_run_folders()
        self.table.setRowCount(0)
        for name in names:
            snapshot = runs_folder.load_snapshot(name)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set(row, 0, name, mono=True)
            self._set(row, 1, snapshot.get("reward_mode", "—"))
            self._set(row, 2, str(snapshot.get("seed", "—")), numeric=True)
            self._set(row, 3, "—", numeric=True)
            self._set(row, 4, "—")
            self._set(
                row, 5,
                theme.truncate_hash(snapshot.get("topology_config_hash")), mono=True,
            )
            self.table.setCellWidget(row, 6, status_chip("on disk", theme.TEXT_TERTIARY))
        self.footer.setText(f"showing {len(names)} run folder(s) · database offline")

    # -- table -----------------------------------------------------------

    def _apply_filter(self) -> None:
        text = self.filter_box.text().strip().lower()
        reward = self.reward_filter.currentText()
        visible = [
            r for r in self._runs
            if (not text or text in r.name.lower())
            and (reward.startswith("Reward:") or r.reward_mode == reward)
        ]
        self._populate(visible)
        self.count_label.setText("Experiments")
        self.footer.setText(f"showing {len(visible)} of {len(self._runs)}")

    def _populate(self, runs: list[RunSummary]) -> None:
        self.table.setRowCount(0)
        for run in runs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set(row, 0, run.name, mono=True, payload=run)
            self._set(row, 1, run.reward_mode)
            self._set(row, 2, run.seed_label, numeric=True)
            self._set(row, 3, theme.fmt_num(run.episode_count), numeric=True)
            self._set(
                row, 4,
                run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "—",
                mono=True,
            )
            self._set(
                row, 5,
                f"{theme.truncate_hash(run.topology_config_hash)}  "
                f"{theme.truncate_hash(run.cve_manifest_sha256)}",
                mono=True,
            )
            colour = theme.OK if run.status == "complete" else theme.TEXT_TERTIARY
            self.table.setCellWidget(row, 6, status_chip(run.status, colour))

    def _set(
        self, row: int, col: int, text: str, *,
        mono: bool = False, numeric: bool = False, payload=None,
    ) -> None:
        item = QTableWidgetItem(text)
        if mono or numeric:
            item.setFont(QFont(theme.FONT_MONO, int(theme.SIZE_MONO * 0.75)))
        if numeric:
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        if payload is not None:
            item.setData(Qt.ItemDataRole.UserRole, payload)
        self.table.setItem(row, col, item)

    def selected_runs(self) -> list[RunSummary]:
        out = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            run = item.data(Qt.ItemDataRole.UserRole) if item else None
            if run is not None:
                out.append(run)
        return out

    def _on_selection(self) -> None:
        selected = self.selected_runs()
        self.compare_button.setText(f"Compare selected ({len(selected)})")
        self.compare_button.setEnabled(len(selected) == 2)

        self._clear_banners(keep_connection=True)
        if len(selected) == 2:
            self._show_comparability(Comparison(selected[0], selected[1]))
        if selected:
            self._show_detail(selected[-1])
        else:
            self._show_detail_placeholder()

    def _show_comparability(self, comparison: Comparison) -> None:
        pair = f"{comparison.left.name} · {comparison.right.name}"
        problems = comparison.reasons()
        if comparison.comparable and not problems:
            banner = Banner(
                "Comparable — matching topology_config_hash and cve_manifest_sha256",
                pair, kind="ok",
            )
        else:
            banner = Banner(
                "These two runs are not comparable",
                "; ".join(problems) + ". Charting them together would compare "
                "results produced under different conditions.",
                kind="error",
                detail=pair,
            )
        self.banner_slot.addWidget(banner)

    def _compare_selected(self) -> None:
        selected = self.selected_runs()
        if len(selected) == 2:
            self.open_in_results.emit([r.name for r in selected])

    # -- detail panel ----------------------------------------------------

    def _clear_detail(self) -> None:
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_detail_placeholder(self) -> None:
        self._clear_detail()
        self.detail_layout.addWidget(
            label("Select a run to see its provenance.", "muted", wrap=True)
        )
        self.detail_layout.addStretch(1)

    def _show_detail(self, run: RunSummary) -> None:
        self._clear_detail()
        add = self.detail_layout.addWidget

        add(label(run.name, "panelHeading"))
        snapshot = runs_folder.load_snapshot(run.name)
        add(label(
            f"config.snapshot.json · {theme.fmt_num(run.episode_count)} episodes"
            if snapshot else f"{theme.fmt_num(run.episode_count)} episodes",
            "muted",
        ))
        add(hline())

        add(section_label("Provenance"))
        for name, value in (
            ("topology_config_hash", run.topology_config_hash),
            ("cve_manifest_sha256", run.cve_manifest_sha256),
            ("reward_config_hash", run.config_hash),
            ("code_git_sha", run.git_sha or "—"),
        ):
            add(self._hash_row(name, value))
        add(hline())

        add(section_label("Configuration"))
        topo = snapshot.get("topology", {})
        for name, value in (
            ("topology seed", str(run.seeds[0]) if run.seeds else "—"),
            ("hosts / subnets",
             f"{topo.get('num_hosts', '—')} / {topo.get('num_subnets', '—')}"),
            ("timesteps", theme.fmt_num(snapshot.get("timesteps"))),
            ("reward mode", run.reward_mode),
            ("--log-steps", "on" if run.has_steps else "off"),
            ("mean native return", theme.fmt_num(run.mean_native_reward, 1)),
            ("success rate",
             theme.fmt_num(run.success_rate, 2) if run.success_rate is not None else "—"),
        ):
            add(self._kv_row(name, value))
        add(hline())

        add(section_label("Actions"))
        results_button = QPushButton("Open in Results")
        results_button.clicked.connect(lambda: self.open_in_results.emit([run.name]))
        add(results_button)

        replay_button = QPushButton("Replay attack path")
        replay_button.setEnabled(run.has_steps)
        if not run.has_steps:
            replay_button.setToolTip("This run was trained without --log-steps")
        replay_button.clicked.connect(lambda: self.open_in_replay.emit(run.name))
        add(replay_button)

        self.detail_layout.addStretch(1)

    def _kv_row(self, name: str, value: str) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SP_S)
        row.addWidget(label(name, "muted"))
        row.addStretch(1)
        row.addWidget(label(value, "mono"))
        return holder

    def _hash_row(self, name: str, value: str) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(label(name, "muted"))
        top.addStretch(1)
        copy = QPushButton("copy")
        copy.setFixedHeight(20)
        copy.setStyleSheet(
            f"font-size:{theme.SIZE_LABEL}px; padding:1px 7px; "
            f"color:{theme.TEXT_TERTIARY};"
        )
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(value))
        top.addWidget(copy)
        column.addLayout(top)

        # Full value, wrapped. A 64-character hex string contains no space, so
        # word wrap alone cannot break it and it would overflow the panel --
        # dragging the rest of the layout off-screen with it. Chunking into
        # groups of 16 gives real break opportunities and is easier to read
        # besides; the copy button still yields the clean unbroken value.
        chunked = " ".join(
            value[i : i + 16] for i in range(0, len(value), 16)
        ) if value else "—"
        full = QLabel(chunked)
        full.setWordWrap(True)
        full.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        full.setStyleSheet(
            f"color:{theme.TEXT_BRIGHT}; font-family:'{theme.FONT_MONO}'; "
            f"font-size:{theme.SIZE_LABEL}px; background:transparent;"
        )
        column.addWidget(full)
        return holder

    def _clear_banners(self, keep_connection: bool = False) -> None:
        while self.banner_slot.count():
            item = self.banner_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
