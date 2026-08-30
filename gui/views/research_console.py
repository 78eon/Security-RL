from __future__ import annotations

import json
from math import ceil

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.backend import ApplicationBackend, BackendPort, DashboardData
from gui.data.models import TopologyView
from gui.workers.query import run_async

COLORS = {"cyan": "#48dbe2", "lime": "#b7f04a", "amber": "#ffba52", "red": "#ff6174"}


def label(text: str, name: str = "") -> QLabel:
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    return item


def button(text: str, name: str = "Action") -> QPushButton:
    item = QPushButton(text)
    item.setObjectName(name)
    item.setCursor(Qt.CursorShape.PointingHandCursor)
    return item


def panel(layout=None, name: str = "Panel") -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    if layout:
        layout.setContentsMargins(15, 15, 15, 15)
        frame.setLayout(layout)
    return frame


def fill_table(table: QTableWidget, rows) -> None:
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            table.setItem(row_index, column, QTableWidgetItem(str(value)))


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def number(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.1f}{suffix}"


class Metric(QFrame):
    def __init__(self, title: str, value: str = "Loading…", detail: str = "") -> None:
        super().__init__()
        self.setObjectName("Metric")
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 14, 16, 14)
        box.addWidget(label(title, "MetricLabel"))
        self.value = label(value, "MetricValue")
        self.detail = label(detail, "Muted")
        box.addWidget(self.value)
        box.addWidget(self.detail)

    def update_value(self, value: str, detail: str) -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class AttackGraph(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(340)
        self.topology = TopologyView()

    def set_topology(self, topology: TopologyView) -> None:
        self.topology = topology
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#081316"))
        painter.setPen(QPen(QColor("#10282d"), 1))
        for x in range(0, self.width(), 30):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 30):
            painter.drawLine(0, y, self.width(), y)
        subnet_count = max(self.topology.num_subnets, len(self.topology.subnets))
        if not subnet_count:
            painter.setPen(QColor("#698087"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No persisted topology")
            return
        columns = min(4, subnet_count)
        rows = ceil(subnet_count / columns)
        points = []
        for index in range(subnet_count):
            col, row = index % columns, index // columns
            points.append(
                (
                    40 + col * max(150, (self.width() - 140) // columns),
                    45 + row * max(110, (self.height() - 90) // rows),
                )
            )
        adjacency = self.topology.adjacency
        for left in range(min(len(adjacency), subnet_count)):
            for right in range(left + 1, min(len(adjacency[left]), subnet_count)):
                if adjacency[left][right]:
                    painter.setPen(QPen(QColor("#294147"), 1))
                    painter.drawLine(
                        points[left][0] + 105,
                        points[left][1] + 24,
                        points[right][0],
                        points[right][1] + 24,
                    )
        jewels = self.topology.crown_jewels()
        for index, (x, y) in enumerate(points):
            targeted = any(host[0] == index for host in jewels)
            painter.setPen(QPen(QColor(COLORS["red"] if targeted else COLORS["cyan"]), 1))
            painter.setBrush(QColor("#0d1d21"))
            painter.drawRoundedRect(QRectF(x, y, 112, 50), 5, 5)
            painter.setPen(QColor("#e8f0f1"))
            painter.drawText(x + 12, y + 20, f"Subnet {index}")
            size = self.topology.subnets[index] if index < len(self.topology.subnets) else 0
            painter.setPen(QColor("#698087"))
            painter.drawText(x + 12, y + 37, f"{size} hosts" + (" · target" if targeted else ""))


class TrajectoryGraph(QWidget):
    """Replay a trace-derived attack path; no topology oracle is consulted."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.steps: list[dict] = []
        self.visible_steps = 0
        self.timer = QTimer(self)
        self.timer.setInterval(450)
        self.timer.timeout.connect(self._advance)

    def set_steps(self, steps: list[dict]) -> None:
        self.timer.stop()
        self.steps = list(steps)
        self.visible_steps = len(self.steps)
        self.update()

    def replay(self) -> None:
        if not self.steps:
            return
        self.visible_steps = 1
        self.timer.start()
        self.update()

    def _advance(self) -> None:
        self.visible_steps += 1
        if self.visible_steps >= len(self.steps):
            self.visible_steps = len(self.steps)
            self.timer.stop()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#081316"))
        visible = self.steps[: self.visible_steps]
        if not visible:
            painter.setPen(QColor("#698087"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a stored attack path"
            )
            return
        columns = min(4, len(visible))
        rows = ceil(len(visible) / columns)
        width = max(150, (self.width() - 60) // columns)
        height = max(90, (self.height() - 40) // rows)
        points = [
            (25 + (index % columns) * width, 25 + (index // columns) * height)
            for index in range(len(visible))
        ]
        for index in range(1, len(points)):
            painter.setPen(QPen(QColor(COLORS["amber"]), 2))
            painter.drawLine(
                points[index - 1][0] + 120,
                points[index - 1][1] + 28,
                points[index][0],
                points[index][1] + 28,
            )
        for index, (step, (x, y)) in enumerate(zip(visible, points, strict=True)):
            final = index == len(self.steps) - 1
            painter.setPen(QPen(QColor(COLORS["red"] if final else COLORS["cyan"]), 2))
            painter.setBrush(QColor("#0d1d21"))
            painter.drawRoundedRect(QRectF(x, y, 122, 58), 5, 5)
            painter.setPen(QColor("#e8f0f1"))
            painter.drawText(x + 8, y + 20, str(step["target"])[:18])
            painter.setPen(QColor("#698087"))
            painter.drawText(x + 8, y + 41, str(step["action"])[:18])


class Page(QWidget):
    def __init__(self, eyebrow: str, title: str, action: str = "") -> None:
        super().__init__()
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 18, 0, 12)
        head, titles = QHBoxLayout(), QVBoxLayout()
        self.eyebrow = label(eyebrow, "Eyebrow")
        titles.addWidget(self.eyebrow)
        titles.addWidget(label(title, "PageTitle"))
        head.addLayout(titles)
        head.addStretch()
        if action:
            self.action = button(action, "Primary")
            head.addWidget(self.action)
        self.root.addLayout(head)


class OverviewPage(Page):
    def __init__(self, notify) -> None:
        super().__init__("BACKEND SNAPSHOT", "Attack Path Discovery")
        self.notify = notify
        campaign = panel(QHBoxLayout())
        self.run = label("Loading backend…", "SectionTitle")
        self.run_detail = label("", "Muted")
        campaign.layout().addWidget(self.run)
        campaign.layout().addStretch()
        campaign.layout().addWidget(self.run_detail)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        campaign.layout().addWidget(self.progress)
        self.root.addWidget(campaign)
        metrics = QHBoxLayout()
        self.success, self.steps, self.cvss, self.tactics = (
            Metric("GOAL ACHIEVEMENT"),
            Metric("MEAN STEPS TO GOAL"),
            Metric("MEAN CVSS EXPLOITED"),
            Metric("LATEST EPISODE MITRE TACTICS"),
        )
        for item in (self.success, self.steps, self.cvss, self.tactics):
            metrics.addWidget(item)
        self.root.addLayout(metrics)
        center = QHBoxLayout()
        graph_box = panel(QVBoxLayout())
        graph_box.layout().addWidget(label("PERSISTED NETWORK TOPOLOGY", "SectionTitle"))
        self.graph = AttackGraph()
        graph_box.layout().addWidget(self.graph, 1)
        center.addWidget(graph_box, 2)
        path_box = panel(QVBoxLayout())
        path_box.layout().addWidget(label("RECENT STORED ATTACK PATHS", "SectionTitle"))
        self.path_table = QTableWidget(0, 2)
        self.path_table.setHorizontalHeaderLabels(["Path", "Target"])
        self.path_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        path_box.layout().addWidget(self.path_table)
        center.addWidget(path_box, 1)
        self.root.addLayout(center)

    def apply(self, data: DashboardData) -> None:
        self.eyebrow.setText(data.source_status.upper())
        self.run.setText(data.run_name)
        self.run_detail.setText(
            f"{data.reward_mode} reward · seed {data.seed} · {data.episodes} episodes"
        )
        self.progress.setVisible(data.progress is not None)
        if data.progress is not None:
            self.progress.setValue(data.progress)
        self.success.update_value(percent(data.success_rate), "Stored run summary")
        self.steps.update_value(number(data.mean_steps), "Successful episodes")
        self.cvss.update_value(number(data.mean_cvss), "Last 10% of training")
        self.tactics.update_value(
            str(data.tactic_count) if data.tactic_count else "—", "From persisted step rows"
        )
        self.graph.set_topology(data.topology)
        fill_table(self.path_table, [(p["id"], p["target"]) for p in data.paths])


class PathsPage(Page):
    def __init__(self, backend: BackendPort, notify) -> None:
        super().__init__(
            "PERSISTED STEP EVIDENCE", "Prioritise exploitable routes", "Refresh paths"
        )
        self.backend, self.notify = backend, notify
        self.action.clicked.connect(self.refresh)
        self.metrics = Metric("DISCOVERED PATHS", "—", "Waiting for PostgreSQL")
        self.root.addWidget(self.metrics)
        replay_bar = QHBoxLayout()
        replay_bar.addWidget(label("TRACE-DERIVED ATTACK GRAPH", "SectionTitle"))
        replay_bar.addStretch()
        self.replay_button = button("Replay selected")
        self.replay_button.clicked.connect(self._replay)
        replay_bar.addWidget(self.replay_button)
        self.root.addLayout(replay_bar)
        self.graph = TrajectoryGraph()
        self.root.addWidget(self.graph)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Path", "Target", "Risk", "Steps", "Detection", "Confidence"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._select_path)
        self.root.addWidget(self.table)
        self.paths: list[dict] = []

    def apply_paths(self, paths: list[dict]) -> None:
        self.paths = list(paths)
        fill_table(
            self.table,
            [
                (p["id"], p["target"], p["risk"], p["steps"], p["detection"], p["confidence"])
                for p in paths
            ],
        )
        self.metrics.update_value(
            str(len(paths)),
            "No synthetic fallback" if not paths else "Loaded from PostgreSQL steps",
        )
        self.graph.set_steps(paths[0].get("trajectory", []) if paths else [])
        if paths:
            self.table.selectRow(0)

    def _select_path(self, row: int, column: int) -> None:
        del column
        if 0 <= row < len(self.paths):
            self.graph.set_steps(self.paths[row].get("trajectory", []))

    def _replay(self) -> None:
        self.graph.replay()

    def refresh(self) -> None:
        self._task = run_async(
            self.backend.refresh_paths,
            self._refreshed,
            lambda error, _detail: self.notify(f"Path refresh failed: {error}"),
        )

    def _refreshed(self, paths) -> None:
        self.apply_paths(paths)
        self.notify(f"Loaded {len(paths)} stored attack path(s)")


class SimulationPage(Page):
    """Run and replay a backend-generated, simulation-only enterprise graph."""

    def __init__(self, backend: BackendPort, notify) -> None:
        super().__init__("OFFLINE TYPED-GRAPH BACKEND", "Enterprise Simulation")
        self.backend, self.notify = backend, notify
        controls = panel(QHBoxLayout())
        controls.layout().addWidget(label("Environment profile", "Muted"))
        self.profile = QComboBox()
        self.profile.setMinimumWidth(190)
        controls.layout().addWidget(self.profile)
        controls.layout().addWidget(label("Topology seed", "Muted"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(2001)
        controls.layout().addWidget(self.seed)
        controls.layout().addStretch()
        self.run_button = button("Run offline simulation", "Primary")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run)
        controls.layout().addWidget(self.run_button)
        self.replay_button = button("Replay causal path")
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.graph_replay)
        controls.layout().addWidget(self.replay_button)
        self.root.addWidget(controls)

        self.summary = label(
            "Loading backend profiles… · no live discovery or network traffic",
            "Muted",
        )
        self.root.addWidget(self.summary)
        self.metrics = Metric("SIMULATION STATUS", "READY", "Backend not yet executed")
        self.root.addWidget(self.metrics)
        self.graph = TrajectoryGraph()
        self.root.addWidget(self.graph)
        self.nodes = QTableWidget(0, 4)
        self.nodes.setHorizontalHeaderLabels(["Entity", "Type", "Name", "Attributes"])
        self.nodes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.root.addWidget(self.nodes)

        if hasattr(self.backend, "simulation_profiles"):
            self._profiles_task = run_async(
                self.backend.simulation_profiles,
                self._profiles_loaded,
                self._failed,
            )
        else:
            self.summary.setText("Simulation service is not available from this backend")

    def _profiles_loaded(self, profiles: list[dict]) -> None:
        self.profile.clear()
        for item in profiles:
            self.profile.addItem(str(item["label"]), str(item["id"]))
        self.run_button.setEnabled(bool(profiles))
        self.summary.setText(
            f"{len(profiles)} configured simulation profiles · no live network discovery"
        )

    def run(self) -> None:
        profile = self.profile.currentData()
        if not profile:
            self.notify("No simulation profile is available")
            return
        seed = self.seed.value()
        self.run_button.setEnabled(False)
        self.metrics.update_value("RUNNING", f"{profile} · seed {seed}")
        self._simulation_task = run_async(
            lambda: self.backend.run_simulation(str(profile), seed),
            self._completed,
            self._failed,
        )

    def _completed(self, result) -> None:
        self.run_button.setEnabled(True)
        self.replay_button.setEnabled(bool(result.trajectory))
        outcome = "GOAL REACHED" if result.goal_reached else "STEP LIMIT"
        self.metrics.update_value(
            outcome,
            f"{result.episode_steps} raw steps · {len(result.trajectory)} causal events",
        )
        self.summary.setText(
            f"{result.topology_name} · {len(result.nodes)} entities · "
            f"{len(result.edges)} relationships · "
            f"coverage {100 * result.discovery_coverage:.1f}% · "
            f"hash {result.topology_hash[:12]}… · {result.agent}"
        )
        self.graph.set_steps(result.trajectory)
        fill_table(
            self.nodes,
            [
                (
                    item["id"],
                    item["type"],
                    item["name"],
                    json.dumps(item["attributes"], sort_keys=True),
                )
                for item in result.nodes
            ],
        )
        self.notify(f"Completed {result.profile} simulation for seed {result.topology_seed}")

    def graph_replay(self) -> None:
        self.graph.replay()

    def _failed(self, error: str, detail: str) -> None:
        del detail
        self.run_button.setEnabled(self.profile.count() > 0)
        self.replay_button.setEnabled(False)
        self.metrics.update_value("FAILED", error)
        self.notify(f"Simulation failed: {error}")


class CampaignsPage(Page):
    def __init__(self) -> None:
        super().__init__("STORED RUN ARTEFACTS", "Training campaign history")
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Reward", "Status", "Episodes", "Progress", "Seed", "Success"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.root.addWidget(self.table)

    def apply(self, data: DashboardData) -> None:
        self.eyebrow.setText(f"{len(data.campaigns)} STORED RUNS")
        fill_table(
            self.table,
            [
                (
                    c.name,
                    c.reward_mode,
                    c.status,
                    c.episodes,
                    "—" if c.progress is None else f"{c.progress}%",
                    c.seed,
                    percent(c.success_rate),
                )
                for c in data.campaigns
            ],
        )


class AgentLabPage(Page):
    def __init__(self) -> None:
        super().__init__("READ-ONLY RUN CONFIGURATION", "Agent Lab")
        body = QHBoxLayout()
        details = panel(QFormLayout())
        self.fields = {
            name: label("Loading…", "Muted")
            for name in (
                "Algorithm",
                "Environment",
                "Learning rate",
                "Discount factor",
                "Batch size",
            )
        }
        for name, item in self.fields.items():
            details.layout().addRow(name, item)
        body.addWidget(details, 2)
        self.spec = QPlainTextEdit()
        self.spec.setReadOnly(True)
        body.addWidget(self.spec, 3)
        self.root.addLayout(body)
        self.root.addWidget(
            label(
                "Configuration is loaded from the run snapshot. "
                "Editing requires a backend configuration service.",
                "Muted",
            )
        )

    def apply(self, data: DashboardData) -> None:
        agent = data.agent
        values = (
            agent.algorithm,
            agent.environment,
            agent.learning_rate,
            agent.gamma,
            agent.batch_size,
        )
        for item, value in zip(self.fields.values(), values, strict=True):
            item.setText("—" if value is None else str(value))
        self.spec.setPlainText(agent.specification)


class ExperimentsPage(Page):
    def __init__(self) -> None:
        super().__init__("ANALYSIS ARTEFACT", "Controlled experiment matrix")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Run",
                "Reward",
                "Seed",
                "Episodes",
                "Success",
                "Native return",
                "Steps to goal",
                "Converged",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.root.addWidget(self.table)

    def apply(self, data: DashboardData) -> None:
        self.eyebrow.setText(f"{len(data.experiments)} ANALYSED RUNS")
        fill_table(
            self.table,
            [
                (
                    r.get("run_name", "—"),
                    r.get("reward_mode", "—"),
                    r.get("seed", "—"),
                    r.get("episodes", "—"),
                    percent(r.get("success_rate")),
                    number(r.get("native_return")),
                    number(r.get("steps_to_goal")),
                    str(r.get("converged", "—")),
                )
                for r in data.experiments
            ],
        )


class DatasetsPage(Page):
    def __init__(self) -> None:
        super().__init__("BACKEND DATA SOURCES", "Research data catalogue")
        self.grid = QGridLayout()
        self.root.addLayout(self.grid)

    def apply(self, data: DashboardData) -> None:
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for index, source in enumerate(data.datasets):
            box = panel(QVBoxLayout())
            box.layout().addWidget(label(source.integrity, "Eyebrow"))
            box.layout().addWidget(label(source.title, "SectionTitle"))
            box.layout().addWidget(label(source.count, "MetricValue"))
            box.layout().addWidget(label(source.detail, "Muted"))
            self.grid.addWidget(box, index // 2, index % 2)


class LogsPage(Page):
    def __init__(self) -> None:
        super().__init__("PERSISTED EVENTS", "Event logs")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter stored events…")
        self.root.addWidget(self.search)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Sequence", "Level", "Source", "Message", "Context"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.root.addWidget(self.table)
        self.search.textChanged.connect(self.filter)

    def apply(self, data: DashboardData) -> None:
        fill_table(
            self.table, [(e.sequence, e.level, e.source, e.message, e.context) for e in data.events]
        )
        self.eyebrow.setText(f"{len(data.events)} PERSISTED EVENTS")

    def filter(self, text: str) -> None:
        for row in range(self.table.rowCount()):
            values = " ".join(
                (self.table.item(row, col).text() if self.table.item(row, col) else "")
                for col in range(self.table.columnCount())
            )
            self.table.setRowHidden(row, text.lower() not in values.lower())


class SettingsPage(Page):
    def __init__(self) -> None:
        super().__init__("READ-ONLY BACKEND STATUS", "Platform configuration")
        box = panel(QFormLayout())
        self.database = label("Loading…", "Muted")
        self.mode = label("Loading…", "Muted")
        self.boundary = label("Offline typed graph / NASim only", "Muted")
        box.layout().addRow("PostgreSQL", self.database)
        box.layout().addRow("Data mode", self.mode)
        box.layout().addRow("Execution boundary", self.boundary)
        self.root.addWidget(box)
        self.root.addWidget(
            label(
                "Secrets are read from container environment variables and are never displayed.",
                "Muted",
            )
        )

    def apply(self, data: DashboardData) -> None:
        self.database.setText(data.database_label)
        self.mode.setText(data.source_status)


class MainWindow(QMainWindow):
    PAGE_DATA = [
        ("Overview", "Attack Path Discovery"),
        ("Simulation", "Enterprise Simulation"),
        ("Attack Paths", "Attack Paths"),
        ("Live Campaigns", "Campaign History"),
        ("Agent Lab", "Agent Lab"),
        ("Experiments", "Experiments"),
        ("Datasets", "Datasets"),
        ("Event Logs", "Event Logs"),
        ("Configuration", "Configuration"),
    ]

    def __init__(self, backend: BackendPort | None = None) -> None:
        super().__init__()
        self.backend = backend or ApplicationBackend()
        self.setWindowTitle("RLRedTeam Research Console")
        self.resize(1440, 920)
        self.setMinimumSize(1120, 720)
        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QFrame(objectName="Sidebar")
        sidebar.setFixedWidth(230)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(15, 24, 15, 18)
        nav.addWidget(label("⬡  RLREDTEAM", "Brand"))
        nav.addWidget(label("RESEARCH CONSOLE", "BrandSub"))
        nav.addSpacing(18)
        self.nav_buttons = []
        for index, (name, _) in enumerate(self.PAGE_DATA):
            item = button(name, "Nav")
            item.setCheckable(True)
            item.clicked.connect(lambda checked=False, i=index: self.select_page(i))
            self.nav_buttons.append(item)
            nav.addWidget(item)
        nav.addStretch()
        safety = panel(QVBoxLayout())
        safety.layout().addWidget(label("✓  SIMULATION BOUNDARY", "Eyebrow"))
        safety.layout().addWidget(label("Offline graph / NASim only", "Muted"))
        nav.addWidget(safety)
        shell.addWidget(sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 0, 24, 0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 18, 0, 14)
        self.header = label("Attack Path Discovery", "PageTitle")
        top.addWidget(self.header)
        top.addStretch()
        export = button("Open analysis report", "Primary")
        export.clicked.connect(self.export_report)
        top.addWidget(export)
        content_layout.addLayout(top)
        self.stack = QStackedWidget()
        self.pages = [
            OverviewPage(self.notify),
            SimulationPage(self.backend, self.notify),
            PathsPage(self.backend, self.notify),
            CampaignsPage(),
            AgentLabPage(),
            ExperimentsPage(),
            DatasetsPage(),
            LogsPage(),
            SettingsPage(),
        ]
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        content_layout.addWidget(self.stack)
        shell.addWidget(content, 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Loading backend snapshot…")
        self.select_page(0)
        if hasattr(self.backend, "load_dashboard"):
            self._load_task = run_async(
                self.backend.load_dashboard,
                self.apply_dashboard,
                lambda error, _detail: self.statusBar().showMessage(
                    f"Backend unavailable: {error}"
                ),
            )

    def apply_dashboard(self, data: DashboardData) -> None:
        for page in self.pages:
            apply = getattr(page, "apply", None)
            if apply:
                apply(data)
        self.pages[2].apply_paths(data.paths)
        self.statusBar().showMessage(data.source_status)

    def select_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.header.setText(self.PAGE_DATA[index][1])
        for item_index, item in enumerate(self.nav_buttons):
            item.setChecked(item_index == index)

    def notify(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)

    def export_report(self) -> None:
        self._export_task = run_async(
            self.backend.export_report,
            lambda path: self.notify(f"Analysis report: {path}"),
            lambda error, _detail: self.notify(f"Report unavailable: {error}"),
        )
