"""Native PySide6 research console backed by current repository evidence."""

from __future__ import annotations

from math import ceil

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.backend import ApplicationBackend, BackendPort, DashboardData
from gui.data.models import StudySummary
from gui.widgets.enterprise_graph import EnterpriseGraph
from gui.widgets.mitre_navigator import MitreNavigator
from gui.workers.query import run_async


def label(text: str, name: str = "", *, wrap: bool = False) -> QLabel:
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


def button(text: str, name: str = "Action") -> QPushButton:
    item = QPushButton(text)
    item.setObjectName(name)
    item.setCursor(Qt.CursorShape.PointingHandCursor)
    return item


def panel(layout=None, name: str = "Panel") -> QFrame:
    frame = QFrame()
    frame.setObjectName(name)
    if layout is not None:
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        frame.setLayout(layout)
    return frame


def configure_table(table: QTableWidget, *, stretch_column: int = 0) -> None:
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(
        stretch_column, QHeaderView.ResizeMode.Stretch
    )


def fill_table(table: QTableWidget, rows) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            text = str(value)
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(row_index, column, item)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def number(value: float | None, decimals: int = 1) -> str:
    return "—" if value is None else theme.fmt_num(value, decimals)


def metric_name(value: str) -> str:
    return value.replace("_", " ").title()


class Metric(QFrame):
    def __init__(self, title: str, value: str = "—", detail: str = "") -> None:
        super().__init__()
        self.setObjectName("Metric")
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(3)
        box.addWidget(label(title, "MetricLabel"))
        self.value = label(value, "MetricValue")
        self.detail = label(detail, "Muted", wrap=True)
        box.addWidget(self.value)
        box.addWidget(self.detail)

    def update_value(self, value: str, detail: str) -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class StateBanner(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("StateBanner")
        box = QHBoxLayout(self)
        box.setContentsMargins(14, 11, 14, 11)
        self.state = label("LOADING", "StateLabel")
        self.message = label("Loading backend snapshot…", "Muted", wrap=True)
        box.addWidget(self.state)
        box.addWidget(self.message, 1)

    def update_state(self, state: str, message: str, kind: str = "ok") -> None:
        self.state.setText(state.upper())
        self.state.setProperty("kind", kind)
        self.state.style().unpolish(self.state)
        self.state.style().polish(self.state)
        self.message.setText(message)


class Page(QWidget):
    def __init__(self, eyebrow: str, title: str, subtitle: str = "") -> None:
        super().__init__()
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 18, 0, 18)
        self.root.setSpacing(12)
        self.eyebrow = label(eyebrow, "Eyebrow")
        self.title = label(title, "ViewTitle")
        self.subtitle = label(subtitle, "Muted", wrap=True)
        self.root.addWidget(self.eyebrow)
        self.root.addWidget(self.title)
        if subtitle:
            self.root.addWidget(self.subtitle)


class OverviewPage(Page):
    def __init__(self) -> None:
        super().__init__(
            "RESEARCH STATUS",
            "Mission control",
            "One consistent view of the latest canonical study and live storage backend.",
        )
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        self.study_panel = panel(QVBoxLayout(), "HeroPanel")
        self.study_phase = label("NO CANONICAL STUDY", "Eyebrow")
        self.study_name = label("Generated evidence has not been found", "HeroTitle")
        self.study_outcome = label(
            "Run a canonical study or mount the local results directory.", "Muted", wrap=True
        )
        self.study_panel.layout().addWidget(self.study_phase)
        self.study_panel.layout().addWidget(self.study_name)
        self.study_panel.layout().addWidget(self.study_outcome)
        self.root.addWidget(self.study_panel)

        metric_row = QHBoxLayout()
        self.studies = Metric("COMPLETED PHASES", "—", "Canonical local evidence")
        self.episodes = Metric("EVALUATION EPISODES", "—", "Latest study")
        self.signals = Metric("PRIMARY SIGNALS", "—", "Multiplicity controlled")
        self.runs = Metric("STORED RUNS", "—", "PostgreSQL + artifact-only")
        for item in (self.studies, self.episodes, self.signals, self.runs):
            metric_row.addWidget(item)
        self.root.addLayout(metric_row)

        metrics_panel = panel(QVBoxLayout())
        metrics_panel.layout().addWidget(label("LATEST PRIMARY COMPARISONS", "SectionTitle"))
        self.metric_table = QTableWidget(0, 6)
        self.metric_table.setHorizontalHeaderLabels(
            ["Metric", "Reference", "Candidate", "Difference", "Adjusted p", "Verdict"]
        )
        configure_table(self.metric_table, stretch_column=0)
        self.metric_table.setMinimumHeight(220)
        metrics_panel.layout().addWidget(self.metric_table)
        self.root.addWidget(metrics_panel)
        self.root.addStretch(1)

    def apply(self, data: DashboardData) -> None:
        kind = "warn" if data.source_status.startswith("Artefact mode") else "ok"
        self.banner.update_state(
            "DEGRADED" if kind == "warn" else "CONNECTED", data.source_status, kind
        )
        complete = [study for study in data.studies if study.complete]
        latest = data.studies[0] if data.studies else None
        self.studies.update_value(str(len(complete)), "Phase 7–13 packages")
        self.runs.update_value(str(len(data.campaigns)), "Deduplicated records")
        if latest is None:
            self.episodes.update_value("—", "No current study")
            self.signals.update_value("—", "No primary statistics")
            fill_table(self.metric_table, [])
            return
        state = "COMPLETE" if latest.complete else "PARTIAL"
        self.study_phase.setText(f"PHASE {latest.phase} · {state}")
        self.study_name.setText(latest.title)
        self.study_outcome.setText(
            f"{latest.outcome} · {latest.training_seeds} matched training seeds · "
            f"commit {latest.code_commit[:8] or 'unavailable'}"
        )
        self.episodes.update_value(f"{latest.evaluation_episodes:,}", "Frozen-policy outcomes")
        signals = sum(metric.significant for metric in latest.primary_metrics)
        self.signals.update_value(
            str(signals), f"of {len(latest.primary_metrics)} primary metrics"
        )
        fill_table(
            self.metric_table,
            [
                (
                    metric_name(metric.name),
                    number(metric.arm_a_mean, 3),
                    number(metric.arm_b_mean, 3),
                    number(metric.difference, 3),
                    number(metric.p_adjusted, 4),
                    "SIGNIFICANT" if metric.significant else "NO SIGNAL",
                )
                for metric in latest.primary_metrics
            ],
        )


class SimulationPage(Page):
    """Reactive MITRE workspace for one offline enterprise episode."""

    def __init__(self, backend: BackendPort, notify) -> None:
        super().__init__(
            "EVENT-DRIVEN ATT&CK + ATLAS",
            "MITRE Navigator workspace",
            "Replay policy-visible progress through ATT&CK and ATLAS semantics while "
            "keeping Ground Truth isolated from AgentKnowledge.",
        )
        self.backend, self.notify = backend, notify
        self.result = None
        self.events: list[dict] = []
        self.replay_timer = QTimer(self)
        self.replay_timer.setInterval(480)
        self.replay_timer.timeout.connect(self._advance_replay)

        controls = panel(QHBoxLayout())
        controls.layout().addWidget(label("Environment", "FieldLabel"))
        self.profile = QComboBox()
        self.profile.setMinimumWidth(190)
        controls.layout().addWidget(self.profile)
        controls.layout().addWidget(label("Topology seed", "FieldLabel"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(2001)
        controls.layout().addWidget(self.seed)
        controls.layout().addStretch()
        self.run_button = button("Run offline episode", "Primary")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run)
        controls.layout().addWidget(self.run_button)
        self.replay_button = button("Replay episode")
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.graph_replay)
        controls.layout().addWidget(self.replay_button)
        self.root.addWidget(controls)
        self.banner = StateBanner()
        self.banner.update_state("READY", "Loading supported profiles…")
        self.root.addWidget(self.banner)

        metric_row = QHBoxLayout()
        self.outcome = Metric("OUTCOME", "READY", "Backend not yet executed")
        self.entities = Metric("GROUND TRUTH", "—", "Analyst-only graph entities")
        self.knowledge_count = Metric("AGENT KNOWLEDGE", "—", "Discovered entities")
        self.coverage = Metric("DISCOVERY COVERAGE", "—", "AgentKnowledge")
        for item in (self.outcome, self.entities, self.knowledge_count, self.coverage):
            metric_row.addWidget(item)
        self.root.addLayout(metric_row)

        replay_panel = panel(QVBoxLayout())
        replay_head = QHBoxLayout()
        replay_head.addWidget(label("EPISODE REPLAY", "SectionTitle"))
        self.step_label = label("Step 0 / 0", "MonoDetail")
        replay_head.addStretch()
        replay_head.addWidget(self.step_label)
        replay_panel.layout().addLayout(replay_head)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self._render_step)
        replay_panel.layout().addWidget(self.timeline)
        context = QHBoxLayout()
        self.rl_context = label("RL ACTION  —", "ActionLayer")
        self.simulator_context = label("SIMULATOR  —", "ActionLayer")
        self.mitre_context = label("MITRE  —", "ActionLayer", wrap=True)
        context.addWidget(self.rl_context, 1)
        context.addWidget(self.simulator_context, 2)
        context.addWidget(self.mitre_context, 3)
        replay_panel.layout().addLayout(context)
        self.root.addWidget(replay_panel)

        matrix_panel = panel(QVBoxLayout())
        matrix_head = QHBoxLayout()
        matrix_head.addWidget(label("MITRE TACTIC VIEW", "SectionTitle"))
        matrix_head.addStretch()
        matrix_head.addWidget(
            label(
                "Event mappings only · hidden topology cannot activate cells",
                "Muted",
            )
        )
        matrix_panel.layout().addLayout(matrix_head)
        self.navigator = MitreNavigator()
        self.navigator.setMinimumHeight(330)
        matrix_panel.layout().addWidget(self.navigator)
        self.root.addWidget(matrix_panel)

        views_panel = panel(QVBoxLayout())
        views_panel.layout().addWidget(label("ENVIRONMENT VIEWS", "SectionTitle"))
        self.views = QTabWidget()
        self.views.setObjectName("EnvironmentViews")
        self.knowledge_graph = EnterpriseGraph()
        self.graph = self.knowledge_graph
        self.path_graph = TrajectoryGraph()
        self.truth_graph = EnterpriseGraph()
        self.views.addTab(self.knowledge_graph, "AgentKnowledge")
        self.views.addTab(self.path_graph, "Attack path")
        self.views.addTab(self.truth_graph, "Ground Truth · analyst only")
        views_panel.layout().addWidget(self.views)
        self.root.addWidget(views_panel)

        if hasattr(self.backend, "simulation_profiles"):
            self._profiles_task = run_async(
                self.backend.simulation_profiles, self._profiles_loaded, self._failed
            )
        else:
            self.banner.update_state(
                "UNAVAILABLE", "This backend does not provide simulation profiles.", "warn"
            )

    def _profiles_loaded(self, profiles: list[dict]) -> None:
        self.profile.clear()
        for item in profiles:
            self.profile.addItem(str(item["label"]), str(item["id"]))
        self.run_button.setEnabled(bool(profiles))
        self.banner.update_state(
            "READY", f"{len(profiles)} backend-defined profiles · execution remains offline"
        )

    def run(self) -> None:
        profile = self.profile.currentData()
        if not profile:
            self.notify("No simulation profile is available")
            return
        seed = self.seed.value()
        self.run_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.outcome.update_value("RUNNING", f"{profile} · seed {seed}")
        self.banner.update_state("RUNNING", "Executing one backend simulation on a Qt worker…")
        self._simulation_task = run_async(
            lambda: self.backend.run_simulation(str(profile), seed), self._completed, self._failed
        )

    def _completed(self, result) -> None:
        self.result = result
        self.events = list(result.events)
        self.run_button.setEnabled(True)
        self.replay_button.setEnabled(bool(self.events))
        outcome = "GOAL REACHED" if result.goal_reached else "STEP LIMIT"
        self.outcome.update_value(
            outcome,
            f"{result.episode_steps} events · {len(result.trajectory)} causal events",
        )
        self.entities.update_value(
            str(len(result.nodes)), f"{len(result.edges)} typed relationships"
        )
        self.coverage.update_value(
            percent(result.discovery_coverage), "Policy-visible discovered entities"
        )
        self.banner.update_state(
            "COMPLETE", f"hash {result.topology_hash} · {result.agent}"
        )
        self.timeline.setRange(0, len(self.events))
        self.timeline.setValue(len(self.events))
        self._render_step(len(self.events))
        self.notify(f"Completed {result.profile} simulation for seed {result.topology_seed}")

    def graph_replay(self) -> None:
        if not self.events:
            return
        self.replay_timer.stop()
        self.timeline.setValue(0)
        self.replay_timer.start()

    def _advance_replay(self) -> None:
        next_step = self.timeline.value() + 1
        self.timeline.setValue(next_step)
        if next_step >= len(self.events):
            self.replay_timer.stop()

    def _render_step(self, visible_count: int) -> None:
        self.navigator.set_events(self.events, visible_count)
        self.step_label.setText(f"Step {visible_count} / {len(self.events)}")
        if self.result is None:
            return
        visible = self.events[:visible_count]
        current = visible[-1] if visible else None
        if current is None:
            self.rl_context.setText("RL ACTION  —")
            self.simulator_context.setText("SIMULATOR  —")
            self.mitre_context.setText("MITRE  —")
            self.knowledge_count.update_value("0", "No replayed discoveries")
            self.knowledge_graph.clear_graph("Replay has not started")
            self.path_graph.set_steps([])
            self.truth_graph.set_graph(self.result.nodes, self.result.edges, [])
            return

        action_index = current.get("rl_action_index")
        self.rl_context.setText(
            f"RL ACTION  #{action_index}" if action_index is not None else "RL ACTION  scripted"
        )
        self.simulator_context.setText(
            f"SIMULATOR  {current.get('simulator_action') or current.get('action')}"
        )
        mappings = current.get("framework_mappings") or []
        mapped = " · ".join(
            f"{item.get('technique_id')} {item.get('technique_name')}"
            for item in mappings
            if isinstance(item, dict)
        )
        self.mitre_context.setText(f"MITRE  {mapped or 'UNMAPPED · unsupported semantic'}")

        knowledge = current.get("knowledge") or {}
        known_ids = set(knowledge.get("nodes") or [])
        known_nodes = [node for node in self.result.nodes if node["id"] in known_ids]
        known_edges = list(knowledge.get("edges") or [])
        self.knowledge_count.update_value(
            str(len(known_nodes)),
            f"{len(knowledge.get('access') or {})} accessed · "
            f"{len(knowledge.get('credentials') or [])} credentials",
        )
        self.knowledge_graph.set_graph(known_nodes, known_edges, visible)
        current_step = int(current.get("step", visible_count - 1))
        visible_path = [
            step
            for step in self.result.trajectory
            if int(step.get("step", 0)) <= current_step
        ]
        self.path_graph.set_steps(visible_path)
        self.truth_graph.set_graph(self.result.nodes, self.result.edges, visible)

    def _failed(self, error: str, detail: str) -> None:
        self.replay_timer.stop()
        self.run_button.setEnabled(self.profile.count() > 0)
        self.replay_button.setEnabled(False)
        self.outcome.update_value("FAILED", error)
        self.banner.update_state("FAILED", detail or error, "warn")
        self.notify(f"Simulation failed: {error}")


class TrajectoryGraph(QWidget):
    """Compact native replay of a stored trace-derived path."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.steps: list[dict] = []
        self.visible_steps = 0
        self.timer = QTimer(self)
        self.timer.setInterval(420)
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

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.PANEL))
        visible = self.steps[: self.visible_steps]
        if not visible:
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a stored attack path"
            )
            return
        columns = min(4, len(visible))
        rows = ceil(len(visible) / columns)
        cell_width = max(160, (self.width() - 70) // columns)
        cell_height = max(86, (self.height() - 40) // rows)
        points = []
        for index in range(len(visible)):
            row, offset = divmod(index, columns)
            column = offset if row % 2 == 0 else columns - 1 - offset
            points.append((25 + column * cell_width, 22 + row * cell_height))
        for index in range(1, len(points)):
            painter.setPen(QPen(QColor(theme.WARN), 2))
            painter.drawLine(
                points[index - 1][0] + 132,
                points[index - 1][1] + 28,
                points[index][0],
                points[index][1] + 28,
            )
        for index, (step, (x, y)) in enumerate(zip(visible, points, strict=True)):
            final = index == len(self.steps) - 1
            painter.setPen(QPen(QColor(theme.ERROR if final else theme.ARM_1), 2))
            painter.setBrush(QColor(theme.SURFACE))
            painter.drawRoundedRect(QRectF(x, y, 134, 58), 4, 4)
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(x + 8, y + 20, str(step.get("target", "environment"))[:20])
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            painter.drawText(x + 8, y + 41, str(step.get("action", "action"))[:20])


class PathsPage(Page):
    def __init__(self, backend: BackendPort, notify) -> None:
        super().__init__(
            "POSTGRESQL STEP EVIDENCE",
            "Attack paths",
            "Routes are reconstructed from successful state-changing events, "
            "never hidden topology.",
        )
        self.backend, self.notify = backend, notify
        toolbar = QHBoxLayout()
        self.count = Metric("DISCOVERED PATHS", "—", "Waiting for PostgreSQL")
        toolbar.addWidget(self.count)
        toolbar.addStretch()
        self.refresh_button = button("Refresh paths", "Primary")
        self.refresh_button.clicked.connect(self.refresh)
        self.replay_button = button("Replay selected")
        self.replay_button.clicked.connect(self._replay)
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.replay_button)
        self.root.addLayout(toolbar)

        graph_panel = panel(QVBoxLayout())
        graph_panel.layout().addWidget(label("TRACE-DERIVED CAUSAL ROUTE", "SectionTitle"))
        self.graph = TrajectoryGraph()
        graph_panel.layout().addWidget(self.graph)
        self.root.addWidget(graph_panel)

        matrix_panel = panel(QVBoxLayout())
        matrix_panel.layout().addWidget(
            label("TRACE-DERIVED MITRE TACTIC VIEW", "SectionTitle")
        )
        matrix_panel.layout().addWidget(
            label(
                "Only semantics stored with episode events are highlighted; topology is not read.",
                "Muted",
                wrap=True,
            )
        )
        self.navigator = MitreNavigator()
        self.navigator.setMinimumHeight(300)
        matrix_panel.layout().addWidget(self.navigator)
        self.root.addWidget(matrix_panel)

        table_panel = panel(QVBoxLayout())
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Path", "Target", "Risk", "Raw steps", "Detection", "Confidence"]
        )
        configure_table(self.table, stretch_column=1)
        self.table.cellClicked.connect(self._select_path)
        table_panel.layout().addWidget(self.table)
        self.root.addWidget(table_panel)
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
        self.count.update_value(
            str(len(paths)), "Loaded from PostgreSQL" if paths else "No replayable step rows"
        )
        self.graph.set_steps(paths[0].get("trajectory", []) if paths else [])
        self.navigator.set_events(paths[0].get("trajectory", []) if paths else [])
        if paths:
            self.table.selectRow(0)

    def _select_path(self, row: int, column: int) -> None:
        del column
        if 0 <= row < len(self.paths):
            trajectory = self.paths[row].get("trajectory", [])
            self.graph.set_steps(trajectory)
            self.navigator.set_events(trajectory)

    def _replay(self) -> None:
        self.graph.replay()

    def refresh(self) -> None:
        self.refresh_button.setEnabled(False)
        self._task = run_async(self.backend.refresh_paths, self._refreshed, self._failed)

    def _refreshed(self, paths) -> None:
        self.refresh_button.setEnabled(True)
        self.apply_paths(paths)
        self.notify(f"Loaded {len(paths)} stored attack path(s)")

    def _failed(self, error: str, detail: str) -> None:
        self.refresh_button.setEnabled(True)
        self.notify(f"Path refresh failed: {error} · {detail}")


class AttackPathReportPage(Page):
    """Render stored deterministic facts without consulting hidden topology."""

    def __init__(self) -> None:
        super().__init__(
            "PHASE 14 · STORED DERIVED FACTS",
            "Explainable attack-path analysis",
            "MITRE mappings, observed targets and criticality come from persisted report facts. "
            "Normal mode never reads hidden TrueTopology.",
        )
        selector = panel(QHBoxLayout())
        selector.layout().addWidget(label("Report", "FieldLabel"))
        self.selector = QComboBox()
        self.selector.setMinimumWidth(430)
        self.selector.currentIndexChanged.connect(self._selected)
        selector.layout().addWidget(self.selector)
        selector.layout().addStretch()
        self.mode = label("NO STORED REPORT", "StateLabel")
        selector.layout().addWidget(self.mode)
        self.root.addWidget(selector)

        metric_row = QHBoxLayout()
        self.trajectories = Metric("TRAJECTORIES")
        self.techniques = Metric("TECHNIQUES")
        self.crown_reach = Metric("CROWN-JEWEL REACH")
        self.failed_actions = Metric("FAILED ACTIONS")
        for item in (
            self.trajectories,
            self.techniques,
            self.crown_reach,
            self.failed_actions,
        ):
            metric_row.addWidget(item)
        self.root.addLayout(metric_row)

        timeline_panel = panel(QVBoxLayout())
        timeline_panel.layout().addWidget(
            label("MITRE TACTIC / TECHNIQUE TIMELINE", "SectionTitle")
        )
        self.navigator = MitreNavigator()
        self.navigator.setMinimumHeight(300)
        timeline_panel.layout().addWidget(self.navigator)
        self.root.addWidget(timeline_panel)

        graph_panel = panel(QVBoxLayout())
        graph_panel.layout().addWidget(label("OBSERVED AGENTKNOWLEDGE TARGETS", "SectionTitle"))
        graph_panel.layout().addWidget(
            label(
                "Nodes are recorded targets only. Missing causal edges are left absent rather "
                "than inferred from simulator ground truth.",
                "Muted",
                wrap=True,
            )
        )
        self.graph = EnterpriseGraph()
        graph_panel.layout().addWidget(self.graph)
        self.root.addWidget(graph_panel)

        critical_panel = panel(QVBoxLayout())
        critical_panel.layout().addWidget(label("OBSERVED PATH-CRITICAL CVES", "SectionTitle"))
        self.criticality = QTableWidget(0, 6)
        self.criticality.setHorizontalHeaderLabels(
            ["CVE", "Observed criticality", "Broken paths", "Containing paths", "CVSS", "MITRE"]
        )
        configure_table(self.criticality, stretch_column=0)
        critical_panel.layout().addWidget(self.criticality)
        self.root.addWidget(critical_panel)

        detail_panel = panel(QVBoxLayout())
        detail_head = QHBoxLayout()
        detail_head.addWidget(label("SELECTED TRAJECTORY FACTS", "SectionTitle"))
        detail_head.addStretch()
        self.episode = QComboBox()
        self.episode.currentIndexChanged.connect(self._episode_selected)
        detail_head.addWidget(self.episode)
        detail_panel.layout().addLayout(detail_head)
        self.details = QTableWidget(0, 10)
        self.details.setHorizontalHeaderLabels(
            [
                "Step",
                "Simulator action",
                "Target",
                "Result",
                "Access",
                "CVE",
                "CVSS",
                "Technique",
                "Reward",
                "Knowledge Δ",
            ]
        )
        configure_table(self.details, stretch_column=1)
        detail_panel.layout().addWidget(self.details)
        self.root.addWidget(detail_panel)
        self.provenance = label("", "MonoDetail", wrap=True)
        self.root.addWidget(self.provenance)
        self.reports: list[dict] = []
        self.current_report: dict = {}

    def apply(self, data: DashboardData) -> None:
        self.reports = list(data.attack_path_reports)
        self.selector.blockSignals(True)
        self.selector.clear()
        for report in self.reports:
            provenance = report.get("provenance") or {}
            self.selector.addItem(
                f"{provenance.get('experiment_id', 'unknown')} · "
                f"{str(report.get('report_id', ''))[:12]}",
                report.get("report_id"),
            )
        self.selector.blockSignals(False)
        self._selected(0)

    def _selected(self, index: int) -> None:
        if not 0 <= index < len(self.reports):
            self.current_report = {}
            self.mode.setText("NO STORED REPORT")
            self.navigator.set_events([])
            self.graph.clear_graph("No persisted Phase 14 report")
            fill_table(self.criticality, [])
            fill_table(self.details, [])
            return
        report = self.reports[index]
        self.current_report = report
        mode = str(report.get("report_mode", "unknown"))
        self.mode.setText(mode.upper())
        analysis = report.get("cross_episode_analysis") or {}
        techniques = analysis.get("technique_coverage") or []
        self.trajectories.update_value(
            str(analysis.get("trajectory_count", 0)), "Recorded episodes"
        )
        self.techniques.update_value(str(len(techniques)), "Catalogue-mapped")
        self.crown_reach.update_value(
            percent(analysis.get("crown_jewel_reach_rate")), "Observed outcomes"
        )
        self.failed_actions.update_value(
            (
                str(analysis["failed_action_count"])
                if analysis.get("failed_action_count") is not None
                else "—"
            ),
            str(analysis.get("failed_action_observation", "When recorded")).replace("_", " "),
        )
        facts = list(report.get("facts") or [])
        events = [_fact_event(fact) for fact in facts]
        self.navigator.set_events(events)
        observed_graph = report.get("observed_graph") or {}
        self.graph.set_graph(
            list(observed_graph.get("nodes") or []),
            list(observed_graph.get("edges") or []),
            events,
        )
        criticality = list(report.get("observed_path_criticality") or [])
        fill_table(
            self.criticality,
            [
                (
                    item.get("cve_id", "—"),
                    f"{float(item.get('observed_path_criticality_percentage', 0)):.1f}%",
                    item.get("observed_successful_paths_broken", 0),
                    item.get("successful_trajectories_containing", 0),
                    item.get("cvss_score", "—"),
                    item.get("technique_id", "—"),
                )
                for item in criticality
            ],
        )
        episode_ids = list(dict.fromkeys(str(fact.get("episode_id")) for fact in facts))
        self.episode.blockSignals(True)
        self.episode.clear()
        self.episode.addItems(episode_ids)
        self.episode.blockSignals(False)
        self._episode_selected(0)
        provenance = report.get("provenance") or {}
        self.provenance.setText(
            f"REPORT  {report.get('report_id', '—')}\n"
            f"SOURCE  {provenance.get('source_trajectory_sha256', '—')}\n"
            f"FACTS   {provenance.get('facts_payload_sha256', '—')}\n"
            f"MITRE   {provenance.get('mitre_catalogue_version', '—')} · "
            f"{provenance.get('mitre_catalogue_sha256', '—')}"
        )

    def _episode_selected(self, index: int) -> None:
        if index < 0:
            fill_table(self.details, [])
            return
        episode_id = self.episode.itemText(index)
        facts = [
            fact
            for fact in self.current_report.get("facts", [])
            if str(fact.get("episode_id")) == episode_id
        ]
        fill_table(
            self.details,
            [
                (
                    fact.get("trajectory_step", "—"),
                    fact.get("simulator_action", "—"),
                    fact.get("target", "—"),
                    "SUCCESS" if fact.get("success") else "FAILED",
                    fact.get("access_gained", "—"),
                    fact.get("cve_id", "—"),
                    fact.get("cvss_score", "—"),
                    fact.get("technique_id", "UNMAPPED"),
                    "—" if fact.get("reward") is None else fact["reward"],
                    "—"
                    if fact.get("knowledge_delta") is None
                    else fact["knowledge_delta"],
                )
                for fact in facts
            ],
        )


def _fact_event(fact: dict) -> dict:
    mapping = []
    if fact.get("framework") and fact.get("technique_id"):
        mapping = [
            {
                "framework": fact["framework"],
                "tactic_id": fact.get("tactic_id"),
                "tactic_name": fact.get("tactic_name"),
                "technique_id": fact["technique_id"],
                "technique_name": fact.get("technique_name"),
            }
        ]
    return {
        "step": fact.get("trajectory_step"),
        "action": fact.get("action_kind"),
        "simulator_action": fact.get("simulator_action"),
        "target": fact.get("target"),
        "target_entity": fact.get("target"),
        "success": bool(fact.get("success")),
        "state_changed": bool(fact.get("state_changed")),
        "framework_mappings": mapping,
        "outcomes": list(fact.get("outcomes") or []),
    }


class MitigationComparisonPage(Page):
    """Render persisted paired reruns without performing evaluation in Qt."""

    def __init__(self) -> None:
        super().__init__(
            "PHASE 15 · FROZEN-POLICY RERUNS",
            "Mitigation counterfactual comparison",
            "Observed path criticality is trace-derived. Mitigation effect is measured "
            "separately by paired reruns with one CVE disabled.",
        )
        selector = panel(QHBoxLayout())
        selector.layout().addWidget(label("Counterfactual report", "FieldLabel"))
        self.selector = QComboBox()
        self.selector.setMinimumWidth(430)
        self.selector.currentIndexChanged.connect(self._selected)
        selector.layout().addWidget(self.selector)
        selector.layout().addStretch()
        self.cve = label("NO STORED REPORT", "StateLabel")
        selector.layout().addWidget(self.cve)
        self.root.addWidget(selector)

        metrics = QHBoxLayout()
        self.original = Metric("ORIGINAL SUCCESS")
        self.mitigated = Metric("MITIGATED SUCCESS")
        self.change = Metric("ABSOLUTE CHANGE")
        self.significance = Metric("PAIRED EXACT P")
        for item in (self.original, self.mitigated, self.change, self.significance):
            metrics.addWidget(item)
        self.root.addLayout(metrics)

        comparison = panel(QVBoxLayout())
        comparison.layout().addWidget(label("IDENTICAL-SEED OUTCOME PAIRS", "SectionTitle"))
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Seed",
                "Original success",
                "Mitigated success",
                "Original steps",
                "Mitigated steps",
                "Original reward",
                "Mitigated reward",
                "Original crown",
                "Mitigated crown",
                "Blocked attempts",
            ]
        )
        configure_table(self.table, stretch_column=0)
        comparison.layout().addWidget(self.table)
        self.root.addWidget(comparison)

        paths = panel(QVBoxLayout())
        paths.layout().addWidget(label("MITRE / PATH COMPARISON", "SectionTitle"))
        self.paths = QTableWidget(0, 5)
        self.paths.setHorizontalHeaderLabels(
            [
                "Seed",
                "Original techniques",
                "Mitigated techniques",
                "Original path",
                "Mitigated path",
            ]
        )
        configure_table(self.paths, stretch_column=1)
        paths.layout().addWidget(self.paths)
        self.root.addWidget(paths)
        self.provenance = label("", "MonoDetail", wrap=True)
        self.root.addWidget(self.provenance)
        self.reports: list[dict] = []

    def apply(self, data: DashboardData) -> None:
        self.reports = list(data.mitigation_reports)
        self.selector.blockSignals(True)
        self.selector.clear()
        for report in self.reports:
            intervention = report.get("intervention") or {}
            self.selector.addItem(
                f"{intervention.get('selected_cve', 'unknown')} · "
                f"{str(report.get('report_id', ''))[:12]}",
                report.get("report_id"),
            )
        self.selector.blockSignals(False)
        self._selected(0)

    def _selected(self, index: int) -> None:
        if not 0 <= index < len(self.reports):
            self.cve.setText("NO STORED REPORT")
            fill_table(self.table, [])
            fill_table(self.paths, [])
            self.provenance.clear()
            return
        report = self.reports[index]
        intervention = report.get("intervention") or {}
        analysis = report.get("analysis") or {}
        pairs = list(report.get("pairs") or [])
        self.cve.setText(str(intervention.get("selected_cve", "UNKNOWN")))
        self.original.update_value(
            percent(analysis.get("original_success_rate")), "Frozen policy · original environment"
        )
        self.mitigated.update_value(
            percent(analysis.get("mitigated_success_rate")), "Same policy and episode seeds"
        )
        self.change.update_value(
            percent(analysis.get("absolute_success_rate_change")), "Mitigated minus original"
        )
        exact = analysis.get("mcnemar_exact") or {}
        p_value = exact.get("p_value")
        self.significance.update_value(
            "—" if p_value is None else f"{float(p_value):.4f}",
            f"{exact.get('discordant_pairs', 0)} discordant pairs",
        )
        fill_table(
            self.table,
            [
                (
                    pair.get("evaluation_seed"),
                    "YES" if pair["original"].get("success") else "NO",
                    "YES" if pair["mitigated"].get("success") else "NO",
                    pair["original"].get("steps"),
                    pair["mitigated"].get("steps"),
                    pair["original"].get("reward"),
                    pair["mitigated"].get("reward"),
                    "YES" if pair["original"].get("crown_jewel_reach") else "NO",
                    "YES" if pair["mitigated"].get("crown_jewel_reach") else "NO",
                    pair["mitigated"].get("mitigation_blocked_attempts", 0),
                )
                for pair in pairs
            ],
        )
        fill_table(
            self.paths,
            [
                (
                    pair.get("evaluation_seed"),
                    ", ".join(
                        item.get("technique_id", "")
                        for item in pair["original"].get("mitre_techniques", [])
                    )
                    or "UNMAPPED",
                    ", ".join(
                        item.get("technique_id", "")
                        for item in pair["mitigated"].get("mitre_techniques", [])
                    )
                    or "UNMAPPED",
                    len(pair["original"].get("observed_path", [])),
                    len(pair["mitigated"].get("observed_path", [])),
                )
                for pair in pairs
            ],
        )
        provenance = report.get("provenance") or {}
        promotion = report.get("phase14_promotion") or {}
        self.provenance.setText(
            f"REPORT       {report.get('report_id', '—')}\n"
            f"CHECKPOINT   {provenance.get('checkpoint_sha256', '—')}\n"
            f"POLICY       {provenance.get('policy_sha256_before', '—')} · unchanged\n"
            f"INTERVENTION {intervention.get('intervention_version', '—')} · "
            f"{intervention.get('intervention_sha256', '—')}\n"
            f"PHASE 14     {promotion.get('source_report_id', 'not promoted')}"
        )


class ResearchPage(Page):
    def __init__(self) -> None:
        super().__init__(
            "CANONICAL LOCAL EVIDENCE",
            "Research studies",
            "Browse the frozen Phase 7–13 outcomes. Generated results stay local and ignored.",
        )
        selector = panel(QHBoxLayout())
        selector.layout().addWidget(label("Study", "FieldLabel"))
        self.selector = QComboBox()
        self.selector.setMinimumWidth(380)
        self.selector.currentIndexChanged.connect(self._selected)
        selector.layout().addWidget(self.selector)
        selector.layout().addStretch()
        self.status = label("No canonical results loaded", "StateLabel")
        selector.layout().addWidget(self.status)
        self.root.addWidget(selector)
        self.banner = StateBanner()
        self.root.addWidget(self.banner)

        metric_row = QHBoxLayout()
        self.seed_count = Metric("MATCHED SEEDS")
        self.episode_count = Metric("EVALUATION EPISODES")
        self.signal_count = Metric("PRIMARY SIGNALS")
        self.commit = Metric("CODE COMMIT")
        for item in (self.seed_count, self.episode_count, self.signal_count, self.commit):
            metric_row.addWidget(item)
        self.root.addLayout(metric_row)

        panel_widget = panel(QVBoxLayout())
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Metric",
                "Family",
                "Reference",
                "Candidate",
                "Difference",
                "Adjusted p",
                "Effect size",
                "Verdict",
            ]
        )
        configure_table(self.table, stretch_column=0)
        panel_widget.layout().addWidget(self.table)
        self.root.addWidget(panel_widget)
        self.provenance = label("", "MonoDetail", wrap=True)
        self.root.addWidget(self.provenance)
        self.studies: list[StudySummary] = []

    def apply(self, data: DashboardData) -> None:
        self.studies = list(data.studies)
        self.selector.blockSignals(True)
        self.selector.clear()
        for study in self.studies:
            self.selector.addItem(f"Phase {study.phase} · {study.title}", study.phase)
        self.selector.blockSignals(False)
        self._selected(0)

    def _selected(self, index: int) -> None:
        if not 0 <= index < len(self.studies):
            self.status.setText("NO EVIDENCE")
            self.banner.update_state("EMPTY", "No canonical result package is mounted.", "warn")
            fill_table(self.table, [])
            return
        study = self.studies[index]
        self.status.setText("COMPLETE" if study.complete else "PARTIAL")
        self.banner.update_state(
            "COMPLETE" if study.complete else "INCOMPLETE",
            f"{study.arm_a.replace('_', ' ')} versus {study.arm_b.replace('_', ' ')} · "
            f"{study.outcome}",
            "ok" if study.complete else "warn",
        )
        self.seed_count.update_value(str(study.training_seeds), "Primary paired unit")
        self.episode_count.update_value(f"{study.evaluation_episodes:,}", "Frozen-policy episodes")
        signals = sum(metric.significant for metric in study.primary_metrics)
        self.signal_count.update_value(
            str(signals), f"of {len(study.primary_metrics)} primary metrics"
        )
        self.commit.update_value(study.code_commit[:8] or "—", "Recorded producer")
        fill_table(
            self.table,
            [
                (
                    metric_name(metric.name),
                    "PRIMARY" if metric.primary else "DESCRIPTIVE",
                    number(metric.arm_a_mean, 3),
                    number(metric.arm_b_mean, 3),
                    number(metric.difference, 3),
                    number(metric.p_adjusted, 4),
                    number(metric.effect_size, 3),
                    "SIGNIFICANT" if metric.significant else "NO SIGNAL",
                )
                for metric in sorted(study.metrics, key=lambda item: not item.primary)
            ],
        )
        self.provenance.setText(
            f"CONFIG  {study.config_hash or 'unavailable'}\nRESULT  {study.result_path}"
        )


class RunsPage(Page):
    def __init__(self) -> None:
        super().__init__(
            "DEDUPLICATED BACKEND RECORDS",
            "Runs",
            "PostgreSQL is authoritative; artifact-only runs are appended without duplicates.",
        )
        toolbar = QHBoxLayout()
        self.count = Metric("STORED RUNS")
        toolbar.addWidget(self.count)
        toolbar.addStretch()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by run, reward or seed…")
        self.search.setMinimumWidth(300)
        self.search.textChanged.connect(self.filter)
        toolbar.addWidget(self.search)
        self.root.addLayout(toolbar)
        panel_widget = panel(QVBoxLayout())
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Reward / condition", "Status", "Episodes", "Progress", "Seed", "Success"]
        )
        configure_table(self.table, stretch_column=0)
        panel_widget.layout().addWidget(self.table)
        self.root.addWidget(panel_widget)

    def apply(self, data: DashboardData) -> None:
        self.count.update_value(str(len(data.campaigns)), "Current backend snapshot")
        fill_table(
            self.table,
            [
                (
                    campaign.name,
                    campaign.reward_mode,
                    campaign.status,
                    f"{campaign.episodes:,}",
                    "—" if campaign.progress is None else f"{campaign.progress}%",
                    campaign.seed,
                    percent(campaign.success_rate),
                )
                for campaign in data.campaigns
            ],
        )
        self.filter(self.search.text())

    def filter(self, text: str) -> None:
        query = text.casefold().strip()
        for row in range(self.table.rowCount()):
            values = " ".join(
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
                if self.table.item(row, column) is not None
            )
            self.table.setRowHidden(row, bool(query) and query not in values.casefold())


class SystemPage(Page):
    def __init__(self) -> None:
        super().__init__(
            "RUNTIME AND TRUST BOUNDARIES",
            "System",
            "Read-only operational state. Secrets and checkpoint contents are never displayed.",
        )
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        self.grid = QGridLayout()
        self.root.addLayout(self.grid)
        boundary = panel(QVBoxLayout())
        boundary.layout().addWidget(label("EXECUTION BOUNDARY", "SectionTitle"))
        boundary.layout().addWidget(
            label(
                "Offline typed graph and NASim simulation only. Training and GUI run in "
                "separate Podman images. Model weights remain under ignored runs/ paths.",
                "Muted",
                wrap=True,
            )
        )
        self.database = label("PostgreSQL: loading", "MonoDetail", wrap=True)
        boundary.layout().addWidget(self.database)
        self.root.addWidget(boundary)
        self.root.addStretch(1)

    def apply(self, data: DashboardData) -> None:
        kind = "warn" if data.source_status.startswith("Artefact mode") else "ok"
        self.banner.update_state(
            "DEGRADED" if kind == "warn" else "HEALTHY", data.source_status, kind
        )
        self.database.setText(f"DATABASE  {data.database_label}\nMODE      {data.source_status}")
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for index, source in enumerate(data.datasets):
            card = panel(QVBoxLayout(), "DatasetCard")
            card.layout().addWidget(label(source.integrity, "Eyebrow"))
            card.layout().addWidget(label(source.title, "SectionTitle"))
            card.layout().addWidget(label(source.count, "DatasetValue"))
            card.layout().addWidget(label(source.detail, "Muted", wrap=True))
            self.grid.addWidget(card, index // 2, index % 2)


class MainWindow(QMainWindow):
    PAGE_DATA = [
        ("Overview", "Mission control"),
        ("MITRE Workspace", "MITRE Navigator workspace"),
        ("Attack Paths", "Attack paths"),
        ("Path Report", "Explainable attack-path report"),
        ("Mitigation", "Mitigation counterfactual comparison"),
        ("Research", "Research studies"),
        ("Runs", "Stored runs"),
        ("System", "System status"),
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
        sidebar.setFixedWidth(218)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(16, 24, 16, 18)
        nav.setSpacing(4)
        nav.addWidget(label("RLREDTEAM", "Brand"))
        nav.addWidget(label("RESEARCH INSTRUMENT", "BrandSub"))
        nav.addSpacing(22)
        self.nav_buttons: list[QPushButton] = []
        for index, (name, _) in enumerate(self.PAGE_DATA):
            item = button(name, "Nav")
            item.setCheckable(True)
            item.clicked.connect(lambda checked=False, page=index: self.select_page(page))
            self.nav_buttons.append(item)
            nav.addWidget(item)
        nav.addStretch()
        safety = panel(QVBoxLayout(), "BoundaryPanel")
        safety.layout().addWidget(label("SIMULATION BOUNDARY", "Eyebrow"))
        safety.layout().addWidget(label("Offline · no live exploitation", "Muted", wrap=True))
        nav.addWidget(safety)
        shell.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 0, 24, 0)
        content_layout.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 16, 0, 12)
        self.header = label("Mission control", "PageTitle")
        top.addWidget(self.header)
        top.addStretch()
        self.refresh_button = button("Refresh backend")
        self.refresh_button.clicked.connect(self.refresh)
        top.addWidget(self.refresh_button)
        self.report_button = button("Locate latest statistics", "Primary")
        self.report_button.clicked.connect(self.export_report)
        top.addWidget(self.report_button)
        content_layout.addLayout(top)

        self.stack = QStackedWidget()
        self.pages = [
            OverviewPage(),
            SimulationPage(self.backend, self.notify),
            PathsPage(self.backend, self.notify),
            AttackPathReportPage(),
            MitigationComparisonPage(),
            ResearchPage(),
            RunsPage(),
            SystemPage(),
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
        self.refresh()

    def refresh(self) -> None:
        if not hasattr(self.backend, "load_dashboard"):
            return
        self.refresh_button.setEnabled(False)
        self.statusBar().showMessage("Refreshing backend snapshot…")
        self._load_task = run_async(
            self.backend.load_dashboard, self.apply_dashboard, self._dashboard_failed
        )

    def apply_dashboard(self, data: DashboardData) -> None:
        for page in self.pages:
            apply = getattr(page, "apply", None)
            if apply:
                apply(data)
        self.pages[2].apply_paths(data.paths)
        self.refresh_button.setEnabled(True)
        self.statusBar().showMessage(data.source_status)

    def _dashboard_failed(self, error: str, detail: str) -> None:
        self.refresh_button.setEnabled(True)
        self.statusBar().showMessage(f"Backend unavailable: {error} · {detail}")

    def select_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.header.setText(self.PAGE_DATA[index][1])
        scroll = self.stack.currentWidget()
        if isinstance(scroll, QScrollArea):
            scroll.verticalScrollBar().setValue(0)
            scroll.horizontalScrollBar().setValue(0)
        for item_index, item in enumerate(self.nav_buttons):
            item.setChecked(item_index == index)

    def notify(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def export_report(self) -> None:
        self._export_task = run_async(
            self.backend.export_report,
            lambda path: self.notify(f"Latest statistics: {path}"),
            lambda error, _detail: self.notify(f"Statistics unavailable: {error}"),
        )
