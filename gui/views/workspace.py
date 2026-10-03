"""Original native workspace composed from existing research-console widgets."""

from __future__ import annotations

import json

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data.models import (
    AgentKnowledgeView,
    AttackTrajectorySummary,
    ExperimentReport,
    ScenarioSummary,
)
from gui.data.workspace import DIAGNOSTICS, compare_options
from gui.views.research_console import (
    Metric,
    Page,
    StateBanner,
    TrajectoryGraph,
    button,
    configure_table,
    fill_table,
    label,
    number,
)
from gui.widgets.enterprise_graph import EnterpriseGraph

HELP = {
    "approx_kl": "Approximate change in the PPO policy after an update.",
    "clip_fraction": "Fraction of PPO updates limited by the clipping rule.",
    "explained_variance": "How well the learned value function predicts observed returns.",
    "entropy_loss": "Negative entropy term; indicator of policy randomness/exploration.",
    "policy_gradient_loss": "Recorded PPO policy optimisation loss; not task success.",
    "value_loss": "Recorded value-function prediction loss.",
    "learning_rate": "Recorded optimiser step size.",
    "mean_episode_reward": "Recorded training episode reward mean; not held-out evaluation.",
    "mean_episode_length": "Recorded mean training episode length.",
    "Native reward": "Reward produced by the simulator before research shaping.",
    "Shaped reward": "Training reward after configured research reward terms.",
}


def text_box():
    box = QPlainTextEdit()
    box.setReadOnly(True)
    return box


def table(headers):
    item = QTableWidget(0, len(headers))
    item.setHorizontalHeaderLabels(headers)
    configure_table(item)
    return item


def knowledge_text(value: AgentKnowledgeView):
    def items(values):
        return "\n".join(map(str, values)) or "Unknown / not recorded"

    return (
        f"{value.status}\n\nKNOWN NODES / ACCESS\n"
        + items(f"{node['id']}: {node['attributes']['access']}" for node in value.nodes)
        + "\n\nDISCOVERED SERVICES\n"
        + items(value.services)
        + "\n\nKNOWN VULNERABILITIES (native IDs may not be CVEs)\n"
        + items(value.vulnerabilities)
        + "\n\nCREDENTIALS\n"
        + items(value.credentials)
        + "\n\nAVAILABLE SEMANTIC ACTIONS\n"
        + items(value.available_actions)
    )


class ActionInspector(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(label("CURRENT ACTION", "SectionTitle"))
        self.fields = table(["Field", "Recorded value"])
        self.fields.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.fields.setColumnWidth(0, 115)
        self.fields.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.knowledge = text_box()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.fields, "Action")
        self.tabs.addTab(self.knowledge, "Before / after")
        layout.addWidget(self.tabs)
        self.set_entry(None)

    def set_entry(self, entry):
        if entry is None:
            fill_table(self.fields, [])
            self.knowledge.setPlainText("Not yet available: select a recorded step")
            return
        a = entry.action
        rows = [
            ("Step", a.step),
            ("Action type", a.kind),
            ("Action name", a.name),
            ("Target", a.target),
            ("CVE", a.cve),
            ("CVSS", number(a.cvss, 2)),
            ("MITRE tactic / technique", "\n".join(a.mitre) or "Unknown / unmapped"),
            ("Native reward", number(a.native_reward, 4)),
            ("Shaped reward", number(a.shaped_reward, 4)),
            ("Recorded policy / demo reward", number(a.recorded_reward, 4)),
            ("Success", "Unknown" if a.success is None else "Yes" if a.success else "No"),
            ("Access gained", a.access_gained),
            (
                "Newly discovered count",
                str(a.newly_discovered) if a.newly_discovered is not None else "Unknown / withheld",
            ),
            ("Crown-jewel event", "Yes" if entry.crown_jewel else "Not recorded"),
            ("Reward breakdown", json.dumps(dict(a.breakdown)) if a.breakdown else "Unknown"),
            ("Evidence", a.source),
        ]
        fill_table(self.fields, rows)
        for i, (name, _) in enumerate(rows):
            if name in HELP:
                self.fields.item(i, 0).setToolTip(HELP[name])
        self.knowledge.setPlainText(
            "BEFORE ACTION\n"
            + knowledge_text(entry.before)
            + "\n\nKNOWLEDGE CHANGE\n"
            + a.knowledge_delta
            + "\n\nAFTER ACTION\n"
            + knowledge_text(entry.after)
        )


class ReplayWorkspace(Page):
    step_selected = Signal(object)

    def __init__(self, title="Recorded attack trajectory"):
        super().__init__(
            "EVIDENCE REPLAY",
            title,
            "Actual recorded actions, not a hidden shortest path. "
            "Lines show chronological order, not network links. "
            "Knowledge snapshots and trajectory evidence are separate from True Environment.",
        )
        self.data = AttackTrajectorySummary()
        self.eyebrow.hide()
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.knowledge = text_box()
        self.knowledge.setMinimumWidth(190)
        split.addWidget(self.knowledge)
        self.graphs = QTabWidget()
        self.path_graph = TrajectoryGraph()
        self.knowledge_graph = EnterpriseGraph()
        self.knowledge_graph.setMinimumHeight(300)
        self.graphs.addTab(self.path_graph, "Recorded attack trajectory")
        self.graphs.addTab(self.knowledge_graph, "Agent Knowledge — recorded snapshots")
        split.addWidget(self.graphs)
        self.inspector = ActionInspector()
        split.addWidget(self.inspector)
        split.setSizes([220, 620, 310])
        split.setMinimumHeight(330)
        split.setMaximumHeight(370)
        self.root.addWidget(split)
        self.node_details = label("Select a node for recorded details", "Muted", wrap=True)
        self.node_details.setTextFormat(Qt.TextFormat.PlainText)
        self.knowledge_graph.node_selected.connect(
            lambda node: self.node_details.setText(json.dumps(node, sort_keys=True))
        )
        self.root.addWidget(self.node_details)
        self.timeline = table(["Step", "Action", "Target / host transition", "Outcome", "MITRE"])
        self.timeline.setMinimumHeight(150)
        self.timeline.currentCellChanged.connect(lambda row, *_: self.select_step(row))
        self.root.addWidget(self.timeline)
        self.set_trajectory(self.data)

    def set_trajectory(self, data):
        self.data = data
        self.banner.update_state(data.title, data.status)
        self.timeline.blockSignals(True)
        fill_table(
            self.timeline,
            [
                (
                    e.action.step,
                    e.action.name,
                    e.action.target,
                    "Unknown"
                    if e.action.success is None
                    else "Success"
                    if e.action.success
                    else "Failure",
                    " · ".join(e.action.mitre),
                )
                for e in data.entries
            ],
        )
        self.timeline.blockSignals(False)
        self.node_details.setText("Select a node for recorded details")
        self.select_step(0 if data.entries else -1)

    def select_step(self, index):
        entry = self.data.entries[index] if 0 <= index < len(self.data.entries) else None
        self.inspector.set_entry(entry)
        knowledge = entry.after if entry else AgentKnowledgeView()
        self.knowledge.setPlainText(knowledge_text(knowledge))
        visible = self.data.entries[: index + 1] if entry else ()
        trace = [
            {"step": e.action.step, "target": e.action.target, "action": e.action.name}
            for e in visible
        ]
        self.path_graph.set_steps(trace)
        self.knowledge_graph.set_graph(list(knowledge.nodes), list(knowledge.edges), trace)
        if entry:
            self.timeline.blockSignals(True)
            self.timeline.selectRow(index)
            self.timeline.blockSignals(False)
        self.step_selected.emit(entry)


class HomePage(Page):
    def __init__(self, navigate):
        super().__init__(
            "SECURITY-RL",
            "Security-RL Attack Simulation Workspace",
            "Simulation-only research environment",
        )
        actions = QHBoxLayout()
        for name, target in (
            ("Open Scenario", 1),
            ("Run / View Simulation", 2),
            ("View Research Evidence", 5),
        ):
            item = button(name, "Primary" if target == 1 else "Action")
            if target == 2:
                item.setToolTip("Open recorded replay or the separate offline demo; does not start a run.")
            item.clicked.connect(lambda checked=False, i=target: navigate(i))
            actions.addWidget(item)
        self.root.addLayout(actions)
        self.cards = {}
        cards = QGridLayout()
        for index, name in enumerate(
            (
                "Current simulator",
                "Current topology seed",
                "Current reward mode",
                "Frozen baseline status",
                "Confirmatory study status",
            )
        ):
            card = Metric(name.upper(), "Not yet available")
            self.cards[name] = card
            cards.addWidget(card, index // 2, index % 2)
        self.root.addLayout(cards)
        self.root.addWidget(label("Simulation-only. No live-network execution.", "StateLabel"))
        self.root.addWidget(
            label(
                "Explore the environment → inspect recorded actions → follow the trajectory → "
                "read the outcome → check the research evidence.\n"
                "Recorded-policy replay is not a new training run. The separate offline demo uses "
                "a deterministic feasibility agent, not PPO.",
                "Muted",
                wrap=True,
            )
        )

    def set_scenario(self, data):
        for name, value in (
            ("Current simulator", data.simulator),
            ("Current topology seed", data.topology_seed),
            ("Current reward mode", data.reward_mode),
        ):
            self.cards[name].update_value(
                str(value) if value is not None else "Not yet available", ""
            )


class ScenarioPage(Page):
    def __init__(self):
        super().__init__(
            "SCENARIO",
            "What environment is being simulated?",
            "True Environment is research inspection only and never feeds PPO.",
        )
        self.data, self.knowledge = ScenarioSummary(), AgentKnowledgeView()
        self.mode = QComboBox()
        self.mode.addItems(["Agent Knowledge", "True Environment"])
        self.mode.currentIndexChanged.connect(self.render)
        self.root.addWidget(self.mode)
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        self.summary = table(["Scenario field", "Value"])
        self.summary.setMaximumHeight(250)
        self.root.addWidget(self.summary)
        self.graph = EnterpriseGraph()
        self.root.addWidget(self.graph)
        self.details = text_box()
        self.details.setMaximumHeight(120)
        self.graph.node_selected.connect(
            lambda value: self.details.setPlainText(json.dumps(value, indent=2))
        )
        self.root.addWidget(self.details)
        provenance = QGroupBox("Scenario provenance — expand")
        provenance.setCheckable(True)
        provenance.setChecked(False)
        content = QVBoxLayout(provenance)
        self.provenance = text_box()
        self.provenance.setVisible(False)
        content.addWidget(self.provenance)
        provenance.toggled.connect(self.provenance.setVisible)
        self.root.addWidget(provenance)
        self.render()

    def set_scenario(self, data):
        self.data = data
        self.knowledge = AgentKnowledgeView()  # never carry knowledge across environments
        self.render()

    def set_knowledge(self, entry):
        self.knowledge = entry.after if entry else AgentKnowledgeView()
        self.render()

    def render(self, *_):
        self.details.clear()
        true = self.mode.currentIndex() == 1
        if true:
            self.banner.update_state("Research True Environment View", self.data.status, "warn")
            self.graph.set_graph(list(self.data.true_nodes), list(self.data.true_edges))
            rows = [
                ("Simulator", self.data.simulator),
                ("Topology seed", self.data.topology_seed),
                ("Hosts", self.data.host_count),
                ("Subnets", self.data.subnet_count),
                ("Crown jewels", ", ".join(self.data.crown_jewels) or "Unknown"),
                ("CVE catalogue", self.data.catalogue),
                ("Reward mode", self.data.reward_mode),
                ("Training seed", self.data.training_seed),
            ]
        else:
            self.banner.update_state("Agent Knowledge — no hidden topology", self.knowledge.status)
            self.graph.set_graph(list(self.knowledge.nodes), list(self.knowledge.edges))
            rows = [
                (
                    "Recorded known nodes",
                    len(self.knowledge.nodes) if self.knowledge.nodes else "Not yet available",
                ),
                ("Recorded services", ", ".join(self.knowledge.services) or "Unknown"),
                ("Known vulnerabilities", ", ".join(self.knowledge.vulnerabilities) or "Unknown"),
            ]
        fill_table(
            self.summary,
            [(key, str(value) if value is not None else "Unknown") for key, value in rows],
        )
        self.provenance.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in self.data.provenance) or "Not yet available"
        )


class OutcomePage(Page):
    def __init__(self, navigate):
        super().__init__(
            "REPORT",
            "What was the outcome?",
            "Recorded simulation outcomes, with scope and provenance.",
        )
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        self.metrics = table(["Outcome metric", "Recorded value"])
        self.root.addWidget(self.metrics)
        self.limitation = label("", "Muted", wrap=True)
        self.limitation.setTextFormat(Qt.TextFormat.PlainText)
        self.root.addWidget(self.limitation)
        self.provenance = text_box()
        self.root.addWidget(self.provenance)
        actions = QHBoxLayout()
        for name, target in (
            ("View Attack Path", 3),
            ("Compare Experiment", 5),
            ("View Research Evidence", 5),
        ):
            item = button(name)
            item.clicked.connect(lambda checked=False, i=target, n=name: navigate(i, n))
            actions.addWidget(item)
        self.root.addLayout(actions)
        self.set_report(ExperimentReport())

    def set_report(self, data):
        self.data = data
        self.banner.update_state(data.outcome, f"{data.title}\n{data.scope}")
        # Fail closed even if another backend accidentally supplies legacy metrics.
        allowed = {
            "steps to success",
            "native return",
            "failed actions",
            "exploited cves",
            "mean cvss",
            "compromised hosts",
            "mitre tactics observed",
            "recorded demo reward",
            "native / shaped decomposition",
        }
        fill_table(self.metrics, [(k, v) for k, v in data.metrics if k.lower() in allowed])
        self.limitation.setText(data.limitation)
        self.provenance.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in data.provenance) or "Not yet available"
        )


class Curve(QWidget):
    """Display recorded diagnostic points, without smoothing or normalisation."""

    def __init__(self):
        super().__init__()
        self.points = []
        self.setMinimumHeight(220)

    def paintEvent(self, event):  # noqa: N802
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.PANEL))
        painter.setPen(QColor(theme.TEXT_SECONDARY))
        if not self.points:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Not yet available")
            return
        xs, ys = zip(*self.points, strict=True)
        lo, hi = min(ys), max(ys)
        painter.drawText(
            5, 18, f"Recorded range {lo:.5g} … {hi:.5g} · timesteps {min(xs):g} … {max(xs):g}"
        )
        painter.setPen(QPen(QColor(theme.ARM_1), 1.5))
        points = [
            QPointF(
                18 + (x - min(xs)) / max(1, max(xs) - min(xs)) * (self.width() - 36),
                self.height() - 20 - (y - lo) / max(1e-12, hi - lo) * (self.height() - 55),
            )
            for x, y in self.points
        ]
        for a, b in zip(points, points[1:], strict=False):
            painter.drawLine(a, b)


class DiagnosticsPage(Page):
    def __init__(self):
        super().__init__(
            "RESEARCH",
            "Convergence and PPO diagnostics",
            "Frozen shaped training diagnostics — not held-out evaluation outcomes.",
        )
        self.rows = {}
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        self.aggregate = QLabel("Aggregate reward curve: Not yet available")
        self.root.addWidget(self.aggregate)
        controls = QHBoxLayout()
        self.seed, self.metric_selector = QComboBox(), QComboBox()
        self.metric_selector.addItems(DIAGNOSTICS)
        self.metric_selector.setCurrentText("mean_episode_reward")
        for i, name in enumerate(DIAGNOSTICS):
            self.metric_selector.setItemData(i, HELP[name], Qt.ItemDataRole.ToolTipRole)
        controls.addWidget(self.seed)
        controls.addWidget(self.metric_selector)
        self.root.addLayout(controls)
        self.curve = Curve()
        self.root.addWidget(self.curve)
        self.values = table(["Timesteps", *DIAGNOSTICS])
        for i, name in enumerate(DIAGNOSTICS, 1):
            self.values.horizontalHeaderItem(i).setToolTip(HELP[name])
        self.values.setMinimumHeight(200)
        self.root.addWidget(self.values)
        self.convergence = table(["Seed", "Registered criterion result", "Recorded explanation"])
        self.root.addWidget(self.convergence)
        self.criterion = text_box()
        self.criterion.setMaximumHeight(160)
        self.root.addWidget(self.criterion)
        self.seed.currentIndexChanged.connect(self.render)
        self.metric_selector.currentIndexChanged.connect(self.render)

    def set_evidence(self, data):
        self.rows = dict(data.diagnostics)
        self.banner.update_state("Frozen training evidence", data.status)
        self.seed.blockSignals(True)
        self.seed.clear()
        for seed in self.rows:
            self.seed.addItem(f"Training seed {seed}", seed)
        self.seed.blockSignals(False)
        self.aggregate.clear()
        if data.images:
            pixmap = QPixmap()
            if pixmap.loadFromData(data.images[0][1]):
                self.aggregate.setPixmap(
                    pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
                )
        else:
            self.aggregate.setText("Aggregate reward curve: Not yet available")
        fill_table(self.convergence, data.convergence)
        self.criterion.setPlainText(data.criterion)
        self.render()

    def render(self, *_):
        rows = self.rows.get(self.seed.currentData(), ())
        key = self.metric_selector.currentText()
        self.metric_selector.setToolTip(HELP.get(key, ""))
        fill_table(
            self.values,
            [
                (number(r.get("timesteps"), 0), *(number(r.get(k), 6) for k in DIAGNOSTICS))
                for r in rows
            ],
        )
        self.curve.points = [
            (r["timesteps"], r[key])
            for r in rows
            if r.get("timesteps") is not None and r.get(key) is not None
        ]
        self.curve.update()


class ComparisonPage(Page):
    def __init__(self):
        super().__init__(
            "DESCRIPTIVE COMPARISON",
            "Compare recorded experiments",
            "Difference = B − A. No ranking or equivalence claim. "
            "Cross-simulator comparison is unavailable without a shared recorded protocol.",
        )
        self.options = []
        self.left, self.right = QComboBox(), QComboBox()
        controls = QHBoxLayout()
        controls.addWidget(self.left)
        controls.addWidget(self.right)
        self.root.addLayout(controls)
        self.note = label("Not yet available", "Muted", wrap=True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.root.addWidget(self.note)
        self.table = table(["Metric", "Experiment A", "Experiment B", "Difference (B − A)"])
        self.root.addWidget(self.table)
        self.left.currentIndexChanged.connect(self.render)
        self.right.currentIndexChanged.connect(self.render)

    def set_options(self, options):
        self.options = list(options)
        for combo in (self.left, self.right):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([value.label for value in self.options])
            combo.blockSignals(False)
        self.right.setCurrentIndex(1 if len(options) > 1 else 0)
        self.render()

    def render(self, *_):
        if not self.options:
            fill_table(self.table, [])
            self.note.setText("Not yet available")
            return
        a, b = self.options[self.left.currentIndex()], self.options[self.right.currentIndex()]
        rows = compare_options(a, b)
        self.note.setText(
            f"A: {a.label}\nB: {b.label}\n{a.basis}\n{b.basis}"
            if rows
            else "Not comparable: no shared recorded evaluation cohort. "
            "Choose two aggregate arms or two policy seeds."
        )
        fill_table(
            self.table,
            [
                (r.metric, number(r.value_a, 5), number(r.value_b, 5), number(r.difference, 5))
                for r in rows
            ],
        )


class Section(Page):
    def __init__(self, title, description, tabs):
        super().__init__("WORKSPACE", title, description)
        self.eyebrow.hide()
        self.title.hide()  # primary header already identifies this section
        self.tabs = QTabWidget()
        self.root.addWidget(self.tabs)
        for name, widget in tabs:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.tabs.addTab(scroll, name)
