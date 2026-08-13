"""Replay — the attack path.

QGraphicsScene network, transport, step detail. Hosts change state as the
episode is stepped through; subnets are QGraphicsItemGroups laid out in a row.

Failure is a red numeral and a status word, never a red panel: roughly two
thirds of steps fail, and colouring the whole panel would make the view
unreadable exactly when it is most used.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.data import runs as runs_folder
from gui.data.models import EpisodeRow, StepRow, TopologyView
from gui.data.repository import Repository
from gui.widgets.common import Banner, hline, label, section_label
from gui.workers.query import run_async

HOST_W, HOST_H = 78, 34
COL_GAP, ROW_GAP = 46, 14
SUBNET_PAD = 16


class NetworkCanvas(QGraphicsView):
    """Hosts grouped by subnet, edges from the adjacency matrix."""

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(self.renderHints().Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(theme.SURFACE)))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._items: dict[tuple[int, int], QGraphicsEllipseItem] = {}
        self._labels: dict[tuple[int, int], QGraphicsSimpleTextItem] = {}
        self._crowns: set[tuple[int, int]] = set()

    def build(self, topology: TopologyView) -> None:
        self._scene.clear()
        self._items.clear()
        self._labels.clear()
        self._crowns = topology.crown_jewels()

        # Group by real host addresses. Deriving columns from the subnet-size
        # list is off by one, because subnets[0] is NASim's internet subnet.
        grouped: dict[int, list[int]] = {}
        for subnet, host in topology.hosts:
            grouped.setdefault(subnet, []).append(host)
        if not grouped and topology.num_hosts:
            # Degraded layout for runs predating topology persistence.
            grouped = {1: list(range(topology.num_hosts))}

        centres: dict[int, tuple[float, float]] = {}
        x = 0.0
        for subnet_index in sorted(grouped):
            size = len(grouped[subnet_index])
            group = QGraphicsItemGroup()
            height = size * (HOST_H + ROW_GAP) + SUBNET_PAD
            box = QGraphicsEllipseItem()  # placeholder, replaced by rect below
            self._scene.addItem(group)

            frame = self._scene.addRect(
                QRectF(x, 0, HOST_W + SUBNET_PAD * 2, height + 26),
                QPen(QColor(theme.BORDER)), QBrush(QColor(theme.PANEL)),
            )
            frame.setZValue(-2)
            caption = self._scene.addSimpleText(f"subnet {subnet_index}")
            caption.setBrush(QBrush(QColor(theme.TEXT_TERTIARY)))
            caption.setFont(QFont(theme.FONT_MONO, 7))
            caption.setPos(x + SUBNET_PAD, 6)

            for host_index in sorted(grouped[subnet_index]):
                y = 26 + host_index * (HOST_H + ROW_GAP)
                addr = (subnet_index, host_index)
                rect = self._scene.addRect(
                    QRectF(x + SUBNET_PAD, y, HOST_W, HOST_H),
                    QPen(QColor(theme.BORDER)), QBrush(QColor(theme.WINDOW)),
                )
                text = self._scene.addSimpleText(f"({subnet_index},{host_index})")
                text.setFont(QFont(theme.FONT_MONO, 7))
                text.setPos(x + SUBNET_PAD + 8, y + 11)
                self._items[addr] = rect
                self._labels[addr] = text
                self.set_state(addr, "undiscovered")

            centres[subnet_index] = (x + SUBNET_PAD + HOST_W / 2, height / 2)
            x += HOST_W + SUBNET_PAD * 2 + COL_GAP
            box.hide()

        self._draw_edges(topology, centres)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-12, -12, 12, 12))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_edges(self, topology: TopologyView, centres: dict) -> None:
        if not topology.adjacency:
            return
        for i, row in enumerate(topology.adjacency):
            for j, connected in enumerate(row):
                if not connected or i >= j or i == 0 or j == 0:
                    continue
                if i not in centres or j not in centres:
                    continue
                x1, y1 = centres[i]
                x2, y2 = centres[j]
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(QPen(QColor(theme.RULE), 1))
                line.setZValue(-3)
                self._scene.addItem(line)

    def set_state(self, addr: tuple[int, int], state: str) -> None:
        rect = self._items.get(addr)
        if rect is None:
            return
        if addr in self._crowns and state in ("undiscovered", "discovered"):
            state = "crown"
        fill, border, text_colour = theme.HOST_STATE[state]
        rect.setBrush(QBrush(QColor(fill)))
        rect.setPen(QPen(QColor(border), 2 if state in ("root", "crown") else 1))
        text = self._labels.get(addr)
        if text is not None:
            text.setBrush(QBrush(QColor(text_colour)))

    def reset_states(self) -> None:
        for addr in self._items:
            self.set_state(addr, "undiscovered")

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._scene.items():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ReplayView(QWidget):
    def __init__(self, repository: Repository) -> None:
        super().__init__()
        self.repository = repository
        self._steps: list[StepRow] = []
        self._episodes: list[EpisodeRow] = []
        self._index = 0
        self._run_name = ""

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SP_L, theme.SP_L, theme.SP_L, theme.SP_L)
        outer.setSpacing(theme.SP_M)

        head = QVBoxLayout()
        head.setSpacing(2)
        head.addWidget(label("Replay", "viewTitle"))
        head.addWidget(label("QGraphicsScene, transport, step detail", "viewSubtitle"))
        outer.addLayout(head)

        outer.addLayout(self._build_toolbar())

        self.banner_slot = QVBoxLayout()
        outer.addLayout(self.banner_slot)

        body = QHBoxLayout()
        body.setSpacing(theme.SP_M)
        self.canvas = NetworkCanvas()
        self.canvas.setMinimumHeight(260)
        body.addWidget(self.canvas, 1)
        body.addWidget(self._build_detail())
        outer.addLayout(body, 1)

        outer.addLayout(self._build_transport())

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.SP_S)
        self.run_label = label("No run loaded", "panelHeading")
        row.addWidget(self.run_label)

        self.episode_box = QComboBox()
        self.episode_box.setFixedWidth(200)
        self.episode_box.currentIndexChanged.connect(self._on_episode_change)
        row.addWidget(self.episode_box)
        row.addStretch(1)

        for state, text in (
            ("undiscovered", "undiscovered"), ("discovered", "discovered"),
            ("user", "user"), ("root", "root"), ("crown", "crown jewel"),
        ):
            fill, border, _ = theme.HOST_STATE[state]
            chip = label(text, "sectionLabel")
            chip.setStyleSheet(
                f"background:{fill}; border:1px solid {border}; padding:2px 7px; "
                f"color:{theme.TEXT_SECONDARY}; font-family:'{theme.FONT_MONO}'; "
                f"font-size:{theme.SIZE_LABEL}px;"
            )
            row.addWidget(chip)
        return row

    def _build_detail(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(theme.DETAIL_WIDTH)
        self.detail_layout = QVBoxLayout(panel)
        self.detail_layout.setContentsMargins(
            theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD, theme.PANEL_PAD
        )
        self.detail_layout.setSpacing(theme.SP_S)
        self.detail_layout.addStretch(1)
        return panel

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.SP_S)

        self.step_label = label("step 0/0", "mono")
        row.addWidget(self.step_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._seek)
        row.addWidget(self.slider, 1)

        # ASCII-safe glyphs: IBM Plex has no U+23EE/U+23ED, so the media-control
        # codepoints render as tofu boxes.
        for text, slot in (
            ("|◀", self._first), ("◀", self._back),
            ("▶", self._toggle_play), ("▶|", self._last),
        ):
            button = QPushButton(text)
            button.setFixedWidth(44)
            button.clicked.connect(slot)
            row.addWidget(button)
            if text == "▶":
                self.play_button = button

        self.speed_box = QComboBox()
        self.speed_box.addItems(["1×", "2×", "4×", "8×"])
        self.speed_box.setFixedWidth(70)
        row.addWidget(self.speed_box)

        self.cumulative_label = label("", "mono")
        row.addWidget(self.cumulative_label)
        return row

    # -- loading ---------------------------------------------------------

    def load_run(self, run_name: str) -> None:
        self._run_name = run_name
        self.run_label.setText(run_name)
        self._clear_banners()
        run_async(lambda: self._fetch(run_name), self._on_loaded, self._on_error)

    def _fetch(self, run_name: str):
        runs = [r for r in self.repository.list_runs() if r.name == run_name]
        if not runs:
            return None, [], TopologyView()
        run = runs[0]
        episodes = self.repository.replayable_episodes(run.experiment_id)
        return run, episodes, runs_folder.load_topology(run_name)

    def _on_loaded(self, payload) -> None:
        run, episodes, topology = payload
        self._episodes = episodes
        self.canvas.build(topology)

        if not episodes:
            self._show_banner(
                Banner(
                    "This run has no step data",
                    f"{self._run_name} was trained without --log-steps, so the "
                    "steps table holds nothing to replay.",
                    kind="warn",
                )
            )
            self.episode_box.clear()
            return

        if not topology.has_structure:
            self._show_banner(
                Banner(
                    "Adjacency matrix missing from config.snapshot.json",
                    "Showing a grouped layout without edges. Runs created after "
                    "the topology-persistence fix draw full edges.",
                    kind="warn",
                )
            )

        self.episode_box.blockSignals(True)
        self.episode_box.clear()
        for episode in episodes:
            mark = "goal" if episode.goal_reached else "limit"
            self.episode_box.addItem(
                f"episode {episode.episode_idx} · {episode.length} steps · {mark}",
                episode.episode_id,
            )
        self.episode_box.blockSignals(False)
        self._on_episode_change(0)

    def _on_episode_change(self, index: int) -> None:
        if index < 0 or index >= len(self._episodes):
            return
        episode_id = self._episodes[index].episode_id
        run_async(
            lambda: self.repository.steps(episode_id), self._on_steps, self._on_error
        )

    def _on_steps(self, steps: list[StepRow]) -> None:
        self._steps = steps
        self.slider.setMaximum(max(0, len(steps) - 1))
        self._seek(0)

    def _on_error(self, message: str, detail: str) -> None:
        self._show_banner(Banner(message, "", kind="error", detail=detail))

    # -- transport -------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("▶")
        else:
            speed = int(self.speed_box.currentText().rstrip("×"))
            self._timer.start(max(30, 300 // speed))
            self.play_button.setText("⏸")

    def _advance(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._timer.stop()
            self.play_button.setText("▶")
            return
        self.slider.setValue(self._index + 1)

    def _first(self) -> None:
        self.slider.setValue(0)

    def _last(self) -> None:
        self.slider.setValue(max(0, len(self._steps) - 1))

    def _back(self) -> None:
        self.slider.setValue(max(0, self._index - 1))

    def _seek(self, index: int) -> None:
        if not self._steps:
            self.step_label.setText("step 0/0")
            return
        index = max(0, min(index, len(self._steps) - 1))
        self._index = index

        # Replay host states from the start so scrubbing backwards is correct.
        self.canvas.reset_states()
        cumulative = 0.0
        for step in self._steps[: index + 1]:
            cumulative += step.native_reward
            addr = step.target
            if addr is None:
                continue
            if step.success:
                if step.action_kind in ("exploit", "privesc"):
                    state = "root" if step.action_kind == "privesc" else "user"
                    self.canvas.set_state(addr, state)
                else:
                    self.canvas.set_state(addr, "discovered")

        self.step_label.setText(f"step {index + 1}/{len(self._steps)}")
        self.cumulative_label.setText(
            f"cumulative native {theme.fmt_num(cumulative, 0)}"
        )
        self._show_step(self._steps[index])

    # -- step detail -----------------------------------------------------

    def _show_step(self, step: StepRow) -> None:
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        add = self.detail_layout.addWidget

        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(label(f"Step {step.step_idx}", "panelHeading"))
        row.addStretch(1)
        # Failure is a word plus a coloured numeral, not a red panel.
        verdict = label("success" if step.success else "failed", "sectionLabel")
        verdict.setStyleSheet(
            f"color:{theme.OK_TEXT if step.success else theme.ERROR_TEXT}; "
            f"font-family:'{theme.FONT_MONO}'; font-size:{theme.SIZE_LABEL}px;"
        )
        row.addWidget(verdict)
        add(header)
        add(hline())

        target = f"({step.target_subnet},{step.target_host})" if step.target else "—"
        for name, value in (
            ("action", step.action_kind),
            ("name", step.action_name),
            ("target", target),
            ("tactic", step.tactic or "—"),
            ("technique", step.technique_id or "—"),
        ):
            add(self._kv(name, value))

        add(hline())
        add(self._kv("native reward", theme.fmt_num(step.native_reward, 1),
                     colour=theme.OK_TEXT if step.native_reward > 0 else
                     theme.ERROR_TEXT if step.native_reward < 0 else None))
        add(self._kv("shaped reward", theme.fmt_num(step.reward, 2)))

        if step.cve_id:
            add(hline())
            add(section_label("Vulnerability"))
            severity_colour = theme.SEVERITY.get(step.severity, theme.TEXT_SECONDARY)
            chip = label(f"{step.cve_id}  {step.cvss_base}  {step.severity}", "mono")
            chip.setStyleSheet(
                f"color:{severity_colour}; font-family:'{theme.FONT_MONO}'; "
                f"font-size:{theme.SIZE_MONO}px; background:transparent;"
            )
            add(chip)
            detail = runs_folder.load_cve(step.cve_id)
            if detail is not None:
                if detail.note:
                    add(label(detail.note, "muted", wrap=True))
                add(self._kv("vector", detail.vector.replace("CVSS:3.1/", "")))
                add(self._kv("cwe", detail.cwe or "—"))

        self.detail_layout.addStretch(1)

    def _kv(self, name: str, value: str, colour: str | None = None) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(label(name, "muted"))
        row.addStretch(1)
        value_label = label(value, "mono")
        if colour:
            value_label.setStyleSheet(
                f"color:{colour}; font-family:'{theme.FONT_MONO}'; "
                f"font-size:{theme.SIZE_MONO}px; background:transparent;"
            )
        row.addWidget(value_label)
        return holder

    def _show_banner(self, banner: Banner) -> None:
        self._clear_banners()
        self.banner_slot.addWidget(banner)

    def _clear_banners(self) -> None:
        while self.banner_slot.count():
            item = self.banner_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
