"""Native Qt canvas for a typed enterprise graph.

The widget renders backend-returned nodes and edges.  Policy decisions remain
knowledge-only; this is a post-episode analyst view and the causal highlight is
derived only from the returned trajectory.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from gui import theme

LANE_LABELS = ("NETWORK", "COMPUTE", "SERVICES & IDENTITY", "DATA & ASSETS")
NETWORK_TYPES = {"entry_point", "security_control", "network_segment", "cloud_network"}
COMPUTE_TYPES = {"host", "legacy_host", "cloud_workload", "workstation", "server"}
SERVICE_TYPES = {
    "service",
    "application",
    "api",
    "identity",
    "iam_role",
    "domain_controller",
}
DATA_TYPES = {"storage", "database", "data_store", "asset"}


def lane_for(node_type: str) -> int:
    if node_type in NETWORK_TYPES:
        return 0
    if node_type in COMPUTE_TYPES:
        return 1
    if node_type in SERVICE_TYPES:
        return 2
    if node_type in DATA_TYPES:
        return 3
    return 2


def causal_entities(nodes: list[dict], trajectory: list[dict]) -> tuple[set[str], str | None]:
    """Return node ids evidenced by the causal trace and its final graph node."""
    node_ids = {str(node.get("id", "")) for node in nodes}
    selected: set[str] = set()
    final: str | None = None
    for step in trajectory:
        candidates = [step.get("target"), step.get("target_entity")]
        candidates.extend(str(item).partition(":")[2] for item in step.get("outcomes", []))
        for candidate in candidates:
            value = str(candidate or "")
            if value in node_ids:
                selected.add(value)
                final = value
    return selected, final


class EnterpriseGraph(QGraphicsView):
    """Scrollable, zoomable graph grouped by enterprise entity type."""

    node_selected = Signal(dict)
    edge_selected = Signal(dict)

    NODE_WIDTH = 184
    NODE_HEIGHT = 58
    LANE_WIDTH = 248
    ROW_HEIGHT = 78

    def __init__(self) -> None:
        self.graph_scene = QGraphicsScene()
        super().__init__(self.graph_scene)
        self.setObjectName("EnterpriseGraph")
        self.setMinimumHeight(430)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(theme.PANEL)))
        self.node_count = 0
        self.edge_count = 0
        self.highlighted_entities: set[str] = set()
        self.final_entity: str | None = None
        self._nodes: list[dict] = []
        self._edges: list[dict] = []
        self._trajectory: list[dict] = []
        self._visible_trajectory: list[dict] = []
        self._edge_categories: set[str] = set()
        self.replay_index = 0
        self.replay_timer = QTimer(self)
        self.replay_timer.setInterval(420)
        self.replay_timer.timeout.connect(self._advance_replay)
        self._draw_empty("Run a simulation to build the enterprise graph")

    def set_graph(
        self,
        nodes: list[dict],
        edges: list[dict],
        trajectory: list[dict] | None = None,
    ) -> None:
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._trajectory = list(trajectory or [])
        self._visible_trajectory = list(self._trajectory)
        self.replay_index = len(self._trajectory)
        self.node_count = len(self._nodes)
        self.edge_count = len(self._edges)
        self.highlighted_entities, self.final_entity = causal_entities(
            self._nodes, self._trajectory
        )
        self._render()

    def set_edge_categories(self, categories: set[str] | None) -> None:
        """Show edges matching all requested evidence categories."""
        self._edge_categories = set(categories or ())
        self._render()

    def clear_graph(self, message: str = "No topology is available") -> None:
        self._nodes = []
        self._edges = []
        self._trajectory = []
        self._visible_trajectory = []
        self.replay_timer.stop()
        self.replay_index = 0
        self.node_count = self.edge_count = 0
        self.highlighted_entities = set()
        self.final_entity = None
        self._draw_empty(message)

    def replay(self) -> None:
        if not self._trajectory:
            return
        self.replay_timer.stop()
        self.replay_index = 1
        self._visible_trajectory = self._trajectory[:1]
        self.highlighted_entities, self.final_entity = causal_entities(
            self._nodes, self._visible_trajectory
        )
        self._render()
        if len(self._trajectory) > 1:
            self.replay_timer.start()

    def _advance_replay(self) -> None:
        self.replay_index += 1
        self._visible_trajectory = self._trajectory[: self.replay_index]
        self.highlighted_entities, self.final_entity = causal_entities(
            self._nodes, self._visible_trajectory
        )
        self._render()
        if self.replay_index >= len(self._trajectory):
            self.replay_timer.stop()

    def _draw_empty(self, message: str) -> None:
        self.graph_scene.clear()
        self.graph_scene.setSceneRect(QRectF(0, 0, 900, 420))
        item = self.graph_scene.addSimpleText(message)
        item.setBrush(QBrush(QColor(theme.TEXT_SECONDARY)))
        item.setPos(300, 195)

    def _render(self) -> None:
        self.graph_scene.clear()
        if not self._nodes:
            self._draw_empty("The backend returned no topology entities")
            return

        lanes: dict[int, list[dict]] = {index: [] for index in range(4)}
        for node in self._nodes:
            lanes[lane_for(str(node.get("type", "")))].append(node)
        for nodes in lanes.values():
            nodes.sort(
                key=lambda item: (
                    str(item.get("id", "")) not in self.highlighted_entities,
                    str(item.get("type", "")),
                    str(item.get("id", "")),
                )
            )

        max_rows = max(len(nodes) for nodes in lanes.values())
        scene_height = max(520, 86 + max_rows * self.ROW_HEIGHT)
        scene_width = 4 * self.LANE_WIDTH + 50
        self.graph_scene.setSceneRect(QRectF(0, 0, scene_width, scene_height))

        positions: dict[str, QPointF] = {}
        for lane, nodes in lanes.items():
            x = 28 + lane * self.LANE_WIDTH
            heading = self.graph_scene.addSimpleText(LANE_LABELS[lane])
            heading.setBrush(QBrush(QColor(theme.TEXT_SECONDARY)))
            heading.setPos(x, 18)
            separator = self.graph_scene.addLine(
                x - 10,
                48,
                x - 10,
                scene_height - 24,
                QPen(QColor(theme.RULE), 1),
            )
            separator.setZValue(-3)
            for row, node in enumerate(nodes):
                positions[str(node.get("id", ""))] = QPointF(
                    x, 64 + row * self.ROW_HEIGHT
                )

        path_edges = {
            (str(left), str(right))
            for left, right in zip(
                [step.get("target") for step in self._visible_trajectory],
                [step.get("target") for step in self._visible_trajectory[1:]],
                strict=False,
            )
        }
        for edge in self._edges:
            categories = set(map(str, edge.get("categories", ())))
            if self._edge_categories and not self._edge_categories <= categories:
                continue
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in positions or target not in positions:
                continue
            start = positions[source] + QPointF(self.NODE_WIDTH, self.NODE_HEIGHT / 2)
            end = positions[target] + QPointF(0, self.NODE_HEIGHT / 2)
            if end.x() < start.x():
                start = positions[source] + QPointF(0, self.NODE_HEIGHT / 2)
                end = positions[target] + QPointF(self.NODE_WIDTH, self.NODE_HEIGHT / 2)
            span = max(40.0, abs(end.x() - start.x()) * 0.45)
            curve = QPainterPath(start)
            curve.cubicTo(
                QPointF(start.x() + span, start.y()),
                QPointF(end.x() - span, end.y()),
                end,
            )
            active = (source, target) in path_edges or (
                source in self.highlighted_entities and target in self.highlighted_entities
            )
            item = QGraphicsPathItem(curve)
            item.setPen(
                QPen(
                    QColor(theme.WARN if active else theme.BORDER_SOFT),
                    2 if active else 1,
                )
            )
            item.setToolTip(str(edge.get("type", "relationship")))
            item.setData(0, "edge")
            item.setData(1, edge)
            item.setZValue(-2 if active else -3)
            self.graph_scene.addItem(item)
            relationship = str(edge.get("type", "unknown"))
            edge_label = QGraphicsSimpleTextItem(relationship)
            edge_label.setBrush(QBrush(QColor(theme.TEXT_SECONDARY)))
            edge_label.setPos((start.x() + end.x()) / 2, (start.y() + end.y()) / 2 - 16)
            edge_label.setToolTip(json.dumps(edge, indent=2, sort_keys=True))
            edge_label.setData(0, "edge")
            edge_label.setData(1, edge)
            self.graph_scene.addItem(edge_label)

        for node in self._nodes:
            node_id = str(node.get("id", ""))
            position = positions[node_id]
            final = node_id == self.final_entity
            active = node_id in self.highlighted_entities
            border = theme.ERROR if final else theme.ARM_1 if active else theme.BORDER_SOFT
            fill = theme.TINT_ERROR if final else theme.HEADER if active else theme.SURFACE
            card = QGraphicsRectItem(QRectF(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT))
            card.setPos(position)
            card.setBrush(QBrush(QColor(fill)))
            card.setPen(QPen(QColor(border), 2 if active or final else 1))
            card.setToolTip(
                json.dumps(node.get("attributes", {}), indent=2, sort_keys=True)
                or "No attributes"
            )
            card.setData(0, "node")
            card.setData(1, node)
            self.graph_scene.addItem(card)

            title = QGraphicsSimpleTextItem(node_id, card)
            title.setBrush(QBrush(QColor(theme.TEXT)))
            title.setPos(10, 8)
            subtitle = QGraphicsSimpleTextItem(
                str(node.get("type", "entity")).replace("_", " "), card
            )
            subtitle.setBrush(QBrush(QColor(theme.TEXT_SECONDARY)))
            subtitle.setPos(10, 31)

        self.resetTransform()
        available_width = max(1, self.viewport().width() - 20)
        scale = min(1.0, available_width / scene_width)
        if scale < 1.0:
            self.scale(scale, scale)
        self.verticalScrollBar().setValue(0)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current = self.transform().m11()
        if 0.35 <= current * factor <= 2.5:
            self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        item = self.itemAt(event.position().toPoint())
        while item is not None and item.data(0) is None:
            item = item.parentItem()
        if item is not None:
            kind, payload = item.data(0), item.data(1)
            if kind == "node" and isinstance(payload, dict):
                self.node_selected.emit(payload)
            elif kind == "edge" and isinstance(payload, dict):
                self.edge_selected.emit(payload)
        super().mousePressEvent(event)
