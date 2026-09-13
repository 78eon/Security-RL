"""Native, event-driven MITRE ATT&CK and ATLAS matrix widgets."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rlredteam.frameworks import (
    Framework,
    TechniqueMapping,
    TechniqueProgress,
    framework_catalog,
    technique_progress,
)


def _label(text: str, name: str = "") -> QLabel:
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(True)
    return item


class TechniqueCell(QFrame):
    def __init__(self, mapping: TechniqueMapping) -> None:
        super().__init__()
        self.mapping = mapping
        self.setObjectName("TechniqueCell")
        self.setProperty("state", "inactive")
        self.setToolTip(
            f"{mapping.technique_id} · {mapping.technique_name}\n{mapping.source_url}"
        )
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(_label(mapping.technique_id, "TechniqueId"))
        row.addStretch()
        self.count = _label("", "TechniqueCount")
        row.addWidget(self.count)
        box.addLayout(row)
        box.addWidget(_label(mapping.technique_name, "TechniqueName"))

    def update_progress(self, progress: TechniqueProgress | None) -> None:
        if progress is None:
            state = "inactive"
            count = ""
        elif progress.current:
            state = "current"
            count = str(progress.attempts)
        elif progress.progressed:
            state = "progressed"
            count = str(progress.attempts)
        else:
            state = "attempted"
            count = str(progress.attempts)
        self.setProperty("state", state)
        self.count.setText(count)
        self.style().unpolish(self)
        self.style().polish(self)


class TechniqueMatrix(QWidget):
    """Navigator-style tactic columns driven exclusively by event mappings."""

    def __init__(self, framework: Framework) -> None:
        super().__init__()
        self.framework = framework
        self.cells: dict[str, TechniqueCell] = {}
        self.active_techniques: set[str] = set()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.summary = _label("No mapped behavior observed", "MatrixSummary")
        root.addWidget(self.summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        canvas = QWidget()
        columns = QHBoxLayout(canvas)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(8)

        grouped: dict[tuple[str, str], list[TechniqueMapping]] = defaultdict(list)
        for mapping in framework_catalog(framework):
            grouped[(mapping.tactic_id, mapping.tactic_name)].append(mapping)
        for (tactic_id, tactic_name), mappings in grouped.items():
            column = QFrame()
            column.setObjectName("TacticColumn")
            column.setMinimumWidth(210)
            box = QVBoxLayout(column)
            box.setContentsMargins(10, 10, 10, 10)
            box.setSpacing(7)
            box.addWidget(_label(tactic_id, "TacticId"))
            box.addWidget(_label(tactic_name, "TacticName"))
            for mapping in mappings:
                cell = TechniqueCell(mapping)
                self.cells[mapping.technique_id] = cell
                box.addWidget(cell)
            box.addStretch()
            columns.addWidget(column)
        columns.addStretch()
        scroll.setWidget(canvas)
        root.addWidget(scroll)

    def set_events(self, events: list[dict], visible_count: int | None = None) -> None:
        visible = events if visible_count is None else events[:visible_count]
        progress = technique_progress(visible, self.framework)
        self.active_techniques = set(progress)
        for technique_id, cell in self.cells.items():
            cell.update_progress(progress.get(technique_id))
        progressed = sum(item.progressed > 0 for item in progress.values())
        attempts = sum(item.attempts for item in progress.values())
        if attempts:
            self.summary.setText(
                f"{len(progress)} techniques observed · {progressed} progressed · "
                f"{attempts} mapped events"
            )
        elif self.framework is Framework.ATLAS:
            self.summary.setText(
                "No AI-targeting behavior observed · RL control alone is not ATLAS"
            )
        else:
            self.summary.setText("No mapped ATT&CK behavior observed at this replay step")


class MitreNavigator(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("MitreNavigator")
        self.attack = TechniqueMatrix(Framework.ATTACK_ENTERPRISE)
        self.atlas = TechniqueMatrix(Framework.ATLAS)
        self.addTab(self.attack, "ATT&CK Enterprise")
        self.addTab(self.atlas, "ATLAS")

    def set_events(self, events: list[dict], visible_count: int | None = None) -> None:
        self.attack.set_events(events, visible_count)
        self.atlas.set_events(events, visible_count)
