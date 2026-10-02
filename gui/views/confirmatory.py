"""Read-only, database-independent confirmatory study presentation."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressBar, QTableWidget, QTabWidget, QVBoxLayout, QWidget

from gui.data.confirmatory import METRICS
from gui.data.models import ConfirmatoryStudySummary
from gui.views.research_console import Page, StateBanner, configure_table, fill_table, label, number


class ConfirmatoryPage(Page):
    def __init__(self):
        super().__init__(
            "SEPARATELY VERSIONED EVIDENCE",
            "Corrected Confirmatory Study",
            "Read-only persisted evidence. Frozen Baseline remains separate and unchanged. "
            "Results will be reported regardless of direction.",
        )
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        tabs = QTabWidget()
        self.root.addWidget(tabs)

        def tab(title):
            widget = QWidget()
            tabs.addTab(widget, title)
            return QVBoxLayout(widget)

        def table(layout, headers):
            widget = QTableWidget(0, len(headers))
            widget.setHorizontalHeaderLabels(headers)
            configure_table(widget)
            widget.setMinimumHeight(300)
            layout.addWidget(widget)
            return widget

        status = tab("Study status / progress")
        self.status = table(status, ["Field", "Recorded value"])
        self.progress = QProgressBar()
        status.addWidget(self.progress)
        self.progress_note = label("", "Muted", wrap=True)
        status.addWidget(self.progress_note)
        audit = tab("Discovery-counting defect audit")
        self.audit = table(audit, ["Audit quantity", "Recorded value"])
        self.audit_note = label("", "Muted", wrap=True)
        self.audit_note.setTextFormat(Qt.TextFormat.PlainText)
        audit.addWidget(self.audit_note)
        comparison = tab("Corrected vs Frozen")
        self.comparison_note = label("", "Muted", wrap=True)
        self.comparison_note.setTextFormat(Qt.TextFormat.PlainText)
        comparison.addWidget(self.comparison_note)
        self.comparison = table(comparison, ["Arm", "Metric", "Frozen", "Corrected", "Difference"])
        comparison.addWidget(
            label(
                "Descriptive corrected-vs-frozen comparison only. Discovery-derived baseline "
                "metrics are withheld. No equivalence or superiority is inferred.",
                "Muted",
                wrap=True,
            )
        )
        self.apply_confirmatory(ConfirmatoryStudySummary())

    def apply_confirmatory(self, data):
        self.data = data
        self.banner.update_state(data.stage, data.status_message, "warn")

        def seeds(values):
            return ", ".join(map(str, values)) or "Not yet available"

        fill_table(
            self.status,
            [
                ("Study ID", data.study_id),
                ("Stage", data.stage),
                ("Registered", "Yes" if data.registered else "No"),
                ("Corrected source commit", data.corrected_commit),
                ("Parent frozen commit", data.parent_commit),
                ("Training seeds", seeds(data.training_seeds)),
                ("Evaluation seeds", seeds(data.evaluation_seeds)),
                ("Requested timesteps per policy", number(data.total_timesteps, 0)),
                ("Convergence", data.convergence_status),
                ("Evaluation", data.evaluation_status),
            ],
        )
        self.progress.setRange(0, max(1, len(data.training_seeds)))
        self.progress.setValue(len(data.completed_training_seeds))
        self.progress.setFormat(
            "%v / %m recorded shaped completions" if data.training_seeds else "Not yet available"
        )
        self.progress_note.setText(
            "Verified completed seeds: " + seeds(data.completed_training_seeds)
        )
        audit = data.audit
        rows = []
        if audit.evidence_available:
            rows = [
                ("Confirmed affected payments", number(audit.affected_scan_count, 0)),
                ("Affected reward points", number(audit.affected_reward, 0)),
                ("Total positive shaping reward", number(audit.total_positive_reward, 3)),
                ("Share of positive shaping reward", f"{number(audit.affected_share_pct, 4)}%"),
                ("Share of tactic reward", f"{number(audit.tactic_share_pct, 3)}%"),
            ]
        fill_table(self.audit, rows)
        self.audit_note.setText(audit.interpretation + "\n\n" + audit.source)
        self.comparison_note.setText(
            "Descriptive corrected-vs-frozen comparison\n" + data.comparison_path
            if data.comparison_available
            else "Confirmatory comparison not yet available."
        )
        fill_table(
            self.comparison,
            [
                (
                    row.arm,
                    row.metric,
                    number(row.frozen, 5),
                    number(row.corrected, 5),
                    number(row.corrected_minus_frozen, 5),
                )
                for row in data.comparisons
                if row.metric in METRICS
            ]
            if data.comparison_available
            else [],
        )
