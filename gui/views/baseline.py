"""Native read-only presentation of reviewed baseline evidence."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPlainTextEdit,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.data.baseline import BaselineData
from gui.views.research_console import (
    Metric,
    Page,
    StateBanner,
    button,
    configure_table,
    fill_table,
    label,
    metric_name,
    number,
    percent,
)


class BaselinePage(Page):
    def __init__(self, notify) -> None:
        super().__init__(
            "REVIEWED FROZEN EVIDENCE",
            "Frozen baseline",
            "Recorded policy outcomes, limitations and supervisor handoff. "
            "Read-only: no training, correction or result generation.",
        )
        self.notify = notify
        self.data = BaselineData()
        self.banner = StateBanner()
        self.root.addWidget(self.banner)
        cards = QHBoxLayout()
        self.pairs = Metric("TRAINED-POLICY PAIRS")
        self.episodes = Metric("EVALUATION EPISODES")
        self.signals = Metric("SIGNIFICANT PRIMARIES")
        for card in (self.pairs, self.episodes, self.signals):
            cards.addWidget(card)
        self.root.addLayout(cards)
        self.warning = label("", "Muted", wrap=True)
        self.warning.setTextFormat(Qt.TextFormat.PlainText)
        self.root.addWidget(self.warning)
        tabs = QTabWidget()
        self.root.addWidget(tabs)

        def tab(title):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            tabs.addTab(widget, title)
            return layout

        def table(layout, headers):
            item = QTableWidget(0, len(headers))
            item.setHorizontalHeaderLabels(headers)
            configure_table(item)
            item.setMinimumHeight(250)
            layout.addWidget(item)
            return item

        outcomes = tab("Evaluation")
        self.outcomes = table(outcomes, ["Metric", "Sparse", "Shaped", "Interpretation"])
        outcomes.addWidget(
            label(
                "Native outcomes describe the frozen implementation, "
                "not a corrected shaping algorithm. "
                "Discovery counts and discovery-based path length are WITHHELD. "
                "MITRE coverage describes attempted semantic actions, "
                "not real-world technique success.",
                "Muted",
                wrap=True,
            )
        )
        stats = tab("Statistics")
        self.statistics = table(
            stats,
            [
                "Primary metric",
                "n",
                "Difference (shaped − sparse)",
                "Raw p",
                "Bonferroni p",
                "Marginal 95% CI",
                "Verdict",
            ],
        )
        stats.addWidget(
            label(
                "Policy pairs, not nested episodes, are the independent unit. "
                "Registered paired t-test; five-comparison Bonferroni family. "
                "Bootstrap intervals are marginal, not multiplicity-adjusted. "
                "All-success p=1 / [0,0] is a zero-variance reporting convention, not equivalence. "
                "Descriptive differences do not establish shaping superiority.",
                "Muted",
                wrap=True,
            )
        )
        training = tab("Convergence")
        self.training = table(
            training,
            ["Condition", "Training seeds", "Timesteps per policy", "Passed", "Assessment"],
        )
        training.addWidget(
            label(
                "Sparse block stability is not the shaped convergence gate. "
                "Gate passage does not establish optimality or correct reward semantics. "
                "Normalization settings are recorded separately under Provenance.",
                "Muted",
                wrap=True,
            )
        )
        provenance = tab("Provenance")
        self.provenance = QPlainTextEdit()
        self.provenance.setReadOnly(True)
        self.provenance.setMinimumHeight(360)
        provenance.addWidget(self.provenance)
        documents = tab("Supervisor files")
        self.documents = table(documents, ["Document", "Availability", "Repository-relative path"])
        self.copy_button = button("Copy selected host file path")
        self.copy_button.clicked.connect(self.copy_document_path)
        self.documents.itemSelectionChanged.connect(self._document_selected)
        documents.addWidget(self.copy_button)
        documents.addWidget(
            label(
                "Files stay local. Copy the host path and open it in your desktop Word/Excel "
                "application. No file is executed, regenerated or uploaded by this page. "
                "A: preserve the frozen baseline with disclosed limitation. "
                "B: separately version and rerun corrected shaping. "
                "Neither option is recommended; approval is not inferred from this screen.",
                "Muted",
                wrap=True,
            )
        )
        self.apply_baseline(BaselineData())

    def apply(self, data) -> None:
        self.apply_baseline(data.baseline)

    def apply_baseline(self, data: BaselineData) -> None:
        self.data = data
        self.banner.update_state(data.state, data.detail, "warn")
        self.warning.setText(data.warning or "No reviewed comparison to display.")
        self.pairs.update_value(
            str(data.policy_pairs) if data.summary else "—", "Independent paired unit"
        )
        self.episodes.update_value(
            f"{data.total_episodes:,}" if data.summary else "—",
            "Held-out episode seeds; one fixed topology",
        )
        significant = sum(row["significant"] for row in data.comparisons)
        self.signals.update_value(
            str(significant) if data.summary else "—",
            f"of {len(data.comparisons)} registered primaries",
        )
        rows = []
        if data.state == "REVIEWED":
            for metric in (
                "success_rate",
                "steps_to_goal",
                "native_return",
                "native_return_std",
                "failed_actions",
                "mean_cvss_exploited",
                "mitre_technique_count",
            ):
                fmt = percent if metric == "success_rate" else lambda value: number(value, 4)
                rows.append(
                    (
                        metric_name(metric),
                        fmt(data.summary["sparse"].get(metric)),
                        fmt(data.summary["shaped"].get(metric)),
                        "Descriptive only"
                        if metric in {"failed_actions", "mitre_technique_count"}
                        else "Recorded frozen-policy outcome",
                    )
                )
            rows.extend(
                (name, "WITHHELD", "WITHHELD", "Discovery-counting defect")
                for name in ("Discovered hosts / coverage", "Discovery-based path length")
            )
        fill_table(self.outcomes, rows)
        fill_table(
            self.statistics,
            [
                (
                    metric_name(row["metric"]),
                    row["n_pairs"],
                    number(row["difference"], 5),
                    number(row["p_value"], 6),
                    number(row["p_bonferroni"], 6),
                    f"[{number(row['ci_low'], 5)}, {number(row['ci_high'], 5)}]",
                    "SIGNIFICANT" if row["significant"] else "NOT SIGNIFICANT",
                )
                for row in data.comparisons
            ],
        )
        fill_table(self.training, data.training)
        self.provenance.setPlainText(
            "\n".join(f"{key}\n{value}\n" for key, value in data.provenance)
            + (f"\n{data.interpretation}" if data.interpretation else "")
        )
        fill_table(
            self.documents,
            [
                (title, "Available" if exists else "Not mounted", path)
                for title, path, exists in data.documents
            ],
        )
        self._document_selected()

    def _document_selected(self) -> None:
        index = self.documents.currentRow()
        self.copy_button.setEnabled(
            0 <= index < len(self.data.documents) and self.data.documents[index][2]
        )

    def copy_document_path(self) -> None:
        index = self.documents.currentRow()
        if not 0 <= index < len(self.data.documents) or not self.data.documents[index][2]:
            return
        root = Path(os.environ.get("RLREDTEAM_HOST_REPO") or Path(__file__).resolve().parents[2])
        path = str(root / self.data.documents[index][1])
        QApplication.clipboard().setText(path)
        self.notify(f"Copied local file path: {path}")
