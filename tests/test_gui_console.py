from __future__ import annotations

import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from gui.backend import CampaignData, DashboardData  # noqa: E402
from gui.views.main_window import MainWindow  # noqa: E402


class FakeBackend:
    def pause_campaign(self, campaign_id: str) -> None:
        pass

    def resume_campaign(self, campaign_id: str) -> None:
        pass

    def refresh_paths(self) -> list[dict]:
        return []

    def save_agent_config(self, config: dict) -> None:
        pass

    def export_report(self) -> str:
        return "runs/_analysis/results_table.txt"


def test_all_eight_desktop_workspaces_navigate() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=FakeBackend())

    assert window.stack.count() == 8
    assert len(window.nav_buttons) == 8
    for index, nav_button in enumerate(window.nav_buttons):
        nav_button.click()
        app.processEvents()
        assert window.stack.currentIndex() == index
        assert nav_button.isChecked()

    labels = [item.text() for item in window.findChildren(QLabel)]
    assert "✓  SIMULATION BOUNDARY" in labels
    assert "NaSim / CybORG only" in labels
    window.close()


def test_dashboard_values_are_rendered_from_backend_snapshot() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=FakeBackend())
    data = DashboardData(
        source_status="test backend connected",
        run_name="stored-run-123",
        reward_mode="shaped",
        seed="731",
        episodes=19,
        success_rate=0.25,
        campaigns=[
            CampaignData("stored-run-123", "shaped", "complete", 19, 100, "731", -4.5, 0.25)
        ],
    )

    window.apply_dashboard(data)
    app.processEvents()
    labels = [item.text() for item in window.findChildren(QLabel)]

    assert "stored-run-123" in labels
    assert "25.0%" in labels
    assert "EXP-09 · PPO / Large topology" not in labels
    window.close()
