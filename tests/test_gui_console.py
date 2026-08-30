from __future__ import annotations

import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from gui.backend import CampaignData, DashboardData, SimulationData  # noqa: E402
from gui.views.main_window import MainWindow  # noqa: E402
from gui.views.research_console import SimulationPage, TrajectoryGraph  # noqa: E402


class FakeBackend:
    def simulation_profiles(self) -> list[dict]:
        return [{"id": "hybrid", "label": "Hybrid estate"}]

    def run_simulation(self, profile: str, topology_seed: int) -> SimulationData:
        return simulation_result(profile, topology_seed)

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


def simulation_result(profile: str = "hybrid", seed: int = 2001) -> SimulationData:
    return SimulationData(
        profile=profile,
        topology_seed=seed,
        topology_hash="a" * 64,
        topology_name=f"enterprise-{profile}-seed-{seed}",
        nodes=[{"id": "asset", "type": "asset", "name": "Data", "attributes": {}}],
        edges=[],
        trajectory=[
            {
                "step": 1,
                "action": "access_asset:asset",
                "action_kind": "access_asset",
                "target": "asset",
            }
        ],
        goal_reached=True,
        episode_steps=10,
        total_reward=100.0,
        discovery_coverage=1.0,
    )


def test_all_nine_desktop_workspaces_navigate() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=FakeBackend())

    assert window.stack.count() == 9
    assert len(window.nav_buttons) == 9
    for index, nav_button in enumerate(window.nav_buttons):
        nav_button.click()
        app.processEvents()
        assert window.stack.currentIndex() == index
        assert nav_button.isChecked()

    labels = [item.text() for item in window.findChildren(QLabel)]
    assert "✓  SIMULATION BOUNDARY" in labels
    assert "Offline graph / NASim only" in labels
    window.close()


def test_simulation_page_renders_backend_result_without_hardcoded_profile() -> None:
    app = QApplication.instance() or QApplication([])
    page = SimulationPage(FakeBackend(), lambda message: None)
    page._profiles_loaded([{"id": "hybrid", "label": "Hybrid estate"}])
    page._completed(simulation_result())
    app.processEvents()

    assert page.profile.currentData() == "hybrid"
    assert page.nodes.rowCount() == 1
    assert page.graph.steps[-1]["target"] == "asset"
    assert page.metrics.value.text() == "GOAL REACHED"
    page.close()


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


def test_trace_graph_replays_backend_steps_without_html() -> None:
    QApplication.instance() or QApplication([])
    graph = TrajectoryGraph()
    graph.set_steps(
        [
            {"target": "service_entry", "action": "exploit"},
            {"target": "asset_crown", "action": "access_asset"},
        ]
    )
    assert graph.visible_steps == 2
    graph.replay()
    assert graph.visible_steps == 1
    graph._advance()
    assert graph.visible_steps == 2
    assert not graph.timer.isActive()
    graph.close()
