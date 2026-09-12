from __future__ import annotations

import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QGraphicsSimpleTextItem, QLabel  # noqa: E402

from gui.backend import CampaignData, DashboardData, SimulationData  # noqa: E402
from gui.data.models import StudyMetric, StudySummary  # noqa: E402
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
        nodes=[
            {"id": "entry", "type": "entry_point", "name": "External", "attributes": {}},
            {"id": "host", "type": "host", "name": "Host", "attributes": {}},
            {"id": "asset", "type": "asset", "name": "Data", "attributes": {}},
        ],
        edges=[
            {"source": "entry", "target": "host", "type": "connects"},
            {"source": "host", "target": "asset", "type": "contains"},
        ],
        trajectory=[
            {
                "step": 0,
                "action": "discover:entry",
                "action_kind": "discover_network",
                "target": "entry",
            },
            {
                "step": 1,
                "action": "pivot:host",
                "action_kind": "pivot",
                "target": "host",
            },
            {
                "step": 2,
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


def test_all_six_desktop_workspaces_navigate() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=FakeBackend())

    assert window.stack.count() == 6
    assert len(window.nav_buttons) == 6
    for index, nav_button in enumerate(window.nav_buttons):
        nav_button.click()
        app.processEvents()
        assert window.stack.currentIndex() == index
        assert nav_button.isChecked()

    labels = [item.text() for item in window.findChildren(QLabel)]
    assert "SIMULATION BOUNDARY" in labels
    assert "Offline · no live exploitation" in labels
    window.close()


def test_simulation_page_renders_backend_result_without_hardcoded_profile() -> None:
    app = QApplication.instance() or QApplication([])
    page = SimulationPage(FakeBackend(), lambda message: None)
    page._profiles_loaded([{"id": "hybrid", "label": "Hybrid estate"}])
    page._completed(simulation_result())
    app.processEvents()

    assert page.profile.currentData() == "hybrid"
    assert page.nodes.rowCount() == 3
    assert page.graph.node_count == 3
    assert page.graph.edge_count == 2
    assert page.graph.highlighted_entities == {"entry", "host", "asset"}
    assert page.graph.final_entity == "asset"
    entry_labels = [
        item
        for item in page.graph.graph_scene.items()
        if isinstance(item, QGraphicsSimpleTextItem) and item.text() == "entry"
    ]
    assert len(entry_labels) == 1
    assert entry_labels[0].scenePos().x() >= 28
    assert entry_labels[0].scenePos().y() >= 64
    assert page.outcome.value.text() == "GOAL REACHED"
    assert page.replay_button.isEnabled()
    page.replay_button.click()
    assert page.graph.replay_index == 1
    assert page.graph.highlighted_entities == {"entry"}
    page.graph._advance_replay()
    page.graph._advance_replay()
    assert page.graph.highlighted_entities == {"entry", "host", "asset"}
    assert not page.graph.replay_timer.isActive()
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

    runs_page = window.pages[4]
    assert runs_page.table.item(0, 0).text() == "stored-run-123"
    assert runs_page.table.item(0, 6).text() == "25.0%"
    window.close()


def test_latest_study_drives_overview_and_research_pages() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=FakeBackend())
    study = StudySummary(
        phase=13,
        study_id="multiagent",
        title="Multi-agent red/blue defence",
        arm_a="static",
        arm_b="adaptive",
        complete=True,
        outcome="No significant primary effect",
        code_commit="1234567890",
        config_hash="abc",
        result_path="results/multiagent/test",
        training_seeds=10,
        evaluation_episodes=1200,
        metrics=[
            StudyMetric(
                "detection_rate", 0.2, 0.24, 0.04, 0.2, 0.6, 0.3, False, True
            )
        ],
    )

    window.apply_dashboard(
        DashboardData(source_status="PostgreSQL connected", studies=[study])
    )
    app.processEvents()

    overview = window.pages[0]
    research = window.pages[3]
    assert overview.study_name.text() == "Multi-agent red/blue defence"
    assert overview.episodes.value.text() == "1,200"
    assert overview.metric_table.item(0, 0).text() == "Detection Rate"
    assert research.selector.itemText(0).startswith("Phase 13")
    assert research.table.item(0, 1).text() == "PRIMARY"
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
