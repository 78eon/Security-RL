"""Workspace presentation tests: synthetic evidence; no research execution."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from gui.data.models import (
    AgentKnowledgeView,
    AttackTrajectorySummary,
    ComparisonOption,
    ExperimentReport,
    ResearchEvidenceSummary,
    ScenarioSummary,
    WorkspaceEvidence,
)
from gui.data.workspace import (
    DIAGNOSTICS,
    METRICS,
    compare_options,
    knowledge_view,
    load_workspace,
    load_workspace_episode,
    simulation_workspace,
    timeline_entries,
)


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def events():
    return [
        {
            "step": 0,
            "action": "service_scan",
            "target": "known",
            "success": True,
            "knowledge": {
                "nodes": ["known"],
                "known_node_types": {"known": "host"},
                "known_services": {"known": ["https"]},
                "access": {"known": "none"},
            },
            "native_reward": -1.0,
            "shaped_reward": 0.5,
        },
        {
            "step": 1,
            "action": "exploit_https",
            "action_kind": "exploit",
            "target": "known",
            "success": True,
            "cve_id": "CVE-2024-1234",
            "cvss_base": 8.1,
            "access_gained": 1,
            "native_reward": 5.0,
            "shaped_reward": 8.0,
            "framework_mappings": [
                {
                    "tactic_id": "TA0008",
                    "tactic_name": "Lateral Movement",
                    "technique_id": "T1210",
                    "technique_name": "Recorded mapping",
                }
            ],
            "knowledge": {
                "nodes": ["known", "later"],
                "access": {"known": "user"},
                "known_vulnerabilities": ["CVE-2024-1234"],
                "edges": [{"source": "known", "target": "later", "type": "REVEALED"}],
            },
            "knowledge_delta": {"discovered_nodes": ["later"]},
        },
    ]


def trajectory():
    return AttackTrajectorySummary(
        "Synthetic recorded episode",
        timeline_entries(events()),
        ExperimentReport(title="Synthetic", outcome="SUCCESS"),
        "Test fixture",
    )


def test_knowledge_projection_has_no_truth_join_or_future_discoveries():
    records = events()
    records[0]["knowledge"]["true_topology"] = {"hidden": "SECRET"}
    records[0]["knowledge"]["attributes"] = {"known": {"secret": "SECRET"}}
    records[0]["knowledge"]["edges"] = [{"source": "known", "target": "hidden", "type": "CONNECTS"}]
    entries = timeline_entries(records)
    assert "SECRET" not in json.dumps(asdict(entries[0]))
    assert "later" not in json.dumps(asdict(entries[0]))
    assert entries[0].before.nodes == ()
    assert entries[1].before == entries[0].after
    assert entries[0].after.edges == ()
    assert entries[0].after.nodes[0]["attributes"]["services"] == ["https"]


def test_legacy_discovery_counts_withheld_not_converted_to_knowledge():
    entry = timeline_entries(
        [
            {
                "step": 0,
                "target": "attempted_hidden",
                "newly_discovered": 8,
                "cve_id": "CVE-2024-1234",
            }
        ],
        legacy=True,
    )[0]
    assert entry.before == AgentKnowledgeView() and entry.after == AgentKnowledgeView()
    assert entry.action.newly_discovered is None
    assert "Withheld" in entry.action.knowledge_delta
    assert not entry.after.vulnerabilities  # action metadata is not an observed CVE inventory


def test_unknown_reward_and_action_mask_not_invented():
    entry = timeline_entries([{"step": 0, "reward": 7.0, "knowledge": {"nodes": []}}])[0]
    assert entry.action.recorded_reward == 7.0
    assert entry.action.native_reward is None and entry.action.shaped_reward is None
    assert entry.after.available_actions == ()
    assert entry.action.cve == "Unknown / not recorded"


@pytest.mark.parametrize("value", [[], "bad", 4])
def test_non_snapshot_knowledge_is_explicitly_unavailable(value):
    assert knowledge_view(value) == AgentKnowledgeView()


def test_malformed_or_missing_evidence_fails_safely(tmp_path):
    assert "not yet available" in load_workspace(tmp_path).status
    path = (
        tmp_path / "results/convergence_v2/final_baseline_evaluation/report/reviewed_manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("[broken")
    assert load_workspace(tmp_path).episode_choices == ()
    assert "not yet available" in load_workspace_episode(tmp_path, "../../outside", 42, 1001).status


def test_frozen_episode_loader_binds_trace_to_reviewed_outcome(tmp_path, monkeypatch):
    from gui.data import workspace as loader

    root = tmp_path
    frozen_path = root / loader.FROZEN
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text("{}")
    directory = root / "fixture"
    trace = directory / "final-evaluation/shaped-42/steps.jsonl"
    trace.parent.mkdir(parents=True)
    rows = [
        dict(e, training_seed=42, evaluation_seed=1001, run_name="fixture-run") for e in events()
    ]
    trace.write_text("\n".join(map(json.dumps, rows)))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    matched = {
        "frozen_sha256": digest(frozen_path),
        "files": {"shaped-42/steps.jsonl": digest(trace)},
    }
    (trace.parent.parent / "matched_outcomes.json").write_text(json.dumps(matched))
    monkeypatch.setattr(
        loader,
        "_context",
        lambda _: (None, {"inputs": {"git_commit": "recorded-source"}}, directory, {}),
    )
    csv = (
        b"training_seed,evaluation_seed,length,goal_reached,run_name,native_return,failed_actions\n"
        b"42,1001,2,True,fixture-run,4.0,0\n"
    )
    monkeypatch.setattr(loader, "_report_blob", lambda *args: csv)
    before = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    data = loader.load_workspace_episode(root, "shaped", 42, 1001)
    assert len(data.entries) == 2 and data.report.outcome == "SUCCESS", data.status
    assert dict(data.report.metrics)["Native return"] == "4.0"
    assert before == {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    wrong = csv.replace(b",4.0,", b",400.0,")
    monkeypatch.setattr(loader, "_report_blob", lambda *args: wrong)
    assert not loader.load_workspace_episode(root, "shaped", 42, 1001).entries
    monkeypatch.setattr(loader, "_report_blob", lambda *args: csv)
    trace.write_text("{}")
    assert not loader.load_workspace_episode(root, "shaped", 42, 1001).entries


def test_comparison_excludes_discovery_and_requires_matching_cohort():
    a = ComparisonOption(
        "A", "fixed", (("native_return", 1.0), ("discovered_hosts", 99.0)), "stored"
    )
    b = replace(a, label="B", metrics=(("native_return", 3.0), ("discovered_hosts", 999.0)))
    rows = compare_options(a, b)
    assert len(rows) == 1 and rows[0].metric == "native_return" and rows[0].difference == 2.0
    assert compare_options(a, replace(b, cohort="other simulator")) == ()


def test_scenario_true_and_knowledge_views_never_merge(app):
    from gui.views.workspace import ScenarioPage

    page = ScenarioPage()
    page.set_scenario(
        ScenarioSummary(
            true_nodes=({"id": "known", "attributes": {"secret": "SECRET"}}, {"id": "hidden"}),
            status="Recorded",
        )
    )
    assert page.graph.node_count == 0
    page.set_knowledge(trajectory().entries[0])
    assert page.graph.node_count == 1
    assert "SECRET" not in json.dumps(page.graph._nodes)
    page.mode.setCurrentIndex(1)
    assert page.graph.node_count == 2
    assert "RESEARCH TRUE ENVIRONMENT VIEW" in page.banner.state.text()
    page.mode.setCurrentIndex(0)
    assert page.graph.node_count == 1 and "hidden" not in json.dumps(page.graph._nodes)
    page.set_scenario(ScenarioSummary())
    assert page.graph.node_count == 0  # no knowledge survives a scenario switch
    page.close()


def test_replay_empty_clickable_timeline_and_action_details(app):
    from gui.views.workspace import ReplayWorkspace

    page = ReplayWorkspace()
    assert page.timeline.rowCount() == 0
    page.set_trajectory(trajectory())
    assert page.timeline.rowCount() == 2
    page.timeline.setCurrentCell(1, 0)
    app.processEvents()
    text = "\n".join(
        page.inspector.fields.item(i, 1).text() for i in range(page.inspector.fields.rowCount())
    )
    for value in ("CVE-2024-1234", "8.10", "T1210", "5.0000", "8.0000"):
        assert value in text
    assert page.knowledge_graph.node_count == 2
    page.timeline.setCurrentCell(0, 0)
    app.processEvents()
    assert page.knowledge_graph.node_count == 1
    page.set_trajectory(AttackTrajectorySummary())
    assert page.knowledge_graph.node_count == 0 and page.inspector.fields.rowCount() == 0
    page.close()


def test_existing_demo_graph_does_not_render_hidden_attributes(app):
    from gui.views.research_console import SimulationPage
    from tests.test_gui_console import FakeBackend, simulation_result

    result = simulation_result()
    result.nodes[0]["attributes"] = {"undiscovered_cve": "SECRET"}
    page = SimulationPage(FakeBackend(), lambda _: None)
    page._completed(result)
    page._render_step(1)
    assert "SECRET" not in json.dumps(page.knowledge_graph._nodes)
    assert "SECRET" in json.dumps(page.truth_graph._nodes)
    scenario, trace = simulation_workspace(result)
    assert "SECRET" not in json.dumps(asdict(trace))
    assert "SECRET" in json.dumps(asdict(scenario))
    page.close()


def test_report_excludes_all_non_allowlisted_outcome_metrics(app):
    from gui.views.workspace import OutcomePage

    page = OutcomePage(lambda *_: None)
    page.set_report(
        ExperimentReport(
            outcome="SUCCESS",
            metrics=(
                ("Native return", "15"),
                ("discovered_hosts", "99"),
                ("discovery coverage", "100%"),
                ("Discovery-derived attack_path_length", "10"),
            ),
        )
    )
    assert page.metrics.rowCount() == 1
    assert page.metrics.item(0, 0).text() == "Native return"
    assert "withheld" in page.limitation.text()
    page.close()


def test_home_actions_follow_research_workflow_without_starting_a_run(app):
    from PySide6.QtWidgets import QPushButton
    from gui.views.workspace import HomePage

    destinations = []
    page = HomePage(destinations.append)
    buttons = {button.text(): button for button in page.findChildren(QPushButton)}
    for name, target in (
        ("Open Scenario", 1),
        ("Run / View Simulation", 2),
        ("View Research Evidence", 5),
    ):
        buttons[name].click()
        assert destinations[-1] == target
    assert destinations == [1, 2, 5]
    page.close()


def test_research_diagnostics_and_tooltips_render_without_database(app):
    from gui.views.workspace import HELP, DiagnosticsPage

    page = DiagnosticsPage()
    rows = tuple({"timesteps": t, **{k: t / 10 for k in DIAGNOSTICS}} for t in (10, 20))
    page.set_evidence(
        ResearchEvidenceSummary(
            status="Synthetic persisted data",
            diagnostics=((42, rows),),
            convergence=(("42", "PASS", "Recorded"),),
        )
    )
    assert page.seed.count() == 1 and page.values.rowCount() == 2
    assert page.convergence.item(0, 1).text() == "PASS"
    page.metric_selector.setCurrentText("approx_kl")
    assert page.metric_selector.toolTip() == HELP["approx_kl"]
    assert page.curve.points == [(10, 1.0), (20, 2.0)]
    page.close()


def test_comparison_view_never_renders_discovery_metrics(app):
    from gui.views.workspace import ComparisonPage

    page = ComparisonPage()
    values = tuple((m, 1.0) for m in METRICS) + (("discovered_hosts", 999.0),)
    page.set_options(
        (
            ComparisonOption("Frozen", "fixed", values, "Stored"),
            ComparisonOption("Corrected", "fixed", values, "Stored"),
        )
    )
    assert page.table.rowCount() == 5
    assert {page.table.item(i, 0).text() for i in range(5)} == METRICS
    page.close()


def test_workspace_offline_and_stale_episode_load_is_discarded(app, monkeypatch):
    from gui.views import research_console as console
    from tests.test_gui_console import FakeBackend

    pending = []

    def deferred(fn, success, failure):
        pending.append((fn, success, failure))

    class Offline(FakeBackend):
        def load_workspace(self):
            return WorkspaceEvidence()

        def load_workspace_episode(self, *args):
            return trajectory()

        def load_dashboard(self):
            raise ConnectionError("No PostgreSQL")

    monkeypatch.setattr(console, "run_async", deferred)
    window = console.MainWindow(Offline())
    assert len(window.workspace_pages) == 6
    assert window.scenario.graph.node_count == 0
    window.apply_workspace(
        WorkspaceEvidence(
            episode_choices=(("Seed 42", "shaped", 42, 1001), ("Seed 43", "shaped", 43, 1001))
        )
    )
    first = pending[-1][1]
    window.episode_selector.setCurrentIndex(1)
    second = pending[-1][1]
    second(replace(trajectory(), title="Current seed 43"))
    first(replace(trajectory(), title="Stale seed 42"))
    assert window.replay.data.title == "Current seed 43"
    assert window.attack_replay.data.title == "Current seed 43"
    window.navigate_to(5, "Compare Experiment")
    assert window.stack.currentIndex() == 5 and window.research_section.tabs.currentIndex() == 3
    window.close()


def test_full_theme_renders_all_sections_without_overriding_qt_metric(app):
    from gui import theme
    from gui.views.main_window import MainWindow
    from tests.test_gui_console import FakeBackend

    original = app.styleSheet()
    window = None
    try:
        app.setStyleSheet(theme.build_stylesheet())
        window = MainWindow(FakeBackend())
        window.show()
        for i in range(6):
            window.select_page(i)
            app.processEvents()
            assert not window.grab().isNull()
        # QPaintDevice.metric is virtual: a combo box named self.metric caused
        # a native crash during themed layout. Keep that Qt method callable.
        assert callable(window.diagnostics.metric)
    finally:
        if window:
            window.close()
        app.setStyleSheet(original)
