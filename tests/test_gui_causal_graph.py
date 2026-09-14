"""Headless PySide6 interaction tests for the causal graph explorer."""

from __future__ import annotations

import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.backend import DashboardData  # noqa: E402
from gui.views.research_console import AttackPathReportPage  # noqa: E402
from rlredteam.causal_graph import build_causal_graph  # noqa: E402
from tests.test_causal_graph import causal_document, causal_report  # noqa: E402


def make_page() -> tuple[QApplication, AttackPathReportPage, dict, dict]:
    app = QApplication.instance() or QApplication([])
    report = causal_report()
    graph = build_causal_graph(report, causal_document()).to_dict()
    page = AttackPathReportPage()
    page.apply(
        DashboardData(
            source_status="PostgreSQL connected",
            attack_path_reports=[report],
            causal_attack_graphs=[graph],
        )
    )
    app.processEvents()
    return app, page, report, graph


def test_two_views_render_only_evidence_backed_labeled_edges() -> None:
    _app, page, _report, graph = make_page()
    assert page.graph_views.tabText(0) == "Attack Path"
    assert page.graph_views.tabText(1) == "Knowledge Flow"
    rendered = [
        item.data(1)
        for item in page.attack_graph.graph_scene.items()
        if item.data(0) == "edge"
    ]
    assert rendered
    assert all(item.get("type") != "unknown" for item in rendered)
    assert all(item.get("evidence_ids") for item in rendered)
    assert 0 < page.attack_graph.edge_count <= len(graph["edges"])
    page.close()


def test_node_and_edge_inspectors_expose_complete_facts() -> None:
    _app, page, _report, graph = make_page()
    node = next(item for item in graph["nodes"] if item["node_id"] == "host-c")
    edge = next(
        item
        for item in graph["edges"]
        if item["target_node"] == "host-c"
        and item["relationship_type"] == "AUTHENTICATED_TO"
    )
    page._inspect_node({"id": "host-c", "attributes": node})
    assert "WHY ACTIONABLE" in page.node_inspector.text()
    assert "cred-backup" in page.node_inspector.text()
    page._inspect_edge({"attributes": edge})
    assert "SSHLogin(host-c,svc-backup)" in page.edge_inspector.text()
    assert "AUTHENTICATED_TO" in page.edge_inspector.text()
    assert page.evidence_items.rowCount() >= 1
    page.close()


def test_evidence_jump_selects_original_trajectory_step() -> None:
    _app, page, _report, graph = make_page()
    evidence = next(item for item in graph["evidence"] if item["step"] == 2)
    assert page.jump_to_evidence(evidence["evidence_id"])
    assert page.details.currentRow() == 2
    page.close()


def test_category_filter_is_applied_to_both_views() -> None:
    _app, page, _report, _graph = make_page()
    page.graph_filter.setCurrentIndex(page.graph_filter.findData("credentials"))
    assert page.attack_graph._edge_categories == {"attack_path", "credentials"}
    assert page.knowledge_graph._edge_categories == {"knowledge_flow", "credentials"}
    page.close()
