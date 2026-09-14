"""Deterministic, evidence-only Neo4j projection contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rlredteam.causal_graph import UNKNOWN, build_causal_graph
from rlredteam.storage.neo4j_projection import (
    MissingNeo4jCredentials,
    Neo4jProjectionError,
    Neo4jSettings,
    build_projection_plan,
)
from tests.test_causal_graph import causal_document, causal_report


def test_projection_plan_is_deterministic_and_does_not_mutate_source() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    before = graph.to_dict()
    first = build_projection_plan(graph)
    second = build_projection_plan(graph)
    assert first == second
    assert first.projection_hash == second.projection_hash
    assert graph.to_dict() == before


def test_every_projected_edge_is_semantic_and_evidence_backed() -> None:
    plan = build_projection_plan(build_causal_graph(causal_report(), causal_document()))
    known_evidence = {row["evidence_id"] for row in plan.evidence_rows}
    assert plan.edge_rows
    for row in plan.edge_rows:
        assert row["relationship_type"] != UNKNOWN
        assert row["evidence_ids"]
        assert set(row["evidence_ids"]) <= known_evidence
        assert row["source_key"]
        assert row["target_key"]


def test_unknown_fields_stay_unknown_instead_of_being_invented() -> None:
    plan = build_projection_plan(build_causal_graph(causal_report(), causal_document()))
    unknown_cve = next(row for row in plan.edge_rows if row["cve_id"] == UNKNOWN)
    assert unknown_cve["cve_id"] == UNKNOWN
    assert "CVE-" not in unknown_cve["mitre_mappings_json"]


def test_projection_rejects_any_non_false_hidden_topology_provenance() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    with pytest.raises(Neo4jProjectionError, match="hidden_topology"):
        build_projection_plan(replace(graph, provenance={"hidden_topology_used": True}))
    with pytest.raises(Neo4jProjectionError, match="hidden_topology"):
        build_projection_plan(replace(graph, provenance={}))


def test_projection_rows_are_scoped_to_exact_graph() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    plan = build_projection_plan(graph)
    for rows in (plan.node_rows, plan.edge_rows, plan.evidence_rows, plan.technique_rows):
        assert all(row["graph_id"] == graph.graph_id for row in rows)


def test_neo4j_settings_fail_closed(monkeypatch) -> None:
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingNeo4jCredentials):
        Neo4jSettings.from_env()
