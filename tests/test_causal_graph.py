"""Evidence, causality and non-CVE coverage for the Phase 19 graph."""

from __future__ import annotations

import pytest

from rlredteam.attack_path_report import build_report
from rlredteam.causal_graph import (
    UNKNOWN,
    CausalGraphError,
    RelationshipType,
    build_causal_graph,
    graph_from_dict,
    why_actionable,
)


def causal_document() -> list[dict]:
    return [
        {
            "run_name": "causal-run",
            "evaluation_seed": 7001,
            "goal_reached": True,
            "events": [
                {
                    "step": 0,
                    "action": "discover-internal",
                    "simulator_action": "DiscoverNetwork(source=host-a)",
                    "action_kind": "discover_network",
                    "source_node": "host-a",
                    "target": "host-b",
                    "success": True,
                    "state_changed": True,
                    "knowledge_delta": {
                        "discovered_nodes": ["host-b", "host-c"],
                        "observed_edges": [["host-a", "host-b", "DISCOVERED"]],
                        "discovered_services": [["host-b", "ssh"]],
                    },
                },
                {
                    "step": 1,
                    "action": "exploit-openssh",
                    "simulator_action": "RemoteExploit(host-b,openssh)",
                    "action_kind": "exploit",
                    "source_node": "host-a",
                    "target": "host-b",
                    "service": "ssh",
                    "port": 22,
                    "protocol": "tcp",
                    "product": "OpenSSH",
                    "version": "8.2",
                    "cve_id": "CVE-2020-1111",
                    "success": True,
                    "state_changed": True,
                    "access_before": "none",
                    "access_after": "root",
                    "knowledge_delta": {
                        "access_changes": [["host-b", "root"]],
                        "credential_targets": ["host-c"],
                    },
                },
                {
                    "step": 2,
                    "action": "ssh-valid-account",
                    "simulator_action": "SSHLogin(host-c,svc-backup)",
                    "action_kind": "authenticate",
                    "source_node": "host-b",
                    "target": "host-c",
                    "service": "ssh",
                    "port": 22,
                    "protocol": "tcp",
                    "credential_id": "cred-backup",
                    "identity": "svc-backup",
                    "success": True,
                    "state_changed": True,
                    "access_before": "none",
                    "access_after": "user",
                    "is_crown_jewel": True,
                    "knowledge_delta": {
                        "access_changes": [["host-c", "user"]],
                        "discovered_properties": [["host-c", "database-role"]],
                    },
                },
            ],
        }
    ]


def causal_report(document=None) -> dict:
    return build_report(
        causal_document() if document is None else document,
        source_trajectory_hash="a" * 64,
        mode="simulation_report",
        experiment_id="causal-test",
    ).to_dict()


def test_every_edge_has_semantics_and_source_evidence() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    evidence_ids = {item.evidence_id for item in graph.evidence}
    assert graph.edges
    assert all(RelationshipType(edge.relationship_type) for edge in graph.edges)
    assert all(edge.evidence_ids for edge in graph.edges)
    assert all(set(edge.evidence_ids) <= evidence_ids for edge in graph.edges)
    assert all(edge.explanation for edge in graph.edges)


def test_complete_service_vulnerability_and_access_chain_is_exposed() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    edge = next(
        item
        for item in graph.edges
        if item.target_node == "host-b" and item.relationship_type == "EXPLOITED"
    )
    assert (edge.service, edge.port, edge.protocol) == ("ssh", "22", "tcp")
    assert (edge.product, edge.version) == ("OpenSSH", "8.2")
    assert edge.vulnerability_id == edge.cve_id == "CVE-2020-1111"
    assert (edge.access_before, edge.access_after) == ("none", "root")
    assert edge.mitre_mappings
    assert 'access_changes:["host-b", "root"]' in edge.new_information


def test_non_cve_identity_path_is_not_fabricated_as_a_cve() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    edge = next(
        item
        for item in graph.edges
        if item.target_node == "host-c" and item.relationship_type == "AUTHENTICATED_TO"
    )
    assert edge.cve_id == edge.vulnerability_id == UNKNOWN
    assert edge.credential_id == "cred-backup"
    assert edge.identity == "svc-backup"
    assert (edge.access_before, edge.access_after) == ("none", "user")


def test_why_actionable_is_deterministic_and_uses_stored_edges() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    first = why_actionable(graph, "host-c")
    restored = graph_from_dict(graph.to_dict())
    second = why_actionable(restored, "host-c")
    assert first == second
    assert first["actionable"]
    assert "cred-backup" in first["explanation"]
    assert set(first["evidence_ids"])


def test_knowledge_propagation_unlocks_later_access_explanation() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    explanation = graph.why_actionable("host-c")
    relationships = {
        edge.relationship_type
        for edge in graph.edges
        if edge.edge_id in explanation["edge_ids"]
    }
    assert {"DISCOVERED", "AUTHENTICATED_TO"} <= relationships


def test_hidden_topology_is_rejected_recursively() -> None:
    document = causal_document()
    document[0]["events"][0]["knowledge_delta"]["true_topology"] = {"secret": {}}
    with pytest.raises(CausalGraphError, match="hidden topology"):
        build_causal_graph(causal_report(), document)


def test_unknown_fields_remain_explicitly_unknown() -> None:
    document = causal_document()
    event = document[0]["events"][0]
    event.pop("source_node")
    graph = build_causal_graph(causal_report(document), document)
    edge = next(
        item
        for item in graph.edges
        if item.target_node == "host-b" and item.source_node == UNKNOWN
    )
    assert edge.source_node == UNKNOWN
    assert edge.product == edge.version == edge.cve_id == UNKNOWN


def test_legacy_scalar_discovery_delta_is_preserved() -> None:
    document = causal_document()
    document[0]["events"][0]["knowledge_delta"] = {"newly_discovered": 2}
    graph = build_causal_graph(causal_report(document), document)
    assert any("newly_discovered:2" in edge.new_information for edge in graph.edges)
