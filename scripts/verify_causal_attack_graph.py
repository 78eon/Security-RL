#!/usr/bin/env python3
"""Verify Phase 19 deterministic derivation and PostgreSQL reconstruction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rlredteam.causal_graph import RelationshipType, build_causal_graph, why_actionable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/phase14/experiment_01_attack_path_report.json"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/experiment_01/trajectories/attack_paths.json"),
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("results/phase19/experiment_01_causal_graph.json"),
    )
    parser.add_argument("--postgres", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tests = subprocess.run(
        [
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_causal_graph.py",
            "tests/test_causal_store.py",
            "tests/test_gui_causal_graph.py",
        ],
        check=False,
    )
    if tests.returncode:
        return tests.returncode
    expected = build_causal_graph(
        json.loads(args.report.read_text()), json.loads(args.source.read_text())
    )
    stored_file = json.loads(args.graph.read_text())
    required_node_fields = {
        "node_type",
        "first_discovery_step",
        "discovered_by",
        "discovered_services",
        "discovered_vulnerabilities",
        "current_access_level",
        "credentials",
        "mitre_techniques",
        "incoming_relationships",
        "outgoing_relationships",
        "evidence_ids",
    }
    required_edge_fields = {
        "source_node",
        "target_node",
        "relationship_type",
        "simulator_action",
        "discovery_method",
        "service",
        "port",
        "protocol",
        "product",
        "version",
        "vulnerability_id",
        "cve_id",
        "credential_id",
        "identity",
        "access_before",
        "access_after",
        "success",
        "mitre_mappings",
        "episode_id",
        "step",
        "evidence_steps",
        "evidence_ids",
        "new_information",
    }
    evidence_ids = {item.evidence_id for item in expected.evidence}
    checks = {
        "deterministic_file_reconstruction": stored_file == expected.to_dict(),
        "node_contract_complete": all(
            required_node_fields <= set(node) for node in stored_file.get("nodes", [])
        ),
        "edge_contract_complete": all(
            required_edge_fields <= set(edge) for edge in stored_file.get("edges", [])
        ),
        "every_edge_has_evidence": all(
            edge.evidence_ids and set(edge.evidence_ids) <= evidence_ids
            for edge in expected.edges
        ),
        "every_edge_is_semantic": all(
            edge.relationship_type in {item.value for item in RelationshipType}
            for edge in expected.edges
        ),
        "why_actionable_deterministic": all(
            why_actionable(expected, node.node_id)
            == why_actionable(stored_file, node.node_id)
            for node in expected.nodes
        ),
        "hidden_topology_used": expected.provenance["hidden_topology_used"],
    }
    if args.postgres:
        import psycopg

        from rlredteam.storage.causal_store import CausalGraphStore
        from rlredteam.storage.postgres_logger import connection_string, ensure_schema

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            reconstructed = CausalGraphStore(connection).load(expected.graph_id)
        checks["postgres_reconstruction"] = reconstructed.to_dict() == expected.to_dict()
    for name, result in checks.items():
        passed = result is False if name == "hidden_topology_used" else bool(result)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(
        (value is False if key == "hidden_topology_used" else bool(value))
        for key, value in checks.items()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
