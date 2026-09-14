#!/usr/bin/env python3
"""Verify Phase 20 source immutability, idempotence and graph analyses."""

from __future__ import annotations

from export_neo4j_projection import load_source_graph

from rlredteam.storage.neo4j_projection import Neo4jProjectionStore, build_projection_plan


def main() -> int:
    graph = load_source_graph(None)
    source_before = graph.to_dict()
    plan = build_projection_plan(graph)
    store = Neo4jProjectionStore.connect()
    try:
        first = store.rebuild(plan)
        first_analysis = store.analyze(graph.graph_id)
        second = store.rebuild(plan)
        second_analysis = store.analyze(graph.graph_id)
    finally:
        store.close()
    source_after = load_source_graph(graph.graph_id).to_dict()
    analyses_present = all(
        name in first_analysis
        for name in (
            "path_frequency",
            "relationship_frequency",
            "bottleneck_assets",
            "bottleneck_vulnerabilities",
            "degree_centrality",
            "mitre_relationships",
        )
    )
    checks = {
        "postgresql_source_unchanged": source_before == source_after,
        "hidden_topology_not_used": graph.provenance.get("hidden_topology_used") is False,
        "projection_hash_stable": first["projection_hash"] == second["projection_hash"],
        "projection_counts_stable": first["counts"] == second["counts"] == plan.counts(),
        "analysis_is_deterministic": first_analysis == second_analysis,
        "analysis_families_present": analyses_present,
        "all_edges_evidence_backed": all(row["evidence_ids"] for row in plan.edge_rows),
        "postgresql_declared_authority": all(
            row["graph_id"] == graph.graph_id
            for rows in (plan.node_rows, plan.edge_rows, plan.evidence_rows)
            for row in rows
        ),
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
