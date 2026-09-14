#!/usr/bin/env python3
"""Rebuild Neo4j solely from a PostgreSQL-authoritative Phase 19 graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from rlredteam.storage.causal_store import CausalGraphStore
from rlredteam.storage.neo4j_projection import Neo4jProjectionStore, build_projection_plan
from rlredteam.storage.postgres_logger import connection_string, ensure_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-id")
    parser.add_argument(
        "--out", type=Path, default=Path("results/phase20/neo4j_projection.json")
    )
    return parser.parse_args()


def load_source_graph(graph_id: str | None):
    with psycopg.connect(connection_string()) as connection:
        ensure_schema(connection)
        if graph_id is None:
            row = connection.execute(
                """
                SELECT graph_id FROM causal_attack_graphs
                ORDER BY created_at DESC, graph_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise SystemExit("no Phase 19 causal graph exists in PostgreSQL")
            graph_id = str(row[0])
        return CausalGraphStore(connection).load(graph_id)


def main() -> int:
    args = parse_args()
    graph = load_source_graph(args.graph_id)
    plan = build_projection_plan(graph)
    store = Neo4jProjectionStore.connect()
    try:
        result = store.rebuild(plan)
    finally:
        store.close()
    document = {
        "schema_version": "security-rl-neo4j-export-result-v1",
        "authority": "postgresql_causal_graph_facts",
        "hidden_topology_used": False,
        "source_graph_id": graph.graph_id,
        "source_graph_sha256": plan.source_graph_sha256,
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
