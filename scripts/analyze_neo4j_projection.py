#!/usr/bin/env python3
"""Run evidence-preserving analyses over one derived Neo4j projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.storage.neo4j_projection import Neo4jProjectionStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-id")
    parser.add_argument(
        "--projection",
        type=Path,
        default=Path("results/phase20/neo4j_projection.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/phase20/neo4j_analysis.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_id = args.graph_id
    if graph_id is None:
        graph_id = str(json.loads(args.projection.read_text())["source_graph_id"])
    store = Neo4jProjectionStore.connect()
    try:
        result = store.analyze(graph_id)
    finally:
        store.close()
    document = {
        "schema_version": "security-rl-neo4j-analysis-v1",
        "authority": "derived_only_postgresql_remains_authoritative",
        "hidden_topology_used": False,
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
