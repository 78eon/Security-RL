#!/usr/bin/env python3
"""Generate and optionally persist the Phase 19 evidence-backed causal graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.causal_graph import build_causal_graph


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
        "--output",
        type=Path,
        default=Path("results/phase19/experiment_01_causal_graph.json"),
    )
    parser.add_argument("--postgres", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text())
    source = json.loads(args.source.read_text())
    graph = build_causal_graph(report, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n")
    if args.postgres:
        import psycopg

        from rlredteam.storage.causal_store import CausalGraphStore
        from rlredteam.storage.postgres_logger import connection_string, ensure_schema

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            CausalGraphStore(connection).save(graph)
    print(
        json.dumps(
            {
                "graph_id": graph.graph_id,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "evidence": len(graph.evidence),
                "postgres": args.postgres,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
