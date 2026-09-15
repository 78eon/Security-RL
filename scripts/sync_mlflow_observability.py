#!/usr/bin/env python3
"""Mirror PostgreSQL run metrics and hash references into optional MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg

from rlredteam.observability import (
    connect_mlflow_client,
    load_observable_runs,
    sync_observable_run,
)
from rlredteam.storage.postgres_logger import connection_string, ensure_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=Path("results/phase21/mlflow_sync.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    with psycopg.connect(connection_string()) as connection:
        ensure_schema(connection)
        records = load_observable_runs(
            connection, repo_root, run_id=args.run_id, limit=args.limit
        )
    if not records:
        raise SystemExit("no PostgreSQL runs matched the synchronization request")
    client = connect_mlflow_client()
    prefix = os.environ.get("MLFLOW_EXPERIMENT_PREFIX") or "Security-RL"
    mirrored = [
        {
            "source_key": record.source_key,
            **sync_observable_run(client, record, experiment_prefix=prefix),
        }
        for record in records
    ]
    document = {
        "schema_version": "security-rl-mlflow-sync-result-v1",
        "authority": "postgresql_and_canonical_files",
        "mlflow_role": "optional_observability_mirror",
        "run_count": len(mirrored),
        "runs": mirrored,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
