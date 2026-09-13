#!/usr/bin/env python3
"""Generate and optionally persist the Phase 14 example report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.attack_path_report import (
    ReportMode,
    checkpoint_hashes_from_metadata,
    generate_report,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/experiment_01/trajectories/attack_paths.json"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("results/experiment_01/metadata/experiment.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase14/experiment_01_attack_path_report.json"),
    )
    parser.add_argument("--experiment-id", default="experiment_01")
    parser.add_argument(
        "--mode", choices=[item.value for item in ReportMode], default=ReportMode.SIMULATION.value
    )
    parser.add_argument("--postgres", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_hashes = (
        checkpoint_hashes_from_metadata(args.metadata) if args.metadata.is_file() else {}
    )
    report = generate_report(
        args.source,
        mode=args.mode,
        experiment_id=args.experiment_id,
        checkpoint_hashes=checkpoint_hashes,
    )
    write_report(report, args.output)
    persisted = False
    if args.postgres:
        import psycopg

        from rlredteam.storage.postgres_logger import connection_string, ensure_schema
        from rlredteam.storage.report_store import ReportStore

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            ReportStore(connection).save(report)
        persisted = True
    print(
        json.dumps(
            {
                "report_id": report.report_id,
                "report_mode": report.report_mode,
                "facts": len(report.facts),
                "trajectories": report.cross_episode_analysis["trajectory_count"],
                "output": str(args.output),
                "postgres": persisted,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
