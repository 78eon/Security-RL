#!/usr/bin/env python3
"""Run Phase 14 tests and verify exact report/database reconstruction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rlredteam.phase14_completion import Phase14VerificationError, verify_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/experiment_01/trajectories/attack_paths.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/phase14/experiment_01_attack_path_report.json"),
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
            "-m",
            "not slow",
            "tests/test_attack_path_report.py",
            "tests/test_phase14_completion.py",
            "tests/test_postgres_logger.py",
            "tests/test_gui_backend.py",
            "tests/test_gui_console.py",
        ],
        check=False,
    )
    if tests.returncode:
        raise Phase14VerificationError("Phase 14 targeted tests failed")
    verification = verify_report(args.source, args.report)
    if args.postgres:
        import psycopg

        from rlredteam.storage.postgres_logger import connection_string, ensure_schema
        from rlredteam.storage.report_store import ReportStore

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            stored = ReportStore(connection).load(verification["report_id"])
        if stored is None:
            raise Phase14VerificationError("report is missing from PostgreSQL")
        if json.loads(args.report.read_text()) != stored:
            raise Phase14VerificationError("PostgreSQL report reconstruction mismatch")
        verification["postgres_reconstruction"] = "pass"
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
