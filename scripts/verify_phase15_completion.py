#!/usr/bin/env python3
"""Run Phase 15 tests and verify an immutable paired report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.phase15_completion import verify_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "results/phase15/experiment_01_mitigation.json",
    )
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args(argv)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not slow",
            "tests/test_mitigation.py",
            "tests/test_phase15_completion.py",
            "tests/test_postgres_logger.py",
            "tests/test_gui_backend.py",
            "tests/test_gui_console.py",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    result = verify_report(args.report)
    if args.postgres:
        import psycopg

        from rlredteam.storage.mitigation_store import MitigationStore
        from rlredteam.storage.postgres_logger import connection_string, ensure_schema

        with psycopg.connect(connection_string()) as connection:
            ensure_schema(connection)
            stored = MitigationStore(connection).load(result["report_id"])
        if stored != json.loads(args.report.read_text()):
            raise RuntimeError("PostgreSQL Phase 15 reconstruction differs from the report")
        result["postgres_reconstruction"] = "pass"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
