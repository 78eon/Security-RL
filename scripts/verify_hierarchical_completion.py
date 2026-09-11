#!/usr/bin/env python3
"""Verify canonical Phase 12 hierarchical-policy evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.hierarchical_completion import (
    HierarchicalCompletionError,
    verify_hierarchical_completion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="also reconstruct every held-out run exactly from PostgreSQL",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        report = verify_hierarchical_completion(repo_root, include_postgres=args.postgres)
    except HierarchicalCompletionError as exc:
        print(json.dumps({"complete": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
