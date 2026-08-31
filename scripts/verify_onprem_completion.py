#!/usr/bin/env python3
"""Verify the canonical hidden-topology on-prem milestone evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.completion import CompletionVerificationError, verify_onprem_completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="also reconstruct canonical validation/test runs from PostgreSQL",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        report = verify_onprem_completion(repo_root, include_postgres=args.postgres)
    except CompletionVerificationError as exc:
        print(json.dumps({"complete": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
