#!/usr/bin/env python3
"""Verify canonical Phase 11 transfer-study evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.transfer_completion import (
    TransferCompletionError,
    verify_transfer_completion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true", help="also reconstruct every run")
    args = parser.parse_args()
    try:
        report = verify_transfer_completion(
            Path(__file__).resolve().parents[1], include_postgres=args.postgres
        )
    except TransferCompletionError as exc:
        print(json.dumps({"complete": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
