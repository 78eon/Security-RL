#!/usr/bin/env python3
"""Verify canonical Phase 13 evidence and optional exact PostgreSQL replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.multiagent_completion import verify_multiagent_completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    result = verify_multiagent_completion(
        args.repo,
        include_postgres=args.postgres,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
