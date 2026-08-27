#!/usr/bin/env python3
"""Evaluate one frozen PPO checkpoint on controlled episode seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.evaluation import evaluate_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args(argv)
    bundle = evaluate_checkpoint(args.run, args.seeds, args.out, postgres=args.postgres)
    successes = sum(episode.goal_reached for episode in bundle.episodes)
    print(f"evaluated {len(bundle.episodes)} episodes; " f"{successes} goals; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
