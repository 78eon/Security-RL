#!/usr/bin/env python3
"""Train mask-aware PPO on the preregistered hidden on-prem distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.generalisation import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/onprem-generalisation"))
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-dirty", action="store_true", help="Development only; recorded")
    args = parser.parse_args()
    manifest = train_policy(
        args.output,
        timesteps=args.timesteps,
        training_seed=args.seed,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
