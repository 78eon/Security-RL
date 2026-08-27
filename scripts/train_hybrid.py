#!/usr/bin/env python3
"""Train PPO across the pre-registered hybrid simulation training split."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rlredteam.enterprise.hybrid import GeneralisationSplit, HybridCurriculumEnv, HybridFamily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("runs/hybrid-ppo"))
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("timesteps must be positive")

    split = GeneralisationSplit()
    env = Monitor(HybridCurriculumEnv(split.train))
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        device="cpu",
        n_steps=512,
        batch_size=64,
    )
    model.learn(total_timesteps=args.timesteps)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"model-seed-{args.seed}"
    model.save(model_path)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "train_topology_seeds": list(split.train),
        "validation_topology_seeds": list(split.validation),
        "held_out_test_topology_seeds": list(split.test),
        "families": [family.value for family in HybridFamily],
        "model_path": str(model_path.with_suffix(".zip")),
        "weights_release": "gated; runs/ is gitignored",
    }
    (args.output / f"metadata-seed-{args.seed}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
