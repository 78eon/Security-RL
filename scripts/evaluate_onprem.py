#!/usr/bin/env python3
"""Evaluate a frozen mask-aware PPO checkpoint on unseen topology seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from rlredteam.enterprise.generalisation import (
    OnPremExperimentConfig,
    evaluate_policy,
    persist_evaluation,
    policy_digest,
    sha256_file,
    validate_checkpoint_manifest,
    write_evaluation_package,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/onprem-generalisation"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/onprem-generalisation"))
    parser.add_argument("--allow-unverifiable", action="store_true", help="Development only")
    parser.add_argument("--postgres", action="store_true", help="Persist reconstructable evidence")
    args = parser.parse_args()
    config = OnPremExperimentConfig.from_yaml()
    checkpoint = args.run / "model.zip"
    manifest = json.loads((args.run / "training_manifest.json").read_text())
    validate_checkpoint_manifest(
        manifest,
        checkpoint,
        config,
        require_current_commit=not args.allow_unverifiable,
        allow_unverifiable=args.allow_unverifiable,
    )
    topology_seeds = getattr(OnPremGeneralisationSplit(), args.split)
    if args.limit:
        topology_seeds = topology_seeds[: args.limit]
    model = MaskablePPO.load(checkpoint, device="cpu")
    before = policy_digest(model)
    episodes, steps = evaluate_policy(
        model,
        topology_seeds=topology_seeds,
        evaluation_episode_seeds=config.evaluation_episode_seeds,
        deterministic=config.deterministic_evaluation,
    )
    after = policy_digest(model)
    metadata = {
        "phase": "frozen_evaluation",
        "split": args.split,
        "topology_seeds": list(topology_seeds),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "deterministic": config.deterministic_evaluation,
        "gradient_updates": False,
        "action_masks_supplied_to_predict": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_before": before,
        "policy_sha256_after": after,
        "training_manifest": manifest,
    }
    if args.postgres:
        metadata.update(
            persist_evaluation(
                episodes=episodes,
                steps=steps,
                training_manifest=manifest,
                checkpoint=checkpoint,
                split_name=args.split,
            )
        )
    write_evaluation_package(
        args.output / args.split,
        episodes=episodes,
        steps=steps,
        metadata=metadata,
    )
    print((args.output / args.split / "summary.json").read_text())


if __name__ == "__main__":
    main()
