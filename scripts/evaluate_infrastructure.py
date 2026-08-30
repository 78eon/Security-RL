#!/usr/bin/env python3
"""Evaluate frozen PPO on unseen legacy, cloud and hybrid topology seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from rlredteam.enterprise.generalisation import (
    persist_evaluation,
    policy_digest,
    sha256_file,
    write_evaluation_package,
)
from rlredteam.enterprise.infrastructure_generalisation import (
    InfrastructureExperimentConfig,
    evaluate_infrastructure_policy,
    validate_infrastructure_checkpoint,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/infrastructure-generalisation")
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/infrastructure-generalisation")
    )
    parser.add_argument("--allow-unverifiable", action="store_true", help="Development only")
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args()

    config = InfrastructureExperimentConfig.from_yaml()
    checkpoint = args.run / "model.zip"
    manifest = json.loads((args.run / "training_manifest.json").read_text())
    validate_infrastructure_checkpoint(
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
    episodes, steps = evaluate_infrastructure_policy(
        model,
        profiles=config.train_profiles,
        topology_seeds=topology_seeds,
        evaluation_episode_seeds=config.evaluation_episode_seeds,
        deterministic=config.deterministic_evaluation,
    )
    after = policy_digest(model)
    metadata = {
        "phase": "frozen_cross_profile_evaluation",
        "split": args.split,
        "profiles": [profile.value for profile in config.train_profiles],
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
    output = args.output / args.split
    write_evaluation_package(output, episodes=episodes, steps=steps, metadata=metadata)
    print((output / "summary.json").read_text())


if __name__ == "__main__":
    main()
