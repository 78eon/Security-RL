#!/usr/bin/env python3
"""Freeze, validate and run the Phase 8 matched recurrent-policy study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlredteam.enterprise.recurrent import RecurrentResearchConfig
from rlredteam.enterprise.recurrent_study import (
    DEFAULT_FROZEN_INPUTS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_RUN_ROOT,
    RecurrentStudyError,
    aggregate_seed_metrics,
    analyse_seed_metrics,
    arm_run_name,
    evaluate_arm,
    freeze_inputs,
    git_commit,
    load_arm_model,
    load_frozen_inputs,
    persist_and_write_run_evaluation,
    train_arm,
    validate_finite_evidence,
    validate_frozen_inputs,
    validate_training_manifest,
    write_study_summary,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action", choices=("freeze", "dry-run", "development", "run")
    )
    result.add_argument(
        "--config", type=Path, default=Path("configs/experiments/recurrent_policy.yaml")
    )
    result.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN_INPUTS)
    result.add_argument("--runs", type=Path, default=DEFAULT_RUN_ROOT)
    result.add_argument("--results", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--timesteps", type=int, help="development action only")
    result.add_argument("--limit-topologies", type=int, default=0)
    result.add_argument("--postgres", action="store_true")
    result.add_argument("--allow-dirty", action="store_true", help="development only")
    return result


def validate_development_gate(
    path: Path, config: RecurrentResearchConfig
) -> dict:
    if not path.is_file():
        raise RecurrentStudyError(
            "canonical run requires the full excluded-seed development gate"
        )
    metadata = json.loads(path.read_text())
    checks = {
        "phase": (metadata.get("phase"), "development"),
        "complete": (metadata.get("complete"), True),
        "study config": (metadata.get("study_config_hash"), config.digest()),
        "training budget": (
            metadata.get("training_timesteps"),
            config.total_timesteps,
        ),
        "validation topology set": (
            metadata.get("topology_seeds"),
            list(config.topology_splits["validation"]),
        ),
    }
    failures = [
        f"{name}: development={actual!r}, required={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    projected = float(metadata.get("projected_canonical_minutes", float("inf")))
    if projected > config.runtime_cap_minutes:
        failures.append(
            f"projected runtime {projected:.1f} exceeds cap {config.runtime_cap_minutes}"
        )
    manifests = metadata.get("training_manifests", [])
    if {
        (item.get("arm"), item.get("training_seed"), item.get("development"))
        for item in manifests
    } != {
        (arm, config.development_seed, True) for arm in config.arms
    }:
        failures.append("development manifests do not cover both arms on the excluded seed")
    if failures:
        raise RecurrentStudyError(
            "Phase 8 development gate failed:\n  " + "\n  ".join(failures)
        )
    return metadata


def run_study(args: argparse.Namespace, config: RecurrentResearchConfig) -> dict:
    development = args.action == "development"
    if args.timesteps is not None and not development:
        raise RecurrentStudyError("--timesteps is allowed only for excluded development")
    if args.limit_topologies and not development:
        raise RecurrentStudyError("canonical test topology set cannot be limited")
    if not development:
        validate_development_gate(
            args.results / "development" / "metadata" / "study.json",
            config,
        )
    frozen = None if development else load_frozen_inputs(args.frozen)
    if frozen is not None:
        validate_frozen_inputs(frozen, config)

    seeds = (config.development_seed,) if development else config.training_seeds
    split_name = "validation" if development else "test"
    topology_seeds = config.topology_splits[split_name]
    if args.limit_topologies:
        topology_seeds = topology_seeds[: args.limit_topologies]
    timesteps = args.timesteps if args.timesteps is not None else config.total_timesteps
    result_root = args.results / ("development" if development else "test")
    all_episodes = []
    training_manifests = []
    evaluation_metadata = []
    for arm in config.arms:
        for seed in seeds:
            run_name = arm_run_name(config, arm, seed)
            run_dir = args.runs / ("development" if development else "canonical") / run_name
            manifest = train_arm(
                run_dir,
                arm=arm,
                training_seed=seed,
                config=config,
                frozen_inputs=frozen,
                timesteps=timesteps,
                development=development,
                allow_dirty=args.allow_dirty,
            )
            checkpoint = run_dir / "model.zip"
            validate_training_manifest(
                manifest,
                checkpoint,
                arm=arm,
                training_seed=seed,
                config=config,
                frozen_inputs=frozen,
                development=development,
                allow_unverifiable=development and args.allow_dirty,
            )
            model = load_arm_model(arm, checkpoint)
            episodes, steps, integrity = evaluate_arm(
                model,
                arm=arm,
                training_seed=seed,
                profiles=config.train_profiles,
                topology_seeds=topology_seeds,
                evaluation_episode_seeds=config.evaluation_episode_seeds,
                deterministic=config.deterministic_evaluation,
            )
            validate_finite_evidence(episodes, steps)
            metadata = persist_and_write_run_evaluation(
                result_root / "raw" / run_name,
                episodes=episodes,
                steps=steps,
                integrity=integrity,
                training_manifest=manifest,
                checkpoint=checkpoint,
                split_name=split_name,
                postgres=args.postgres,
            )
            all_episodes.extend(episodes)
            training_manifests.append(manifest)
            evaluation_metadata.append(metadata)

    seed_metrics = aggregate_seed_metrics(
        all_episodes,
        config,
        expected_topology_seeds=topology_seeds,
    )
    report = analyse_seed_metrics(seed_metrics, config, expected_seeds=seeds)
    expected_complete = all(
        set(report["observed_training_seeds"][arm]) == set(seeds)
        for arm in config.arms
    )
    if not expected_complete:
        raise RecurrentStudyError("Phase 8 study is missing a planned matched arm/seed")
    elapsed_by_arm = {
        arm: sum(
            float(item["training_elapsed_seconds"])
            for item in training_manifests
            if item["arm"] == arm
        )
        for arm in config.arms
    }
    projected_minutes = (
        sum(elapsed_by_arm.values())
        * (len(config.training_seeds) / len(seeds))
        / 60.0
    )
    metadata = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "phase": "development" if development else "canonical_test",
        "complete": report["complete"],
        "code_commit": git_commit(),
        "study_config_hash": config.digest(),
        "frozen_inputs": frozen,
        "topology_split": split_name,
        "topology_seeds": list(topology_seeds),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "training_timesteps": timesteps,
        "training_manifests": training_manifests,
        "evaluation_metadata": evaluation_metadata,
        "training_elapsed_seconds_by_arm": elapsed_by_arm,
        "projected_canonical_minutes": projected_minutes,
        "runtime_cap_minutes": config.runtime_cap_minutes,
    }
    write_study_summary(
        result_root,
        seed_metrics=seed_metrics,
        report=report,
        metadata=metadata,
    )
    (result_root / "README.md").write_text(
        "# Phase 8 recurrent-policy study\n\n"
        f"Phase: `{metadata['phase']}`  \n"
        f"Complete matched grid: `{report['complete']}`  \n"
        f"Topology split: `{split_name}`  \n"
        f"Training seeds: `{list(seeds)}`  \n"
        "See `summaries/analysis.json` and each `raw/` run for evidence.\n"
    )
    if development and timesteps == config.total_timesteps:
        if projected_minutes > config.runtime_cap_minutes:
            raise RecurrentStudyError(
                f"projected canonical runtime {projected_minutes:.1f} minutes exceeds "
                f"cap {config.runtime_cap_minutes}"
            )
    return metadata


def main() -> None:
    args = parser().parse_args()
    config = RecurrentResearchConfig.from_yaml(args.config)
    if args.action == "freeze":
        print(json.dumps(freeze_inputs(args.frozen, config), indent=2, sort_keys=True))
        return
    if args.action == "dry-run":
        frozen = load_frozen_inputs(args.frozen)
        validate_frozen_inputs(frozen, config)
        grid = [
            arm_run_name(config, arm, seed)
            for arm in config.arms
            for seed in config.training_seeds
        ]
        print(json.dumps({"complete_grid": len(grid), "runs": grid}, indent=2))
        return
    metadata = run_study(args, config)
    print(
        json.dumps(
            {
                "complete": metadata["complete"],
                "phase": metadata["phase"],
                "projected_canonical_minutes": metadata[
                    "projected_canonical_minutes"
                ],
                "result": str(
                    args.results
                    / ("development" if args.action == "development" else "test")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
