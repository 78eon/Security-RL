#!/usr/bin/env python3
"""Freeze, validate and run the Phase 10 matched graph-policy study."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rlredteam.enterprise.graph_policy import GraphPolicyResearchConfig
from rlredteam.enterprise.graph_policy_study import (
    DEFAULT_FROZEN_INPUTS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_RUN_ROOT,
    GraphPolicyStudyError,
    aggregate_seed_metrics,
    analyse_seed_metrics,
    arm_run_name,
    evaluate_arm,
    freeze_inputs,
    git_commit,
    load_frozen_inputs,
    load_model,
    persist_and_write_run_evaluation,
    train_arm,
    validate_finite_evidence,
    validate_frozen_inputs,
    validate_paired_training_isolation,
    validate_training_manifest,
    write_study_summary,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("freeze", "dry-run", "development", "run"))
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/graph_policy.yaml"),
    )
    result.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN_INPUTS)
    result.add_argument("--runs", type=Path, default=DEFAULT_RUN_ROOT)
    result.add_argument("--results", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--timesteps", type=int, help="development action only")
    result.add_argument("--limit-topologies", type=int, default=0)
    result.add_argument("--postgres", action="store_true")
    result.add_argument("--allow-dirty", action="store_true", help="development only")
    return result


def project_parallel_training_minutes(
    elapsed_by_arm: dict[str, float],
    *,
    seeds: int,
    arms: tuple[str, ...],
    workers: int,
) -> float:
    if workers <= 0 or seeds <= 0 or set(elapsed_by_arm) != set(arms):
        raise GraphPolicyStudyError("parallel runtime projection inputs are incomplete")
    durations = {arm: float(elapsed_by_arm[arm]) for arm in arms}
    if any(not math.isfinite(value) or value <= 0 for value in durations.values()):
        raise GraphPolicyStudyError("parallel runtime timings must be finite and positive")
    slots = [0.0] * workers
    heapq.heapify(slots)
    for _ in range(seeds):
        for arm in arms:
            available = heapq.heappop(slots)
            heapq.heappush(slots, available + durations[arm])
    return max(slots) / 60.0


def _train_job(payload: tuple) -> dict:
    run_dir, arm, seed, config, frozen, timesteps = payload
    return train_arm(
        run_dir,
        arm=arm,
        training_seed=seed,
        config=config,
        frozen_inputs=frozen,
        timesteps=timesteps,
        development=False,
    )


def validate_development_gate(path: Path, config: GraphPolicyResearchConfig) -> dict:
    if not path.is_file():
        raise GraphPolicyStudyError(
            "canonical run requires the full excluded-seed development gate"
        )
    metadata = json.loads(path.read_text())
    checks = {
        "phase": (metadata.get("phase"), "development"),
        "complete": (metadata.get("complete"), True),
        "scientific config": (
            metadata.get("scientific_config_hash"),
            config.scientific_digest(),
        ),
        "training budget": (metadata.get("training_timesteps"), config.total_timesteps),
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
    elapsed_by_arm = metadata.get("training_elapsed_seconds_by_arm", {})
    try:
        projected = project_parallel_training_minutes(
            {arm: float(elapsed_by_arm[arm]) for arm in config.arms},
            seeds=len(config.training_seeds),
            arms=config.arms,
            workers=config.parallel_training_workers,
        )
    except (KeyError, TypeError, ValueError, GraphPolicyStudyError) as exc:
        failures.append(f"development timing is incomplete: {exc}")
        projected = float("inf")
    if projected > config.runtime_cap_minutes:
        failures.append(
            f"projected runtime {projected:.1f} exceeds cap {config.runtime_cap_minutes}"
        )
    manifests = metadata.get("training_manifests", [])
    observed = {
        (item.get("arm"), item.get("training_seed"), item.get("development")) for item in manifests
    }
    if observed != {(arm, config.development_seed, True) for arm in config.arms}:
        failures.append("development manifests do not cover both arms on the excluded seed")
    if failures:
        raise GraphPolicyStudyError("Phase 10 development gate failed:\n  " + "\n  ".join(failures))
    metadata["parallel_projected_canonical_minutes"] = projected
    return metadata


def run_study(args: argparse.Namespace, config: GraphPolicyResearchConfig) -> dict:
    development = args.action == "development"
    if args.timesteps is not None and not development:
        raise GraphPolicyStudyError("--timesteps is allowed only for development")
    if args.limit_topologies and not development:
        raise GraphPolicyStudyError("canonical test topology set cannot be limited")
    development_gate = None
    if not development:
        development_gate = validate_development_gate(
            args.results / "development/metadata/study.json", config
        )
    frozen = None if development else load_frozen_inputs(args.frozen)
    if frozen is not None:
        validate_frozen_inputs(frozen, config)

    seeds = (config.development_seed,) if development else config.training_seeds
    split_name = "validation" if development else "test"
    topology_seeds = config.topology_splits[split_name]
    if args.limit_topologies:
        topology_seeds = topology_seeds[: args.limit_topologies]
    timesteps = args.timesteps or config.total_timesteps
    result_root = args.results / ("development" if development else "test")
    jobs = [
        (
            args.runs
            / ("development" if development else "canonical")
            / arm_run_name(config, arm, seed),
            arm,
            seed,
        )
        for seed in seeds
        for arm in config.arms
    ]
    training_started = time.monotonic()
    if development:
        training_manifests = [
            train_arm(
                run_dir,
                arm=arm,
                training_seed=seed,
                config=config,
                frozen_inputs=frozen,
                timesteps=timesteps,
                development=True,
                allow_dirty=args.allow_dirty,
            )
            for run_dir, arm, seed in jobs
        ]
    else:
        context = multiprocessing.get_context("spawn")
        completed: dict[tuple[str, int], dict] = {}
        with ProcessPoolExecutor(
            max_workers=config.parallel_training_workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _train_job,
                    (run_dir, arm, seed, config, frozen, timesteps),
                ): (arm, seed)
                for run_dir, arm, seed in jobs
            }
            for future in as_completed(futures):
                arm, seed = futures[future]
                completed[(arm, seed)] = future.result()
        training_manifests = [completed[(arm, seed)] for _, arm, seed in jobs]
    training_wall_seconds = time.monotonic() - training_started
    manifests_by_key = {
        (item["arm"], int(item["training_seed"])): item for item in training_manifests
    }
    for seed in seeds:
        validate_paired_training_isolation(
            manifests_by_key[(config.arms[0], seed)],
            manifests_by_key[(config.arms[1], seed)],
        )

    all_episodes = []
    evaluation_metadata = []
    for run_dir, arm, seed in jobs:
        run_name = arm_run_name(config, arm, seed)
        manifest = manifests_by_key[(arm, seed)]
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
        model = load_model(checkpoint)
        episodes, steps, integrity = evaluate_arm(
            model,
            arm=arm,
            training_seed=seed,
            config=config,
            topology_seeds=topology_seeds,
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
        evaluation_metadata.append(metadata)

    seed_metrics = aggregate_seed_metrics(
        all_episodes, config, expected_topology_seeds=topology_seeds
    )
    report = analyse_seed_metrics(seed_metrics, config, expected_seeds=seeds)
    if not report["complete"]:
        raise GraphPolicyStudyError("Phase 10 study is missing a matched arm/seed")
    elapsed_by_arm = {
        arm: sum(
            float(item["training_elapsed_seconds"])
            for item in training_manifests
            if item["arm"] == arm
        )
        for arm in config.arms
    }
    projected_minutes = project_parallel_training_minutes(
        {arm: elapsed_by_arm[arm] / len(seeds) for arm in config.arms},
        seeds=len(config.training_seeds),
        arms=config.arms,
        workers=config.parallel_training_workers,
    )
    metadata = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "phase": "development" if development else "canonical_test",
        "complete": report["complete"],
        "code_commit": git_commit(),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "frozen_inputs": frozen,
        "topology_split": split_name,
        "topology_seeds": list(topology_seeds),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "training_timesteps": timesteps,
        "training_manifests": training_manifests,
        "evaluation_metadata": evaluation_metadata,
        "training_elapsed_seconds_by_arm": elapsed_by_arm,
        "training_wall_seconds": training_wall_seconds,
        "parallel_training_workers": 1 if development else config.parallel_training_workers,
        "projected_canonical_minutes": projected_minutes,
        "development_gate": development_gate,
        "runtime_cap_minutes": config.runtime_cap_minutes,
    }
    write_study_summary(
        result_root,
        seed_metrics=seed_metrics,
        report=report,
        metadata=metadata,
    )
    (result_root / "README.md").write_text(
        "# Phase 10 graph-policy study\n\n"
        f"Phase: `{metadata['phase']}`  \n"
        f"Complete matched grid: `{report['complete']}`  \n"
        f"Topology split: `{split_name}`  \n"
        f"Training seeds: `{list(seeds)}`  \n"
        "See `summaries/analysis.json` and each `raw/` run for evidence.\n"
    )
    if (
        development
        and timesteps == config.total_timesteps
        and projected_minutes > config.runtime_cap_minutes
    ):
        raise GraphPolicyStudyError(
            f"projected canonical runtime {projected_minutes:.1f} minutes exceeds "
            f"cap {config.runtime_cap_minutes}"
        )
    return metadata


def main() -> None:
    args = parser().parse_args()
    config = GraphPolicyResearchConfig.from_yaml(args.config)
    if args.action == "freeze":
        print(json.dumps(freeze_inputs(args.frozen, config), indent=2, sort_keys=True))
        return
    if args.action == "dry-run":
        frozen = load_frozen_inputs(args.frozen)
        validate_frozen_inputs(frozen, config)
        grid = [
            arm_run_name(config, arm, seed) for arm in config.arms for seed in config.training_seeds
        ]
        print(json.dumps({"complete_grid": len(grid), "runs": grid}, indent=2))
        return
    metadata = run_study(args, config)
    print(
        json.dumps(
            {
                "complete": metadata["complete"],
                "phase": metadata["phase"],
                "projected_canonical_minutes": metadata["projected_canonical_minutes"],
                "result": str(
                    args.results / ("development" if args.action == "development" else "test")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
