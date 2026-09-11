#!/usr/bin/env python3
"""Develop, freeze and run the Phase 13 matched red-blue study."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rlredteam.enterprise.generalisation import git_commit, sha256_file
from rlredteam.enterprise.multiagent import MultiAgentResearchConfig, MultiAgentStudyError
from rlredteam.enterprise.multiagent_study import (
    DEFAULT_FROZEN_INPUTS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_RUN_ROOT,
    aggregate_seed_metrics,
    analyse_seed_metrics,
    arm_run_name,
    evaluate_red_policy,
    freeze_inputs,
    load_frozen_inputs,
    load_maskable_model,
    persist_and_write_run_evaluation,
    train_defender,
    train_red_arm,
    validate_finite_evidence,
    validate_frozen_inputs,
    validate_paired_red_training_isolation,
    validate_red_training_manifest,
    write_study_summary,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("freeze", "dry-run", "development", "run"))
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/multiagent_defense.yaml"),
    )
    result.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN_INPUTS)
    result.add_argument("--runs", type=Path, default=DEFAULT_RUN_ROOT)
    result.add_argument("--results", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--timesteps", type=int, help="development red/blue budget only")
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
        raise MultiAgentStudyError("parallel runtime projection inputs are incomplete")
    durations = {arm: float(elapsed_by_arm[arm]) for arm in arms}
    if any(not math.isfinite(value) or value <= 0 for value in durations.values()):
        raise MultiAgentStudyError("parallel runtime timings must be finite and positive")
    slots = [0.0] * workers
    heapq.heapify(slots)
    for _ in range(seeds):
        for arm in arms:
            available = heapq.heappop(slots)
            heapq.heappush(slots, available + durations[arm])
    return max(slots) / 60.0


def defender_paths(runs_root: Path, config: MultiAgentResearchConfig) -> tuple[Path, Path]:
    directory = runs_root / "development" / f"{config.experiment_id}-defender"
    return directory / "model.zip", directory / "training_manifest.json"


def validate_development_gate(
    path: Path,
    config: MultiAgentResearchConfig,
    *,
    defender_checkpoint: Path,
) -> dict:
    if not path.is_file():
        raise MultiAgentStudyError(
            "canonical run requires the full excluded-seed Phase 13 development gate"
        )
    metadata = json.loads(path.read_text())
    checks = {
        "phase": (metadata.get("phase"), "development"),
        "complete": (metadata.get("complete"), True),
        "scientific config": (
            metadata.get("scientific_config_hash"),
            config.scientific_digest(),
        ),
        "red budget": (metadata.get("red_training_timesteps"), config.red_total_timesteps),
        "blue budget": (
            metadata.get("defender_training_timesteps"),
            config.defender_total_timesteps,
        ),
        "validation topology set": (
            metadata.get("topology_seeds"),
            list(config.topology_splits["validation"]),
        ),
        "defender checkpoint": (
            metadata.get("defender_checkpoint_sha256"),
            sha256_file(defender_checkpoint),
        ),
    }
    failures = [
        f"{name}: development={actual!r}, required={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    elapsed = metadata.get("red_training_elapsed_seconds_by_arm", {})
    try:
        projected = project_parallel_training_minutes(
            {arm: float(elapsed[arm]) for arm in config.arms},
            seeds=len(config.training_seeds),
            arms=config.arms,
            workers=config.parallel_training_workers,
        )
    except (KeyError, TypeError, ValueError, MultiAgentStudyError) as exc:
        failures.append(f"development timing is incomplete: {exc}")
        projected = float("inf")
    if projected > config.runtime_cap_minutes:
        failures.append(
            f"projected runtime {projected:.1f} exceeds cap {config.runtime_cap_minutes}"
        )
    manifests = metadata.get("red_training_manifests", [])
    observed = {
        (item.get("arm"), item.get("training_seed"), item.get("development"))
        for item in manifests
    }
    if observed != {(arm, config.development_seed, True) for arm in config.arms}:
        failures.append("development manifests do not cover both excluded-seed red arms")
    defender = metadata.get("defender_training_manifest", {})
    if defender.get("actual_defender_training_timesteps") != config.defender_total_timesteps:
        failures.append("development defender did not consume the exact full budget")
    if defender.get("policy_sha256_before_training") == defender.get("policy_sha256"):
        failures.append("development defender policy did not change")
    if defender.get("bootstrap_red_policy_sha256_before") != defender.get(
        "bootstrap_red_policy_sha256_after"
    ):
        failures.append("bootstrap red changed during defender optimisation")
    for field in ("threshold_adjustments", "patch_schedules", "patch_activations"):
        if int(defender.get(field, 0)) <= 0:
            failures.append(f"learned defender did not demonstrate {field}")
    evaluations = metadata.get("evaluation_metadata", [])
    if len(evaluations) != 2 or any(
        item.get("evaluation_reset_count")
        != len(config.train_profiles)
        * len(config.topology_splits["validation"])
        * len(config.evaluation_episode_seeds)
        for item in evaluations
    ):
        failures.append("development validation grid is incomplete")
    if failures:
        raise MultiAgentStudyError(
            "Phase 13 development gate failed:\n  " + "\n  ".join(failures)
        )
    metadata["parallel_projected_canonical_minutes"] = projected
    return metadata


def _train_job(payload: tuple) -> dict:
    run_dir, arm, seed, defender_checkpoint, config, frozen, timesteps = payload
    return train_red_arm(
        run_dir,
        arm=arm,
        training_seed=seed,
        defender_checkpoint=defender_checkpoint,
        config=config,
        frozen_inputs=frozen,
        timesteps=timesteps,
        development=False,
    )


def _evaluate_runs(
    jobs: list[tuple[Path, str, int]],
    manifests: list[dict],
    *,
    defender_checkpoint: Path,
    config: MultiAgentResearchConfig,
    frozen: dict | None,
    development: bool,
    allow_dirty: bool,
    topology_seeds: tuple[int, ...],
    split_name: str,
    result_root: Path,
    postgres: bool,
) -> tuple[list[dict], list[dict]]:
    by_key = {(item["arm"], int(item["training_seed"])): item for item in manifests}
    episodes_all: list[dict] = []
    metadata_all: list[dict] = []
    for run_dir, arm, seed in jobs:
        manifest = by_key[(arm, seed)]
        checkpoint = run_dir / "model.zip"
        validate_red_training_manifest(
            manifest,
            checkpoint,
            arm=arm,
            training_seed=seed,
            defender_checkpoint=defender_checkpoint,
            config=config,
            frozen_inputs=frozen,
            development=development,
            allow_unverifiable=development and allow_dirty,
        )
        model = load_maskable_model(checkpoint)
        episodes, steps, integrity = evaluate_red_policy(
            model,
            arm=arm,
            training_seed=seed,
            defender_checkpoint=defender_checkpoint,
            config=config,
            topology_seeds=topology_seeds,
        )
        validate_finite_evidence(episodes, steps)
        metadata = persist_and_write_run_evaluation(
            result_root / "raw" / arm_run_name(config, arm, seed),
            episodes=episodes,
            steps=steps,
            integrity=integrity,
            training_manifest=manifest,
            checkpoint=checkpoint,
            split_name=split_name,
            postgres=postgres,
        )
        episodes_all.extend(episodes)
        metadata_all.append(metadata)
    return episodes_all, metadata_all


def run_development(args: argparse.Namespace, config: MultiAgentResearchConfig) -> dict:
    if args.limit_topologies < 0:
        raise MultiAgentStudyError("topology limit cannot be negative")
    budget = args.timesteps or config.red_total_timesteps
    blue_budget = args.timesteps or config.defender_total_timesteps
    root = args.runs / "development"
    static_arm, adaptive_arm = config.arms
    static_dir = root / arm_run_name(config, static_arm, config.development_seed)
    adaptive_dir = root / arm_run_name(config, adaptive_arm, config.development_seed)

    started = time.monotonic()
    static = train_red_arm(
        static_dir,
        arm=static_arm,
        training_seed=config.development_seed,
        defender_checkpoint=Path("unused-static-defender.zip"),
        config=config,
        timesteps=budget,
        development=True,
        allow_dirty=args.allow_dirty,
    )
    defender_checkpoint, _ = defender_paths(args.runs, config)
    defender = train_defender(
        defender_checkpoint.parent,
        bootstrap_red_checkpoint=static_dir / "model.zip",
        config=config,
        timesteps=blue_budget,
        allow_dirty=args.allow_dirty,
    )
    adaptive = train_red_arm(
        adaptive_dir,
        arm=adaptive_arm,
        training_seed=config.development_seed,
        defender_checkpoint=defender_checkpoint,
        config=config,
        timesteps=budget,
        development=True,
        allow_dirty=args.allow_dirty,
    )
    manifests = [static, adaptive]
    validate_paired_red_training_isolation(static, adaptive)
    topologies = config.topology_splits["validation"]
    if args.limit_topologies:
        topologies = topologies[: args.limit_topologies]
    jobs = [
        (static_dir, static_arm, config.development_seed),
        (adaptive_dir, adaptive_arm, config.development_seed),
    ]
    result_root = args.results / "development"
    all_episodes, evaluation_metadata = _evaluate_runs(
        jobs,
        manifests,
        defender_checkpoint=defender_checkpoint,
        config=config,
        frozen=None,
        development=True,
        allow_dirty=args.allow_dirty,
        topology_seeds=topologies,
        split_name="validation",
        result_root=result_root,
        postgres=args.postgres,
    )
    seed_metrics = aggregate_seed_metrics(
        all_episodes, config, expected_topology_seeds=topologies
    )
    report = analyse_seed_metrics(
        seed_metrics, config, expected_seeds=(config.development_seed,)
    )
    elapsed = {
        arm: float(item["red_training_elapsed_seconds"])
        for arm, item in zip(config.arms, manifests, strict=True)
    }
    projected = project_parallel_training_minutes(
        elapsed,
        seeds=len(config.training_seeds),
        arms=config.arms,
        workers=config.parallel_training_workers,
    )
    full_budget = (
        budget == config.red_total_timesteps
        and blue_budget == config.defender_total_timesteps
        and topologies == config.topology_splits["validation"]
    )
    metadata = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "phase": "development",
        "complete": bool(report["complete"] and full_budget),
        "code_commit": git_commit(),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "topology_split": "validation",
        "topology_seeds": list(topologies),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "red_training_timesteps": budget,
        "defender_training_timesteps": blue_budget,
        "red_training_manifests": manifests,
        "defender_training_manifest": defender,
        "defender_checkpoint": str(defender_checkpoint),
        "defender_checkpoint_sha256": sha256_file(defender_checkpoint),
        "evaluation_metadata": evaluation_metadata,
        "red_training_elapsed_seconds_by_arm": elapsed,
        "defender_training_elapsed_seconds": defender["training_elapsed_seconds"],
        "study_wall_seconds": time.monotonic() - started,
        "parallel_projected_canonical_minutes": projected,
        "parallel_training_workers": 1,
        "runtime_cap_minutes": config.runtime_cap_minutes,
    }
    write_study_summary(
        result_root,
        seed_metrics=seed_metrics,
        report=report,
        metadata=metadata,
    )
    (result_root / "README.md").write_text(
        "# Phase 13 excluded-seed development\n\n"
        f"Full-budget gate complete: `{metadata['complete']}`  \n"
        f"Projected canonical red-training time: `{projected:.2f}` minutes.\n"
    )
    if full_budget and projected > config.runtime_cap_minutes:
        raise MultiAgentStudyError(
            f"projected canonical runtime {projected:.1f} minutes exceeds "
            f"cap {config.runtime_cap_minutes}"
        )
    return metadata


def run_canonical(args: argparse.Namespace, config: MultiAgentResearchConfig) -> dict:
    defender_checkpoint, _ = defender_paths(args.runs, config)
    development_path = args.results / "development/metadata/study.json"
    development_gate = validate_development_gate(
        development_path,
        config,
        defender_checkpoint=defender_checkpoint,
    )
    frozen = load_frozen_inputs(args.frozen)
    validate_frozen_inputs(
        frozen,
        config,
        defender_checkpoint=defender_checkpoint,
    )
    jobs = [
        (
            args.runs / "canonical" / arm_run_name(config, arm, seed),
            arm,
            seed,
        )
        for seed in config.training_seeds
        for arm in config.arms
    ]
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    completed: dict[tuple[str, int], dict] = {}
    with ProcessPoolExecutor(
        max_workers=config.parallel_training_workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _train_job,
                (
                    run_dir,
                    arm,
                    seed,
                    defender_checkpoint,
                    config,
                    frozen,
                    config.red_total_timesteps,
                ),
            ): (arm, seed)
            for run_dir, arm, seed in jobs
        }
        for future in as_completed(futures):
            arm, seed = futures[future]
            completed[(arm, seed)] = future.result()
    training_wall_seconds = time.monotonic() - started
    manifests = [completed[(arm, seed)] for _, arm, seed in jobs]
    by_key = {(item["arm"], int(item["training_seed"])): item for item in manifests}
    for seed in config.training_seeds:
        validate_paired_red_training_isolation(
            by_key[(config.arms[0], seed)], by_key[(config.arms[1], seed)]
        )
    result_root = args.results / "test"
    all_episodes, evaluation_metadata = _evaluate_runs(
        jobs,
        manifests,
        defender_checkpoint=defender_checkpoint,
        config=config,
        frozen=frozen,
        development=False,
        allow_dirty=False,
        topology_seeds=config.topology_splits["test"],
        split_name="test",
        result_root=result_root,
        postgres=args.postgres,
    )
    seed_metrics = aggregate_seed_metrics(
        all_episodes,
        config,
        expected_topology_seeds=config.topology_splits["test"],
    )
    report = analyse_seed_metrics(seed_metrics, config)
    if not report["complete"]:
        raise MultiAgentStudyError("Phase 13 canonical study lacks a matched arm/seed")
    elapsed = {
        arm: sum(
            float(item["red_training_elapsed_seconds"])
            for item in manifests
            if item["arm"] == arm
        )
        for arm in config.arms
    }
    metadata = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "phase": "canonical_test",
        "complete": True,
        "code_commit": git_commit(),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "frozen_inputs": frozen,
        "topology_split": "test",
        "topology_seeds": list(config.topology_splits["test"]),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "red_training_timesteps": config.red_total_timesteps,
        "aggregate_red_training_timesteps": (
            len(jobs) * config.red_total_timesteps
        ),
        "red_training_manifests": manifests,
        "defender_training_manifest": development_gate["defender_training_manifest"],
        "defender_checkpoint": str(defender_checkpoint),
        "defender_checkpoint_sha256": sha256_file(defender_checkpoint),
        "evaluation_metadata": evaluation_metadata,
        "red_training_elapsed_seconds_by_arm": elapsed,
        "training_wall_seconds": training_wall_seconds,
        "parallel_training_workers": config.parallel_training_workers,
        "projected_canonical_minutes": development_gate[
            "parallel_projected_canonical_minutes"
        ],
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
        "# Phase 13 matched red-blue study\n\n"
        "Simulation-only frozen-policy evidence.  \n"
        f"Complete matched grid: `{report['complete']}`  \n"
        "See `summaries/analysis.json` and each `raw/` run.\n"
    )
    if training_wall_seconds > config.runtime_cap_minutes * 60:
        raise MultiAgentStudyError("canonical red training exceeded the frozen runtime cap")
    return metadata


def main() -> None:
    args = parser().parse_args()
    config = MultiAgentResearchConfig.from_yaml(args.config)
    defender_checkpoint, _ = defender_paths(args.runs, config)
    if args.action == "freeze":
        gate_path = args.results / "development/metadata/study.json"
        validate_development_gate(
            gate_path,
            config,
            defender_checkpoint=defender_checkpoint,
        )
        result = freeze_inputs(
            args.frozen,
            development_study=gate_path,
            defender_checkpoint=defender_checkpoint,
            config=config,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.action == "dry-run":
        frozen = load_frozen_inputs(args.frozen)
        validate_development_gate(
            args.results / "development/metadata/study.json",
            config,
            defender_checkpoint=defender_checkpoint,
        )
        validate_frozen_inputs(
            frozen,
            config,
            defender_checkpoint=defender_checkpoint,
        )
        grid = [
            arm_run_name(config, arm, seed)
            for seed in config.training_seeds
            for arm in config.arms
        ]
        print(json.dumps({"complete_grid": len(grid), "runs": grid}, indent=2))
        return
    if args.timesteps is not None and args.action != "development":
        raise MultiAgentStudyError("--timesteps is allowed only for development")
    if args.limit_topologies and args.action != "development":
        raise MultiAgentStudyError("canonical test topology set cannot be limited")
    metadata = (
        run_development(args, config)
        if args.action == "development"
        else run_canonical(args, config)
    )
    result = args.results / ("development" if args.action == "development" else "test")
    print(
        json.dumps(
            {
                "complete": metadata["complete"],
                "phase": metadata["phase"],
                "result": str(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
