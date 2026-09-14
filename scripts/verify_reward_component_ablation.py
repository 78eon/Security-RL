#!/usr/bin/env python3
"""Verify Phase 18 frozen design, paired evaluation and immutable policies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.component_ablation import (  # noqa: E402
    ComponentExperimentConfig,
    validate_frozen,
)
from rlredteam.provenance import ExperimentManifest  # noqa: E402


def verify(config: ComponentExperimentConfig) -> list[tuple[str, bool, str]]:
    frozen = validate_frozen(config)
    root = REPO_ROOT / "results" / config.experiment_id
    checks: list[tuple[str, bool, str]] = []
    analysis_path = root / "summaries/analysis.json"
    experiment_path = root / "metadata/experiment.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else {}
    experiment = json.loads(experiment_path.read_text()) if experiment_path.is_file() else {}
    checks.append(("analysis complete", analysis.get("complete") is True, str(analysis_path)))
    checks.append(
        (
            "earlier artifacts unchanged",
            experiment.get("protected_artifacts_sha256_before")
            == experiment.get("protected_artifacts_sha256_after")
            and bool(experiment.get("protected_artifacts_sha256_before")),
            str(experiment_path),
        )
    )
    expected_eval = list(config.evaluation_seeds)
    ppo_hashes: set[str] = set()
    topology_hashes: set[str] = set()
    complete_runs = 0
    for arm in config.arms:
        for seed in config.training_seeds:
            run_name = config.run_name(arm, seed)
            run_dir = REPO_ROOT / "runs" / run_name
            raw = root / "raw" / run_name
            manifest_path = run_dir / "manifest.json"
            metadata_path = raw / "evaluation_metadata.json"
            episode_path = raw / "evaluation.csv"
            if not all(path.is_file() for path in (manifest_path, metadata_path, episode_path)):
                checks.append((f"{run_name} artifacts", False, "missing run/evaluation file"))
                continue
            manifest = ExperimentManifest.read(manifest_path)
            metadata = json.loads(metadata_path.read_text())
            with episode_path.open(newline="") as handle:
                episode_seeds = [int(row["evaluation_seed"]) for row in csv.DictReader(handle)]
            valid = (
                manifest.reward_mode == arm
                and manifest.reward_config_hash == frozen["reward_config_hash"][arm]
                and metadata["evaluation_seeds"] == expected_eval
                and episode_seeds == expected_eval
                and metadata["gradient_updates"] is False
                and metadata["policy_sha256_before"] == metadata["policy_sha256_after"]
            )
            checks.append(
                (f"{run_name} controlled evaluation", valid, "matched seeds/frozen policy")
            )
            complete_runs += int(valid)
            ppo_hashes.add(manifest.ppo_config_hash)
            topology_hashes.add(manifest.topology_hash)
    expected_runs = len(config.arms) * len(config.training_seeds)
    checks.extend(
        [
            (
                "all planned runs",
                complete_runs == expected_runs,
                f"{complete_runs}/{expected_runs}",
            ),
            (
                "identical PPO protocol",
                ppo_hashes == {frozen["ppo_config_hash"]},
                str(ppo_hashes),
            ),
            (
                "identical fixed topology",
                topology_hashes == {frozen["topology_hash"]},
                str(topology_hashes),
            ),
        ]
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiments/reward_components_01.yaml",
    )
    args = parser.parse_args(argv)
    checks = verify(ComponentExperimentConfig.from_yaml(args.config))
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
