"""Frozen matched-seed evaluation of independently switchable reward terms."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from rlredteam import provenance
from rlredteam.analyse import paired_comparison
from rlredteam.catalogue import CVECatalogue
from rlredteam.evaluation import (
    evaluate_policy,
    persist_bundle,
    policy_digest,
    sha256_file,
    validate_manifest,
    write_bundle,
)
from rlredteam.manifest import digest
from rlredteam.provenance import ExperimentManifest
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig, describe, make_env
from rlredteam.train import ppo_manifest_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


class ComponentAblationError(RuntimeError):
    """The preregistered component study is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ComponentExperimentConfig:
    experiment_id: str
    description: str
    design_version: int
    baseline_arm: str
    full_arm: str
    arms: tuple[str, ...]
    topology_seed: int
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    training_timesteps: int
    learning_rate_schedule: str
    postgres: bool
    log_steps: bool
    reward_config_dir: Path
    frozen_inputs: Path

    @classmethod
    def from_yaml(cls, path: Path) -> ComponentExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text())["experiment"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw["description"]),
            design_version=int(raw["design_version"]),
            baseline_arm=str(raw["baseline_arm"]),
            full_arm=str(raw["full_arm"]),
            arms=tuple(str(value) for value in raw["arms"]),
            topology_seed=int(raw["topology_seed"]),
            training_seeds=tuple(int(value) for value in raw["training_seeds"]),
            evaluation_seeds=tuple(int(value) for value in raw["evaluation_seeds"]),
            training_timesteps=int(raw["training_timesteps"]),
            learning_rate_schedule=str(raw["learning_rate_schedule"]),
            postgres=bool(raw.get("postgres", True)),
            log_steps=bool(raw.get("log_steps", True)),
            reward_config_dir=REPO_ROOT / raw["reward_config_dir"],
            frozen_inputs=REPO_ROOT / raw["frozen_inputs"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "zero_baseline",
            "cvss_only",
            "mitre_only",
            "informative_only",
            "failure_only",
            "objective_only",
            "all_components",
        }
        if self.design_version != 1 or set(self.arms) != expected:
            raise ComponentAblationError("design v1 requires the seven declared component arms")
        if self.baseline_arm not in self.arms or self.full_arm not in self.arms:
            raise ComponentAblationError("baseline and full arms must be in arms")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ComponentAblationError("training seeds must be non-empty and unique")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            raise ComponentAblationError("evaluation seeds must be non-empty and unique")
        if set(self.training_seeds) & set(self.evaluation_seeds):
            raise ComponentAblationError("training and evaluation seeds must be disjoint")
        if self.training_timesteps <= 0:
            raise ComponentAblationError("training_timesteps must be positive")
        for arm in self.arms:
            reward = self.reward_config(arm)
            if reward.components is None:
                raise ComponentAblationError(f"{arm}: explicit components block required")
        common = [self.reward_config(arm) for arm in self.arms]
        invariant_fields = (
            "mode",
            "cve_scale",
            "tactic_bonuses",
            "crown_jewel",
            "failed_action",
            "sparse_goal_reward",
            "weight",
            "first_success_only",
        )
        for field in invariant_fields:
            values = {
                json.dumps(getattr(item, field), sort_keys=True, default=str)
                for item in common
            }
            if len(values) != 1:
                raise ComponentAblationError(f"reward arms differ outside components: {field}")
        singleton = {
            arm: self.reward_config(arm).components.enabled()
            for arm in self.arms
            if arm.endswith("_only")
        }
        if any(len(enabled) != 1 for enabled in singleton.values()):
            raise ComponentAblationError("each *_only arm must enable exactly one component")
        if self.reward_config(self.baseline_arm).components.enabled():
            raise ComponentAblationError("zero baseline must enable no components")
        if len(self.reward_config(self.full_arm).components.enabled()) != 5:
            raise ComponentAblationError("full arm must enable all five components")

    def reward_path(self, arm: str) -> Path:
        return self.reward_config_dir / f"{arm}.yaml"

    def reward_config(self, arm: str) -> RewardConfig:
        if arm not in self.arms:
            raise ComponentAblationError(f"unknown arm: {arm}")
        return RewardConfig.from_yaml(self.reward_path(arm))

    def run_name(self, arm: str, seed: int) -> str:
        return f"{self.experiment_id}-{arm}-s{seed}-t{self.topology_seed}"

    def digest(self) -> str:
        payload = asdict(self)
        payload["reward_config_dir"] = str(self.reward_config_dir.relative_to(REPO_ROOT))
        payload["frozen_inputs"] = str(self.frozen_inputs.relative_to(REPO_ROOT))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def current_inputs(config: ComponentExperimentConfig) -> dict:
    topology = TopologyConfig.from_yaml()
    described = describe(make_env(topology, topology_seed=config.topology_seed))
    return {
        "experiment_id": config.experiment_id,
        "design_version": config.design_version,
        "experiment_config_sha256": config.digest(),
        "topology_seed": config.topology_seed,
        "topology_hash": provenance.topology_hash(described),
        "topology_config_hash": topology.config_hash(),
        "environment_config_hash": provenance.environment_config_hash(described),
        "cve_manifest_sha256": digest(CVECatalogue.open_default()),
        "ppo_config_hash": provenance._digest(
            ppo_manifest_config(config.learning_rate_schedule)
        ),
        "reward_config_hash": {
            arm: config.reward_config(arm).hash() for arm in config.arms
        },
        "training_seeds": list(config.training_seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
        "training_timesteps": config.training_timesteps,
    }


def protected_artifact_hash(config: ComponentExperimentConfig) -> str:
    """Content-address every earlier local result while excluding this study."""
    value = hashlib.sha256()
    for root_name in ("results", "runs"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(REPO_ROOT)
            study_entry = relative.parts[1] if len(relative.parts) > 1 else ""
            if study_entry == config.experiment_id or study_entry.startswith(
                f"{config.experiment_id}-"
            ):
                continue
            value.update(str(relative).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    value.update(chunk)
    return value.hexdigest()


def freeze_inputs(config: ComponentExperimentConfig) -> dict:
    frozen = current_inputs(config)
    frozen["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen["preregistered_from_commit"] = provenance.git_commit()
    config.frozen_inputs.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


def validate_frozen(config: ComponentExperimentConfig) -> dict:
    if not config.frozen_inputs.is_file():
        raise ComponentAblationError(f"missing frozen inputs: {config.frozen_inputs}")
    frozen = json.loads(config.frozen_inputs.read_text())
    current = current_inputs(config)
    mismatches = [
        f"{key}: frozen {frozen.get(key)!r}, current {value!r}"
        for key, value in current.items()
        if frozen.get(key) != value
    ]
    if provenance.git_dirty() is not False:
        mismatches.append("working tree is dirty or unknown")
    if mismatches:
        raise ComponentAblationError(
            "frozen component study mismatch:\n  " + "\n  ".join(mismatches)
        )
    return frozen


def train_run(config: ComponentExperimentConfig, arm: str, seed: int) -> None:
    command = [
        sys.executable,
        "-m",
        "rlredteam.train",
        "--experiment-id",
        config.experiment_id,
        "--condition",
        arm,
        "--seed",
        str(seed),
        "--topology-seed",
        str(config.topology_seed),
        "--timesteps",
        str(config.training_timesteps),
        "--learning-rate-schedule",
        config.learning_rate_schedule,
        "--reward-config",
        str(config.reward_path(arm)),
        "--frozen",
        str(config.frozen_inputs),
    ]
    if config.postgres:
        command.append("--postgres")
    if config.log_steps:
        command.append("--log-steps")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode:
        raise ComponentAblationError(f"training failed for {arm} seed {seed}")


def evaluate_run(
    config: ComponentExperimentConfig, arm: str, seed: int, result_root: Path
) -> None:
    from stable_baselines3 import PPO

    run_name = config.run_name(arm, seed)
    run_dir = RUNS_DIR / run_name
    manifest = ExperimentManifest.read(run_dir / "manifest.json")
    validate_manifest(manifest)
    expected_hash = config.reward_config(arm).hash()
    if manifest.reward_mode != arm or manifest.reward_config_hash != expected_hash:
        raise ComponentAblationError(f"{run_name}: reward provenance mismatch")
    checkpoint = run_dir / "model.zip"
    model = PPO.load(checkpoint, device="cpu")
    before = policy_digest(model)
    bundle = evaluate_policy(
        model,
        run_name=run_name,
        reward_mode=arm,
        training_seed=seed,
        evaluation_seeds=list(config.evaluation_seeds),
        topology_seed=config.topology_seed,
        reward_config=config.reward_config(arm),
    )
    after = policy_digest(model)
    if before != after:
        raise ComponentAblationError(f"{run_name}: policy changed during evaluation")
    metadata = {
        "phase": 18,
        "run_name": run_name,
        "arm": arm,
        "enabled_components": list(config.reward_config(arm).components.enabled()),
        "training_seed": seed,
        "evaluation_seeds": list(config.evaluation_seeds),
        "gradient_updates": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_before": before,
        "policy_sha256_after": after,
        "manifest": manifest.to_dict(),
    }
    if config.postgres:
        metadata.update(persist_bundle(bundle, manifest, checkpoint))
    write_bundle(bundle, result_root / "raw" / run_name, metadata)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _step_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_metrics(config: ComponentExperimentConfig, result_root: Path) -> list[dict]:
    total_hosts = len(
        describe(
            make_env(
                TopologyConfig.from_yaml(), topology_seed=config.topology_seed
            )
        )["hosts"]
    )
    metrics: list[dict] = []
    for arm in config.arms:
        for seed in config.training_seeds:
            run_name = config.run_name(arm, seed)
            raw = result_root / "raw" / run_name
            episodes = _rows(raw / "evaluation.csv")
            steps = _step_rows(raw / "steps.jsonl")
            if len(episodes) != len(config.evaluation_seeds):
                raise ComponentAblationError(f"{run_name}: incomplete evaluation seed set")
            successes = [row for row in episodes if row["goal_reached"] == "True"]
            successful_attack_steps = [
                row
                for row in steps
                if row["success"] and row["action_kind"] in {"exploit", "privesc"}
            ]
            metrics.append(
                {
                    "run_name": run_name,
                    "arm": arm,
                    "training_seed": seed,
                    "topology_seed": config.topology_seed,
                    "evaluation_episodes": len(episodes),
                    "success_rate": len(successes) / len(episodes),
                    "steps_to_goal": (
                        sum(int(row["length"]) for row in successes) / len(successes)
                        if successes
                        else None
                    ),
                    "native_reward": sum(float(row["native_return"]) for row in episodes)
                    / len(episodes),
                    "policy_reward": sum(float(row["policy_return"]) for row in episodes)
                    / len(episodes),
                    "discovery_coverage": sum(
                        int(row["discovered_hosts"]) / total_hosts for row in episodes
                    )
                    / len(episodes),
                    "failed_actions": sum(not bool(row["success"]) for row in steps)
                    / len(episodes),
                    "mitre_coverage": len(
                        {row["technique_id"] for row in steps if row.get("technique_id")}
                    ),
                    "attack_path_steps": len(successful_attack_steps) / len(episodes),
                    "attack_path_unique_targets": len(
                        {
                            tuple(row["target"])
                            for row in successful_attack_steps
                            if row.get("target")
                        }
                    ),
                }
            )
    return metrics


def analyse_metrics(config: ComponentExperimentConfig, metrics: list[dict]) -> dict:
    names = (
        "success_rate",
        "steps_to_goal",
        "native_reward",
        "discovery_coverage",
        "failed_actions",
        "mitre_coverage",
        "attack_path_steps",
        "attack_path_unique_targets",
    )
    protocol = {
        "primary_metrics": list(names),
        "descriptive_metrics": [],
        "statistics": {
            "alpha": 0.05,
            "correction": "bonferroni",
            "family_size": len(names) * (len(config.arms) - 1),
            "bootstrap_samples": 10_000,
            "confidence": 0.95,
        },
    }
    by_arm = {
        arm: {int(row["training_seed"]): row for row in metrics if row["arm"] == arm}
        for arm in config.arms
    }
    comparisons = []
    for arm in config.arms:
        if arm == config.baseline_arm:
            continue
        for metric in names:
            baseline = {
                seed: float(row[metric])
                for seed, row in by_arm[config.baseline_arm].items()
                if row[metric] is not None
            }
            treatment = {
                seed: float(row[metric])
                for seed, row in by_arm[arm].items()
                if row[metric] is not None
            }
            comparison = paired_comparison(metric, baseline, treatment, protocol).to_dict()
            comparison.update({"baseline_arm": config.baseline_arm, "treatment_arm": arm})
            comparisons.append(comparison)
    return {
        "phase": 18,
        "design": "matched_training_seed_component_ablation_v1",
        "unit_of_analysis": "training_seed",
        "baseline_arm": config.baseline_arm,
        "arms": list(config.arms),
        "metrics": metrics,
        "comparisons": comparisons,
        "complete": len(metrics) == len(config.arms) * len(config.training_seeds),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict | list | tuple)
                    else value
                    for key, value in row.items()
                }
            )


def write_results(
    config: ComponentExperimentConfig, result_root: Path, frozen: dict
) -> dict:
    metrics = collect_metrics(config, result_root)
    report = analyse_metrics(config, metrics)
    (result_root / "metadata").mkdir(parents=True, exist_ok=True)
    (result_root / "summaries").mkdir(parents=True, exist_ok=True)
    (result_root / "tables").mkdir(parents=True, exist_ok=True)
    (result_root / "metadata" / "experiment.json").write_text(
        json.dumps(
            {
                "phase": 18,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "code_commit": provenance.git_commit(),
                "experiment_config_sha256": config.digest(),
                "frozen_inputs": frozen,
                "gradient_updates_during_evaluation": False,
                "complete": report["complete"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (result_root / "summaries" / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _write_csv(result_root / "summaries" / "metrics.csv", metrics)
    _write_csv(result_root / "tables" / "paired_statistics.csv", report["comparisons"])
    (result_root / "README.md").write_text(
        "# Phase 18 reward-component ablation\n\n"
        "Frozen-policy evaluation of seven matched-seed PPO arms. The zero-reward "
        "baseline and each single-component arm differ only in the explicit reward "
        "component switches; topology, CVEs, PPO hyperparameters and budgets are fixed.\n"
    )
    return report


def execute(
    config: ComponentExperimentConfig, *, train: bool = True, evaluate: bool = True
) -> dict:
    frozen = validate_frozen(config)
    protected_before = protected_artifact_hash(config)
    result_root = RESULTS_DIR / config.experiment_id
    for arm in config.arms:
        for seed in config.training_seeds:
            if train:
                train_run(config, arm, seed)
            if evaluate:
                evaluate_run(config, arm, seed, result_root)
    report = write_results(config, result_root, frozen)
    protected_after = protected_artifact_hash(config)
    if protected_before != protected_after:
        raise ComponentAblationError("an earlier frozen run/result artifact changed")
    metadata_path = result_root / "metadata" / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["protected_artifacts_sha256_before"] = protected_before
    metadata["protected_artifacts_sha256_after"] = protected_after
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return report
