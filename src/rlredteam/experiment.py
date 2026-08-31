"""Config-driven fixed-topology sparse-versus-shaped experiment harness."""

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
from rlredteam.analyse import RunMetrics, analyse, format_table, load_protocol, metrics_for_run
from rlredteam.catalogue import CVECatalogue
from rlredteam.evaluation import evaluate_checkpoint, sha256_file
from rlredteam.manifest import digest
from rlredteam.reporting import write_figures, write_trajectory_evidence
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig, describe, make_env
from rlredteam.train import ppo_manifest_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


class ExperimentError(RuntimeError):
    """The controlled experiment is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    arms: tuple[str, ...]
    topology_seed: int
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    training_timesteps: int
    postgres: bool
    log_steps: bool
    frozen_inputs: Path
    metrics: Path
    learning_rate_schedule: str = "constant"

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text())["experiment"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw.get("description", "")),
            arms=tuple(raw["arms"]),
            topology_seed=int(raw["topology_seed"]),
            training_seeds=tuple(int(seed) for seed in raw["training_seeds"]),
            evaluation_seeds=tuple(int(seed) for seed in raw["evaluation_seeds"]),
            training_timesteps=int(raw["training_timesteps"]),
            postgres=bool(raw.get("postgres", True)),
            log_steps=bool(raw.get("log_steps", True)),
            frozen_inputs=REPO_ROOT / raw["frozen_inputs"],
            metrics=REPO_ROOT / raw["metrics"],
            learning_rate_schedule=str(raw.get("learning_rate_schedule", "constant")),
        )
        config.validate()
        config.validate_metrics_protocol()
        return config

    def validate(self) -> None:
        if self.arms != ("sparse", "shaped"):
            raise ExperimentError("Essential experiment arms must be [sparse, shaped]")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ExperimentError("training seeds must be non-empty and unique")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            raise ExperimentError("evaluation seeds must be non-empty and unique")
        if set(self.training_seeds) & set(self.evaluation_seeds):
            raise ExperimentError("training and evaluation seeds must be disjoint")
        if self.training_timesteps <= 0:
            raise ExperimentError("training_timesteps must be positive")
        if self.learning_rate_schedule not in ("constant", "linear_to_zero"):
            raise ExperimentError(
                f"unknown learning-rate schedule: {self.learning_rate_schedule}"
            )

    def validate_metrics_protocol(self) -> None:
        """Require config and analysis to name the same experimental units."""
        if not self.metrics.is_file():
            raise ExperimentError(f"missing metrics protocol: {self.metrics}")
        try:
            protocol = load_protocol(self.metrics)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(f"invalid metrics protocol: {exc}") from exc
        evaluation = protocol.get("evaluation", {})
        mismatches = []
        if tuple(evaluation.get("seeds", ())) != self.training_seeds:
            mismatches.append(
                f"training seeds config={list(self.training_seeds)}, "
                f"metrics={evaluation.get('seeds')}"
            )
        if tuple(evaluation.get("episode_seeds", ())) != self.evaluation_seeds:
            mismatches.append(
                f"evaluation seeds config={list(self.evaluation_seeds)}, "
                f"metrics={evaluation.get('episode_seeds')}"
            )
        design = protocol.get("design")
        if design:
            controlled_fields = {
                "training_budget_timesteps": self.training_timesteps,
                "learning_rate_schedule": self.learning_rate_schedule,
                "topology_seed": self.topology_seed,
            }
            for field_name, expected in controlled_fields.items():
                if design.get(field_name) != expected:
                    mismatches.append(
                        f"{field_name} config={expected!r}, "
                        f"metrics={design.get(field_name)!r}"
                    )
        if mismatches:
            raise ExperimentError("metrics protocol mismatch:\n  " + "\n  ".join(mismatches))

    def run_name(self, arm: str, seed: int) -> str:
        return f"{self.experiment_id}-{arm}-s{seed}-t{self.topology_seed}"

    def digest(self) -> str:
        payload = asdict(self)
        # Preserve every already-frozen historical constant-rate digest.
        if payload["learning_rate_schedule"] == "constant":
            payload.pop("learning_rate_schedule")
        payload["frozen_inputs"] = str(self.frozen_inputs.relative_to(REPO_ROOT))
        payload["metrics"] = str(self.metrics.relative_to(REPO_ROOT))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def current_inputs(config: ExperimentConfig) -> dict:
    topology_config = TopologyConfig.from_yaml()
    described = describe(make_env(topology_config, topology_seed=config.topology_seed))
    return {
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "topology_seed": config.topology_seed,
        "topology_hash": provenance.topology_hash(described),
        "topology_config_hash": topology_config.config_hash(),
        "environment_config_hash": provenance.environment_config_hash(described),
        "cve_manifest_sha256": digest(CVECatalogue.open_default()),
        "ppo_config_hash": provenance._digest(
            ppo_manifest_config(config.learning_rate_schedule)
        ),
        "reward_config_hash": {
            arm: RewardConfig.from_yaml(REPO_ROOT / "configs" / f"{arm}.yaml").hash()
            for arm in config.arms
        },
        "dependency_lock_hash": provenance.dependency_lock_hash(),
    }


def freeze_inputs(config: ExperimentConfig) -> dict:
    frozen = current_inputs(config)
    frozen["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # This is the commit that preregistered the inputs. It cannot equal the
    # later commit containing this generated file (Git hashes are content
    # addressed). Exact runtime commits are recorded in every run manifest.
    frozen["preregistered_from_commit"] = provenance.git_commit()
    config.frozen_inputs.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def validate_frozen(config: ExperimentConfig) -> dict:
    if not config.frozen_inputs.is_file():
        raise ExperimentError(f"missing frozen inputs: {config.frozen_inputs}")
    frozen = json.loads(config.frozen_inputs.read_text())
    current = current_inputs(config)
    mismatches = []
    for key, value in current.items():
        if frozen.get(key) != value:
            mismatches.append(f"{key}: frozen {frozen.get(key)!r}, current {value!r}")
    if provenance.git_dirty() is not False:
        mismatches.append("working tree is dirty or its state is unknown")
    if mismatches:
        raise ExperimentError("frozen experiment mismatch:\n  " + "\n  ".join(mismatches))
    return frozen


def train_run(config: ExperimentConfig, arm: str, seed: int) -> None:
    command = [
        sys.executable,
        "-m",
        "rlredteam.train",
        "--experiment-id",
        config.experiment_id,
        "--seed",
        str(seed),
        "--topology-seed",
        str(config.topology_seed),
        "--timesteps",
        str(config.training_timesteps),
        "--learning-rate-schedule",
        config.learning_rate_schedule,
        "--reward-config",
        str(REPO_ROOT / "configs" / f"{arm}.yaml"),
        "--frozen",
        str(config.frozen_inputs),
    ]
    if config.postgres:
        command.append("--postgres")
    if config.log_steps:
        command.append("--log-steps")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode:
        raise ExperimentError(f"training failed for {arm} seed {seed}")


def evaluate_run(config: ExperimentConfig, arm: str, seed: int, result_root: Path) -> None:
    name = config.run_name(arm, seed)
    evaluate_checkpoint(
        RUNS_DIR / name,
        list(config.evaluation_seeds),
        result_root / "raw" / name,
        postgres=config.postgres,
    )


def _read_training(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "native_return": float(row["native_return"]),
            "shaped_return": float(row["shaped_return"]),
            "length": int(row["length"]),
            "goal_reached": row["goal_reached"] == "True",
            "mean_cvss_exploited": _optional_float(row["mean_cvss_exploited"]),
            "max_cvss_exploited": _optional_float(row["max_cvss_exploited"]),
            "hosts_compromised": int(row.get("hosts_compromised") or 0),
        }
        for row in rows
    ]


def _read_evaluation(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "native_return": float(row["native_return"]),
            "shaped_return": float(row["policy_return"]),
            "length": int(row["length"]),
            "goal_reached": row["goal_reached"] == "True",
            "terminal_reason": row["terminal_reason"],
            "mean_cvss_exploited": _optional_float(row["mean_cvss_exploited"]),
            "max_cvss_exploited": _optional_float(row["max_cvss_exploited"]),
            "hosts_compromised": int(row["hosts_compromised"]),
            "discovered_hosts": int(row["discovered_hosts"]),
            "tactics": json.loads(row["tactics"]),
        }
        for row in rows
    ]


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def collect_metrics(config: ExperimentConfig, result_root: Path) -> list[RunMetrics]:
    protocol = load_protocol(config.metrics)
    metrics = []
    for arm in config.arms:
        for seed in config.training_seeds:
            name = config.run_name(arm, seed)
            evaluation_path = result_root / "raw" / name / "evaluation.csv"
            training_path = RUNS_DIR / name / "episodes.csv"
            if not evaluation_path.is_file() or not training_path.is_file():
                raise ExperimentError(f"missing training/evaluation raw data for {name}")
            metrics.append(
                metrics_for_run(
                    name,
                    arm,
                    seed,
                    config.topology_seed,
                    _read_evaluation(evaluation_path),
                    protocol,
                    training_episodes=_read_training(training_path),
                )
            )
    return metrics


def write_results(
    config: ExperimentConfig, result_root: Path, metrics: list[RunMetrics], frozen: dict
) -> dict:
    protocol = load_protocol(config.metrics)
    report = analyse(metrics, protocol)
    for folder in ("metadata", "summaries", "tables", "figures", "trajectories"):
        (result_root / folder).mkdir(parents=True, exist_ok=True)
    (result_root / "summaries" / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (result_root / "tables" / "results_table.txt").write_text(format_table(report) + "\n")
    _write_csv(
        result_root / "summaries" / "metrics.csv",
        [run.to_dict() for run in metrics],
    )
    _write_csv(result_root / "tables" / "statistics.csv", report["comparisons"])
    _write_csv(
        result_root / "tables" / f"{config.experiment_id}_results.csv",
        [run.to_dict() for run in metrics],
    )
    manifests = []
    for arm in config.arms:
        for seed in config.training_seeds:
            name = config.run_name(arm, seed)
            manifest_path = RUNS_DIR / name / "manifest.json"
            checkpoint_path = RUNS_DIR / name / "model.zip"
            manifests.append(
                {
                    "run_name": name,
                    "manifest": json.loads(manifest_path.read_text()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
    metadata = {
        "result_phase": "dedicated_frozen_policy_evaluation",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": provenance.git_commit(),
        "experiment_config": asdict(config)
        | {
            "frozen_inputs": str(config.frozen_inputs.relative_to(REPO_ROOT)),
            "metrics": str(config.metrics.relative_to(REPO_ROOT)),
        },
        "experiment_config_sha256": config.digest(),
        "frozen_inputs": frozen,
        "runs": manifests,
        "complete": report["complete"],
    }
    (result_root / "metadata" / "experiment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=list)
    )
    hash_files = {
        "config_hash.txt": config.digest(),
        "topology_hash.txt": frozen["topology_hash"],
        "cve_snapshot_hash.txt": frozen["cve_manifest_sha256"],
        "code_commit.txt": provenance.git_commit() or "unknown",
    }
    for filename, value in hash_files.items():
        (result_root / "metadata" / filename).write_text(str(value) + "\n")
    write_figures(metrics, result_root / "figures")
    write_trajectory_evidence(config, result_root)
    _write_readme(config, result_root, report)
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
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


def _write_readme(config: ExperimentConfig, result_root: Path, report: dict) -> None:
    content = f"""# {config.experiment_id}

This package was generated from dedicated frozen-policy evaluation, not from
training returns. It compares sparse and shaped PPO across
{len(config.training_seeds)} matched training seeds on topology seed
{config.topology_seed}, with {len(config.evaluation_seeds)} held-out episode
seeds per checkpoint. Policy parameters are hashed before and after evaluation;
no gradient updates occur.

- `metadata/`: exact code, configuration, topology, CVE and checkpoint provenance
- `raw/`: evaluation episodes and step-level evidence for every checkpoint
- `summaries/`: analysis JSON and per-run metrics
- `tables/`: paired statistical comparisons and dissertation-ready CSV
- `figures/`: plots generated only from evaluation outcomes
- `trajectories/`: trace-derived attack paths, examples and validation findings

Analysis complete: `{report['complete']}`. See `summaries/analysis.json` for
assumption warnings, missing pairs and convergence status.
"""
    interpretation = report.get("fixed_budget_interpretation")
    if interpretation:
        content += f"""

The preregistered estimand is policy performance after the fixed training
budget. Every completed planned seed remains in the analysis; training
stability is a diagnostic, not an exclusion rule. Interpretation status:
`{interpretation['status']}`.
"""
    (result_root / "README.md").write_text(content)


def execute_experiment(
    config: ExperimentConfig,
    *,
    train: bool = True,
    evaluate: bool = True,
) -> dict:
    frozen = validate_frozen(config)
    result_root = RESULTS_DIR / config.experiment_id
    failures = []
    for arm in config.arms:
        for seed in config.training_seeds:
            try:
                if train:
                    train_run(config, arm, seed)
                if evaluate:
                    evaluate_run(config, arm, seed, result_root)
            except Exception as exc:  # preserve every failed planned run in metadata
                failures.append({"arm": arm, "seed": seed, "error": str(exc)})
                break
        if failures:
            break
    if failures:
        (result_root / "metadata").mkdir(parents=True, exist_ok=True)
        (result_root / "metadata" / "failures.json").write_text(json.dumps(failures, indent=2))
        raise ExperimentError(f"experiment stopped after failure: {failures[0]}")
    metrics = collect_metrics(config, result_root)
    return write_results(config, result_root, metrics, frozen)
