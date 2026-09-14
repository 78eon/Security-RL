"""Training, frozen evaluation, evidence generation and verification helpers."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from rlredteam.attack_path_report import ReportMode, generate_report, write_report
from rlredteam.catalogues import mitre_catalogue_digest
from rlredteam.cyberbattle_adapter import (
    ADAPTER_SCHEMA_VERSION,
    CYBERBATTLE_SOURCE_REVISION,
    CyberBattleAdapter,
    CyberBattleScenario,
    cyberbattle_adapter_source_hash,
)
from rlredteam.enterprise.trajectory import reconstruct_attack_path
from rlredteam.evaluation import policy_digest, sha256_file
from rlredteam.frameworks import map_simulator_behavior
from rlredteam.provenance import git_commit, git_dirty
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cyberbattle.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "cyberbattle"
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "cyberbattle"


class CyberBattleStudyError(RuntimeError):
    """A CyberBattle artifact cannot support the claimed result."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise CyberBattleStudyError("CyberBattle study config must be an object")
    if raw.get("simulator_source_revision") != CYBERBATTLE_SOURCE_REVISION:
        raise CyberBattleStudyError("config and adapter CyberBattle revisions differ")
    train = CyberBattleScenario(**raw["train_scenario"])
    held_out = CyberBattleScenario(**raw["held_out_scenario"])
    if train.digest() == held_out.digest():
        raise CyberBattleStudyError("held-out scenario must differ from training scenario")
    train_seeds = tuple(map(int, raw.get("train_seeds", ())))
    evaluation_seeds = tuple(map(int, raw.get("evaluation_seeds", ())))
    if not train_seeds or not evaluation_seeds:
        raise CyberBattleStudyError("fixed train and evaluation seed sets are required")
    if set(train_seeds) & set(evaluation_seeds):
        raise CyberBattleStudyError("train and held-out evaluation seeds must be disjoint")
    return raw


def scenario_from(raw: dict[str, Any], key: str) -> CyberBattleScenario:
    return CyberBattleScenario(**raw[key])


def config_hash(raw: dict[str, Any]) -> str:
    return canonical_sha256(raw)


def protected_artifact_hash() -> str:
    """Guard frozen results by bytes and historical run files by metadata."""
    digest = hashlib.sha256()
    for root_name in ("results", "runs"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(REPO_ROOT)
            if len(relative.parts) > 1 and relative.parts[1] == "cyberbattle":
                continue
            digest.update(str(relative).encode())
            if root_name == "results":
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
            else:
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "cyberbattlesim": version("cyberbattlesim"),
        "gymnasium": version("gymnasium"),
        "numpy": version("numpy"),
        "stable_baselines3": version("stable-baselines3"),
    }


def provenance(
    adapter: CyberBattleAdapter,
    *,
    raw_config: dict[str, Any],
    policy_hash: str | None,
) -> dict[str, Any]:
    return {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "cyberbattle_source_revision": CYBERBATTLE_SOURCE_REVISION,
        "package_versions": _runtime_versions(),
        "scenario": adapter.scenario.to_dict(),
        "scenario_config_sha256": adapter.scenario.digest(),
        "study_config_sha256": config_hash(raw_config),
        "adapter_source_sha256": cyberbattle_adapter_source_hash(),
        "security_rl_git_commit": git_commit(),
        "security_rl_git_dirty": git_dirty(),
        "policy_sha256": policy_hash,
        "train_seeds": list(map(int, raw_config["train_seeds"])),
        "evaluation_seeds": list(map(int, raw_config["evaluation_seeds"])),
        "spaces": adapter.space_summary(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _scripted_action_indices(adapter: CyberBattleAdapter) -> list[int]:
    index_by_vulnerability = {
        item.simulator_vulnerability_id: item.index
        for item in adapter.action_catalogue()
        if item.simulator_vulnerability_id
    }
    connect = next(
        item.index for item in adapter.action_catalogue() if item.native_kind == "connect"
    )
    return [
        index_by_vulnerability["ScanExplorerRecentFiles"],
        connect,
        index_by_vulnerability["CrackKeepPassX"],
        connect,
        index_by_vulnerability["CrackKeepPass"],
        connect,
    ]


def scripted_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG,
    result_dir: Path = DEFAULT_RESULT_DIR,
    seed: int = 1001,
) -> dict[str, Any]:
    raw = load_config(config_path)
    before = protected_artifact_hash()
    adapter = CyberBattleAdapter(scenario_from(raw, "train_scenario"))
    adapter.reset(seed=seed)
    events: list[dict[str, Any]] = []
    goal_reached = False
    for action in _scripted_action_indices(adapter):
        _, _, terminated, truncated, info = adapter.step(action)
        transition = info["simulator_transition"]
        events.append(transition.to_trajectory_event())
        if terminated or truncated:
            goal_reached = bool(terminated)
            break
    if not goal_reached:
        raise CyberBattleStudyError("scripted small-chain smoke did not reach its goal")
    document = {
        "schema_version": "security-rl-cyberbattle-trajectory-v1",
        "simulator_backend": "CyberBattleSim",
        "provenance": provenance(adapter, raw_config=raw, policy_hash=None),
        "trajectories": [
            {
                "run_name": "cyberbattle-scripted-smoke",
                "evaluation_seed": seed,
                "goal_reached": True,
                "events": events,
                "agent_knowledge_graph": adapter.agent_knowledge_graph(),
                "reconstructed_attack_path": reconstruct_attack_path(events),
            }
        ],
    }
    trajectory_path = Path(result_dir) / "example_trajectory.json"
    report_path = Path(result_dir) / "example_phase14_report.json"
    _write_json(trajectory_path, document)
    report = generate_report(
        trajectory_path,
        mode=ReportMode.SIMULATION,
        experiment_id="cyberbattle-adapter-smoke",
    )
    write_report(report, report_path)
    after = protected_artifact_hash()
    if before != after:
        raise CyberBattleStudyError("a pre-existing frozen artifact changed during smoke")
    summary = {
        "goal_reached": True,
        "event_count": len(events),
        "trajectory": str(trajectory_path),
        "report": str(report_path),
        "report_id": report.report_id,
        "mitre_techniques": sorted(
            {
                mapping["technique_id"]
                for event in events
                for mapping in event["framework_mappings"]
            }
        ),
        "cve_fields_absent": all(event["cve_id"] is None for event in events),
        "protected_artifacts_sha256_before": before,
        "protected_artifacts_sha256_after": after,
    }
    _write_json(Path(result_dir) / "smoke_summary.json", summary)
    adapter.close()
    return summary


def train_ppo(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    raw = load_config(config_path)
    ppo = dict(raw["ppo"])
    budget = int(total_timesteps or ppo.pop("total_timesteps"))
    ppo.pop("total_timesteps", None)
    train_seeds = tuple(map(int, raw["train_seeds"]))
    set_all_seeds(train_seeds[0])
    before = protected_artifact_hash()
    adapter = CyberBattleAdapter(
        scenario_from(raw, "train_scenario"), reset_seeds=train_seeds
    )
    model = PPO("MlpPolicy", adapter, seed=train_seeds[0], verbose=0, **ppo)
    model.learn(total_timesteps=budget)
    policy_hash = policy_digest(model)
    training_validation: list[dict[str, Any]] = []
    for seed in train_seeds:
        validation_env = CyberBattleAdapter(scenario_from(raw, "train_scenario"))
        observation, _ = validation_env.reset(seed=seed)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = validation_env.step(int(action))
            steps += 1
        training_validation.append(
            {"seed": seed, "goal_reached": bool(terminated), "steps": steps}
        )
        validation_env.close()
    if policy_digest(model) != policy_hash:
        raise CyberBattleStudyError("policy changed during training-chain validation")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint = run_path / "ppo_chain.zip"
    model.save(checkpoint)
    after = protected_artifact_hash()
    if before != after:
        raise CyberBattleStudyError("a pre-existing frozen artifact changed during training")
    metadata = {
        "algorithm": "PPO",
        "cross_simulator_transfer": False,
        "training_budget": budget,
        "ppo": ppo,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": policy_hash,
        "training_validation": training_validation,
        "training_validation_deterministic": True,
        "training_validation_updates_performed": False,
        "provenance": provenance(adapter, raw_config=raw, policy_hash=policy_hash),
        "protected_artifacts_sha256_before": before,
        "protected_artifacts_sha256_after": after,
    }
    _write_json(run_path / "training_metadata.json", metadata)
    adapter.close()
    return metadata


def evaluate_ppo(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    result_dir: Path = DEFAULT_RESULT_DIR,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    raw = load_config(config_path)
    checkpoint = Path(run_dir) / "ppo_chain.zip"
    training_metadata_path = Path(run_dir) / "training_metadata.json"
    if not checkpoint.is_file() or not training_metadata_path.is_file():
        raise CyberBattleStudyError("run cyberbattle-train before evaluation")
    training_metadata = json.loads(training_metadata_path.read_text())
    if training_metadata["provenance"]["study_config_sha256"] != config_hash(raw):
        raise CyberBattleStudyError("checkpoint config provenance does not match evaluation")
    if training_metadata["provenance"][
        "adapter_source_sha256"
    ] != cyberbattle_adapter_source_hash():
        raise CyberBattleStudyError("checkpoint adapter source does not match evaluation")
    if training_metadata["checkpoint_sha256"] != sha256_file(checkpoint):
        raise CyberBattleStudyError("checkpoint bytes differ from training provenance")

    before_artifacts = protected_artifact_hash()
    model = PPO.load(checkpoint, device="cpu")
    before_policy = policy_digest(model)
    if before_policy != training_metadata["policy_sha256"]:
        raise CyberBattleStudyError("loaded policy differs from training policy hash")
    episodes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    evaluation_seeds = list(map(int, raw["evaluation_seeds"]))
    last_adapter: CyberBattleAdapter | None = None
    for seed in evaluation_seeds:
        set_all_seeds(seed)
        model.set_random_seed(seed)
        adapter = CyberBattleAdapter(scenario_from(raw, "held_out_scenario"))
        observation, _ = adapter.reset(seed=seed)
        terminated = truncated = False
        events: list[dict[str, Any]] = []
        total_reward = native_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = adapter.step(int(action))
            transition = info["simulator_transition"]
            events.append(transition.to_trajectory_event())
            total_reward += float(reward)
            native_return += float(transition.event.native_reward)
        episodes.append(
            {
                "evaluation_seed": seed,
                "scenario_config_sha256": adapter.scenario.digest(),
                "goal_reached": bool(terminated),
                "steps": len(events),
                "policy_return": total_reward,
                "native_return": native_return,
            }
        )
        trajectories.append(
            {
                "run_name": "cyberbattle-ppo-held-out",
                "evaluation_seed": seed,
                "goal_reached": bool(terminated),
                "events": events,
                "agent_knowledge_graph": adapter.agent_knowledge_graph(),
                "reconstructed_attack_path": reconstruct_attack_path(events),
            }
        )
        if last_adapter is not None:
            last_adapter.close()
        last_adapter = adapter

    after_policy = policy_digest(model)
    if before_policy != after_policy:
        raise CyberBattleStudyError("policy changed during dedicated evaluation")
    if last_adapter is None:
        raise CyberBattleStudyError("evaluation seed set is empty")
    result_path = Path(result_dir)
    trajectory_document = {
        "schema_version": "security-rl-cyberbattle-trajectory-v1",
        "simulator_backend": "CyberBattleSim",
        "provenance": provenance(
            last_adapter, raw_config=raw, policy_hash=before_policy
        ),
        "trajectories": trajectories,
    }
    trajectory_path = result_path / "ppo_evaluation_trajectories.json"
    _write_json(trajectory_path, trajectory_document)
    report = generate_report(
        trajectory_path,
        mode=ReportMode.SIMULATION,
        experiment_id="cyberbattle-ppo-held-out",
        checkpoint_hashes={"cyberbattle-ppo-held-out": before_policy},
    )
    report_path = result_path / "ppo_phase14_report.json"
    write_report(report, report_path)
    after_artifacts = protected_artifact_hash()
    if before_artifacts != after_artifacts:
        raise CyberBattleStudyError("a pre-existing frozen artifact changed during evaluation")
    result = {
        "algorithm": "PPO",
        "deterministic": True,
        "training_updates_performed": False,
        "evaluation_seeds": evaluation_seeds,
        "held_out_scenario_config_sha256": last_adapter.scenario.digest(),
        "training_scenario_config_sha256": training_metadata["provenance"][
            "scenario_config_sha256"
        ],
        "policy_sha256_before": before_policy,
        "policy_sha256_after": after_policy,
        "episodes": episodes,
        "trajectory": str(trajectory_path),
        "phase14_report": str(report_path),
        "report_id": report.report_id,
        "provenance": trajectory_document["provenance"],
        "protected_artifacts_sha256_before": before_artifacts,
        "protected_artifacts_sha256_after": after_artifacts,
    }
    _write_json(result_path / "evaluation.json", result)
    last_adapter.close()
    return result


def verify_completion(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    result_dir: Path = DEFAULT_RESULT_DIR,
) -> dict[str, Any]:
    raw = load_config(config_path)
    run_path, result_path = Path(run_dir), Path(result_dir)
    required = {
        "training": run_path / "training_metadata.json",
        "checkpoint": run_path / "ppo_chain.zip",
        "smoke": result_path / "smoke_summary.json",
        "example_trajectory": result_path / "example_trajectory.json",
        "example_report": result_path / "example_phase14_report.json",
        "evaluation": result_path / "evaluation.json",
        "evaluation_trajectory": result_path / "ppo_evaluation_trajectories.json",
        "evaluation_report": result_path / "ppo_phase14_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CyberBattleStudyError("missing CyberBattle artifacts: " + ", ".join(missing))
    training = json.loads(required["training"].read_text())
    smoke = json.loads(required["smoke"].read_text())
    evaluation = json.loads(required["evaluation"].read_text())
    trajectory = json.loads(required["evaluation_trajectory"].read_text())
    report = json.loads(required["evaluation_report"].read_text())

    checks: dict[str, bool] = {
        "cyberbattle_revision_pinned": training["provenance"][
            "cyberbattle_source_revision"
        ]
        == CYBERBATTLE_SOURCE_REVISION,
        "adapter_source_current": training["provenance"]["adapter_source_sha256"]
        == cyberbattle_adapter_source_hash(),
        "study_config_current": training["provenance"]["study_config_sha256"]
        == config_hash(raw),
        "checkpoint_bytes_match": training["checkpoint_sha256"]
        == sha256_file(required["checkpoint"]),
        "ppo_solves_training_chain": all(
            episode["goal_reached"] for episode in training["training_validation"]
        ),
        "evaluation_is_deterministic": evaluation["deterministic"] is True,
        "evaluation_has_no_updates": evaluation["training_updates_performed"] is False,
        "policy_unchanged": evaluation["policy_sha256_before"]
        == evaluation["policy_sha256_after"]
        == training["policy_sha256"],
        "evaluation_seeds_fixed": evaluation["evaluation_seeds"]
        == list(map(int, raw["evaluation_seeds"])),
        "held_out_scenario": evaluation["held_out_scenario_config_sha256"]
        != evaluation["training_scenario_config_sha256"],
        "scripted_goal_reached": smoke["goal_reached"] is True,
        "no_cve_invented": True,
        "mitre_catalogue_only": True,
        "phase14_catalogue_provenance": report["provenance"][
            "mitre_catalogue_sha256"
        ]
        == mitre_catalogue_digest(),
        "frozen_artifacts_unchanged": all(
            item["protected_artifacts_sha256_before"]
            == item["protected_artifacts_sha256_after"]
            for item in (training, smoke, evaluation)
        ),
    }
    for episode in trajectory["trajectories"]:
        for event in episode["events"]:
            if event.get("simulator_vulnerability_id") and event.get("cve_id") is not None:
                checks["no_cve_invented"] = False
            expected = {
                (item.framework.value, item.technique_id)
                for item in map_simulator_behavior(event["action_kind"])
            }
            recorded = {
                (item["framework"], item["technique_id"])
                for item in event["framework_mappings"]
            }
            if expected != recorded:
                checks["mitre_catalogue_only"] = False
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise CyberBattleStudyError("CyberBattle verification failed: " + ", ".join(failures))
    result = {
        "status": "PASS",
        "checks": checks,
        "example_report_id": smoke["report_id"],
        "evaluation_report_id": evaluation["report_id"],
        "policy_sha256": training["policy_sha256"],
        "environment": (
            "Podman"
            if Path("/.dockerenv").exists() or os.getuid() != 1000
            else "unknown"
        ),
    }
    _write_json(result_path / "verification.json", result)
    return result
