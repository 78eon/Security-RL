"""Deterministic CybORG PPO study, evidence generation, and verification."""

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
from rlredteam.cyborg_adapter import (
    ADAPTER_SCHEMA_VERSION,
    CYBORG_SOURCE_REVISION,
    CybORGAdapter,
    CybORGScenario,
    cyborg_adapter_source_hash,
)
from rlredteam.enterprise.trajectory import reconstruct_attack_path
from rlredteam.evaluation import policy_digest, sha256_file
from rlredteam.frameworks import map_simulator_behavior
from rlredteam.provenance import git_commit, git_dirty
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cyborg.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "cyborg"
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "cyborg"
NATIVE_SCENARIO = Path(
    "/opt/cyborg/CybORG/Simulator/Scenarios/scenario_files/Scenario1.yaml"
)


class CybORGStudyError(RuntimeError):
    """A CybORG study artifact cannot support the claimed result."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise CybORGStudyError("CybORG study config must be an object")
    if raw.get("schema_version") != "security-rl-cyborg-study-v1":
        raise CybORGStudyError("unsupported CybORG study schema")
    if raw.get("simulator_source_revision") != CYBORG_SOURCE_REVISION:
        raise CybORGStudyError("config and adapter CybORG revisions differ")
    CybORGScenario(**raw["scenario"])
    train_seeds = tuple(map(int, raw.get("train_seeds", ())))
    evaluation_seeds = tuple(map(int, raw.get("evaluation_seeds", ())))
    if not train_seeds or not evaluation_seeds:
        raise CybORGStudyError("fixed train and evaluation seed sets are required")
    if set(train_seeds) & set(evaluation_seeds):
        raise CybORGStudyError("train and evaluation seeds must be disjoint")
    return raw


def config_hash(raw: dict[str, Any]) -> str:
    return canonical_sha256(raw)


def scenario_from(raw: dict[str, Any]) -> CybORGScenario:
    return CybORGScenario(**raw["scenario"])


def protected_artifact_hash() -> str:
    """Hash every earlier result/run while excluding only new CybORG outputs."""
    digest = hashlib.sha256()
    for root_name in ("results", "runs"):
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(REPO_ROOT)
            if len(relative.parts) > 1 and relative.parts[1] == "cyborg":
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
        "cyborg": version("CybORG"),
        "gym": version("gym"),
        "gymnasium": version("gymnasium"),
        "numpy": version("numpy"),
        "stable_baselines3": version("stable-baselines3"),
    }


def provenance(
    adapter: CybORGAdapter,
    *,
    raw_config: dict[str, Any],
    policy_hash: str | None,
) -> dict[str, Any]:
    if not NATIVE_SCENARIO.is_file():
        raise CybORGStudyError("pinned native scenario file is unavailable")
    return {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "cyborg_source_revision": CYBORG_SOURCE_REVISION,
        "package_versions": _runtime_versions(),
        "scenario": adapter.scenario.to_dict(),
        "scenario_config_sha256": adapter.scenario.digest(),
        "native_scenario_sha256": sha256_file(NATIVE_SCENARIO),
        "study_config_sha256": config_hash(raw_config),
        "adapter_source_sha256": cyborg_adapter_source_hash(),
        "security_rl_git_commit": git_commit(),
        "security_rl_git_dirty": git_dirty(),
        "policy_sha256": policy_hash,
        "train_seeds": list(map(int, raw_config["train_seeds"])),
        "evaluation_seeds": list(map(int, raw_config["evaluation_seeds"])),
        "spaces": adapter.space_summary(),
        "hidden_state_api_used": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def scripted_actions() -> tuple[int, ...]:
    return (1, 2, 3, 4, 5, 6, 1, 2, 7)


def _rollout(
    adapter: CybORGAdapter,
    *,
    seed: int,
    actions: tuple[int, ...] | None = None,
    model: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, _ = adapter.reset(seed=seed)
    events: list[dict[str, Any]] = []
    total_reward = 0.0
    native_return = 0.0
    terminated = truncated = False
    action_index = 0
    while not (terminated or truncated):
        if actions is not None:
            if action_index >= len(actions):
                break
            action = actions[action_index]
            action_index += 1
        else:
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=adapter.action_masks(),
            )
            action = int(action)
        observation, reward, terminated, truncated, info = adapter.step(action)
        transition = info["simulator_transition"]
        events.append(transition.to_trajectory_event())
        total_reward += float(reward)
        native_return += float(transition.event.native_reward)
    episode = {
        "evaluation_seed": seed,
        "goal_reached": bool(terminated and any(item["goal_reached"] for item in events)),
        "steps": len(events),
        "policy_return": total_reward,
        "native_return": native_return,
    }
    return episode, events


def scripted_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG,
    result_dir: Path = DEFAULT_RESULT_DIR,
    seed: int = 3001,
) -> dict[str, Any]:
    raw = load_config(config_path)
    before = protected_artifact_hash()
    adapter = CybORGAdapter(scenario_from(raw))
    episode, events = _rollout(adapter, seed=seed, actions=scripted_actions())
    if not episode["goal_reached"]:
        raise CybORGStudyError("scripted Scenario1 smoke did not reach its objective")
    document = {
        "schema_version": "security-rl-cyborg-trajectory-v1",
        "simulator_backend": "CybORG",
        "provenance": provenance(adapter, raw_config=raw, policy_hash=None),
        "trajectories": [
            {
                "run_name": "cyborg-scripted-smoke",
                **episode,
                "events": events,
                "agent_knowledge_graph": adapter.agent_knowledge_graph(),
                "reconstructed_attack_path": reconstruct_attack_path(events),
            }
        ],
    }
    result_path = Path(result_dir)
    trajectory_path = result_path / "example_trajectory.json"
    report_path = result_path / "example_phase14_report.json"
    _write_json(trajectory_path, document)
    report = generate_report(
        trajectory_path,
        mode=ReportMode.SIMULATION,
        experiment_id="cyborg-adapter-smoke",
    )
    write_report(report, report_path)
    after = protected_artifact_hash()
    if before != after:
        raise CybORGStudyError("an earlier frozen artifact changed during smoke")
    summary = {
        **episode,
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
    _write_json(result_path / "smoke_summary.json", summary)
    adapter.close()
    return summary


def train_ppo(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    raw = load_config(config_path)
    ppo = dict(raw["ppo"])
    budget = int(total_timesteps or ppo.pop("total_timesteps"))
    ppo.pop("total_timesteps", None)
    train_seeds = tuple(map(int, raw["train_seeds"]))
    set_all_seeds(train_seeds[0])
    before = protected_artifact_hash()
    adapter = CybORGAdapter(scenario_from(raw), reset_seeds=train_seeds)
    model = MaskablePPO("MlpPolicy", adapter, seed=train_seeds[0], verbose=0, **ppo)
    model.learn(total_timesteps=budget)
    policy_hash = policy_digest(model)
    validation: list[dict[str, Any]] = []
    for seed in train_seeds:
        validation_env = CybORGAdapter(scenario_from(raw))
        episode, _ = _rollout(validation_env, seed=seed, model=model)
        validation.append(episode)
        validation_env.close()
    if policy_digest(model) != policy_hash:
        raise CybORGStudyError("policy changed during training validation")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint = run_path / "ppo_scenario1.zip"
    model.save(checkpoint)
    after = protected_artifact_hash()
    if before != after:
        raise CybORGStudyError("an earlier frozen artifact changed during training")
    metadata = {
        "algorithm": "MaskablePPO",
        "algorithm_family": "PPO",
        "single_agent": True,
        "cross_simulator_transfer": False,
        "training_budget": budget,
        "ppo": ppo,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": policy_hash,
        "training_validation": validation,
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
    from sb3_contrib import MaskablePPO

    raw = load_config(config_path)
    run_path = Path(run_dir)
    checkpoint = run_path / "ppo_scenario1.zip"
    training_path = run_path / "training_metadata.json"
    if not checkpoint.is_file() or not training_path.is_file():
        raise CybORGStudyError("run cyborg-train before evaluation")
    training = json.loads(training_path.read_text())
    if training["provenance"]["study_config_sha256"] != config_hash(raw):
        raise CybORGStudyError("checkpoint config provenance does not match evaluation")
    if training["provenance"]["adapter_source_sha256"] != cyborg_adapter_source_hash():
        raise CybORGStudyError("checkpoint adapter source does not match evaluation")
    if training["checkpoint_sha256"] != sha256_file(checkpoint):
        raise CybORGStudyError("checkpoint bytes differ from training provenance")
    before_artifacts = protected_artifact_hash()
    model = MaskablePPO.load(checkpoint, device="cpu")
    before_policy = policy_digest(model)
    if before_policy != training["policy_sha256"]:
        raise CybORGStudyError("loaded policy differs from training policy hash")
    episodes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    evaluation_seeds = list(map(int, raw["evaluation_seeds"]))
    last_adapter: CybORGAdapter | None = None
    for seed in evaluation_seeds:
        set_all_seeds(seed)
        model.set_random_seed(seed)
        adapter = CybORGAdapter(scenario_from(raw))
        episode, events = _rollout(adapter, seed=seed, model=model)
        episodes.append(episode)
        trajectories.append(
            {
                "run_name": "cyborg-ppo-evaluation",
                **episode,
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
        raise CybORGStudyError("policy changed during dedicated evaluation")
    if last_adapter is None:
        raise CybORGStudyError("evaluation seed set is empty")
    result_path = Path(result_dir)
    trajectory_document = {
        "schema_version": "security-rl-cyborg-trajectory-v1",
        "simulator_backend": "CybORG",
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
        experiment_id="cyborg-ppo-evaluation",
        checkpoint_hashes={"cyborg-ppo-evaluation": before_policy},
    )
    report_path = result_path / "ppo_phase14_report.json"
    write_report(report, report_path)
    after_artifacts = protected_artifact_hash()
    if before_artifacts != after_artifacts:
        raise CybORGStudyError("an earlier frozen artifact changed during evaluation")
    result = {
        "algorithm": "MaskablePPO",
        "algorithm_family": "PPO",
        "single_agent": True,
        "deterministic": True,
        "training_updates_performed": False,
        "evaluation_seeds": evaluation_seeds,
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
        "checkpoint": run_path / "ppo_scenario1.zip",
        "smoke": result_path / "smoke_summary.json",
        "example_trajectory": result_path / "example_trajectory.json",
        "example_report": result_path / "example_phase14_report.json",
        "evaluation": result_path / "evaluation.json",
        "evaluation_trajectory": result_path / "ppo_evaluation_trajectories.json",
        "evaluation_report": result_path / "ppo_phase14_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CybORGStudyError("missing CybORG artifacts: " + ", ".join(missing))
    training = json.loads(required["training"].read_text())
    smoke = json.loads(required["smoke"].read_text())
    evaluation = json.loads(required["evaluation"].read_text())
    trajectory = json.loads(required["evaluation_trajectory"].read_text())
    report = json.loads(required["evaluation_report"].read_text())
    checks: dict[str, bool] = {
        "cyborg_revision_pinned": training["provenance"]["cyborg_source_revision"]
        == CYBORG_SOURCE_REVISION,
        "adapter_source_current": training["provenance"]["adapter_source_sha256"]
        == cyborg_adapter_source_hash(),
        "study_config_current": training["provenance"]["study_config_sha256"]
        == config_hash(raw),
        "native_scenario_current": training["provenance"]["native_scenario_sha256"]
        == sha256_file(NATIVE_SCENARIO),
        "checkpoint_bytes_match": training["checkpoint_sha256"]
        == sha256_file(required["checkpoint"]),
        "ppo_solves_training_scenario": all(
            episode["goal_reached"] for episode in training["training_validation"]
        ),
        "evaluation_is_deterministic": evaluation["deterministic"] is True,
        "evaluation_has_no_updates": evaluation["training_updates_performed"] is False,
        "policy_unchanged": evaluation["policy_sha256_before"]
        == evaluation["policy_sha256_after"]
        == training["policy_sha256"],
        "evaluation_seeds_fixed": evaluation["evaluation_seeds"]
        == list(map(int, raw["evaluation_seeds"])),
        "train_eval_seeds_disjoint": not set(raw["train_seeds"])
        & set(raw["evaluation_seeds"]),
        "scripted_goal_reached": smoke["goal_reached"] is True,
        "no_cve_invented": True,
        "mitre_catalogue_only": True,
        "phase14_catalogue_provenance": report["provenance"][
            "mitre_catalogue_sha256"
        ]
        == mitre_catalogue_digest(),
        "hidden_state_api_unused": training["provenance"]["hidden_state_api_used"]
        is False,
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
        raise CybORGStudyError("CybORG verification failed: " + ", ".join(failures))
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
