"""Dedicated frozen-policy evaluation for the Essential experiment.

Training callbacks are deliberately absent.  A policy receives only the
partial NASim observation returned by ``reset``/``step``; environment ground
truth is used after decisions only to produce research metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from rlredteam.catalogue import CVECatalogue
from rlredteam.manifest import digest
from rlredteam.nasim_adapter import RewardWrapper
from rlredteam.provenance import ExperimentManifest, git_commit, topology_hash
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig, describe, make_env
from rlredteam.train import set_all_seeds


class EvaluationError(RuntimeError):
    """The checkpoint or its provenance cannot support a valid evaluation."""


class Policy(Protocol):
    def predict(self, observation, deterministic: bool = False): ...

    def set_random_seed(self, seed: int) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    run_name: str
    reward_mode: str
    training_seed: int
    evaluation_seed: int
    topology_seed: int
    goal_reached: bool
    terminal_reason: str
    length: int
    native_return: float
    policy_return: float
    hosts_compromised: int
    discovered_hosts: int
    mean_cvss_exploited: float | None
    max_cvss_exploited: float | None
    high_value_target_hit: bool
    distinct_actions: int
    repeated_actions: int
    tactics: tuple[str, ...]
    techniques: tuple[str, ...]

    def to_row(self) -> dict:
        row = asdict(self)
        row["tactics"] = json.dumps(self.tactics)
        row["techniques"] = json.dumps(self.techniques)
        return row


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    episodes: tuple[EvaluationEpisode, ...]
    steps: tuple[dict, ...]


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def policy_digest(model) -> str:
    """Hash policy parameters without serialising or mutating the model."""
    value = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def validate_manifest(manifest: ExperimentManifest) -> None:
    """Fail when current immutable inputs do not match the trained checkpoint."""
    topology_config = TopologyConfig.from_yaml()
    environment = make_env(topology_config, topology_seed=manifest.topology_seed)
    actual_topology_hash = topology_hash(describe(environment))
    actual_cve_hash = digest(CVECatalogue.open_default())
    checks = {
        "code commit": (manifest.git_commit, git_commit()),
        "topology config": (manifest.topology_config_hash, topology_config.config_hash()),
        "realised topology": (manifest.topology_hash, actual_topology_hash),
        "CVE snapshot": (manifest.cve_manifest_sha256, actual_cve_hash),
    }
    mismatches = [
        f"{name}: checkpoint {expected}, runtime {actual}"
        for name, (expected, actual) in checks.items()
        if expected != actual
    ]
    if manifest.git_dirty:
        mismatches.append("checkpoint was trained from a dirty working tree")
    if mismatches:
        raise EvaluationError("evaluation provenance mismatch:\n  " + "\n  ".join(mismatches))


def evaluate_policy(
    model: Policy,
    *,
    run_name: str,
    reward_mode: str,
    training_seed: int,
    evaluation_seeds: list[int],
    topology_seed: int,
    topology_config: TopologyConfig | None = None,
    reward_config: RewardConfig | None = None,
    hvt_threshold: float = 9.0,
) -> EvaluationBundle:
    """Evaluate a policy without calling ``learn`` or exposing ground truth."""
    if not evaluation_seeds:
        raise EvaluationError("at least one evaluation seed is required")
    topology_config = topology_config or TopologyConfig.from_yaml()
    reward_config = reward_config or RewardConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / f"{reward_mode}.yaml"
    )
    episode_rows: list[EvaluationEpisode] = []
    step_rows: list[dict] = []

    for evaluation_seed in evaluation_seeds:
        set_all_seeds(evaluation_seed)
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(evaluation_seed)
        environment = RewardWrapper(
            make_env(topology_config, topology_seed=topology_seed),
            CVECatalogue.open_default(),
            topology_seed=topology_seed,
            reward_config=reward_config,
        )
        observation, _ = environment.reset(seed=evaluation_seed)
        terminated = truncated = False
        native_return = policy_return = 0.0
        compromised: dict[tuple[int, int], float] = {}
        discovered_hosts = 0
        actions: list[str] = []
        tactics: set[str] = set()
        techniques: set[str] = set()
        episode_index = len(episode_rows)

        while not (terminated or truncated):
            # This is the policy boundary: no env, scenario, info, adapter,
            # reward breakdown or topology object crosses it.
            action, _state = model.predict(observation, deterministic=False)
            observation, reward, terminated, truncated, info = environment.step(action)
            event = info["attack_event"]
            breakdown = info["reward_breakdown"]
            native_return += float(info["native_reward"])
            policy_return += float(reward)
            discovered_hosts += event.newly_discovered
            actions.append(event.action_name)
            if breakdown.tactic_name:
                tactics.add(breakdown.tactic_name)
            if breakdown.technique_id:
                techniques.add(breakdown.technique_id)
            if event.success and event.access_gained > 0 and event.target is not None:
                compromised[event.target] = max(
                    compromised.get(event.target, 0.0), event.cvss_base or 0.0
                )
            step_rows.append(
                {
                    "run_name": run_name,
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "episode_index": episode_index,
                    "step": event.step,
                    "action": event.action_name,
                    "action_kind": str(event.kind),
                    "target": list(event.target) if event.target else None,
                    "success": event.success,
                    "native_reward": event.native_reward,
                    "policy_reward": breakdown.total,
                    "cve_term": breakdown.cve,
                    "tactic_term": breakdown.tactic,
                    "crown_jewel_term": breakdown.crown_jewel,
                    "penalty_term": breakdown.penalty,
                    "cve_id": event.cve_id,
                    "cvss_base": event.cvss_base,
                    "access_gained": int(event.access_gained),
                    "newly_discovered": event.newly_discovered,
                    "is_crown_jewel": event.is_crown_jewel,
                    "tactic": breakdown.tactic_name,
                    "technique_id": breakdown.technique_id,
                    "error": event.error,
                }
            )
            if not math.isfinite(native_return) or not math.isfinite(policy_return):
                raise EvaluationError("non-finite reward encountered during evaluation")

        scores = list(compromised.values())
        repeated = sum(left == right for left, right in zip(actions, actions[1:], strict=False))
        episode_rows.append(
            EvaluationEpisode(
                run_name=run_name,
                reward_mode=reward_mode,
                training_seed=training_seed,
                evaluation_seed=evaluation_seed,
                topology_seed=topology_seed,
                goal_reached=bool(terminated and not truncated),
                terminal_reason="goal" if terminated and not truncated else "step_limit",
                length=len(actions),
                native_return=native_return,
                policy_return=policy_return,
                hosts_compromised=len(compromised),
                discovered_hosts=discovered_hosts,
                mean_cvss_exploited=sum(scores) / len(scores) if scores else None,
                max_cvss_exploited=max(scores) if scores else None,
                high_value_target_hit=any(score >= hvt_threshold for score in scores),
                distinct_actions=len(set(actions)),
                repeated_actions=repeated,
                tactics=tuple(sorted(tactics)),
                techniques=tuple(sorted(techniques)),
            )
        )
    return EvaluationBundle(tuple(episode_rows), tuple(step_rows))


def write_bundle(bundle: EvaluationBundle, out_dir: Path, metadata: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_path = out_dir / "evaluation.csv"
    rows = [episode.to_row() for episode in bundle.episodes]
    if rows:
        with episode_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / "steps.jsonl").open("w") as handle:
        for step in bundle.steps:
            handle.write(json.dumps(step, sort_keys=True) + "\n")
    (out_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def evaluate_checkpoint(
    run_dir: Path, evaluation_seeds: list[int], out_dir: Path
) -> EvaluationBundle:
    """Validate, load and evaluate one persisted PPO checkpoint."""
    from stable_baselines3 import PPO

    run_dir = Path(run_dir)
    manifest = ExperimentManifest.read(run_dir / "manifest.json")
    validate_manifest(manifest)
    checkpoint = run_dir / "model.zip"
    if not checkpoint.is_file():
        raise EvaluationError(f"checkpoint not found: {checkpoint}")
    model = PPO.load(checkpoint, device="cpu")
    before = policy_digest(model)
    bundle = evaluate_policy(
        model,
        run_name=manifest.experiment_id,
        reward_mode=manifest.reward_mode,
        training_seed=manifest.training_seed,
        evaluation_seeds=evaluation_seeds,
        topology_seed=manifest.topology_seed,
    )
    after = policy_digest(model)
    if before != after:
        raise EvaluationError("policy parameters changed during evaluation")
    metadata = {
        "run_name": manifest.experiment_id,
        "training_seed": manifest.training_seed,
        "evaluation_seeds": evaluation_seeds,
        "topology_seed": manifest.topology_seed,
        "action_selection": "stochastic",
        "gradient_updates": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_before": before,
        "policy_sha256_after": after,
        "manifest": manifest.to_dict(),
    }
    write_bundle(bundle, out_dir, metadata)
    return bundle
