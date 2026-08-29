"""Mask-aware PPO training and frozen evaluation across on-prem topologies.

This module is simulation-only. Policy decisions receive an Observation and an
AgentKnowledge-derived mask; realised topology is read only after a decision to
produce provenance and evaluation evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from rlredteam.enterprise.onprem import (
    OnPremCurriculumEnv,
    OnPremGeneralisationSplit,
    OnPremTopologyConfig,
    topology_digest,
)
from rlredteam.provenance import dependency_lock_hash, git_commit, git_dirty
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiments" / "onprem_generalisation.yaml"


class GeneralisationError(RuntimeError):
    """Raised when provenance or evaluation invariants are violated."""


class MaskedPolicy(Protocol):
    policy: object

    def predict(self, observation, *, action_masks, deterministic: bool): ...

    def set_random_seed(self, seed: int) -> None: ...


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_digest(model: MaskedPolicy) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OnPremExperimentConfig:
    experiment_id: str
    description: str
    training_seed: int
    total_timesteps: int
    evaluation_episode_seeds: tuple[int, ...]
    deterministic_evaluation: bool
    ppo: dict

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> OnPremExperimentConfig:
        raw = yaml.safe_load((path or DEFAULT_EXPERIMENT_CONFIG).read_text())["experiment"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw["description"]),
            training_seed=int(raw["training_seed"]),
            total_timesteps=int(raw["total_timesteps"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            ppo=dict(raw["ppo"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.experiment_id or self.training_seed < 0 or self.total_timesteps <= 0:
            raise ValueError("experiment id, seed and timesteps must be valid")
        if not self.evaluation_episode_seeds or len(set(self.evaluation_episode_seeds)) != len(
            self.evaluation_episode_seeds
        ):
            raise ValueError("evaluation episode seeds must be non-empty and unique")
        required = {
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
            "policy_layers",
        }
        if set(self.ppo) != required:
            raise ValueError(f"PPO configuration fields differ: {sorted(set(self.ppo) ^ required)}")
        if int(self.ppo["n_steps"]) % int(self.ppo["batch_size"]):
            raise ValueError("PPO n_steps must be divisible by batch_size")

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


def distribution_manifest(
    split: OnPremGeneralisationSplit, topology_config: OnPremTopologyConfig
) -> dict:
    """Hash every preregistered topology so split membership is auditable."""
    from rlredteam.enterprise.onprem import generate_onprem_topology

    groups = {}
    for name in ("train", "validation", "test"):
        seeds = getattr(split, name)
        groups[name] = {
            str(seed): topology_digest(generate_onprem_topology(seed, topology_config))
            for seed in seeds
        }
    return groups


def synthetic_vulnerability_manifest(
    split: OnPremGeneralisationSplit, topology_config: OnPremTopologyConfig
) -> str:
    """Hash the synthetic vulnerability definitions across the frozen split."""
    from rlredteam.enterprise.onprem import generate_onprem_topology

    records = {}
    for seed in (*split.train, *split.validation, *split.test):
        topology = generate_onprem_topology(seed, topology_config).to_dict()
        records[str(seed)] = topology["vulnerabilities"]
    return _canonical_digest({"source": "synthetic-onprem-v1", "records": records})


def train_policy(
    output_dir: Path,
    *,
    config: OnPremExperimentConfig | None = None,
    timesteps: int | None = None,
    training_seed: int | None = None,
    allow_dirty: bool = False,
) -> dict:
    """Train MaskablePPO across only the preregistered training topology seeds."""
    from sb3_contrib import MaskablePPO

    config = config or OnPremExperimentConfig.from_yaml()
    split = OnPremGeneralisationSplit()
    topology_config = OnPremTopologyConfig.from_yaml()
    budget = int(timesteps if timesteps is not None else config.total_timesteps)
    seed = int(training_seed if training_seed is not None else config.training_seed)
    dirty = git_dirty()
    if dirty is not False and not allow_dirty:
        state = "dirty" if dirty else "unknown"
        raise GeneralisationError(
            f"refusing canonical training while working-tree state is {state}"
        )
    if budget <= 0:
        raise ValueError("timesteps must be positive")
    if budget % int(config.ppo["n_steps"]):
        raise GeneralisationError(
            "training budget must be divisible by PPO n_steps to prevent silent rounding"
        )

    set_all_seeds(seed)
    env = OnPremCurriculumEnv(split.train, config=topology_config)
    ppo = dict(config.ppo)
    layers = list(ppo.pop("policy_layers"))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=1,
        policy_kwargs={"net_arch": layers},
        **ppo,
    )
    model.learn(total_timesteps=budget)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    manifest = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "description": config.description,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "action_mask_source": "AgentKnowledge",
        "training_seed": seed,
        "training_timesteps": budget,
        "actual_training_timesteps": int(model.num_timesteps),
        "train_topology_seeds": list(split.train),
        "validation_topology_seeds": list(split.validation),
        "test_topology_seeds": list(split.test),
        "experiment_config_hash": config.digest(),
        "topology_config_hash": topology_config.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "ppo": config.ppo,
        "distribution": distribution_manifest(split, topology_config),
        "synthetic_vulnerability_manifest_sha256": synthetic_vulnerability_manifest(
            split, topology_config
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": policy_digest(model),
        "weights_release": "gated; runs/ is gitignored",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    env.close()
    return manifest


def validate_checkpoint_manifest(
    manifest: dict,
    checkpoint: Path,
    config: OnPremExperimentConfig,
    *,
    require_current_commit: bool = True,
    allow_unverifiable: bool = False,
) -> None:
    split = OnPremGeneralisationSplit()
    topology_config = OnPremTopologyConfig.from_yaml()
    checks = {
        "algorithm": (manifest.get("algorithm"), "MaskablePPO"),
        "experiment config": (manifest.get("experiment_config_hash"), config.digest()),
        "topology config": (manifest.get("topology_config_hash"), topology_config.digest()),
        "dependency lock": (manifest.get("dependency_lock_hash"), dependency_lock_hash()),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "training split": (manifest.get("train_topology_seeds"), list(split.train)),
        "validation split": (manifest.get("validation_topology_seeds"), list(split.validation)),
        "test split": (manifest.get("test_topology_seeds"), list(split.test)),
    }
    if require_current_commit:
        checks["code commit"] = (manifest.get("git_commit"), git_commit())
    failures = [
        f"{name}: manifest={actual!r}, runtime={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if manifest.get("git_dirty") is not False and not allow_unverifiable:
        failures.append("checkpoint working-tree state was dirty or unknown")
    if git_dirty() is not False and not allow_unverifiable:
        failures.append("evaluation working-tree state is dirty or unknown")
    if failures:
        raise GeneralisationError("checkpoint provenance mismatch:\n  " + "\n  ".join(failures))


def evaluate_policy(
    model: MaskedPolicy,
    *,
    topology_seeds: tuple[int, ...],
    evaluation_episode_seeds: tuple[int, ...],
    deterministic: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Evaluate frozen weights on unseen topologies with mask-aware inference."""
    if not topology_seeds or not evaluation_episode_seeds:
        raise GeneralisationError("evaluation topology and episode seeds are required")
    if set(topology_seeds) & set(OnPremGeneralisationSplit().train):
        raise GeneralisationError("evaluation topology seeds overlap the training split")

    topology_config = OnPremTopologyConfig.from_yaml()
    episodes: list[dict] = []
    steps: list[dict] = []
    before = policy_digest(model)
    for topology_seed in topology_seeds:
        env = OnPremCurriculumEnv((topology_seed,), config=topology_config)
        for episode_seed in evaluation_episode_seeds:
            set_all_seeds(episode_seed)
            model.set_random_seed(episode_seed)
            observation, reset_info = env.reset(
                seed=episode_seed, options={"topology_seed": topology_seed}
            )
            terminated = truncated = False
            total_reward = 0.0
            invalid_mask_selections = 0
            failed_actions = 0
            episode_steps = 0
            while not (terminated or truncated):
                mask = env.action_masks()
                if not mask.any():
                    raise GeneralisationError("no valid agent-visible action before termination")
                action, _ = model.predict(
                    observation,
                    action_masks=mask,
                    deterministic=deterministic,
                )
                action = int(np.asarray(action).item())
                invalid_mask_selections += int(not mask[action])
                observation, reward, terminated, truncated, info = env.step(action)
                event = info["event"]
                vulnerability = env.true_topology.vulnerabilities.get(event.action.target)
                total_reward += float(reward)
                failed_actions += int(not event.success)
                steps.append(
                    {
                        "topology_seed": topology_seed,
                        "topology_hash": reset_info["topology_hash"],
                        "evaluation_seed": episode_seed,
                        "step": event.step,
                        "action": event.action.name,
                        "action_kind": event.action.type.value,
                        "target_entity": event.action.target,
                        "success": event.success,
                        "state_changed": event.state_changed,
                        "reward": event.reward,
                        "prerequisites": list(event.prerequisites),
                        "outcomes": list(event.outcomes),
                        "reason": event.reason,
                        "goal_reached": event.goal_reached,
                        "cve_id": vulnerability.id if vulnerability else None,
                        "cvss_base": vulnerability.cvss if vulnerability else None,
                    }
                )
                episode_steps += 1
            known_nodes = len(env.knowledge.discovered)
            episodes.append(
                {
                    "topology_seed": topology_seed,
                    "topology_hash": reset_info["topology_hash"],
                    "evaluation_seed": episode_seed,
                    "goal_reached": bool(terminated and not truncated),
                    "terminal_reason": "goal" if terminated else "step_limit",
                    "steps_to_goal": episode_steps if terminated else None,
                    "episode_length": episode_steps,
                    "total_reward": total_reward,
                    "known_nodes": known_nodes,
                    "true_nodes": len(env.true_topology.nodes),
                    "discovery_coverage": known_nodes / len(env.true_topology.nodes),
                    "invalid_mask_selections": invalid_mask_selections,
                    "failed_actions": failed_actions,
                    "hosts_compromised": len(env.knowledge.access),
                    "path_events": len(env.attack_path()),
                }
            )
        env.close()
    after = policy_digest(model)
    if before != after:
        raise GeneralisationError("policy parameters changed during frozen evaluation")
    return episodes, steps


def _normalise_atom(atom: str) -> set[str]:
    prefix, separator, entity = atom.partition(":")
    if not separator:
        return {atom}
    atoms = {atom}
    if prefix in {
        "known_host",
        "known_service",
        "known_component",
        "known_target",
        "known_identity",
    }:
        atoms.add(f"known:{entity}")
    if prefix in {"authenticated", "pivoted", "root_access", "user_access", "read_access"}:
        atoms.add(f"access:{entity}")
    if prefix == "discovered":
        atoms.update({f"known:{entity}", f"reachable:{entity}"})
    if prefix == "access_target":
        atoms.add(f"reachable:{entity}")
    if prefix == "network":
        atoms.add(f"reachable:{entity}")
    if prefix == "vulnerability":
        atoms.add(f"known_vulnerability:{entity}")
    if prefix == "credential_source":
        atoms.add(f"known:{entity}")
    return atoms


def reconstruct_attack_path(episode_steps: list[dict]) -> list[dict]:
    """Back-chain prerequisites from the observed goal event.

    This uses only recorded actions, prerequisites and outcomes. It never asks
    the topology for a shortest path, so the resulting graph is evidence of
    what the policy actually did rather than a post-hoc omniscient route.
    """
    progress = [
        step for step in episode_steps if step.get("success") and step.get("state_changed")
    ]
    goals = [step for step in progress if step.get("goal_reached")]
    if not goals:
        return []
    final = goals[-1]
    selected = [final]
    required = {
        atom
        for item in final.get("prerequisites", [])
        for atom in _normalise_atom(str(item))
    }
    for step in reversed(progress[: progress.index(final)]):
        produced = {
            atom
            for item in step.get("outcomes", [])
            for atom in _normalise_atom(str(item))
        }
        if not produced & required:
            continue
        selected.append(step)
        required -= produced
        required.update(
            atom
            for item in step.get("prerequisites", [])
            for atom in _normalise_atom(str(item))
        )
    return list(reversed(selected))


def persist_evaluation(
    *,
    episodes: list[dict],
    steps: list[dict],
    training_manifest: dict,
    checkpoint: Path,
    split_name: str,
) -> dict:
    """Persist and reconstruct the enterprise evaluation in PostgreSQL."""
    from rlredteam.storage.postgres_logger import EpisodeLogger, EpisodeRecord, StepRecord

    if not episodes:
        raise GeneralisationError("cannot persist an empty evaluation")
    grouped: dict[tuple[int, int], list[dict]] = {}
    for step in steps:
        key = (int(step["topology_seed"]), int(step["evaluation_seed"]))
        grouped.setdefault(key, []).append(step)

    logger = EpisodeLogger.start(
        name=f"{training_manifest['experiment_id']}-{split_name}",
        reward_mode="enterprise_shaped",
        config_hash=str(training_manifest["experiment_config_hash"]),
        topology_config_hash=str(training_manifest["topology_config_hash"]),
        topology_hash=_canonical_digest(training_manifest["distribution"][split_name]),
        cve_manifest_sha256=str(
            training_manifest["synthetic_vulnerability_manifest_sha256"]
        ),
        seed_set=[int(training_manifest["training_seed"])],
        condition="unseen_topology_generalisation",
        algorithm="MaskablePPO",
        hyperparameters=dict(training_manifest["ppo"]),
        designation="evaluation",
        evaluation_seeds=sorted({int(row["evaluation_seed"]) for row in episodes}),
        checkpoint_path=str(checkpoint),
        notes="Simulation-only enterprise evaluation; topology seeds were held out.",
        batch_size=20,
        log_steps=True,
    )
    try:
        for episode_index, episode in enumerate(episodes):
            key = (int(episode["topology_seed"]), int(episode["evaluation_seed"]))
            episode_steps = grouped.get(key, [])
            logger.log_episode(
                EpisodeRecord(
                    seed=int(episode["evaluation_seed"]),
                    topology_seed=int(episode["topology_seed"]),
                    episode_idx=episode_index,
                    total_reward=float(episode["total_reward"]),
                    native_reward=float(episode["total_reward"]),
                    length=int(episode["episode_length"]),
                    terminal_state=str(episode["terminal_reason"]),
                    goal_reached=bool(episode["goal_reached"]),
                    exploited_hosts=[
                        step["target_entity"]
                        for step in episode_steps
                        if step["action_kind"] == "exploit" and step["success"]
                    ],
                    hosts_compromised=int(episode["hosts_compromised"]),
                    topology_hash=str(episode["topology_hash"]),
                    known_nodes=int(episode["known_nodes"]),
                    true_nodes=int(episode["true_nodes"]),
                    discovery_coverage=float(episode["discovery_coverage"]),
                    invalid_mask_selections=int(episode["invalid_mask_selections"]),
                    failed_actions=int(episode["failed_actions"]),
                    steps=[
                        StepRecord(
                            step_idx=int(step["step"]),
                            action_name=str(step["action"]),
                            action_kind=str(step["action_kind"]),
                            success=bool(step["success"]),
                            reward=float(step["reward"]),
                            native_reward=float(step["reward"]),
                            cve_id=step.get("cve_id"),
                            cvss_base=step.get("cvss_base"),
                            target_entity=str(step["target_entity"]),
                            state_changed=bool(step["state_changed"]),
                            prerequisites=list(step["prerequisites"]),
                            outcomes=list(step["outcomes"]),
                            error=step.get("reason"),
                        )
                        for step in episode_steps
                    ],
                )
            )
        logger.flush()
        logger.finish("complete")
        return {
            "database_experiment_id": logger.experiment_id,
            "database_evaluation_run_id": logger.run_id,
            "database_episode_count": logger.episode_count(),
            "database_step_count": logger.step_count(),
        }
    except Exception:
        logger.finish("failed")
        raise
    finally:
        logger._conn.close()


def write_evaluation_package(
    output_dir: Path,
    *,
    episodes: list[dict],
    steps: list[dict],
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if episodes:
        with (output_dir / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
    with (output_dir / "trajectories.jsonl").open("w") as handle:
        for step in steps:
            handle.write(json.dumps(step, sort_keys=True) + "\n")
    attack_paths = []
    for topology_seed, evaluation_seed in sorted(
        {(row["topology_seed"], row["evaluation_seed"]) for row in episodes}
    ):
        episode_steps = [
            step
            for step in steps
            if step["topology_seed"] == topology_seed
            and step["evaluation_seed"] == evaluation_seed
        ]
        attack_paths.append(
            {
                "topology_seed": topology_seed,
                "evaluation_seed": evaluation_seed,
                "steps": reconstruct_attack_path(episode_steps),
            }
        )
    (output_dir / "attack_paths.json").write_text(
        json.dumps(attack_paths, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "episode_count": len(episodes),
        "topology_count": len({row["topology_seed"] for row in episodes}),
        "success_rate": sum(row["goal_reached"] for row in episodes) / len(episodes),
        "mean_steps_to_goal": (
            sum(row["steps_to_goal"] for row in episodes if row["steps_to_goal"] is not None)
            / max(1, sum(row["steps_to_goal"] is not None for row in episodes))
        ),
        "mean_reward": sum(row["total_reward"] for row in episodes) / len(episodes),
        "mean_coverage": sum(row["discovery_coverage"] for row in episodes) / len(episodes),
        "invalid_mask_selections": sum(row["invalid_mask_selections"] for row in episodes),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
