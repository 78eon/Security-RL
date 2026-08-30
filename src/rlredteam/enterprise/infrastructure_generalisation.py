"""Reproducible PPO training and evaluation across enterprise profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from rlredteam.enterprise.generalisation import (
    GeneralisationError,
    MaskedPolicy,
    _canonical_digest,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    policy_digest,
    sha256_file,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit, topology_digest
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
    generate_profile_topology,
)
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "infrastructure_generalisation.yaml"
)


@dataclass(frozen=True, slots=True)
class InfrastructureExperimentConfig:
    experiment_id: str
    description: str
    training_seed: int
    total_timesteps: int
    train_profiles: tuple[DeploymentProfile, ...]
    evaluation_episode_seeds: tuple[int, ...]
    deterministic_evaluation: bool
    ppo: dict

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> InfrastructureExperimentConfig:
        raw = yaml.safe_load((path or DEFAULT_EXPERIMENT_CONFIG).read_text())["experiment"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw["description"]),
            training_seed=int(raw["training_seed"]),
            total_timesteps=int(raw["total_timesteps"]),
            train_profiles=tuple(DeploymentProfile(item) for item in raw["train_profiles"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            ppo=dict(raw["ppo"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        expected_profiles = {
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        }
        if set(self.train_profiles) != expected_profiles:
            raise ValueError("training must include legacy, cloud and hybrid exactly once")
        if len(self.train_profiles) != len(set(self.train_profiles)):
            raise ValueError("training profiles must be unique")
        if not self.experiment_id or self.training_seed < 0 or self.total_timesteps <= 0:
            raise ValueError("experiment id, seed and timesteps must be valid")
        if not self.evaluation_episode_seeds or len(set(self.evaluation_episode_seeds)) != len(
            self.evaluation_episode_seeds
        ):
            raise ValueError("evaluation episode seeds must be non-empty and unique")
        required = {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "policy_layers",
        }
        if set(self.ppo) != required:
            raise ValueError(f"PPO configuration fields differ: {sorted(set(self.ppo) ^ required)}")
        if int(self.ppo["n_steps"]) % int(self.ppo["batch_size"]):
            raise ValueError("PPO n_steps must be divisible by batch_size")

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


def infrastructure_distribution_manifest(
    split: OnPremGeneralisationSplit,
    profile_config: EnterpriseProfileConfig,
    profiles: tuple[DeploymentProfile, ...],
) -> dict:
    """Hash every profile/topology pair in the preregistered distribution."""
    return {
        split_name: {
            profile.value: {
                str(seed): topology_digest(
                    generate_profile_topology(profile, seed, profile_config)
                )
                for seed in getattr(split, split_name)
            }
            for profile in profiles
        }
        for split_name in ("train", "validation", "test")
    }


def infrastructure_vulnerability_manifest(
    split: OnPremGeneralisationSplit,
    profile_config: EnterpriseProfileConfig,
    profiles: tuple[DeploymentProfile, ...],
) -> str:
    records = {
        profile.value: {
            str(seed): generate_profile_topology(profile, seed, profile_config).to_dict()[
                "vulnerabilities"
            ]
            for seed in (*split.train, *split.validation, *split.test)
        }
        for profile in profiles
    }
    return _canonical_digest({"source": "synthetic-enterprise-profiles-v1", "records": records})


def train_infrastructure_policy(
    output_dir: Path,
    *,
    config: InfrastructureExperimentConfig | None = None,
    timesteps: int | None = None,
    training_seed: int | None = None,
    allow_dirty: bool = False,
) -> dict:
    """Train one MaskablePPO policy over legacy, cloud and hybrid samples."""
    from sb3_contrib import MaskablePPO

    config = config or InfrastructureExperimentConfig.from_yaml()
    split = OnPremGeneralisationSplit()
    profile_config = EnterpriseProfileConfig.from_yaml()
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
    env = InfrastructureCurriculumEnv(
        split.train,
        config.train_profiles,
        config=profile_config,
    )
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
    distribution = infrastructure_distribution_manifest(
        split, profile_config, config.train_profiles
    )
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
        "train_profiles": [profile.value for profile in config.train_profiles],
        "train_topology_seeds": list(split.train),
        "validation_topology_seeds": list(split.validation),
        "test_topology_seeds": list(split.test),
        "experiment_config_hash": config.digest(),
        "topology_config_hash": profile_config.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "ppo": config.ppo,
        "distribution": distribution,
        "synthetic_vulnerability_manifest_sha256": infrastructure_vulnerability_manifest(
            split, profile_config, config.train_profiles
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


def validate_infrastructure_checkpoint(
    manifest: dict,
    checkpoint: Path,
    config: InfrastructureExperimentConfig,
    *,
    require_current_commit: bool = True,
    allow_unverifiable: bool = False,
) -> None:
    split = OnPremGeneralisationSplit()
    profile_config = EnterpriseProfileConfig.from_yaml()
    checks = {
        "algorithm": (manifest.get("algorithm"), "MaskablePPO"),
        "experiment config": (manifest.get("experiment_config_hash"), config.digest()),
        "topology config": (manifest.get("topology_config_hash"), profile_config.digest()),
        "dependency lock": (manifest.get("dependency_lock_hash"), dependency_lock_hash()),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "profiles": (
            manifest.get("train_profiles"),
            [profile.value for profile in config.train_profiles],
        ),
        "training split": (manifest.get("train_topology_seeds"), list(split.train)),
        "validation split": (
            manifest.get("validation_topology_seeds"),
            list(split.validation),
        ),
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


def evaluate_infrastructure_policy(
    model: MaskedPolicy,
    *,
    profiles: tuple[DeploymentProfile, ...],
    topology_seeds: tuple[int, ...],
    evaluation_episode_seeds: tuple[int, ...],
    deterministic: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Evaluate frozen weights for each profile on unseen topology seeds."""
    if not profiles or not topology_seeds or not evaluation_episode_seeds:
        raise GeneralisationError("evaluation profiles, topology and episode seeds are required")
    if set(topology_seeds) & set(OnPremGeneralisationSplit().train):
        raise GeneralisationError("evaluation topology seeds overlap the training split")

    profile_config = EnterpriseProfileConfig.from_yaml()
    episodes: list[dict] = []
    steps: list[dict] = []
    before = policy_digest(model)
    for profile in profiles:
        for topology_seed in topology_seeds:
            env = InfrastructureCurriculumEnv(
                (topology_seed,),
                (profile,),
                config=profile_config,
            )
            for episode_seed in evaluation_episode_seeds:
                set_all_seeds(episode_seed)
                model.set_random_seed(episode_seed)
                observation, reset_info = env.reset(
                    seed=episode_seed,
                    options={"profile": profile.value, "topology_seed": topology_seed},
                )
                terminated = truncated = False
                total_reward = 0.0
                invalid_mask_selections = 0
                failed_actions = 0
                episode_steps = 0
                while not (terminated or truncated):
                    mask = env.action_masks()
                    if not mask.any():
                        raise GeneralisationError(
                            "no valid agent-visible action before termination"
                        )
                    action, _ = model.predict(
                        observation,
                        action_masks=mask,
                        deterministic=deterministic,
                    )
                    action = int(np.asarray(action).item())
                    invalid_mask_selections += int(not mask[action])
                    observation, reward, terminated, truncated, info = env.step(action)
                    event = info["event"]
                    vulnerability = env.true_topology.vulnerabilities.get(
                        event.action.target
                    )
                    total_reward += float(reward)
                    failed_actions += int(not event.success)
                    steps.append(
                        {
                            "profile": profile.value,
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
                        "profile": profile.value,
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

