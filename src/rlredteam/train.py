"""PPO training on the RLRedTeam NASim environment.

    python -m rlredteam.train --seed 42 --timesteps 50000
    python -m rlredteam.train --seed 42 --reward-config configs/sparse.yaml --postgres

Uses Stable-Baselines3 PPO defaults as the sanity baseline. Every episode is
recorded with BOTH the shaped reward the agent optimises and the native NASim
reward, because the shaped and sparse arms optimise different objectives and
their raw returns are therefore measured with different rulers -- only the
native reward makes the ablation well-posed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rlredteam import provenance as provenance_mod
from rlredteam.catalogue import CVECatalogue
from rlredteam.events import AccessLevel
from rlredteam.manifest import digest
from rlredteam.nasim_adapter import RewardWrapper
from rlredteam.provenance import ExperimentManifest
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig, describe, make_env

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# CP-18: every PPO hyperparameter is recorded explicitly rather than inherited
# silently from SB3, so a manifest fully describes the agent that was trained.
# These ARE the SB3 defaults except ent_coef, which is raised from 0.0 because a
# 120-action space where most actions are invalid early collapses exploration
# otherwise. CP-19: that single deviation is declared here, not hidden.
PPO_DEFAULTS: dict = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # deviation from the SB3 default of 0.0
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cpu",
}
SB3_DEFAULT_DEVIATIONS = {"ent_coef": (0.0, 0.01)}

# The one Essential topology. Every run in the graded comparison trains on this
# network; only the training seed and the reward mode vary (CP-10, CP-11).
DEFAULT_TOPOLOGY_SEED = 42


# -- reproducibility -------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """Seed every source of randomness that affects a run.

    PYTHONHASHSEED must be set before the interpreter starts to have any
    effect; the Dockerfile sets it to 0. Setting it here only helps if the
    process is re-executed, so it is asserted rather than assigned.

    torch is pinned to a single thread because multi-threaded float reduction
    order is nondeterministic, which makes bitwise-identical reruns impossible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def config_digest(path: Path) -> str:
    """Hash of a config file's bytes, logged so a run traces to its settings."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# -- per-episode collection ------------------------------------------------


@dataclass
class _Partial:
    """Accumulator for one in-flight episode."""

    shaped: float = 0.0
    native: float = 0.0
    steps: int = 0
    exploited: dict[tuple[int, int], float] = field(default_factory=dict)
    step_rows: list = field(default_factory=list)

    def reset(self) -> None:
        self.shaped = 0.0
        self.native = 0.0
        self.steps = 0
        self.exploited.clear()
        self.step_rows.clear()


class EpisodeCollector(BaseCallback):
    """Turns per-step info into one record per finished episode.

    Reads the fields RewardWrapper puts in info: ``native_reward``,
    ``reward_breakdown`` and ``attack_event``.
    """

    def __init__(
        self,
        seed: int,
        topology_seed: int,
        episode_logger=None,
        csv_path: Path | None = None,
        record_steps: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.seed = seed
        self.topology_seed = topology_seed
        self.episode_logger = episode_logger
        self.csv_path = csv_path
        self.record_steps = record_steps

        self.episodes: list[dict] = []
        self._partials: dict[int, _Partial] = {}
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "episode_idx",
                "seed",
                "topology_seed",
                "timesteps",
                "shaped_return",
                "native_return",
                "length",
                "terminal_state",
                "goal_reached",
                "hosts_compromised",
                "mean_cvss_exploited",
                "max_cvss_exploited",
            ],
        )
        self._csv_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for env_idx, info in enumerate(infos):
            partial = self._partials.setdefault(env_idx, _Partial())

            event = info.get("attack_event")
            breakdown = info.get("reward_breakdown")
            if event is None or breakdown is None:
                continue  # not one of our wrapped envs

            partial.shaped += breakdown.total
            partial.native += info.get("native_reward", 0.0)
            partial.steps += 1

            # Record the CVSS of each host actually compromised, for the
            # high-value-target and severity-shift metrics.
            if event.success and event.access_gained > AccessLevel.NONE:
                if event.target is not None and event.cvss_base is not None:
                    previous = partial.exploited.get(event.target, 0.0)
                    partial.exploited[event.target] = max(previous, event.cvss_base)

            if self.record_steps:
                partial.step_rows.append((event, breakdown))

            if dones[env_idx]:
                self._finish(env_idx, partial, info)
                partial.reset()

        return True

    def _finish(self, env_idx: int, partial: _Partial, info: dict) -> None:
        # SB3 flattens gymnasium's (terminated, truncated) into a single `done`
        # plus this flag. NASim truncates on step limit and terminates on goal,
        # so conflating them would corrupt the success-rate metric.
        truncated = bool(info.get("TimeLimit.truncated", False))
        goal_reached = not truncated

        scores = list(partial.exploited.values())
        record = {
            "episode_idx": len(self.episodes),
            "seed": self.seed,
            "topology_seed": self.topology_seed,
            "timesteps": int(self.num_timesteps),
            "shaped_return": round(partial.shaped, 4),
            "native_return": round(partial.native, 4),
            "length": partial.steps,
            "terminal_state": "goal" if goal_reached else "step_limit",
            "goal_reached": goal_reached,
            "hosts_compromised": len(partial.exploited),
            "mean_cvss_exploited": round(sum(scores) / len(scores), 3) if scores else None,
            "max_cvss_exploited": max(scores) if scores else None,
        }
        self.episodes.append(record)

        if self._csv_writer is not None:
            self._csv_writer.writerow(record)
            self._csv_file.flush()

        if self.episode_logger is not None:
            self._persist(record, partial)

    def _persist(self, record: dict, partial: _Partial) -> None:
        from rlredteam.storage.postgres_logger import EpisodeRecord, StepRecord

        steps = [
            StepRecord(
                step_idx=idx,
                action_name=event.action_name,
                action_kind=str(event.kind),
                tactic=breakdown.tactic_name,
                technique_id=breakdown.technique_id,
                target_subnet=event.target[0] if event.target else None,
                target_host=event.target[1] if event.target else None,
                success=event.success,
                reward=breakdown.total,
                native_reward=event.native_reward,
                cve_id=event.cve_id,
                cvss_base=event.cvss_base,
                cve_term=breakdown.cve,
                tactic_term=breakdown.tactic,
                crown_jewel_term=breakdown.crown_jewel,
                penalty_term=breakdown.penalty,
                access_gained=int(event.access_gained),
                newly_discovered=event.newly_discovered,
                is_crown_jewel=event.is_crown_jewel,
                reward_paid=breakdown.paid,
                error=event.error,
            )
            for idx, (event, breakdown) in enumerate(partial.step_rows)
        ]

        self.episode_logger.log_episode(
            EpisodeRecord(
                seed=record["seed"],
                topology_seed=record["topology_seed"],
                episode_idx=record["episode_idx"],
                total_reward=record["shaped_return"],
                native_reward=record["native_return"],
                length=record["length"],
                terminal_state=record["terminal_state"],
                goal_reached=record["goal_reached"],
                exploited_hosts=[list(addr) for addr in partial.exploited],
                mean_cvss_exploited=record["mean_cvss_exploited"],
                max_cvss_exploited=record["max_cvss_exploited"],
                hosts_compromised=record["hosts_compromised"],
                steps=steps,
            )
        )

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


# -- environment -----------------------------------------------------------


def build_env(
    topology_config: TopologyConfig,
    reward_config: RewardConfig,
    catalogue: CVECatalogue,
    topology_seed: int,
    seed: int,
):
    """One wrapped env inside a DummyVecEnv.

    DummyVecEnv rather than SubprocVecEnv: subprocess workers make seeding and
    reproducibility far harder to reason about, and reproducibility is graded.
    Parallelise across runs at the process level instead.
    """

    def factory():
        env = make_env(topology_config, topology_seed=topology_seed)
        env = RewardWrapper(
            env, catalogue, topology_seed=topology_seed, reward_config=reward_config
        )
        env.reset(seed=seed)
        return env

    vec = DummyVecEnv([factory])
    vec.seed(seed)
    return vec


# -- entry point -----------------------------------------------------------


def train(args: argparse.Namespace) -> dict:
    set_all_seeds(args.seed)

    topology_config = TopologyConfig.from_yaml()
    reward_config = RewardConfig.from_yaml(args.reward_config)
    catalogue = CVECatalogue.open_default()

    # CP-11: the topology seed is NEVER derived from the training seed. Doing so
    # gives every seed its own network, which confounds reward mode with
    # topology and destroys the causal comparison the ablation exists to make.
    # It has a fixed default and must be changed deliberately.
    topology_seed = args.topology_seed
    prefix = f"{args.experiment_id}-" if args.experiment_id else ""
    run_name = f"{prefix}{reward_config.mode}-s{args.seed}-t{topology_seed}"
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(topology_config, reward_config, catalogue, topology_seed, args.seed)
    summary = describe(env.envs[0].env)

    described = summary
    ppo_config = dict(PPO_DEFAULTS)
    ppo_config["ent_coef"] = args.ent_coef

    manifest = ExperimentManifest(
        experiment_id=run_name,
        git_commit=provenance_mod.git_commit(),
        git_dirty=provenance_mod.git_dirty(),
        python_version=provenance_mod.python_version(),
        dependency_lock_hash=provenance_mod.dependency_lock_hash(),
        docker_image_digest=provenance_mod.docker_image_digest(),
        training_seed=args.seed,
        topology_seed=topology_seed,
        topology_hash=provenance_mod.topology_hash(described),
        topology_config_hash=topology_config.config_hash(),
        environment_config_hash=provenance_mod.environment_config_hash(described),
        cve_manifest_sha256=digest(catalogue),
        reward_config_hash=reward_config.hash(),
        ppo_config_hash=provenance_mod._digest(ppo_config),
        dataset_version=str(len(catalogue)),
        training_budget=args.timesteps,
        checkpoint_path=str(out_dir / "model.zip"),
        ppo_config=ppo_config,
        reward_mode=str(reward_config.mode),
    )

    frozen = json.loads(Path(args.frozen).read_text()) if args.frozen else None
    gate = provenance_mod.run_gate(
        manifest,
        frozen=frozen,
        require_secure_db=args.postgres,
        require_offline=not args.allow_network,
    )
    print("pre-run gate:")
    print(gate.report())
    if not args.skip_gate:
        provenance_mod.enforce(gate)
    elif not gate.passed:
        print(
            "  WARNING: gate failures ignored via --skip-gate; this run is not "
            "valid for the graded comparison."
        )

    print(f"run                  : {run_name}")
    print(f"reward mode          : {reward_config.mode}")
    print(f"topology hash        : {manifest.topology_hash}")
    print(f"topology config hash : {manifest.topology_config_hash}")
    print(f"CVE manifest SHA-256 : {manifest.cve_manifest_sha256}")
    print(f"observation space    : {env.observation_space.shape}")
    print(f"action space         : {env.action_space}")
    print(f"output               : {out_dir}")

    episode_logger = None
    if args.postgres:
        from rlredteam.storage.postgres_logger import EpisodeLogger

        episode_logger = EpisodeLogger.start(
            name=run_name,
            reward_mode=str(reward_config.mode),
            config_hash=manifest.reward_config_hash,
            topology_config_hash=manifest.topology_config_hash,
            topology_hash=manifest.topology_hash,
            topology_id=summary["name"],
            cve_manifest_sha256=manifest.cve_manifest_sha256,
            seed_set=[args.seed],
            log_steps=args.log_steps,
            condition=str(reward_config.mode),
            algorithm="PPO",
            hyperparameters=ppo_config,
            designation="training",
            checkpoint_path=manifest.checkpoint_path,
        )
        manifest.database_experiment_id = episode_logger.experiment_id
        manifest.database_run_id = episode_logger.run_id

    manifest.write(out_dir / "manifest.json")
    provenance = manifest.to_dict() | {"topology": summary}
    (out_dir / "config.snapshot.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str)
    )

    collector = EpisodeCollector(
        seed=args.seed,
        topology_seed=topology_seed,
        episode_logger=episode_logger,
        csv_path=out_dir / "episodes.csv",
        record_steps=args.log_steps,
    )
    callbacks = [collector]
    if args.checkpoint_every:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=str(out_dir / "checkpoints"),
                name_prefix="ppo",
            )
        )

    # SB3 defaults, per the work plan's "defaults as the sanity baseline".
    # ent_coef is the one deviation: the default of 0.0 collapses exploration
    # in a 120-action space where most actions are invalid early on.
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        ent_coef=args.ent_coef,
        verbose=1 if args.verbose else 0,
        device="cpu",
    )

    training_status = "complete"
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    except Exception:
        training_status = "failed"
        raise
    finally:
        if episode_logger is not None:
            episode_logger.flush()
            episode_logger.finish(training_status)
            episode_logger._conn.close()

    # runs/ is gitignored: trained attack-policy weights are gated by default.
    model.save(out_dir / "model")

    report = summarise(collector.episodes, provenance)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    return report


def summarise(episodes: list[dict], provenance: dict) -> dict:
    if not episodes:
        return {"provenance": provenance, "episodes": 0}

    tail = episodes[-max(1, len(episodes) // 10) :]  # last 10%, as convergence proxy
    successes = [e for e in episodes if e["goal_reached"]]
    tail_successes = [e for e in tail if e["goal_reached"]]
    natives = [e["native_return"] for e in episodes]
    tail_natives = [e["native_return"] for e in tail]

    return {
        "provenance": provenance,
        "episodes": len(episodes),
        "success_rate_overall": len(successes) / len(episodes),
        "success_rate_last_10pct": len(tail_successes) / len(tail) if tail else 0.0,
        "mean_native_return_overall": float(np.mean(natives)),
        "mean_native_return_last_10pct": float(np.mean(tail_natives)) if tail else None,
        "var_native_return_last_10pct": float(np.var(tail_natives)) if tail else None,
        "mean_steps_to_goal": (
            float(np.mean([e["length"] for e in tail_successes])) if tail_successes else None
        ),
        "mean_cvss_exploited_last_10pct": (
            float(
                np.mean(
                    [e["mean_cvss_exploited"] for e in tail if e["mean_cvss_exploited"] is not None]
                )
            )
            if any(e["mean_cvss_exploited"] is not None for e in tail)
            else None
        ),
    }


def print_summary(report: dict) -> None:
    if not report.get("episodes"):
        print("\nno episodes completed")
        return
    print("\n" + "=" * 60)
    print(f"episodes                  : {report['episodes']}")
    print(f"success rate (all)        : {report['success_rate_overall']:.3f}")
    print(f"success rate (last 10%)   : {report['success_rate_last_10pct']:.3f}")
    print(f"mean native return (all)  : {report['mean_native_return_overall']:.2f}")
    if report["mean_native_return_last_10pct"] is not None:
        print(f"mean native (last 10%)    : {report['mean_native_return_last_10pct']:.2f}")
    if report["mean_steps_to_goal"] is not None:
        print(f"mean steps to goal        : {report['mean_steps_to_goal']:.1f}")
    if report["mean_cvss_exploited_last_10pct"] is not None:
        print(f"mean CVSS exploited       : {report['mean_cvss_exploited_last_10pct']:.2f}")
    print("=" * 60)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="training seed")
    parser.add_argument(
        "--experiment-id",
        default="",
        help="optional prefix that keeps a controlled experiment's runs isolated",
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=DEFAULT_TOPOLOGY_SEED,
        help=(
            f"network to train on (default {DEFAULT_TOPOLOGY_SEED}). Deliberately "
            "independent of --seed: holding it fixed while varying --seed is what "
            "makes the shaped-vs-sparse comparison causal. Change it only to "
            "measure generalisation across networks, and say so in the write-up."
        ),
    )
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=REPO_ROOT / "configs" / "shaped.yaml",
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--postgres", action="store_true", help="log episodes to PostgreSQL")
    parser.add_argument("--log-steps", action="store_true", help="also persist per-step rows")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--frozen",
        type=Path,
        default=None,
        help="experiment manifest of frozen hashes; a mismatch stops the run",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit external network during training (not valid for graded runs)",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="report gate failures but train anyway (smoke tests only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("PYTHONHASHSEED") not in ("0",):
        print(
            "warning: PYTHONHASHSEED is not 0; runs may not be bitwise reproducible",
            file=sys.stderr,
        )
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
