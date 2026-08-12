#!/usr/bin/env python3
"""Deterministic random-policy rollout on a generated topology.

Verifies the environment runs end to end (reset -> step -> terminate) and logs
the topology config hash, observation space and action space. This is the
reproducibility artefact: the same seed always yields the same rollout.

    python scripts/rollout_random.py --seed 42
    python scripts/rollout_random.py --seed 42 --reward-config configs/shaped.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.manifest import digest  # noqa: E402
from rlredteam.nasim_adapter import RewardWrapper  # noqa: E402
from rlredteam.reward import RewardConfig  # noqa: E402
from rlredteam.topology import TopologyConfig, describe, make_env  # noqa: E402


def rollout(seed: int, reward_config_path: Path | None, episodes: int, quiet: bool) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    topology_config = TopologyConfig.from_yaml()
    catalogue = CVECatalogue.open_default()
    reward_config = (
        RewardConfig.from_yaml(reward_config_path) if reward_config_path else RewardConfig()
    )

    env = make_env(topology_config, topology_seed=seed)
    summary = describe(env)
    wrapped = RewardWrapper(env, catalogue, topology_seed=seed, reward_config=reward_config)

    if not quiet:
        print("=" * 68)
        print(f"topology config hash : {topology_config.config_hash()}")
        print(f"CVE manifest sha256  : {digest(catalogue)}")
        print(f"topology seed        : {seed}")
        print(f"reward mode          : {reward_config.mode}")
        print("-" * 68)
        print(f"hosts / subnets      : {summary['num_hosts']} / {summary['num_subnets']}")
        print(f"observation space    : Box{tuple(summary['observation_space'])}")
        print(f"action space         : Discrete({summary['action_space_n']})")
        print(f"exploits             : {summary['exploits']}")
        print(f"privescs             : {summary['privescs']}")
        print(f"sensitive hosts      : {summary['sensitive_hosts']}")
        print(
            f"exploit choice       : {summary['hosts_with_exploit_choice']}"
            f"/{summary['num_hosts']} hosts, "
            f"mean {summary['mean_applicable_exploits_per_host']:.2f} applicable"
        )
        print("-" * 68)
        print("CVE assignment:")
        for name, record in sorted(wrapped.adapter.assignment.records.items()):
            print(
                f"  {name:<18} {record.cve_id:<16} "
                f"{record.base_score:>5}  {record.base_severity}"
            )
        print("-" * 68)

    results = []
    for episode in range(episodes):
        obs, _ = wrapped.reset(seed=seed + episode)
        assert obs.shape == env.observation_space.shape

        total = native = 0.0
        steps = 0
        terminated = truncated = False
        rng = random.Random(seed + episode)

        while not (terminated or truncated):
            action = rng.randrange(env.action_space.n)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            total += reward
            native += info["native_reward"]
            steps += 1

        results.append(
            {
                "episode": episode,
                "steps": steps,
                "shaped_return": round(total, 3),
                "native_return": round(native, 3),
                "goal_reached": bool(terminated),
                "terminal": "goal" if terminated else "step_limit",
            }
        )
        if not quiet:
            print(
                f"  episode {episode}: {steps:>4} steps  "
                f"shaped={total:>9.2f}  native={native:>9.2f}  "
                f"{'GOAL' if terminated else 'step_limit'}"
            )

    return {
        "topology_config_hash": topology_config.config_hash(),
        "cve_manifest_sha256": digest(catalogue),
        "topology_seed": seed,
        "reward_mode": str(reward_config.mode),
        "topology": summary,
        "cve_assignment": wrapped.adapter.assignment.mapping,
        "episodes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--reward-config", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = rollout(args.seed, args.reward_config, args.episodes, args.quiet)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")
    elif args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("=" * 68)
        print("rollout OK - env reset, stepped and terminated cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
