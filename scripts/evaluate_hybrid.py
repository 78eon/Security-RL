#!/usr/bin/env python3
"""Evaluate a frozen PPO policy on validation or held-out hybrid topologies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from rlredteam.enterprise.environment import EnterpriseActionType, EnterpriseCyberEnv
from rlredteam.enterprise.hybrid import (
    GeneralisationSplit,
    HybridGeneratorConfig,
    family_for_seed,
    generate_hybrid_enterprise,
)


def heuristic_actions(env: EnterpriseCyberEnv):
    vulnerability = next(iter(env.graph.vulnerabilities))
    sequence = (
        (EnterpriseActionType.DISCOVER_NETWORK, "network_edge"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_entry"),
        (EnterpriseActionType.ENUMERATE_SERVICE, "service_entry"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "application_entry"),
        (EnterpriseActionType.ASSESS_VULNERABILITY, "service_entry"),
        (EnterpriseActionType.EXPLOIT, vulnerability),
        (EnterpriseActionType.OBTAIN_CREDENTIAL, "identity_bridge"),
        (EnterpriseActionType.AUTHENTICATE, "host_pivot"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_pivot"),
        (EnterpriseActionType.PIVOT, "host_data"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_data"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "storage_target"),
        (EnterpriseActionType.AUTHENTICATE, "storage_target"),
        (EnterpriseActionType.ACCESS_ASSET, "asset_crown"),
    )
    return [env.action_index(kind, target) for kind, target in sequence]


def evaluate_seed(seed: int, model: PPO | None, max_steps: int = 200) -> dict:
    config = HybridGeneratorConfig()
    env = EnterpriseCyberEnv(
        generate_hybrid_enterprise(seed, config=config),
        max_steps=max_steps,
        max_nodes=config.max_nodes,
        max_vulnerabilities=config.max_vulnerabilities,
    )
    observation, _ = env.reset(seed=seed)
    planned = iter(heuristic_actions(env)) if model is None else None
    total_reward = 0.0
    invalid_actions = 0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        if model is None:
            try:
                action = next(planned)
            except StopIteration:
                break
        else:
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        invalid_actions += int(not info["event"].success)
        steps += 1
    return {
        "topology_seed": seed,
        "family": family_for_seed(seed).value,
        "goal_reached": terminated,
        "steps": steps,
        "total_reward": total_reward,
        "invalid_actions": invalid_actions,
        "known_nodes": len(env.knowledge.discovered),
        "path_events": len(env.attack_path()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Frozen PPO .zip; omit for feasibility baseline")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test seed limit")
    args = parser.parse_args()

    split = GeneralisationSplit()
    seeds = getattr(split, args.split)
    if args.limit:
        seeds = seeds[: args.limit]
    model = PPO.load(args.model, device="cpu") if args.model else None
    episodes = [evaluate_seed(seed, model) for seed in seeds]
    by_family: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_family[episode["family"]].append(episode)
    report = {
        "policy": "ppo" if model else "scripted_feasibility_baseline",
        "split": args.split,
        "model": str(args.model) if args.model else None,
        "episodes": episodes,
        "summary": {
            "episode_count": len(episodes),
            "success_rate": sum(item["goal_reached"] for item in episodes) / len(episodes),
            "mean_invalid_actions": sum(item["invalid_actions"] for item in episodes)
            / len(episodes),
            "success_rate_by_family": {
                family: sum(item["goal_reached"] for item in items) / len(items)
                for family, items in sorted(by_family.items())
            },
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
