#!/usr/bin/env python3
"""Run the knowledge-only feasibility oracle on one hidden on-prem topology."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rlredteam.enterprise.environment import EnterpriseCyberEnv
from rlredteam.enterprise.onprem import (
    OnPremTopologyConfig,
    generate_onprem_topology,
    knowledge_policy_action,
    topology_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument(
        "--print-topology",
        action="store_true",
        help="Analyst-only: print hidden ground truth after the episode",
    )
    args = parser.parse_args()
    config = OnPremTopologyConfig.from_yaml()
    topology = generate_onprem_topology(args.seed, config)
    env = EnterpriseCyberEnv(
        topology,
        max_steps=config.max_steps,
        max_nodes=config.max_nodes,
        max_vulnerabilities=config.max_vulnerabilities,
    )
    env.reset(seed=args.seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(knowledge_policy_action(env))

    report = {
        "mode": "knowledge_only_feasibility_oracle_not_ppo",
        "topology_seed": args.seed,
        "topology_hash": topology_digest(topology),
        "topology_config_hash": config.digest(),
        "nodes": len(topology.nodes),
        "edges": len(topology.edges),
        "goal_reached": terminated,
        "steps": len(env.events),
        "trajectory": [
            {
                **asdict(event),
                "action": event.action.name,
            }
            for event in env.attack_path()
        ],
    }
    if args.print_topology:
        report["analyst_ground_truth"] = topology.to_dict()
    print(json.dumps(report, indent=2, default=list))


if __name__ == "__main__":
    main()
