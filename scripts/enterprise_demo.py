#!/usr/bin/env python3
"""Run a deterministic end-to-end episode in the typed enterprise simulator."""

from __future__ import annotations

import argparse
import json

from rlredteam.enterprise.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-graph", action="store_true")
    args = parser.parse_args()

    env, path, total_reward = run_demo(args.seed)
    graph = env.graph
    if args.print_graph:
        print(json.dumps(graph.to_dict(), indent=2))
    print(
        f"{graph.name}: nodes={len(graph.nodes)} edges={len(graph.edges)} "
        f"actions={env.action_space.n} observation={env.observation_space.shape}"
    )
    for event in path:
        print(
            f"{event.step:02d} {event.action.name:<45} "
            f"success={str(event.success):<5} reward={event.reward:>5.1f} "
            f"outcomes={list(event.outcomes)}"
        )
    goal_reached = bool(env.knowledge.accessed_assets & set(graph.crown_jewels))
    print(f"goal_reached={goal_reached} total_reward={total_reward:.1f}")
    print("reconstructed attack path:")
    for event in path:
        print(f"  {event.action.name} -> {', '.join(event.outcomes) or 'state updated'}")


if __name__ == "__main__":
    main()
