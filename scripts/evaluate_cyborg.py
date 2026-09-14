"""Evaluate the frozen CybORG PPO policy without updates."""

from __future__ import annotations

import json

from rlredteam.cyborg_study import evaluate_ppo


def main() -> None:
    print(json.dumps(evaluate_ppo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
