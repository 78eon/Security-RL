"""Train standard single-agent PPO through the CybORG adapter."""

from __future__ import annotations

import json

from rlredteam.cyborg_study import train_ppo


def main() -> None:
    print(json.dumps(train_ppo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
