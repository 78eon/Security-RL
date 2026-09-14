#!/usr/bin/env python3
"""Evaluate the frozen CyberBattle PPO without any training callbacks."""

import json

from rlredteam.cyberbattle_study import evaluate_ppo

if __name__ == "__main__":
    print(json.dumps(evaluate_ppo(), indent=2, sort_keys=True))
