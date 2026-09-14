#!/usr/bin/env python3
"""Train standard PPO independently through the CyberBattle adapter."""

import argparse
import json

from rlredteam.cyberbattle_study import train_ppo

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    result = train_ppo(total_timesteps=args.timesteps)
    print(json.dumps(result, indent=2, sort_keys=True))
