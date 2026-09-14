#!/usr/bin/env python3
"""Verify the complete CyberBattle adapter evidence chain."""

import json

from rlredteam.cyberbattle_study import verify_completion

if __name__ == "__main__":
    print(json.dumps(verify_completion(), indent=2, sort_keys=True))
