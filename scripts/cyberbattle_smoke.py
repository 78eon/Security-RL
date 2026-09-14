#!/usr/bin/env python3
"""Generate a successful native CyberBattle trajectory and Phase 14 report."""

import json

from rlredteam.cyberbattle_study import scripted_smoke

if __name__ == "__main__":
    print(json.dumps(scripted_smoke(), indent=2, sort_keys=True))
