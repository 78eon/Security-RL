"""Run the deterministic CybORG Scenario1 adapter smoke trajectory."""

from __future__ import annotations

import json

from rlredteam.cyborg_study import scripted_smoke


def main() -> None:
    print(json.dumps(scripted_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
