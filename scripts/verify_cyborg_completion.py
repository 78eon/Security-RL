"""Fail closed unless all CybORG integration evidence is current."""

from __future__ import annotations

import json

from rlredteam.cyborg_study import verify_completion


def main() -> None:
    print(json.dumps(verify_completion(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
