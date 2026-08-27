#!/usr/bin/env python3
"""Compatibility entry point for the controlled Essential experiment.

The former ablation script analysed training episodes. That is not a valid
evaluation, so execution now goes through the config-driven runner that loads
frozen checkpoints and writes dedicated evaluation outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_experiment import main as run_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in arguments:
        arguments[:0] = [
            "--config",
            str(REPO_ROOT / "configs" / "experiments" / "experiment_01.yaml"),
        ]
    return run_experiment(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
