#!/usr/bin/env python3
"""Run the controlled fixed-topology experiment from one configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.experiment import (  # noqa: E402
    ExperimentConfig,
    execute_experiment,
    freeze_inputs,
    validate_frozen,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    if args.freeze:
        print(json.dumps(freeze_inputs(config), indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        frozen = validate_frozen(config)
        print(f"{config.experiment_id}: controlled inputs valid")
        print(f"topology hash: {frozen['topology_hash']}")
        for arm in config.arms:
            for seed in config.training_seeds:
                print(f"  {config.run_name(arm, seed)}")
        return 0
    report = execute_experiment(
        config,
        train=not args.skip_training,
        evaluate=not args.skip_evaluation,
    )
    print(f"complete: {report['complete']}")
    print(f"wrote results/{config.experiment_id}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
