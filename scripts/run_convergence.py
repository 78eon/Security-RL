"""Explicit convergence operations. No training occurs during verification."""

import argparse
import json
from pathlib import Path

from rlredteam.convergence import BASE_CONFIG, load_config
from rlredteam.convergence_runner import (
    CONFIRMATORY_ID,
    DEFAULT_OUTPUT,
    analyze,
    evaluate,
    register,
    run,
    run_seed,
    study_lock,
    verify,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "freeze", "run", "run-seed", "analyze", "verify",
            "sparse-run", "sparse-run-seed", "evaluate",
        ],
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if load_config(args.config)["id"] == CONFIRMATORY_ID:
        expected = (BASE_CONFIG.parents[2] / "results/confirmatory_discovery_v3").resolve()
        if root != expected:
            parser.error("corrected study requires its separate confirmatory output root")
    if args.command in {"run-seed", "sparse-run-seed"} and args.seed is None:
        parser.error("--seed is required for a single-seed run")
    if args.seed is not None and args.command not in {"run-seed", "sparse-run-seed"}:
        parser.error("--seed is only valid with run-seed or sparse-run-seed")
    with study_lock(root):
        if args.command == "freeze":
            result = register(args.config, root)
        elif args.command in {"run", "sparse-run"}:
            result = run(args.config, root, sparse=args.command == "sparse-run")
        elif args.command in {"run-seed", "sparse-run-seed"}:
            result = run_seed(
                args.config, root, args.seed, sparse=args.command == "sparse-run-seed"
            )
        elif args.command == "analyze":
            result = analyze(args.config, root)
        elif args.command == "evaluate":
            result = evaluate(args.config, root)
        else:
            result = verify(args.config, root)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
