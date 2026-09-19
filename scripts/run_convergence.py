"""Explicit convergence operations. No training occurs during verification."""

import argparse
import json
from pathlib import Path

from rlredteam.convergence import BASE_CONFIG
from rlredteam.convergence_runner import (
    DEFAULT_OUTPUT,
    analyze,
    evaluate,
    register,
    run,
    study_lock,
    verify,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["freeze", "run", "analyze", "verify", "sparse-run", "evaluate"]
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    args = parser.parse_args(argv)
    with study_lock(DEFAULT_OUTPUT):
        if args.command == "freeze":
            result = register(args.config, DEFAULT_OUTPUT)
        elif args.command in {"run", "sparse-run"}:
            result = run(args.config, DEFAULT_OUTPUT, sparse=args.command == "sparse-run")
        elif args.command == "analyze":
            result = analyze(args.config, DEFAULT_OUTPUT)
        elif args.command == "evaluate":
            result = evaluate(args.config, DEFAULT_OUTPUT)
        else:
            result = verify(args.config, DEFAULT_OUTPUT)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
