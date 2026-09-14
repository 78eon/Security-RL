#!/usr/bin/env python3
"""Run a paired Phase 15 frozen-policy CVE mitigation evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.mitigation import (  # noqa: E402
    evaluate_checkpoint_mitigation,
    promote_phase14_cve,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cve")
    parser.add_argument("--phase14-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", default=[])
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args(argv)
    if not args.cve and not args.phase14_report:
        parser.error("supply --cve or --phase14-report")
    selected = args.cve
    if selected is None:
        selected = promote_phase14_cve(args.phase14_report)["selected_cve"]
    report = evaluate_checkpoint_mitigation(
        args.run,
        args.seeds,
        selected,
        args.out,
        phase14_report=args.phase14_report,
        postgres=args.postgres,
        protected_paths=tuple(args.protect),
    )
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "selected_cve": report["intervention"]["selected_cve"],
                "paired_seeds": len(report["pairs"]),
                "original_success_rate": report["analysis"]["original_success_rate"],
                "mitigated_success_rate": report["analysis"]["mitigated_success_rate"],
                "output": str(args.out),
                "postgres": args.postgres,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
