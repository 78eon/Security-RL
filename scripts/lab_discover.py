#!/usr/bin/env python3
"""Plan or execute one authorized, conservative isolated-lab discovery action."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml

from rlredteam.enterprise.live import (
    LabScope,
    LiveKnowledgeGraph,
    NmapLabRunner,
    ScanProfile,
)


def load_scope(path: Path) -> LabScope:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("lab scope must be a YAML mapping")
    return LabScope.from_strings(
        authorization_id=str(raw.get("authorization_id", "")),
        allowed_networks=tuple(raw.get("allowed_networks") or ()),
        max_addresses=int(raw.get("max_addresses", 256)),
        execution_enabled=bool(raw.get("execution_enabled", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Literal authorized private IP/CIDR")
    parser.add_argument("--profile", choices=[item.value for item in ScanProfile], required=True)
    parser.add_argument("--execute", action="store_true", help="Run rather than print the plan")
    parser.add_argument(
        "--authorization",
        default="",
        help="Must exactly match authorization_id when --execute is used",
    )
    args = parser.parse_args()

    scope = load_scope(args.config)
    runner = NmapLabRunner(scope)
    command = runner.plan(args.target, args.profile)
    print(f"authorization_id={scope.authorization_id}")
    print(f"planned_command={shlex.join(command)}")
    if not args.execute:
        print("dry_run=true (no packets sent)")
        return
    if args.authorization != scope.authorization_id:
        raise SystemExit("authorization value does not match the approved lab scope")

    evidence = runner.run(args.target, args.profile, operator_approved=True)
    graph = LiveKnowledgeGraph()
    graph.ingest_nmap(evidence)
    print(json.dumps(graph.to_dict(), indent=2))


if __name__ == "__main__":
    main()
