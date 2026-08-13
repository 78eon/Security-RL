#!/usr/bin/env python3
"""CP-31 — packaging gate for trained policy weights.

A trained attack-path-discovery policy is dual-use, so it is private by default
and packaging requires explicit written approval recorded beside the checkpoint.
This script refuses to package anything lacking that approval.

    python scripts/release_weights.py --list
    python scripts/release_weights.py --approve shaped-s42-t42 \\
        --approved-by "supervisor name" --decision release \\
        --justification "agreed at supervision, simulation-only policy"
    python scripts/release_weights.py --package shaped-s42-t42 --out dist/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
APPROVAL_FILE = "release_approval.json"

VALID_DECISIONS = ("release", "withhold", "redact", "access-control")


class ReleaseRefused(RuntimeError):
    """Raised when a checkpoint may not be packaged."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approve(
    run_name: str, approved_by: str, decision: str, justification: str
) -> dict:
    """Record an explicit release decision beside the checkpoint."""
    if decision not in VALID_DECISIONS:
        raise ReleaseRefused(f"decision must be one of {VALID_DECISIONS}")
    if not approved_by.strip():
        raise ReleaseRefused("an approver must be named")
    if len(justification.strip()) < 10:
        raise ReleaseRefused("a substantive justification is required")

    run_dir = RUNS_DIR / run_name
    checkpoint = run_dir / "model.zip"
    if not checkpoint.exists():
        raise ReleaseRefused(f"no checkpoint at {checkpoint}")

    approval = {
        "run_name": run_name,
        "decision": decision,
        "approved_by": approved_by.strip(),
        "justification": justification.strip(),
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Binds the approval to this exact file: re-training invalidates it.
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    (run_dir / APPROVAL_FILE).write_text(json.dumps(approval, indent=2))
    return approval


def check(run_name: str) -> tuple[bool, list[str]]:
    """Whether this checkpoint may be packaged, and why not."""
    run_dir = RUNS_DIR / run_name
    problems: list[str] = []

    checkpoint = run_dir / "model.zip"
    if not checkpoint.exists():
        return False, [f"no checkpoint at {checkpoint.relative_to(REPO_ROOT)}"]

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        problems.append("no manifest.json — the checkpoint cannot be traced to a run")

    approval_path = run_dir / APPROVAL_FILE
    if not approval_path.exists():
        problems.append(
            "no release approval recorded — weights are private by default; "
            "run with --approve once a decision has been agreed"
        )
        return False, problems

    approval = json.loads(approval_path.read_text())
    if approval.get("decision") != "release":
        problems.append(
            f"decision is '{approval.get('decision')}', not 'release'"
        )
    if approval.get("checkpoint_sha256") != sha256_file(checkpoint):
        problems.append(
            "checkpoint has changed since approval — the approval covered a "
            "different file and no longer applies"
        )
    if not approval.get("approved_by"):
        problems.append("approval names no approver")
    return not problems, problems


def package(run_name: str, out_dir: Path) -> Path:
    ok, problems = check(run_name)
    if not ok:
        raise ReleaseRefused(
            f"refusing to package {run_name}:\n  " + "\n  ".join(problems)
        )

    run_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{run_name}-weights.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.zip", "manifest.json", APPROVAL_FILE, "summary.json"):
            path = run_dir / name
            if path.exists():
                archive.write(path, arcname=name)
    return bundle


def list_runs() -> list[tuple[str, bool, str]]:
    out = []
    if not RUNS_DIR.exists():
        return out
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not (run_dir / "model.zip").exists():
            continue
        ok, problems = check(run_dir.name)
        out.append((run_dir.name, ok, problems[0] if problems else "approved"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", metavar="RUN")
    parser.add_argument("--approve", metavar="RUN")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--decision", default="release", choices=VALID_DECISIONS)
    parser.add_argument("--justification", default="")
    parser.add_argument("--package", metavar="RUN")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args()

    try:
        if args.list:
            rows = list_runs()
            if not rows:
                print("no checkpoints found")
                return 0
            print(f"{'run':<28}{'releasable':<12}reason")
            for name, ok, reason in rows:
                print(f"{name:<28}{'yes' if ok else 'NO':<12}{reason}")
            return 0

        if args.check:
            ok, problems = check(args.check)
            print(f"{args.check}: {'releasable' if ok else 'BLOCKED'}")
            for problem in problems:
                print(f"  - {problem}")
            return 0 if ok else 1

        if args.approve:
            approval = approve(
                args.approve, args.approved_by, args.decision, args.justification
            )
            print(json.dumps(approval, indent=2))
            return 0

        if args.package:
            bundle = package(args.package, args.out)
            try:
                shown = bundle.relative_to(REPO_ROOT)
            except ValueError:
                shown = bundle  # an output path outside the repo is fine
            print(f"packaged {shown}")
            return 0
    except ReleaseRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
