#!/usr/bin/env python3
"""CP-00 — record the baseline the remediation is measured against.

Captures what the system is right now: commit, dependency versions, image
digest, database schema, and the current test result. Without this, "we fixed
it" has nothing to be relative to.

    python scripts/baseline_manifest.py --out runs/_baseline/baseline.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam import provenance as prov  # noqa: E402


def dependency_versions() -> dict:
    """Installed versions of the packages results actually depend on."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in (
        "nasim", "gymnasium", "stable-baselines3", "torch", "numpy",
        "psycopg", "scipy", "pyyaml",
    ):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def schema_version() -> dict:
    """Tables and column counts, as a structural fingerprint of the schema."""
    schema = REPO_ROOT / "src" / "rlredteam" / "storage" / "schema.sql"
    if not schema.exists():
        return {}
    text = schema.read_text()
    tables = {}
    for block in text.split("CREATE TABLE IF NOT EXISTS ")[1:]:
        name = block.split("(")[0].strip()
        body = block.split("(", 1)[1].split(";")[0]
        columns = [
            line.strip().split()[0]
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith(("--", ")", "UNIQUE", "CHECK"))
        ]
        tables[name] = len(columns)
    return {
        "tables": tables,
        "sha256": prov._digest(text),
    }


def run_tests() -> dict:
    """Current test outcome. A baseline that omits it records nothing useful."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-m", "not slow", "--ignore=tests/test_gui_data.py"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False, timeout=900,
    )
    tail = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    return {
        "exit_code": result.returncode,
        "summary": tail[-1] if tail else "",
        "passed": result.returncode == 0,
    }


def build(skip_tests: bool = False) -> dict:
    manifest = {
        "recorded_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   __import__("time").gmtime()),
        "git_commit": prov.git_commit(),
        "git_dirty": prov.git_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock_hash": prov.dependency_lock_hash(),
        "dependency_versions": dependency_versions(),
        "docker_image_digest": prov.docker_image_digest(),
        "database_schema": schema_version(),
    }
    manifest["tests"] = {"skipped": True} if skip_tests else run_tests()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs" / "_baseline" / "baseline.json"
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    manifest = build(skip_tests=args.skip_tests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

    print(f"baseline written to {args.out}")
    print(f"  commit   {manifest['git_commit']}")
    print(f"  python   {manifest['python_version']}")
    print(f"  deps     {sum(1 for v in manifest['dependency_versions'].values() if v)} pinned")
    print(f"  schema   {len(manifest['database_schema'].get('tables', {}))} tables")
    print(f"  tests    {manifest['tests'].get('summary', 'skipped')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
