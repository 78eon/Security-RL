"""Reads the runs/ folder: config snapshots, live episode CSVs, CVE catalogue.

Separate from repository.py because these survive a database outage. When
Postgres is unreachable the app stays usable on folder artefacts alone, which is
the behaviour the connection-failure state promises.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from gui.data.models import TopologyView

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
CATALOGUE_DB = REPO_ROOT / "data" / "cve_catalogue.sqlite"


@dataclass(frozen=True, slots=True)
class CveDetail:
    cve_id: str
    base_score: float
    base_severity: str
    vector: str
    cwe: str | None
    note: str | None
    description: str | None


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def load_snapshot(name: str) -> dict:
    """config.snapshot.json for a run, or {} when the folder is absent."""
    path = run_dir(name) / "config.snapshot.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_topology(name: str) -> TopologyView:
    """Network structure for the replay canvas.

    Runs created before topology persistence carry only counts. The view then
    has no adjacency and draws a grouped layout without edges -- degraded, not
    broken.
    """
    topo = load_snapshot(name).get("topology", {})
    if not topo:
        return TopologyView()

    sensitive = topo.get("sensitive_hosts", {}) or {}
    # Prefer the persisted host list; it is unambiguous. Fall back to deriving
    # from subnet sizes only for older snapshots, remembering that subnets[0]
    # is the internet subnet and carries no real hosts.
    hosts: list[tuple[int, int]] = [
        (int(a), int(b)) for a, b in (topo.get("hosts") or [])
    ]
    subnets = topo.get("subnets") or []
    if not hosts and subnets:
        for subnet_idx, size in enumerate(subnets[1:], start=1):
            hosts.extend((subnet_idx, h) for h in range(size))

    return TopologyView(
        num_hosts=topo.get("num_hosts", 0),
        num_subnets=topo.get("num_subnets", 0),
        subnets=list(subnets),
        adjacency=topo.get("topology") or [],
        sensitive_hosts=sensitive,
        hosts=hosts,
    )


def load_summary(name: str) -> dict:
    path = run_dir(name) / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_episode_csv(name: str) -> list[dict]:
    """Live episode rows for a run in progress.

    train.py flushes this after every episode, so polling it is how the Train
    view shows a curve growing without touching the database.
    """
    path = run_dir(name) / "episodes.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def list_run_folders() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())


def load_cve(cve_id: str) -> CveDetail | None:
    """Look a CVE up in the frozen catalogue, read-only."""
    if not cve_id or not CATALOGUE_DB.exists():
        return None
    try:
        with sqlite3.connect(f"file:{CATALOGUE_DB}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT cve_id, base_score, base_severity, vector, cwe, note, "
                "description FROM cves WHERE cve_id = ?",
                (cve_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return CveDetail(*row)
