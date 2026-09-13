"""Frozen protocol version registry; protocol semantics are never profile data."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_MANIFEST = REPO_ROOT / "configs/protocol_versions.yaml"


def protocol_versions(path: Path | None = None) -> dict[str, int]:
    selected = path or DEFAULT_PROTOCOL_MANIFEST
    document = yaml.safe_load(selected.read_text())
    raw = document.get("protocols") if isinstance(document, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError("protocol version manifest is missing")
    versions = {str(name): int(version) for name, version in raw.items()}
    if any(not name or version <= 0 for name, version in versions.items()):
        raise ValueError("protocol versions must be named positive integers")
    return versions
