"""Verify frozen scientific sources at their historical protocol commit."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any


def frozen_inputs_match(
    repo_root: Path,
    current: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[bool, str]:
    """Check current non-source inputs and historical frozen source bytes.

    Later phases may legitimately modify shared implementation files. Requiring
    HEAD bytes to equal an earlier protocol would make completed evidence fail as
    the project evolves. Frozen source hashes instead belong to protocol_commit.
    """
    non_source = {key: value for key, value in current.items() if key != "source_sha256"}
    if non_source != {key: frozen.get(key) for key in non_source}:
        return False, "current non-source scientific inputs differ from frozen protocol"
    commit = str(frozen.get("protocol_commit", ""))
    sources = frozen.get("source_sha256")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not isinstance(sources, dict):
        return False, "frozen historical source provenance is incomplete"
    for relative_path, expected in sorted(sources.items()):
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            return False, f"invalid frozen source digest: {relative_path}"
        try:
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative_path}"],
                cwd=repo_root,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"cannot read frozen source {relative_path}: {exc}"
        if result.returncode:
            return False, f"frozen source is absent at protocol commit: {relative_path}"
        if hashlib.sha256(result.stdout).hexdigest() != expected:
            return False, f"frozen source hash mismatch at protocol commit: {relative_path}"
    return True, ""
