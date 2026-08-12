"""SHA-256 manifest over the frozen CVE catalogue.

Hashes a CANONICAL DUMP of the rows, not the .sqlite file bytes. SQLite rewrites
page headers, freelists and internal counters between writes, so two builds of
identical data produce different file bytes -- a file-level digest would not be
stable across reloads, which is exactly the property the manifest exists to
guarantee.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rlredteam.catalogue import DEFAULT_DB, CVECatalogue

DEFAULT_MANIFEST = DEFAULT_DB.parent / "cve_catalogue.manifest.json"


def canonical_dump(catalogue: CVECatalogue) -> str:
    """Serialise the catalogue to a byte-stable canonical form.

    Rows sorted by cve_id, keys sorted within each row, no insignificant
    whitespace. Deterministic given the same data, regardless of insertion
    order or SQLite internals.
    """
    rows = [asdict(record) for record in catalogue.all_records()]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(catalogue: CVECatalogue) -> str:
    return hashlib.sha256(canonical_dump(catalogue).encode("utf-8")).hexdigest()


def write_manifest(
    catalogue: CVECatalogue | None = None, path: Path = DEFAULT_MANIFEST
) -> dict:
    catalogue = catalogue or CVECatalogue.open_default()
    manifest = {
        "sha256": digest(catalogue),
        "row_count": len(catalogue),
        "algorithm": "sha256(canonical_json_rows)",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def verify(path: Path = DEFAULT_MANIFEST) -> bool:
    """True when the committed manifest still matches the catalogue."""
    return read_manifest(path)["sha256"] == digest(CVECatalogue.open_default())


if __name__ == "__main__":
    result = write_manifest()
    print(json.dumps(result, indent=2, sort_keys=True))
