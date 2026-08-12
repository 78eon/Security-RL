"""Module 0 -- the frozen CVE catalogue.

Built once from the raw NVD responses in ``data/provenance/`` into a SQLite file
that is committed and opened read-only at training time. Training never touches
the network: NVD re-scores CVEs over time, so a live lookup would make results
irreproducible across dates (charter §7).

Building straight from the provenance JSON rather than a hand-written YAML means
there is no transcription step, so scores cannot drift from their source.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rlredteam.cvss import Severity, severity_band, validate_base_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "data" / "cve_catalogue.sqlite"
DEFAULT_PROVENANCE = REPO_ROOT / "data" / "provenance"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cves (
    cve_id         TEXT PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('exploit', 'privesc')),
    cvss_version   TEXT NOT NULL,
    base_score     REAL NOT NULL CHECK (base_score >= 0.0 AND base_score <= 10.0),
    base_severity  TEXT NOT NULL,
    vector         TEXT NOT NULL,
    metric_source  TEXT NOT NULL,
    cwe            TEXT,
    published      TEXT,
    last_modified  TEXT,
    source_url     TEXT NOT NULL,
    note           TEXT,
    description    TEXT
);
"""


@dataclass(frozen=True, slots=True)
class CVERecord:
    cve_id: str
    kind: str
    cvss_version: str
    base_score: float
    base_severity: str
    vector: str
    metric_source: str
    cwe: str | None
    published: str | None
    last_modified: str | None
    source_url: str
    note: str | None
    description: str | None

    @property
    def severity(self) -> Severity:
        return Severity(self.base_severity)


class CatalogueError(RuntimeError):
    """Raised when the catalogue is missing, malformed, or lacks an entry."""


_COLUMNS = (
    "cve_id, kind, cvss_version, base_score, base_severity, vector, "
    "metric_source, cwe, published, last_modified, source_url, note, description"
)


class CVECatalogue:
    """Read-only view over the frozen catalogue."""

    def __init__(self, records: dict[str, CVERecord]) -> None:
        if not records:
            raise CatalogueError("catalogue is empty")
        self._records = records

    # -- construction ----------------------------------------------------

    @classmethod
    def open_default(cls) -> CVECatalogue:
        return cls.open(DEFAULT_DB)

    @classmethod
    def open(cls, db_path: Path) -> CVECatalogue:
        db_path = Path(db_path)
        if not db_path.exists():
            raise CatalogueError(
                f"catalogue not found at {db_path}. Build it with: "
                "python -m rlredteam.catalogue build"
            )
        # Read-only URI: training must not be able to mutate the frozen artefact.
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT {_COLUMNS} FROM cves").fetchall()
        return cls({r["cve_id"]: CVERecord(**dict(r)) for r in rows})

    # -- access ----------------------------------------------------------

    def lookup(self, cve_id: str) -> CVERecord:
        """Return the record for ``cve_id``, or raise.

        Raises rather than returning None: a missing CVE must never silently
        degrade a shaped-reward run into an unshaped one.
        """
        try:
            return self._records[cve_id]
        except KeyError:
            raise CatalogueError(f"{cve_id} not in catalogue") from None

    def by_kind(self, kind: str) -> list[CVERecord]:
        """All records of a kind, ordered by CVE id for determinism."""
        return sorted(
            (r for r in self._records.values() if r.kind == kind),
            key=lambda r: r.cve_id,
        )

    def all_records(self) -> list[CVERecord]:
        return sorted(self._records.values(), key=lambda r: r.cve_id)

    def scores(self) -> list[float]:
        return [r.base_score for r in self.all_records()]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, cve_id: object) -> bool:
        return cve_id in self._records


# -- build -----------------------------------------------------------------


def build(
    provenance_dir: Path = DEFAULT_PROVENANCE, db_path: Path = DEFAULT_DB
) -> int:
    """Rebuild the SQLite catalogue from raw NVD JSON. Returns the row count."""
    # Imported here so the training image never needs the online tool's deps.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from fetch_nvd import TARGETS, extract_v31  # noqa: PLC0415

    provenance_dir = Path(provenance_dir)
    files = sorted(provenance_dir.glob("CVE-*.json"))
    if not files:
        raise CatalogueError(
            f"no provenance JSON in {provenance_dir}. "
            "Run: python tools/fetch_nvd.py --fetch"
        )

    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        cve_id = path.stem
        record = extract_v31(payload, cve_id)
        meta = TARGETS.get(cve_id, {})

        score = validate_base_score(record["base_score"])
        # NVD's own severity string must agree with the v3.1 band boundaries.
        # A mismatch means either a bad response or a spec misunderstanding.
        derived = severity_band(score)
        if derived.value != record["base_severity"]:
            raise CatalogueError(
                f"{cve_id}: NVD severity {record['base_severity']} != "
                f"band({score}) = {derived.value}"
            )

        rows.append(
            (
                record["cve_id"],
                meta.get("kind", "exploit"),
                record["cvss_version"],
                score,
                record["base_severity"],
                record["vector"],
                record["metric_source"],
                record["cwe"],
                record["published"],
                record["last_modified"],
                record["source_url"],
                meta.get("note"),
                (record["description"] or "")[:500],
            )
        )

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            f"INSERT INTO cves ({_COLUMNS}) VALUES ({', '.join('?' * 13)})", rows
        )
    return len(rows)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "show"])
    args = parser.parse_args()

    if args.command == "build":
        count = build()
        print(f"built {DEFAULT_DB} with {count} CVEs")
        return 0

    catalogue = CVECatalogue.open_default()
    print(f"{'cve_id':<18} {'kind':<8} {'score':>5}  {'severity':<9} {'cwe':<12} note")
    for record in catalogue.all_records():
        print(
            f"{record.cve_id:<18} {record.kind:<8} {record.base_score:>5}  "
            f"{record.base_severity:<9} {record.cwe or '-':<12} {record.note or ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
