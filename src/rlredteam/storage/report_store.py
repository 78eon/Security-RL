"""PostgreSQL persistence for immutable Phase 14 derived reports."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from rlredteam.attack_path_report import AttackPathReport


class ReportPersistenceError(RuntimeError):
    """A report ID collision or persistence integrity failure."""


class ReportStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def save(self, report: AttackPathReport) -> str:
        payload = report.to_dict()
        provenance = report.provenance
        existing = self.connection.execute(
            "SELECT report_data FROM attack_path_reports WHERE report_id = %s",
            (report.report_id,),
        ).fetchone()
        if existing is not None:
            if _canonical(existing[0]) != _canonical(payload):
                raise ReportPersistenceError(
                    f"immutable report ID collision: {report.report_id}"
                )
            return report.report_id
        try:
            self.connection.execute(
                """
                INSERT INTO attack_path_reports (
                    report_id, report_mode, experiment_key, run_keys,
                    source_trajectory_sha256, facts_payload_sha256,
                    mitre_catalogue_sha256, mitre_catalogue_version,
                    source_checkpoint_hashes, report_schema_version, report_data
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    report.report_id,
                    report.report_mode,
                    provenance.experiment_id,
                    list(provenance.run_ids),
                    provenance.source_trajectory_sha256,
                    provenance.facts_payload_sha256,
                    provenance.mitre_catalogue_sha256,
                    provenance.mitre_catalogue_version,
                    Jsonb(provenance.source_checkpoint_hashes),
                    provenance.report_schema_version,
                    Jsonb(payload),
                ),
            )
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO attack_path_report_facts (
                        report_id, fact_index, episode_key, fact_data
                    ) VALUES (%s,%s,%s,%s)
                    """,
                    [
                        (report.report_id, index, fact.episode_id, Jsonb(fact.to_dict()))
                        for index, fact in enumerate(report.facts)
                    ],
                )
                cursor.executemany(
                    """
                    INSERT INTO attack_path_report_criticalities (
                        report_id, cve_id, criticality_data
                    ) VALUES (%s,%s,%s)
                    """,
                    [
                        (report.report_id, item.cve_id, Jsonb(_asdict(item)))
                        for item in report.observed_path_criticality
                    ],
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return report.report_id

    def load(self, report_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT report_data FROM attack_path_reports WHERE report_id = %s",
            (report_id,),
        ).fetchone()
        return None if row is None else dict(row[0])

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= int(limit) <= 500:
            raise ValueError("report limit must be between 1 and 500")
        rows = self.connection.execute(
            """
            SELECT report_data FROM attack_path_reports
            ORDER BY created_at DESC, report_id LIMIT %s
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row[0]) for row in rows]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _asdict(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)
