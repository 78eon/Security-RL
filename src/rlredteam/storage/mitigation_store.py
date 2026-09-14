"""Immutable PostgreSQL persistence for Phase 15 paired reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from rlredteam.mitigation import validate_report_identity


class MitigationPersistenceError(RuntimeError):
    """A report ID collision or persistence integrity failure."""


class MitigationStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def save(self, report: Mapping[str, Any]) -> str:
        validate_report_identity(report)
        report_id = str(report["report_id"])
        existing = self.connection.execute(
            "SELECT report_data FROM mitigation_counterfactual_reports WHERE report_id = %s",
            (report_id,),
        ).fetchone()
        if existing is not None:
            if _canonical(existing[0]) != _canonical(report):
                raise MitigationPersistenceError(f"immutable report ID collision: {report_id}")
            return report_id
        intervention = report["intervention"]
        provenance = report["provenance"]
        promotion = report.get("phase14_promotion") or {}
        try:
            self.connection.execute(
                """
                INSERT INTO mitigation_counterfactual_reports (
                    report_id, selected_cve, intervention_version,
                    intervention_sha256, checkpoint_sha256,
                    source_attack_report_id, evaluation_seeds,
                    report_schema_version, report_data
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    report_id,
                    intervention["selected_cve"],
                    intervention["intervention_version"],
                    intervention["intervention_sha256"],
                    provenance["checkpoint_sha256"],
                    promotion.get("source_report_id"),
                    list(provenance["evaluation_seeds_original"]),
                    report["schema_version"],
                    Jsonb(dict(report)),
                ),
            )
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO mitigation_counterfactual_pairs (
                        report_id, evaluation_seed, original_data,
                        mitigated_data, delta_data
                    ) VALUES (%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            report_id,
                            pair["evaluation_seed"],
                            Jsonb(pair["original"]),
                            Jsonb(pair["mitigated"]),
                            Jsonb(pair["delta"]),
                        )
                        for pair in report["pairs"]
                    ],
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return report_id

    def load(self, report_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT report_data FROM mitigation_counterfactual_reports WHERE report_id = %s",
            (report_id,),
        ).fetchone()
        return None if row is None else dict(row[0])

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= int(limit) <= 500:
            raise ValueError("report limit must be between 1 and 500")
        rows = self.connection.execute(
            """
            SELECT report_data FROM mitigation_counterfactual_reports
            ORDER BY created_at DESC, report_id LIMIT %s
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row[0]) for row in rows]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["MitigationPersistenceError", "MitigationStore"]
