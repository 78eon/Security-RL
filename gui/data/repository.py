"""All SQL lives here. No Qt import, so this is testable headlessly.

Every method either returns data or raises :class:`RepositoryError`. Nothing
returns a partial result silently: a view that cannot tell "no runs" from
"database unreachable" would show an empty table during an outage, which is the
one thing the Runs view must never do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import psycopg

from gui.data.models import EpisodeRow, RunSummary, StepRow
from rlredteam.frameworks import map_simulator_behavior


class RepositoryError(RuntimeError):
    """Any failure reaching or reading the database, with a usable message."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class ConnectionSettings:
    """CP-01: no default credentials. Missing secrets are reported, not guessed."""

    host: str = "localhost"
    port: int = 5433
    user: str = ""
    password: str = ""
    dbname: str = ""

    @classmethod
    def from_env(cls) -> ConnectionSettings:
        return cls(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5433")),
            user=os.environ.get("POSTGRES_USER", ""),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            dbname=os.environ.get("POSTGRES_DB", ""),
        )

    @property
    def complete(self) -> bool:
        return bool(self.user and self.password and self.dbname)

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={self.dbname}"
        )

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


class Repository:
    def __init__(self, settings: ConnectionSettings | None = None) -> None:
        self.settings = settings or ConnectionSettings.from_env()

    def _connect(self):
        if not self.settings.complete:
            raise RepositoryError(
                "Database credentials not supplied",
                "POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB must be set. "
                "No default is provided.",
            )
        try:
            return psycopg.connect(self.settings.conninfo, connect_timeout=4)
        except psycopg.OperationalError as exc:
            raise RepositoryError(
                f"Cannot reach Postgres on {self.settings.host}:{self.settings.port}",
                str(exc).strip(),
            ) from exc

    def ping(self) -> bool:
        with self._connect() as conn:
            conn.execute("SELECT 1")
        return True

    # -- runs ------------------------------------------------------------

    def list_runs(self) -> list[RunSummary]:
        sql = """
            SELECT e.id, e.name, e.reward_mode, e.seed_set, e.created_at,
                   e.topology_config_hash, e.cve_manifest_sha256,
                   e.config_hash, e.git_sha,
                   count(ep.id)                          AS episode_count,
                   avg(ep.native_reward)                 AS mean_native_reward,
                   avg(ep.goal_reached::int)             AS success_rate,
                   avg(ep.length)                        AS mean_length,
                   EXISTS (SELECT 1 FROM steps s
                           JOIN episodes e2 ON e2.id = s.episode_id
                           WHERE e2.experiment_id = e.id) AS has_steps
            FROM experiments e
            LEFT JOIN episodes ep ON ep.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.id DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            RunSummary(
                experiment_id=r[0], name=r[1], reward_mode=r[2],
                seeds=list(r[3] or []), created_at=r[4],
                topology_config_hash=r[5] or "", cve_manifest_sha256=r[6] or "",
                config_hash=r[7] or "", git_sha=r[8],
                episode_count=r[9],
                mean_native_reward=float(r[10]) if r[10] is not None else None,
                success_rate=float(r[11]) if r[11] is not None else None,
                mean_length=float(r[12]) if r[12] is not None else None,
                has_steps=bool(r[13]),
            )
            for r in rows
        ]

    # -- episodes --------------------------------------------------------

    def episodes(self, experiment_ids: list[int]) -> list[EpisodeRow]:
        if not experiment_ids:
            return []
        sql = """
            SELECT ep.experiment_id, e.name, e.reward_mode, ep.seed,
                   ep.topology_seed, ep.episode_idx, ep.total_reward,
                   ep.native_reward, ep.length, ep.terminal_state,
                   ep.goal_reached, ep.exploited_hosts,
                   ep.mean_cvss_exploited, ep.id
            FROM episodes ep
            JOIN experiments e ON e.id = ep.experiment_id
            WHERE ep.experiment_id = ANY(%s)
            ORDER BY ep.experiment_id, ep.episode_idx
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (experiment_ids,)).fetchall()
        return [
            EpisodeRow(
                experiment_id=r[0], run_name=r[1], reward_mode=r[2], seed=r[3],
                topology_seed=r[4], episode_idx=r[5],
                total_reward=float(r[6]), native_reward=float(r[7]),
                length=r[8], terminal_state=r[9], goal_reached=r[10],
                exploited_hosts=r[11] or [],
                mean_cvss_exploited=float(r[12]) if r[12] is not None else None,
                episode_id=r[13],
            )
            for r in rows
        ]

    def replayable_episodes(self, experiment_id: int) -> list[EpisodeRow]:
        """Episodes that actually have step rows -- the only ones replay can use."""
        sql = """
            SELECT ep.experiment_id, e.name, e.reward_mode, ep.seed,
                   ep.topology_seed, ep.episode_idx, ep.total_reward,
                   ep.native_reward, ep.length, ep.terminal_state,
                   ep.goal_reached, ep.exploited_hosts,
                   ep.mean_cvss_exploited, ep.id
            FROM episodes ep
            JOIN experiments e ON e.id = ep.experiment_id
            WHERE ep.experiment_id = %s
              AND EXISTS (SELECT 1 FROM steps s WHERE s.episode_id = ep.id)
            ORDER BY ep.episode_idx
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (experiment_id,)).fetchall()
        return [
            EpisodeRow(
                experiment_id=r[0], run_name=r[1], reward_mode=r[2], seed=r[3],
                topology_seed=r[4], episode_idx=r[5],
                total_reward=float(r[6]), native_reward=float(r[7]),
                length=r[8], terminal_state=r[9], goal_reached=r[10],
                exploited_hosts=r[11] or [],
                mean_cvss_exploited=float(r[12]) if r[12] is not None else None,
                episode_id=r[13],
            )
            for r in rows
        ]

    def steps(self, episode_id: int) -> list[StepRow]:
        sql = """
            SELECT step_idx, action_name, action_kind,
                   (to_jsonb(steps)->>'rl_action_index')::integer,
                   COALESCE(to_jsonb(steps)->>'simulator_action', action_name),
                   COALESCE(to_jsonb(steps)->'framework_mappings', '[]'::jsonb),
                   tactic, technique_id, target_subnet, target_host, success,
                   reward, native_reward, cve_id, cvss_base, target_entity,
                   state_changed, prerequisites, outcomes
            FROM steps WHERE episode_id = %s ORDER BY step_idx
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (episode_id,)).fetchall()
        output = []
        for row in rows:
            persisted = list(row[5] or [])
            mappings = persisted or [
                mapping.as_dict() for mapping in map_simulator_behavior(str(row[2]))
            ]
            output.append(
                StepRow(
                    step_idx=row[0],
                    action_name=row[1],
                    action_kind=row[2],
                    tactic=row[6],
                    technique_id=row[7],
                    target_subnet=row[8],
                    target_host=row[9],
                    success=row[10],
                    reward=float(row[11]),
                    native_reward=float(row[12]),
                    cve_id=row[13],
                    cvss_base=float(row[14]) if row[14] is not None else None,
                    target_entity=row[15],
                    state_changed=bool(row[16]),
                    prerequisites=list(row[17] or []),
                    outcomes=list(row[18] or []),
                    rl_action_index=row[3],
                    simulator_action=row[4],
                    framework_mappings=mappings,
                    mapping_persisted=bool(persisted),
                )
            )
        return output

    # -- derived attack-path reports -------------------------------------

    def attack_path_reports(self, limit: int = 50) -> list[dict]:
        """Return stored Phase 14 payloads; never derive facts in the GUI."""
        if not 1 <= int(limit) <= 500:
            raise ValueError("report limit must be between 1 and 500")
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT to_regclass('public.attack_path_reports')"
            ).fetchone()[0]
            if exists is None:
                return []
            rows = conn.execute(
                """
                SELECT report_data FROM attack_path_reports
                ORDER BY created_at DESC, report_id LIMIT %s
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row[0]) for row in rows]

    def mitigation_counterfactual_reports(self, limit: int = 50) -> list[dict]:
        """Return stored Phase 15 paired reports without rerunning evaluation."""
        if not 1 <= int(limit) <= 500:
            raise ValueError("report limit must be between 1 and 500")
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT to_regclass('public.mitigation_counterfactual_reports')"
            ).fetchone()[0]
            if exists is None:
                return []
            rows = conn.execute(
                """
                SELECT report_data FROM mitigation_counterfactual_reports
                ORDER BY created_at DESC, report_id LIMIT %s
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row[0]) for row in rows]
