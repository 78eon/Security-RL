"""Module 4 -- persist episodes and steps to PostgreSQL.

Exposes ``EpisodeLogger.log_episode()``, a hook callable from any training or
evaluation loop. Inserts are batched: at one row per step and ~1000 steps per
episode, per-row inserts would dominate wall clock on a CPU-only budget.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import psycopg
from psycopg.types.json import Jsonb

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class MissingCredentials(RuntimeError):
    """Raised when a required database secret is not supplied."""


def connection_string() -> str:
    """Build a libpq connection string from the environment.

    CP-01: there is no fallback password. A default credential is worse than no
    credential -- it is published in the repository, works everywhere, and gets
    silently relied upon. Missing secrets stop startup instead.
    """
    required = {
        "POSTGRES_USER": os.environ.get("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD"),
        "POSTGRES_DB": os.environ.get("POSTGRES_DB"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise MissingCredentials(
            "Database credentials not supplied: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set them; no default is provided."
        )
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5433')} "
        f"user={required['POSTGRES_USER']} "
        f"password={required['POSTGRES_PASSWORD']} "
        f"dbname={required['POSTGRES_DB']}"
    )


def git_sha() -> str | None:
    """Current commit, recorded so a result can be traced to the code."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class StepRecord:
    step_idx: int
    action_name: str
    action_kind: str
    success: bool
    reward: float
    native_reward: float
    tactic: str | None = None
    technique_id: str | None = None
    target_subnet: int | None = None
    target_host: int | None = None
    cve_id: str | None = None
    cvss_base: float | None = None
    cve_term: float = 0.0
    tactic_term: float = 0.0
    crown_jewel_term: float = 0.0
    penalty_term: float = 0.0
    access_gained: int = 0
    newly_discovered: int = 0
    is_crown_jewel: bool = False
    reward_paid: bool = False
    error: str | None = None


@dataclass(slots=True)
class EpisodeRecord:
    seed: int
    topology_seed: int
    episode_idx: int
    total_reward: float
    native_reward: float
    length: int
    terminal_state: str
    goal_reached: bool
    exploited_hosts: list = field(default_factory=list)
    mean_cvss_exploited: float | None = None
    max_cvss_exploited: float | None = None
    hosts_compromised: int = 0
    steps: list[StepRecord] = field(default_factory=list)


class EpisodeLogger:
    """Batched writer for one experiment.

    Usage::

        with EpisodeLogger.start(
            name="shaped-seed42", reward_mode="shaped",
            config_hash=..., topology_config_hash=...,
            cve_manifest_sha256=..., seed_set=[42],
        ) as logger:
            logger.log_episode(record)
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        experiment_id: int,
        run_id: int,
        batch_size: int = 20,
        log_steps: bool = True,
    ) -> None:
        self._conn = conn
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.batch_size = batch_size
        self.log_steps = log_steps
        self._pending: list[EpisodeRecord] = []

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        name: str,
        reward_mode: str,
        config_hash: str,
        topology_config_hash: str,
        cve_manifest_sha256: str,
        seed_set: list[int],
        notes: str | None = None,
        conninfo: str | None = None,
        batch_size: int = 20,
        log_steps: bool = True,
        condition: str | None = None,
        algorithm: str = "PPO",
        topology_id: str | None = None,
        topology_hash: str | None = None,
        hyperparameters: dict | None = None,
        designation: str = "training",
        evaluation_seeds: list[int] | None = None,
        checkpoint_path: str | None = None,
        experiment_id: int | None = None,
    ) -> EpisodeLogger:
        conn = psycopg.connect(conninfo or connection_string())
        ensure_schema(conn)
        with conn.cursor() as cur:
            if experiment_id is None:
                cur.execute(
                    """
                    INSERT INTO experiments (
                        name, condition, algorithm, reward_mode, config_hash,
                        topology_config_hash, topology_id, topology_hash,
                        cve_manifest_sha256, hyperparameters, git_sha, seed_set, notes
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (
                        name,
                        condition or reward_mode,
                        algorithm,
                        reward_mode,
                        config_hash,
                        topology_config_hash,
                        topology_id,
                        topology_hash,
                        cve_manifest_sha256,
                        Jsonb(hyperparameters or {}),
                        git_sha(),
                        seed_set,
                        notes,
                    ),
                )
                experiment_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO runs (
                    experiment_id, seed, designation, status, evaluation_seeds,
                    checkpoint_path
                ) VALUES (%s,%s,%s,'running',%s,%s)
                RETURNING id
                """,
                (
                    experiment_id,
                    seed_set[0],
                    designation,
                    evaluation_seeds or [],
                    checkpoint_path,
                ),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
        return cls(
            conn,
            experiment_id,
            run_id,
            batch_size=batch_size,
            log_steps=log_steps,
        )

    def __enter__(self) -> EpisodeLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Flush even on error: a crashed run's completed episodes are still data.
        status = "complete" if exc_type is None else "failed"
        try:
            self.flush()
            self.finish(status)
        finally:
            self._conn.close()

    def finish(self, status: str = "complete") -> None:
        if status not in {"complete", "failed"}:
            raise ValueError("run status must be complete or failed")
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, ended_at = now() WHERE id = %s",
                (status, self.run_id),
            )
        self._conn.commit()

    # -- writing ---------------------------------------------------------

    def log_episode(self, record: EpisodeRecord) -> None:
        """Queue one episode. Flushes automatically every ``batch_size``."""
        self._pending.append(record)
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Write queued episodes. Returns how many were written."""
        if not self._pending:
            return 0

        written = len(self._pending)
        try:
            self._write_batch()
        except Exception:
            # Roll back so the connection stays usable. Without this an aborted
            # transaction poisons every later statement with InFailedSqlTransaction,
            # turning one bad episode into a silently dead logger for the rest
            # of the run.
            self._conn.rollback()
            self._pending.clear()
            raise
        self._conn.commit()
        self._pending.clear()
        return written

    def _write_batch(self) -> None:
        with self._conn.cursor() as cur:
            for record in self._pending:
                cur.execute(
                    """
                    INSERT INTO episodes (
                        experiment_id, run_id, seed, topology_seed, episode_idx,
                        total_reward, native_reward, length, terminal_state,
                        goal_reached, exploited_hosts, mean_cvss_exploited,
                        max_cvss_exploited, hosts_compromised
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (
                        self.experiment_id,
                        self.run_id,
                        record.seed,
                        record.topology_seed,
                        record.episode_idx,
                        record.total_reward,
                        record.native_reward,
                        record.length,
                        record.terminal_state,
                        record.goal_reached,
                        Jsonb(record.exploited_hosts),
                        record.mean_cvss_exploited,
                        record.max_cvss_exploited,
                        record.hosts_compromised,
                    ),
                )
                episode_id = cur.fetchone()[0]

                if self.log_steps and record.steps:
                    cur.executemany(
                        """
                        INSERT INTO steps (
                            episode_id, step_idx, action_name, action_kind,
                            tactic, technique_id, target_subnet, target_host,
                            success, reward, native_reward, cve_id, cvss_base
                            , cve_term, tactic_term, crown_jewel_term, penalty_term,
                            access_gained, newly_discovered, is_crown_jewel,
                            reward_paid, error
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        [
                            (
                                episode_id,
                                s.step_idx,
                                s.action_name,
                                s.action_kind,
                                s.tactic,
                                s.technique_id,
                                s.target_subnet,
                                s.target_host,
                                s.success,
                                s.reward,
                                s.native_reward,
                                s.cve_id,
                                s.cvss_base,
                                s.cve_term,
                                s.tactic_term,
                                s.crown_jewel_term,
                                s.penalty_term,
                                s.access_gained,
                                s.newly_discovered,
                                s.is_crown_jewel,
                                s.reward_paid,
                                s.error,
                            )
                            for s in record.steps
                        ],
                    )

    # -- reading back ----------------------------------------------------

    def episode_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM episodes WHERE experiment_id = %s",
                (self.experiment_id,),
            )
            return cur.fetchone()[0]

    def step_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM steps s
                JOIN episodes e ON e.id = s.episode_id
                WHERE e.experiment_id = %s
                """,
                (self.experiment_id,),
            )
            return cur.fetchone()[0]


def ensure_schema(conn: psycopg.Connection) -> None:
    """Apply schema.sql. Idempotent -- every statement is IF NOT EXISTS."""
    conn.execute(SCHEMA_PATH.read_text())
    conn.commit()


def summarise(conninfo: str | None = None) -> str:
    """Human-readable dump of what has been logged. Used by `make db-summary`."""
    with psycopg.connect(conninfo or connection_string()) as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.name, e.reward_mode, count(ep.id) AS episodes,
                   avg(ep.native_reward) AS mean_native,
                   avg(ep.goal_reached::int) AS success_rate
            FROM experiments e
            LEFT JOIN episodes ep ON ep.experiment_id = e.id
            GROUP BY e.id ORDER BY e.id
            """
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0],
                "name": r[1],
                "reward_mode": r[2],
                "episodes": r[3],
                "mean_native_reward": float(r[4]) if r[4] is not None else None,
                "success_rate": float(r[5]) if r[5] is not None else None,
            }
            for r in rows
        ],
        indent=2,
    )


if __name__ == "__main__":
    print(summarise())
