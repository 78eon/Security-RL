"""Module 4 -- PostgreSQL episode logger.

Marked `postgres`: needs a live database (`make db-up`). Skipped automatically
when one is not reachable, so the rest of the suite stays runnable offline.
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")

from rlredteam.storage.postgres_logger import (  # noqa: E402
    EpisodeLogger,
    EpisodeRecord,
    StepRecord,
    connection_string,
    ensure_schema,
)

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def conninfo() -> str:
    info = connection_string()
    try:
        with psycopg.connect(info, connect_timeout=3) as conn:
            ensure_schema(conn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no PostgreSQL reachable: {exc}")
    return info


@pytest.fixture
def logger(conninfo: str):
    """A logger against a throwaway experiment, removed afterwards."""
    instance = EpisodeLogger.start(
        name="pytest-synthetic",
        reward_mode="shaped",
        config_hash="cfg-deadbeef",
        topology_config_hash="topo-cafe",
        cve_manifest_sha256="manifest-1234",
        seed_set=[42, 43],
        notes="created by the test suite",
        conninfo=conninfo,
        batch_size=2,
    )
    yield instance
    experiment_id = instance.experiment_id
    try:
        instance.flush()
    finally:
        instance._conn.close()
    # ON DELETE CASCADE removes the episodes and steps with it.
    with psycopg.connect(conninfo) as conn:
        conn.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
        conn.commit()


def make_episode(idx: int, *, steps: int = 3) -> EpisodeRecord:
    return EpisodeRecord(
        seed=42,
        topology_seed=42,
        episode_idx=idx,
        total_reward=123.5 + idx,
        native_reward=98.0 + idx,
        length=steps,
        terminal_state="goal" if idx % 2 == 0 else "step_limit",
        goal_reached=idx % 2 == 0,
        exploited_hosts=[[1, 0], [2, 0]],
        mean_cvss_exploited=8.4,
        steps=[
            StepRecord(
                step_idx=s,
                action_name=f"e_srv_{s}",
                action_kind="exploit",
                tactic="exploit",
                technique_id="T1210",
                target_subnet=1,
                target_host=s,
                success=True,
                reward=2.5,
                native_reward=1.0,
                cve_id="CVE-2021-42013",
                cvss_base=9.8,
                cve_term=2.5,
                tactic_term=0.5,
                crown_jewel_term=100.0 if s == steps - 1 else 0.0,
                penalty_term=0.0,
                access_gained=1,
                newly_discovered=1,
                is_crown_jewel=s == steps - 1,
                reward_paid=True,
            )
            for s in range(steps)
        ],
    )


def test_synthetic_run_writes_all_three_tables(logger: EpisodeLogger, conninfo: str) -> None:
    """The work plan's acceptance test."""
    for idx in range(4):
        logger.log_episode(make_episode(idx))
    logger.flush()

    assert logger.episode_count() == 4
    assert logger.step_count() == 12

    with psycopg.connect(conninfo) as conn:
        row = conn.execute(
            "SELECT name, reward_mode, seed_set, cve_manifest_sha256 "
            "FROM experiments WHERE id = %s",
            (logger.experiment_id,),
        ).fetchone()
    assert row[0] == "pytest-synthetic"
    assert row[1] == "shaped"
    assert row[2] == [42, 43]
    assert row[3] == "manifest-1234"


def test_episode_reads_back_with_values_intact(logger: EpisodeLogger, conninfo: str) -> None:
    logger.log_episode(make_episode(0))
    logger.flush()

    with psycopg.connect(conninfo) as conn:
        row = conn.execute(
            """
            SELECT total_reward, native_reward, length, terminal_state,
                   goal_reached, exploited_hosts, mean_cvss_exploited
            FROM episodes WHERE experiment_id = %s AND episode_idx = 0
            """,
            (logger.experiment_id,),
        ).fetchone()

    assert row[0] == pytest.approx(123.5)
    assert row[1] == pytest.approx(98.0)
    assert row[2] == 3
    assert row[3] == "goal"
    assert row[4] is True
    assert row[5] == [[1, 0], [2, 0]]
    assert row[6] == pytest.approx(8.4)


def test_experiment_reconstructs_run_episode_and_attributed_steps(
    logger: EpisodeLogger, conninfo: str
) -> None:
    logger.log_episode(make_episode(0))
    logger.flush()

    with psycopg.connect(conninfo) as conn:
        rows = conn.execute(
            """
            SELECT e.name, r.designation, r.status, ep.episode_idx,
                   ep.terminal_state, s.step_idx, s.action_name,
                   s.cve_term, s.tactic_term, s.crown_jewel_term,
                   s.access_gained, s.is_crown_jewel
            FROM experiments e
            JOIN runs r ON r.experiment_id = e.id
            JOIN episodes ep ON ep.run_id = r.id
            JOIN steps s ON s.episode_id = ep.id
            WHERE e.id = %s
            ORDER BY ep.episode_idx, s.step_idx
            """,
            (logger.experiment_id,),
        ).fetchall()

    assert len(rows) == 3
    assert rows[0][:7] == (
        "pytest-synthetic",
        "training",
        "running",
        0,
        "goal",
        0,
        "e_srv_0",
    )
    assert rows[0][7:11] == pytest.approx((2.5, 0.5, 0.0, 1))
    assert rows[-1][9] == pytest.approx(100.0)
    assert rows[-1][11] is True


def test_evaluation_run_attaches_to_training_experiment(
    logger: EpisodeLogger, conninfo: str
) -> None:
    logger.log_episode(make_episode(0))
    logger.flush()
    with EpisodeLogger.start(
        name="pytest-synthetic",
        reward_mode="shaped",
        config_hash="cfg-deadbeef",
        topology_config_hash="topo-cafe",
        cve_manifest_sha256="manifest-1234",
        seed_set=[42],
        conninfo=conninfo,
        designation="evaluation",
        evaluation_seeds=[1001],
        checkpoint_path="runs/model.zip",
        experiment_id=logger.experiment_id,
        batch_size=1,
    ) as evaluation:
        record = make_episode(0)
        record.seed = 1001
        evaluation.log_episode(record)

    with psycopg.connect(conninfo) as conn:
        rows = conn.execute(
            """
            SELECT r.designation, r.status, r.evaluation_seeds, ep.seed,
                   ep.terminal_state, count(s.id)
            FROM runs r
            JOIN episodes ep ON ep.run_id = r.id
            JOIN steps s ON s.episode_id = ep.id
            WHERE r.experiment_id = %s
            GROUP BY r.id, ep.id
            ORDER BY r.designation DESC
            """,
            (logger.experiment_id,),
        ).fetchall()

    assert rows[0][:5] == ("training", "running", [], 42, "goal")
    assert rows[1][:5] == ("evaluation", "complete", [1001], 1001, "goal")
    assert rows[1][5] == 3


def test_batching_defers_writes_until_threshold(logger: EpisodeLogger) -> None:
    logger.log_episode(make_episode(0))
    assert logger.episode_count() == 0, "should still be buffered"
    logger.log_episode(make_episode(1))
    assert logger.episode_count() == 2, "batch_size=2 should have flushed"


def test_flush_on_close_loses_nothing(conninfo: str) -> None:
    with EpisodeLogger.start(
        name="pytest-close",
        reward_mode="sparse",
        config_hash="c",
        topology_config_hash="t",
        cve_manifest_sha256="m",
        seed_set=[42],
        conninfo=conninfo,
        batch_size=1000,
    ) as instance:
        experiment_id = instance.experiment_id
        instance.log_episode(make_episode(0))
        assert instance.episode_count() == 0

    with psycopg.connect(conninfo) as conn:
        count, status = conn.execute(
            """
            SELECT count(ep.id), max(r.status)
            FROM runs r LEFT JOIN episodes ep ON ep.run_id = r.id
            WHERE r.experiment_id = %s
            """,
            (experiment_id,),
        ).fetchone()
        conn.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
        conn.commit()
    assert count == 1, "context manager exit must flush"
    assert status == "complete"


def test_duplicate_episode_index_is_rejected(logger: EpisodeLogger) -> None:
    """Guards against double-logging the same episode on a resumed run."""
    logger.log_episode(make_episode(0))
    logger.flush()
    logger.log_episode(make_episode(0))
    with pytest.raises(psycopg.errors.UniqueViolation):
        logger.flush()


def test_steps_cascade_delete_with_experiment(conninfo: str) -> None:
    instance = EpisodeLogger.start(
        name="pytest-cascade",
        reward_mode="shaped",
        config_hash="c",
        topology_config_hash="t",
        cve_manifest_sha256="m",
        seed_set=[42],
        conninfo=conninfo,
        batch_size=1,
    )
    instance.log_episode(make_episode(0))
    experiment_id = instance.experiment_id
    instance._conn.close()

    with psycopg.connect(conninfo) as conn:
        conn.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
        conn.commit()
        orphans = conn.execute(
            """
            SELECT count(*) FROM steps s
            LEFT JOIN episodes e ON e.id = s.episode_id
            WHERE e.id IS NULL
            """
        ).fetchone()[0]
    assert orphans == 0


# -- 2. end-to-end logger integrity from a real training run ---------------


@pytest.mark.integration
@pytest.mark.slow
def test_training_run_writes_consistent_rows(conninfo: str) -> None:
    """A real PPO run must land in all three tables with intact relationships.

    The synthetic tests above prove the logger works; this proves it is
    actually wired into training and that the counts agree with what the run
    reported, rather than silently dropping episodes.
    """
    pytest.importorskip("nasim")
    pytest.importorskip("stable_baselines3")

    from pathlib import Path

    import rlredteam.train as train_module
    from rlredteam.catalogue import CVECatalogue
    from rlredteam.manifest import digest
    from rlredteam.reward import RewardConfig
    from rlredteam.topology import TopologyConfig

    configs = Path(__file__).resolve().parents[1] / "configs"
    args = train_module.parse_args(
        [
            "--seed",
            "42",
            "--timesteps",
            "2048",
            "--reward-config",
            str(configs / "shaped.yaml"),
            "--postgres",
            "--log-steps",
        ]
    )
    report = train_module.train(args)
    reported = report["episodes"]

    with psycopg.connect(conninfo) as conn:
        experiment = conn.execute(
            "SELECT id, config_hash, topology_config_hash, topology_hash, "
            "cve_manifest_sha256 "
            "FROM experiments ORDER BY id DESC LIMIT 1"
        ).fetchone()
        experiment_id = experiment[0]

        episodes = conn.execute(
            "SELECT count(*) FROM episodes WHERE experiment_id = %s", (experiment_id,)
        ).fetchone()[0]
        steps = conn.execute(
            "SELECT count(*) FROM steps s JOIN episodes e ON e.id = s.episode_id "
            "WHERE e.experiment_id = %s",
            (experiment_id,),
        ).fetchone()[0]
        step_sum = conn.execute(
            "SELECT coalesce(sum(length),0) FROM episodes WHERE experiment_id = %s",
            (experiment_id,),
        ).fetchone()[0]
        orphans = conn.execute(
            "SELECT count(*) FROM steps s LEFT JOIN episodes e ON e.id = s.episode_id "
            "WHERE e.id IS NULL"
        ).fetchone()[0]

    try:
        assert episodes == reported, "episodes in DB disagree with the run's own count"
        assert steps == step_sum, "step rows disagree with the episode lengths"
        assert orphans == 0, "steps exist with no parent episode"
        # Test 14, checked at the storage layer as well as the snapshot file.
        assert experiment[1] == RewardConfig.from_yaml(configs / "shaped.yaml").hash()
        assert experiment[2] == TopologyConfig.from_yaml().config_hash()
        assert experiment[3] == "f875a25adf37ee34"
        assert experiment[4] == digest(CVECatalogue.open_default())
    finally:
        with psycopg.connect(conninfo) as conn:
            conn.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
            conn.commit()
