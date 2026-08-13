"""GUI data layer — headless.

No QApplication and no Qt import: gui.data deliberately depends on neither, so
the risky part (SQL, comparability logic, snapshot parsing) is testable without
a display.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from gui.data.models import Comparison, RunSummary, StepRow, TopologyView


def make_run(**overrides) -> RunSummary:
    base = dict(
        experiment_id=1, name="shaped-s42-t42", reward_mode="shaped", seeds=[42],
        episode_count=100, created_at=datetime(2026, 8, 12, 15, 28),
        topology_config_hash="2aa12404d40a23d0",
        cve_manifest_sha256="c14575707519311e",
        config_hash="8aa4aee9074f423d", git_sha="508ac4f",
        mean_native_reward=-747.0, success_rate=0.82, mean_length=63.4,
        has_steps=True,
    )
    base.update(overrides)
    return RunSummary(**base)


# -- comparability: the app's central integrity guard -----------------------


def test_matching_provenance_is_comparable() -> None:
    left = make_run(name="sparse-s42-t42", reward_mode="sparse")
    right = make_run(name="shaped-s42-t42", reward_mode="shaped")
    assert Comparison(left, right).comparable
    assert Comparison(left, right).reasons() == []


def test_different_topology_is_not_comparable() -> None:
    left = make_run()
    right = make_run(topology_config_hash="deadbeefdeadbeef")
    comparison = Comparison(left, right)
    assert not comparison.comparable
    assert "different topology configuration" in comparison.reasons()


def test_different_catalogue_is_not_comparable() -> None:
    comparison = Comparison(make_run(), make_run(cve_manifest_sha256="0000"))
    assert not comparison.comparable
    assert "different CVE catalogue" in comparison.reasons()


def test_same_reward_on_both_sides_is_flagged() -> None:
    """Provenance matches, but comparing a condition against itself is useless."""
    comparison = Comparison(make_run(), make_run(name="shaped-s43-t42"))
    assert comparison.comparable
    assert any("same reward" in r for r in comparison.reasons())


def test_both_failures_reported_together() -> None:
    comparison = Comparison(
        make_run(),
        make_run(topology_config_hash="x", cve_manifest_sha256="y"),
    )
    assert len(comparison.reasons()) == 2


# -- run summary ------------------------------------------------------------


def test_run_with_no_episodes_reports_empty() -> None:
    assert make_run(episode_count=0).status == "empty"
    assert make_run(episode_count=1).status == "complete"


@pytest.mark.parametrize(
    ("seeds", "expected"), [([], "—"), ([42], "42"), ([42, 43, 51], "42–51")]
)
def test_seed_label(seeds: list[int], expected: str) -> None:
    assert make_run(seeds=seeds).seed_label == expected


# -- severity banding -------------------------------------------------------


def make_step(score: float | None) -> StepRow:
    return StepRow(
        step_idx=0, action_name="e", action_kind="exploit", tactic=None,
        technique_id=None, target_subnet=1, target_host=0, success=True,
        reward=1.0, native_reward=1.0, cve_id="CVE-X", cvss_base=score,
    )


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (None, "NONE"), (0.0, "NONE"), (3.7, "LOW"), (3.9, "LOW"),
        (4.0, "MEDIUM"), (6.9, "MEDIUM"), (7.0, "HIGH"), (8.9, "HIGH"),
        (9.0, "CRITICAL"), (10.0, "CRITICAL"),
    ],
)
def test_severity_bands_match_cvss_v31(score: float | None, band: str) -> None:
    assert make_step(score).severity == band


def test_step_target_is_none_when_unaddressed() -> None:
    step = StepRow(
        step_idx=0, action_name="noop", action_kind="noop", tactic=None,
        technique_id=None, target_subnet=None, target_host=None, success=False,
        reward=0.0, native_reward=0.0, cve_id=None, cvss_base=None,
    )
    assert step.target is None


# -- topology ---------------------------------------------------------------


def test_topology_without_structure_degrades_not_crashes() -> None:
    """Runs predating topology persistence must still open the replay view."""
    view = TopologyView(num_hosts=8, num_subnets=5)
    assert not view.has_structure
    assert view.crown_jewels() == set()


def test_crown_jewels_parsed_from_string_keys() -> None:
    view = TopologyView(sensitive_hosts={"(2, 0)": 100.0, "(4, 0)": 100.0})
    assert view.crown_jewels() == {(2, 0), (4, 0)}


def test_load_topology_prefers_host_list_over_subnet_sizes(tmp_path, monkeypatch) -> None:
    """subnets[0] is NASim's internet subnet and holds no real hosts.

    Deriving addresses from subnet sizes without accounting for that shifts
    every host into the wrong subnet.
    """
    from gui.data import runs as runs_folder

    run = tmp_path / "shaped-s44-t44"
    run.mkdir()
    (run / "config.snapshot.json").write_text(
        json.dumps(
            {
                "topology": {
                    "num_hosts": 8, "num_subnets": 5,
                    "subnets": [1, 1, 1, 5, 1],
                    "topology": [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]],
                    "hosts": [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3],
                              [3, 4], [4, 0]],
                    "sensitive_hosts": {"(2, 0)": 100.0, "(4, 0)": 100.0},
                }
            }
        )
    )
    monkeypatch.setattr(runs_folder, "RUNS_DIR", tmp_path)

    view = runs_folder.load_topology("shaped-s44-t44")
    assert view.has_structure
    assert (1, 0) in view.hosts and (3, 4) in view.hosts
    # No host may sit in subnet 0 — that is the internet.
    assert all(subnet >= 1 for subnet, _ in view.hosts)
    assert view.crown_jewels() == {(2, 0), (4, 0)}


def test_missing_snapshot_returns_empty_topology(tmp_path, monkeypatch) -> None:
    from gui.data import runs as runs_folder

    monkeypatch.setattr(runs_folder, "RUNS_DIR", tmp_path)
    assert runs_folder.load_topology("does-not-exist").num_hosts == 0


# -- formatting -------------------------------------------------------------


def test_numbers_use_a_real_minus_sign() -> None:
    """So −1,041 and −747 align on the decimal in a tabular column."""
    from gui import theme

    assert theme.fmt_num(-1041) == "−1,041"
    assert "-" not in theme.fmt_num(-747)


def test_missing_number_renders_as_dash_not_zero() -> None:
    """'nothing exploited' and 'mean CVSS of zero' are different facts."""
    from gui import theme

    assert theme.fmt_num(None) == "—"
    assert theme.fmt_num(0) == "0"


def test_hash_truncation_keeps_both_ends() -> None:
    from gui import theme

    assert theme.truncate_hash("2aa12404d40a23d0") == "2aa1…d0"
    assert theme.truncate_hash("") == "—"
    assert theme.truncate_hash("abc") == "abc"


def test_stylesheet_has_no_unsubstituted_tokens() -> None:
    """A stray @TOKEN@ silently voids the whole QSS rule it sits in."""
    from gui import theme

    qss = theme.build_stylesheet()
    assert "@" not in qss, "unsubstituted token left in the stylesheet"
    assert theme.ARM_1 in qss


# -- Train view: the displayed command must not leak secrets ----------------


def test_command_preview_redacts_the_database_password() -> None:
    """The command is shown on screen and projected during demos."""
    import os

    os.environ["POSTGRES_PASSWORD"] = "hunter2-should-not-appear"
    from gui.workers.trainer import TrainRequest

    request = TrainRequest(seed=42, topology_seed=42, timesteps=1000)
    assert "hunter2-should-not-appear" not in request.command_line
    assert "POSTGRES_PASSWORD=********" in request.command_line
    # The process itself still receives the real value.
    assert any("hunter2-should-not-appear" in a for a in request.argv())


def test_training_never_launches_on_host_networking() -> None:
    """CP-03. Host networking reaches the database but also the internet, and
    the pre-run gate refuses to train with an open network."""
    from gui.workers.trainer import TrainRequest

    argv = TrainRequest().argv()
    network = argv[argv.index("--network") + 1]
    assert network != "host", "training would have internet access"
    assert "internal" in network
