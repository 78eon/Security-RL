"""CP-09/21/25/27/28/29/31 — validation, provenance, gating, analysis.

Pure-logic tests: no training, no database, no display.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rlredteam import provenance as prov
from rlredteam.analyse import (
    assess_convergence,
    load_protocol,
    metrics_for_run,
    paired_comparison,
)
from rlredteam.catalogue import ValidationError, validate_record

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- CP-09 CVE validation ---------------------------------------------------


def valid_record(**overrides) -> dict:
    base = {
        "cve_id": "CVE-2021-42013",
        "kind": "exploit",
        "cvss_version": "3.1",
        "base_score": 9.8,
        "base_severity": "CRITICAL",
        "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-42013",
    }
    base.update(overrides)
    return base


def test_valid_record_is_accepted() -> None:
    assert validate_record(valid_record())["cve_id"] == "CVE-2021-42013"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cve_id", "CVE-BAD"),
        ("cve_id", "2021-42013"),
        ("base_score", 11.0),
        ("base_score", -1.0),
        ("base_score", "not a number"),
        ("base_severity", "SEVERE"),
        ("base_severity", "LOW"),  # disagrees with 9.8
        ("kind", "something-else"),
        ("cvss_version", "3.0"),  # disagrees with the vector prefix
        ("vector", "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
        ("source_url", "https://example.com/other"),
    ],
)
def test_malformed_records_are_rejected(field: str, value) -> None:
    with pytest.raises(ValidationError):
        validate_record(valid_record(**{field: value}))


@pytest.mark.parametrize("field", ["cve_id", "vector", "base_score", "source_url"])
def test_missing_required_fields_are_rejected(field: str) -> None:
    record = valid_record()
    record.pop(field)
    with pytest.raises(ValidationError, match="missing required fields"):
        validate_record(record)


def test_committed_catalogue_passes_validation() -> None:
    from rlredteam.catalogue import CVECatalogue

    for record in CVECatalogue.open_default().all_records():
        validate_record(
            {
                "cve_id": record.cve_id,
                "kind": record.kind,
                "cvss_version": record.cvss_version,
                "base_score": record.base_score,
                "base_severity": record.base_severity,
                "vector": record.vector,
                "source_url": record.source_url,
            }
        )


# -- CP-10/21 topology hashing ----------------------------------------------


def described(**overrides) -> dict:
    base = {
        "subnets": [1, 1, 1, 5, 1],
        "topology": [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
        "hosts": [[1, 0], [2, 0]],
        "services": ["a", "b"],
        "os": ["linux"],
        "processes": ["p"],
        "exploits": ["e_0"],
        "privescs": ["pe_0"],
        "sensitive_hosts": {"(2, 0)": 100.0},
        "step_limit": 1000,
        "observation_space": [234],
        "action_space_n": 120,
    }
    base.update(overrides)
    return base


def test_topology_hash_is_stable() -> None:
    assert prov.topology_hash(described()) == prov.topology_hash(described())


@pytest.mark.parametrize(
    "change",
    [
        {"subnets": [1, 1, 2, 5, 1]},
        {"exploits": ["e_1"]},
        {"sensitive_hosts": {"(3, 0)": 100.0}},
        {"topology": [[1, 0, 0], [0, 1, 1], [0, 1, 1]]},
        {"hosts": [[1, 0], [3, 0]]},
    ],
)
def test_topology_hash_changes_with_the_network(change: dict) -> None:
    """The whole point: a different network must hash differently.

    The topology *config* hash cannot do this -- it covers generation rules and
    is identical for every seed, so two runs on different networks share it.
    """
    assert prov.topology_hash(described()) != prov.topology_hash(described(**change))


def test_environment_hash_ignores_network_details() -> None:
    """The agent's interface is unchanged by which hosts exist."""
    a = prov.environment_config_hash(described())
    b = prov.environment_config_hash(described(exploits=["e_9"], hosts=[[9, 9]]))
    assert a == b


# -- CP-21 the pre-run gate -------------------------------------------------


def manifest(**overrides) -> prov.ExperimentManifest:
    base = dict(
        experiment_id="shaped-s42-t42",
        git_commit="abc123",
        git_dirty=False,
        python_version="3.11.15",
        dependency_lock_hash="lock",
        docker_image_digest="",
        training_seed=42,
        topology_seed=42,
        topology_hash="topo",
        topology_config_hash="cfg",
        environment_config_hash="env",
        cve_manifest_sha256="cve",
        reward_config_hash="rew",
        ppo_config_hash="ppo",
        dataset_version="16",
        training_budget=200_000,
        ppo_config={"n_steps": 2048},
    )
    base.update(overrides)
    return prov.ExperimentManifest(**base)


def test_gate_passes_a_well_formed_run(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    assert prov.run_gate(manifest()).passed


@pytest.mark.parametrize("declared", ["", "unknown", "unexpected"])
def test_unknown_declared_git_state_is_not_treated_as_clean(monkeypatch, declared) -> None:
    monkeypatch.setenv("RLREDTEAM_GIT_DIRTY", declared)
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert prov.git_dirty() is None


def test_gate_stops_on_a_frozen_hash_mismatch(monkeypatch) -> None:
    """CP-25 — the guardrail against running an arm on different inputs."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    result = prov.run_gate(manifest(), frozen={"topology_hash": "different"})
    assert not result.passed
    assert any("topology_hash" in f for f in result.failures())


def test_gate_stops_without_database_credentials(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    result = prov.run_gate(manifest())
    assert not result.passed
    assert any("credentials" in f for f in result.failures())


def test_gate_stops_when_the_network_is_open(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: True)
    result = prov.run_gate(manifest())
    assert not result.passed
    assert any("offline" in f for f in result.failures())


def test_gate_stops_on_a_dirty_tree(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    assert not prov.run_gate(manifest(git_dirty=True)).passed


def test_gate_stops_when_tree_state_is_unknown(monkeypatch) -> None:
    """A false 'clean' would certify a run whose commit does not describe it."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    result = prov.run_gate(manifest(git_dirty=None))
    assert not result.passed
    assert any("working tree state known" in f for f in result.failures())


def test_gate_stops_without_ppo_configuration(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "x")
    monkeypatch.setattr(prov, "_internet_reachable", lambda timeout=1.5: False)
    assert not prov.run_gate(manifest(ppo_config={})).passed


def test_enforce_raises_with_the_failing_checks_named() -> None:
    result = prov.GateResult()
    result.add("something", False, "because")
    with pytest.raises(prov.GateFailure, match="something"):
        prov.enforce(result)


def test_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest().write(path)
    assert prov.ExperimentManifest.read(path).topology_hash == "topo"


def test_git_dirty_prefers_the_host_declaration(monkeypatch) -> None:
    monkeypatch.setenv("RLREDTEAM_GIT_DIRTY", "1")
    assert prov.git_dirty() is True
    monkeypatch.setenv("RLREDTEAM_GIT_DIRTY", "0")
    assert prov.git_dirty() is False


# -- CP-29 convergence ------------------------------------------------------


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_protocol()


def episodes_from(values: list[float]) -> list[dict]:
    return [
        {
            "native_return": v,
            "shaped_return": v,
            "length": 100,
            "goal_reached": True,
            "mean_cvss_exploited": 8.0,
            "max_cvss_exploited": 9.5,
            "hosts_compromised": 3,
        }
        for v in values
    ]


def test_short_run_is_not_converged(protocol: dict) -> None:
    converged, detail = assess_convergence(episodes_from([1.0] * 5), protocol)
    assert not converged
    assert "episodes" in detail["reason"]


def test_flat_run_is_converged(protocol: dict) -> None:
    converged, _ = assess_convergence(episodes_from([-500.0] * 60), protocol)
    assert converged


def test_still_improving_run_is_not_converged(protocol: dict) -> None:
    """A run still climbing steeply has not plateaued, whatever its last window
    happens to average."""
    values = [-2000.0 + 30 * i for i in range(60)]
    converged, detail = assess_convergence(episodes_from(values), protocol)
    assert not converged
    assert not detail["checks"]["plateaued"]


def test_declining_run_is_not_converged(protocol: dict) -> None:
    values = [-500.0] * 40 + [-500.0 - 40 * i for i in range(20)]
    converged, detail = assess_convergence(episodes_from(values), protocol)
    assert not converged


def test_wildly_noisy_run_is_not_converged(protocol: dict) -> None:
    values = [-100.0 if i % 2 else -3000.0 for i in range(60)]
    converged, detail = assess_convergence(episodes_from(values), protocol)
    assert not converged
    assert not detail["checks"]["spread_within_threshold"]


def test_convergence_reports_its_criterion(protocol: dict) -> None:
    """CP-29 — the criterion travels with the verdict, so 'converged' is a
    claim someone else can check."""
    _, detail = assess_convergence(episodes_from([-500.0] * 60), protocol)
    assert "criterion" in detail and "checks" in detail


# -- CP-27/28 metrics and statistics ----------------------------------------


def test_steps_to_goal_ignores_failed_episodes(protocol: dict) -> None:
    """Averaging over failures rewards an arm that fails fast, since a
    truncated episode has a fixed length."""
    episodes = episodes_from([-500.0] * 20)
    episodes[0]["goal_reached"] = False
    episodes[0]["length"] = 1000
    metrics = metrics_for_run("r", "shaped", 42, 42, episodes, protocol)
    assert metrics.steps_to_goal == 100.0
    assert metrics.success_rate == pytest.approx(19 / 20)


def test_metrics_handle_absent_cvss(protocol: dict) -> None:
    episodes = episodes_from([-500.0] * 20)
    for e in episodes:
        e["mean_cvss_exploited"] = None
        e["max_cvss_exploited"] = None
    metrics = metrics_for_run("r", "sparse", 42, 42, episodes, protocol)
    assert metrics.mean_cvss_exploited is None
    assert metrics.hvt_hit_rate == 0.0


def test_paired_comparison_pairs_by_seed(protocol: dict) -> None:
    # Differences vary across pairs, as real data does. A constant difference
    # gives zero variance and an undefined effect size -- covered separately.
    a = {42: 1.0, 43: 2.0, 44: 3.0}
    b = {42: 2.2, 43: 2.9, 44: 4.4}
    result = paired_comparison("native_return", a, b, protocol)
    assert result.n_pairs == 3
    assert result.difference == pytest.approx(1.1667, abs=1e-3)
    assert result.cohens_d is not None
    assert result.ci_low is not None and result.ci_high is not None


def test_constant_difference_reports_no_effect_size(protocol: dict) -> None:
    """A perfectly constant difference has zero variance, so Cohen's d is
    undefined rather than infinite. Reported as None instead of a number that
    would look like a real measurement."""
    a = {42: 1.0, 43: 2.0, 44: 3.0}
    b = {42: 2.0, 43: 3.0, 44: 4.0}
    result = paired_comparison("native_return", a, b, protocol)
    assert result.difference == pytest.approx(1.0)
    assert result.cohens_d is None


def test_unpaired_seeds_are_dropped_not_compared(protocol: dict) -> None:
    """Comparing unpaired data would silently change the test being run."""
    result = paired_comparison(
        "native_return", {42: 1.0, 43: 2.0, 99: 9.0}, {42: 2.0, 43: 3.0}, protocol
    )
    assert result.n_pairs == 2


def test_comparison_survives_a_single_pair(protocol: dict) -> None:
    result = paired_comparison("native_return", {42: 1.0}, {42: 2.0}, protocol)
    assert result.n_pairs == 1
    assert result.p_value is None


def test_bonferroni_applies_only_to_primary_metrics(protocol: dict) -> None:
    a, b = {42: 1.0, 43: 2.0, 44: 3.0}, {42: 2.0, 43: 3.0, 44: 4.5}
    primary = paired_comparison("native_return", a, b, protocol)
    descriptive = paired_comparison("hvt_hit_rate", a, b, protocol)
    assert primary.p_bonferroni is not None
    assert descriptive.p_bonferroni is None
    assert primary.p_bonferroni >= primary.p_value


def test_protocol_family_size_matches_primary_metric_count(protocol: dict) -> None:
    assert protocol["statistics"]["family_size"] == len(protocol["primary_metrics"])


def test_protocol_pins_stochastic_native_evaluation(protocol: dict) -> None:
    assert protocol["evaluation"]["action_selection"] == "stochastic"
    assert protocol["evaluation"]["reward_scale"] == "native"
    assert protocol["evaluation"]["seeds"] == list(range(42, 52))
    assert protocol["evaluation"]["episode_seeds"] == list(range(1001, 1011))


def test_convergence_uses_training_rows_not_evaluation_rows(protocol: dict) -> None:
    evaluation = episodes_from([-500.0] * 10)
    training = episodes_from([-2000.0 + 30 * index for index in range(60)])
    metrics = metrics_for_run(
        "r",
        "shaped",
        42,
        42,
        evaluation,
        protocol,
        training_episodes=training,
    )
    assert not metrics.converged
