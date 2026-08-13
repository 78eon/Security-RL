"""Topology generation and the NASim <-> reward-core bridge.

Marked `integration`: needs nasim installed. The reward core's own tests run
without it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nasim")

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.cvss import contrast_ratio  # noqa: E402
from rlredteam.events import AccessLevel, ActionKind  # noqa: E402
from rlredteam.nasim_adapter import (  # noqa: E402
    AdapterError,
    NASimEventAdapter,
    RewardWrapper,
    _access_level,
    _count,
)
from rlredteam.reward import RewardConfig, RewardMode  # noqa: E402
from rlredteam.topology import TopologyConfig, describe, make_env  # noqa: E402

pytestmark = pytest.mark.integration

SEEDS = list(range(42, 52))


@pytest.fixture(scope="module")
def catalogue() -> CVECatalogue:
    return CVECatalogue.open_default()


@pytest.fixture(scope="module")
def config() -> TopologyConfig:
    return TopologyConfig.from_yaml()


# -- topology --------------------------------------------------------------


def test_config_hash_is_stable_and_seed_independent(config: TopologyConfig) -> None:
    assert config.config_hash() == TopologyConfig.from_yaml().config_hash()
    assert len(config.config_hash()) == 16


def test_same_seed_reproduces_same_topology(config: TopologyConfig) -> None:
    """'Random' and 'reproducible' must not be in tension."""
    first = describe(make_env(config, topology_seed=42))
    second = describe(make_env(config, topology_seed=42))
    assert first == second


def test_different_seeds_give_different_topologies(config: TopologyConfig) -> None:
    a = describe(make_env(config, topology_seed=42))
    b = describe(make_env(config, topology_seed=43))
    assert a != b


@pytest.mark.parametrize("seed", SEEDS)
def test_every_seed_gives_the_agent_a_real_exploit_choice(
    config: TopologyConfig, seed: int
) -> None:
    """Without a choice the CVE weighting is a constant and cannot steer.

    This is the property that the stock `small` benchmark lacked: 7 of its 8
    hosts admitted exactly one applicable exploit.
    """
    summary = describe(make_env(config, topology_seed=seed))
    assert summary["mean_applicable_exploits_per_host"] > 1.5
    assert summary["hosts_with_exploit_choice"] >= summary["num_hosts"] * 0.5


@pytest.mark.parametrize("seed", SEEDS)
def test_crown_jewel_value_matches_locked_reward(config: TopologyConfig, seed: int) -> None:
    env = make_env(config, topology_seed=seed)
    assert env.scenario.sensitive_hosts
    for value in env.scenario.sensitive_hosts.values():
        assert value == pytest.approx(100.0)


# -- info-field normalisation ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, AccessLevel.NONE),
        (0, AccessLevel.NONE),
        (1, AccessLevel.USER),
        (2, AccessLevel.ROOT),
        ({}, AccessLevel.NONE),
        ({(1, 0): 2}, AccessLevel.ROOT),
        ({(1, 0): 1, (2, 0): 2}, AccessLevel.ROOT),
    ],
)
def test_access_level_handles_int_and_dict(raw: object, expected: AccessLevel) -> None:
    """NASim documents `access` as a dict but passes an int; accept both."""
    assert _access_level(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, 0), ({}, 0), (3, 3), ({(1, 0): True, (2, 0): True}, 2)]
)
def test_count_handles_int_and_collection(raw: object, expected: int) -> None:
    assert _count(raw) == expected


# -- adapter ---------------------------------------------------------------


def test_step_returns_the_documented_five_tuple(config: TopologyConfig) -> None:
    """Guards against a silent NASim upgrade changing the step API."""
    env = make_env(config, topology_seed=42)
    env.reset(seed=42)
    result = env.step(0)
    assert len(result) == 5


def test_every_action_resolves_to_a_kind(config: TopologyConfig, catalogue) -> None:
    env = make_env(config, topology_seed=42)
    adapter = NASimEventAdapter(env, catalogue, topology_seed=42)
    for idx in range(env.action_space.n):
        action = env.action_space.get_action(idx)
        assert isinstance(adapter.kind_of(action), ActionKind)


def test_every_exploit_and_privesc_gets_a_cve(config: TopologyConfig, catalogue) -> None:
    env = make_env(config, topology_seed=42)
    adapter = NASimEventAdapter(env, catalogue, topology_seed=42)
    for name in list(env.scenario.exploits) + list(env.scenario.privescs):
        assert adapter.assignment.cve_for(name) is not None


def test_missing_cve_raises_at_construction(
    config: TopologyConfig, catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing CVE must never silently degrade a shaped run into a native one."""
    env = make_env(config, topology_seed=42)
    dropped = next(iter(env.scenario.exploits))

    from rlredteam import nasim_adapter as adapter_module

    real_assign = adapter_module.assign_cves

    def assign_with_a_gap(exploits, privescs, cat, topology_seed):
        assignment = real_assign(exploits, privescs, cat, topology_seed)
        assignment.records.pop(dropped)
        assignment.mapping.pop(dropped)
        return assignment

    monkeypatch.setattr(adapter_module, "assign_cves", assign_with_a_gap)

    with pytest.raises(AdapterError, match=dropped):
        NASimEventAdapter(env, catalogue, topology_seed=42)


def test_missing_cve_tolerated_when_not_required(
    config: TopologyConfig, catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(config, topology_seed=42)
    from rlredteam import nasim_adapter as adapter_module

    real_assign = adapter_module.assign_cves
    dropped = next(iter(env.scenario.exploits))

    def assign_with_a_gap(exploits, privescs, cat, topology_seed):
        assignment = real_assign(exploits, privescs, cat, topology_seed)
        assignment.records.pop(dropped)
        assignment.mapping.pop(dropped)
        return assignment

    monkeypatch.setattr(adapter_module, "assign_cves", assign_with_a_gap)
    adapter = NASimEventAdapter(env, catalogue, topology_seed=42, require_cve=False)
    assert adapter.assignment.cve_for(dropped) is None


@pytest.mark.parametrize("seed", SEEDS)
def test_assigned_cves_keep_contrast_on_every_seed(
    config: TopologyConfig, catalogue, seed: int
) -> None:
    env = make_env(config, topology_seed=seed)
    adapter = NASimEventAdapter(env, catalogue, topology_seed=seed)
    assert contrast_ratio(adapter.assignment.scores()) >= 1.8


# -- wrapper ---------------------------------------------------------------


def test_native_mode_is_bit_identical_to_unwrapped_env(
    config: TopologyConfig, catalogue
) -> None:
    """The baseline really is the baseline.

    Compares only until the first divergence opportunity: two separately
    constructed NASim envs sample exploit success independently, so a long
    lockstep comparison tests luck rather than the wrapper. The invariant that
    holds for all time is asserted in test_environment_integrity.py.
    """
    plain = make_env(config, topology_seed=42)
    wrapped = RewardWrapper(
        make_env(config, topology_seed=42),
        catalogue,
        topology_seed=42,
        reward_config=RewardConfig(mode=RewardMode.NATIVE),
    )
    plain.reset(seed=42)
    wrapped.reset(seed=42)

    import random

    rng = random.Random(7)
    for _ in range(40):
        action = rng.randrange(plain.action_space.n)
        _, expected, done_a, trunc_a, _ = plain.step(action)
        _, actual, done_b, trunc_b, info = wrapped.step(action)
        assert actual == pytest.approx(expected)
        assert info["native_reward"] == pytest.approx(expected)
        assert (done_a, trunc_a) == (done_b, trunc_b)
        if done_a or trunc_a:
            break


def test_wrapper_exposes_native_reward_in_shaped_mode(
    config: TopologyConfig, catalogue
) -> None:
    """Evaluation reports native units even when training on shaped reward."""
    wrapped = RewardWrapper(
        make_env(config, topology_seed=42), catalogue, topology_seed=42
    )
    wrapped.reset(seed=42)
    _, _, _, _, info = wrapped.step(0)
    assert "native_reward" in info
    assert "reward_breakdown" in info
    assert "attack_event" in info


def test_goal_dominance_checked_at_construction(config: TopologyConfig, catalogue) -> None:
    from rlredteam.reward import GoalDominanceError

    with pytest.raises(GoalDominanceError):
        RewardWrapper(
            make_env(config, topology_seed=42),
            catalogue,
            topology_seed=42,
            reward_config=RewardConfig(cve_scale=500.0),
        )


def test_full_episode_shaping_stays_within_analytic_budget(
    config: TopologyConfig, catalogue
) -> None:
    """Empirical check that the anti-farming bound actually holds."""
    import random

    wrapped = RewardWrapper(make_env(config, topology_seed=42), catalogue, topology_seed=42)
    budget = wrapped.engine.shaping_budget(num_hosts=len(wrapped.env.scenario.hosts))

    for episode in range(5):
        wrapped.reset(seed=42 + episode)
        rng = random.Random(episode)
        shaping = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.randrange(wrapped.action_space.n)
            _, _, terminated, truncated, info = wrapped.step(action)
            breakdown = info["reward_breakdown"]
            shaping += breakdown.cve + breakdown.tactic
        assert shaping <= budget, f"episode {episode} shaping {shaping} exceeded {budget}"


def test_reset_clears_anti_farming_state(config: TopologyConfig, catalogue) -> None:
    import random

    wrapped = RewardWrapper(make_env(config, topology_seed=42), catalogue, topology_seed=42)
    wrapped.reset(seed=42)

    # Random actions rather than a fixed index: action 0 is one specific scan
    # that may never be informative, so it would accumulate no state at all.
    rng = random.Random(0)
    for _ in range(400):
        _, _, terminated, truncated, _ = wrapped.step(rng.randrange(wrapped.action_space.n))
        if terminated or truncated:
            break
    assert wrapped.engine._paid_scans or wrapped.engine._best_access, (
        "expected some shaping state to accumulate over 400 random steps"
    )

    wrapped.reset(seed=42)
    assert not wrapped.engine._paid_scans
    assert not wrapped.engine._best_access
