"""CP-08/12/13/17 — offline CVE use, environment integrity, action validation,
and reward reachability.

Marked `integration`: needs nasim.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

pytest.importorskip("nasim")

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.events import AccessLevel  # noqa: E402
from rlredteam.nasim_adapter import RewardWrapper  # noqa: E402
from rlredteam.reward import RewardConfig, RewardMode  # noqa: E402
from rlredteam.topology import TopologyConfig, make_env  # noqa: E402

pytestmark = pytest.mark.integration

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEED = 42


@pytest.fixture(scope="module")
def wrapped():
    env = make_env(TopologyConfig.from_yaml(), topology_seed=SEED)
    return RewardWrapper(env, CVECatalogue.open_default(), topology_seed=SEED)


# -- CP-08 offline CVE use --------------------------------------------------


def test_catalogue_loads_without_network(monkeypatch) -> None:
    """Training must consume only the frozen local catalogue.

    Any socket use during load is a silent refresh, which would make results
    depend on the day they were produced.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted while loading the catalogue")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    catalogue = CVECatalogue.open_default()
    assert len(catalogue) >= 10
    assert catalogue.lookup("CVE-2021-42013").base_score == pytest.approx(9.8)


def test_environment_builds_without_network(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted while building the env")

    monkeypatch.setattr(socket, "create_connection", refuse)
    env = make_env(TopologyConfig.from_yaml(), topology_seed=SEED)
    assert env.action_space.n > 0


def test_catalogue_is_opened_read_only() -> None:
    """Training must not be able to mutate the frozen artefact."""
    import sqlite3

    from rlredteam.catalogue import DEFAULT_DB

    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM cves")


def test_training_code_never_imports_the_online_tool() -> None:
    """tools/fetch_nvd.py is the only networked component and must stay out of
    the training path."""
    src = Path(__file__).resolve().parents[1] / "src" / "rlredteam"
    offenders = [
        path.name for path in src.rglob("*.py")
        if "import fetch_nvd" in path.read_text()
        and path.name != "catalogue.py"  # build-time only, not train-time
    ]
    assert not offenders, f"training modules import the online tool: {offenders}"


# -- CP-12 environment integrity --------------------------------------------


def test_reset_returns_a_valid_observation(wrapped) -> None:
    obs, info = wrapped.reset(seed=SEED)
    assert obs.shape == wrapped.observation_space.shape
    assert wrapped.observation_space.contains(obs)


def test_reset_is_repeatable(wrapped) -> None:
    a, _ = wrapped.reset(seed=SEED)
    b, _ = wrapped.reset(seed=SEED)
    assert (a == b).all()


def test_step_returns_the_documented_five_tuple(wrapped) -> None:
    wrapped.reset(seed=SEED)
    result = wrapped.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert wrapped.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_terminates_within_the_step_limit(wrapped) -> None:
    import random

    limit = wrapped.env.scenario.step_limit
    wrapped.reset(seed=SEED)
    rng = random.Random(0)
    for step in range(limit + 5):
        _, _, terminated, truncated, _ = wrapped.step(
            rng.randrange(wrapped.action_space.n)
        )
        if terminated or truncated:
            assert step < limit, "ran past the declared step limit"
            return
    pytest.fail("episode never terminated")


def test_goal_and_step_limit_are_distinguishable(wrapped) -> None:
    """Conflating them would corrupt the success-rate metric."""
    import random

    seen = set()
    for episode in range(12):
        wrapped.reset(seed=SEED + episode)
        rng = random.Random(episode)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(
                rng.randrange(wrapped.action_space.n)
            )
        seen.add("goal" if terminated else "step_limit")
        assert not (terminated and truncated), "both flags set at once"
    assert seen, "no episodes completed"


# -- CP-13 action validation ------------------------------------------------


@pytest.mark.parametrize("action", [-1, 10**6])
def test_out_of_range_actions_are_rejected(wrapped, action: int) -> None:
    """An out-of-range action must fail loudly, not silently do something."""
    wrapped.reset(seed=SEED)
    with pytest.raises((AssertionError, IndexError, ValueError)):
        wrapped.step(action)


def test_state_survives_a_rejected_action(wrapped) -> None:
    """A rejected action must not corrupt the environment."""
    wrapped.reset(seed=SEED)
    with pytest.raises(ValueError, match="outside"):
        wrapped.step(10**6)
    # The environment must still be usable and consistent afterwards.
    obs_after, reward, terminated, truncated, _ = wrapped.step(0)
    assert wrapped.observation_space.contains(obs_after)
    assert isinstance(float(reward), float)


def test_every_valid_action_index_is_accepted(wrapped) -> None:
    for index in range(wrapped.action_space.n):
        wrapped.reset(seed=SEED)
        obs, _, _, _, _ = wrapped.step(index)
        assert wrapped.observation_space.contains(obs)


def test_numpy_action_types_are_accepted(wrapped) -> None:
    """SB3 hands down numpy integers; NASim asserts a Python int."""
    import numpy as np

    wrapped.reset(seed=SEED)
    for value in (np.int64(3), np.int32(3), 3):
        obs, _, _, _, _ = wrapped.step(value)
        assert wrapped.observation_space.contains(obs)


# -- CP-17 reachable rewards ------------------------------------------------


def test_positive_reward_only_follows_a_real_event(wrapped) -> None:
    """Every positive reward must correspond to a legitimate, successful,
    information-bearing action -- never to a failure or a no-op."""
    import random

    wrapped.reset(seed=SEED)
    rng = random.Random(3)
    checked = 0
    for _ in range(1500):
        _, reward, terminated, truncated, info = wrapped.step(
            rng.randrange(wrapped.action_space.n)
        )
        event = info["attack_event"]
        breakdown = info["reward_breakdown"]

        if breakdown.cve > 0 or breakdown.tactic > 0:
            assert event.success, "shaping paid on a failed action"
            assert event.is_informative, "shaping paid for no new information"
            checked += 1
        if breakdown.crown_jewel > 0:
            assert event.is_crown_jewel and event.success
        if not event.success:
            assert breakdown.total == pytest.approx(
                wrapped.engine.config.failed_action
            ), "a failed action earned something other than the penalty"
        if terminated or truncated:
            wrapped.reset(seed=SEED)
            wrapped.engine.reset()
    assert checked > 0, "no shaped rewards observed — test proved nothing"


def test_episode_shaping_stays_within_the_analytic_budget(wrapped) -> None:
    """The anti-farming bound, verified empirically rather than asserted."""
    import random

    budget = wrapped.engine.shaping_budget(
        num_hosts=len(wrapped.env.scenario.hosts)
    )
    for episode in range(4):
        wrapped.reset(seed=SEED + episode)
        rng = random.Random(episode)
        shaping = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = wrapped.step(
                rng.randrange(wrapped.action_space.n)
            )
            breakdown = info["reward_breakdown"]
            shaping += breakdown.cve + breakdown.tactic
        assert shaping <= budget


def test_native_mode_pays_exactly_what_the_simulator_pays() -> None:
    """CP-17 for the baseline arm: no invented reward at all.

    Tested on a SINGLE environment. Two separately constructed NASim envs
    diverge even when identically seeded -- exploit success is sampled per
    instance -- so comparing two of them step for step tests luck, not the
    wrapper. The real invariant is that the reward the wrapper returns is
    exactly the value it received, which info["native_reward"] carries verbatim.
    """
    import random

    wrapped = RewardWrapper(
        make_env(TopologyConfig.from_yaml(), topology_seed=SEED),
        CVECatalogue.open_default(),
        topology_seed=SEED,
        reward_config=RewardConfig(mode=RewardMode.NATIVE),
    )
    wrapped.reset(seed=SEED)
    rng = random.Random(11)
    for _ in range(400):
        _, reward, terminated, truncated, info = wrapped.step(
            rng.randrange(wrapped.action_space.n)
        )
        assert reward == pytest.approx(info["native_reward"])
        assert info["reward_breakdown"].cve == 0.0
        assert info["reward_breakdown"].tactic == 0.0
        if terminated or truncated:
            break


def test_crown_jewel_requires_access_not_merely_contact(wrapped) -> None:
    """The +100 must not be payable by touching a sensitive host without
    actually gaining access to it."""
    from rlredteam.events import ActionKind, AttackEvent

    sensitive = next(iter(wrapped.env.scenario.sensitive_hosts))
    event = AttackEvent(
        step=0, kind=ActionKind.EXPLOIT, action_name="e", target=tuple(sensitive),
        success=True, access_gained=AccessLevel.NONE, is_crown_jewel=False,
    )
    assert wrapped.engine.score(event).crown_jewel == 0.0
