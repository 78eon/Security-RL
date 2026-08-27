"""The Essential reward engine: locked constants, anti-farming, sparse toggle."""

from __future__ import annotations

import pytest

from rlredteam.events import AccessLevel, ActionKind, AttackEvent
from rlredteam.reward import (
    GoalDominanceError,
    RewardConfig,
    RewardEngine,
    RewardMode,
)


def exploit_event(
    *,
    target: tuple[int, int] = (1, 0),
    name: str = "e_srv_0",
    success: bool = True,
    cvss: float | None = 9.8,
    access: AccessLevel = AccessLevel.USER,
    crown: bool = False,
    native: float = 0.0,
    step: int = 0,
) -> AttackEvent:
    return AttackEvent(
        step=step,
        kind=ActionKind.EXPLOIT,
        action_name=name,
        target=target,
        success=success,
        cvss_base=cvss,
        cve_id="CVE-TEST",
        access_gained=access,
        is_crown_jewel=crown,
        native_reward=native,
    )


def scan_event(
    *, target: tuple[int, int] = (1, 0), name: str = "service_scan", found: int = 1
) -> AttackEvent:
    return AttackEvent(
        step=0,
        kind=ActionKind.SERVICE_SCAN,
        action_name=name,
        target=target,
        success=True,
        newly_discovered=found,
    )


@pytest.fixture
def engine() -> RewardEngine:
    return RewardEngine(RewardConfig())


# -- locked constants ------------------------------------------------------


def test_failed_action_scores_exactly_minus_five(engine: RewardEngine) -> None:
    result = engine.score(exploit_event(success=False))
    assert result.total == -5.0
    assert result.penalty == -5.0
    assert result.cve == 0.0 and result.tactic == 0.0


def test_crown_jewel_pays_one_hundred(engine: RewardEngine) -> None:
    result = engine.score(exploit_event(crown=True))
    assert result.crown_jewel == 100.0
    assert result.total >= 100.0


def test_tactic_bonuses_match_locked_values() -> None:
    cases = [
        (ActionKind.SERVICE_SCAN, 1.0),
        (ActionKind.OS_SCAN, 1.0),
        (ActionKind.SUBNET_SCAN, 1.0),
        (ActionKind.PROCESS_SCAN, 1.0),
        (ActionKind.EXPLOIT, 0.5),
        (ActionKind.PRIVESC, 2.0),
    ]
    for kind, expected in cases:
        engine = RewardEngine(RewardConfig())
        if kind.is_scan:
            event = AttackEvent(
                step=0, kind=kind, action_name=str(kind), target=(1, 0),
                success=True, newly_discovered=1,
            )
        else:
            event = AttackEvent(
                step=0, kind=kind, action_name=str(kind), target=(1, 0),
                success=True, access_gained=AccessLevel.ROOT, cvss_base=None,
            )
        assert engine.score(event).tactic == pytest.approx(expected), kind


def test_higher_cvss_scores_strictly_higher() -> None:
    """The core ordering property the whole study rests on."""
    low = RewardEngine(RewardConfig()).score(exploit_event(cvss=3.7))
    high = RewardEngine(RewardConfig()).score(exploit_event(cvss=9.8))
    assert high.total > low.total
    assert high.cve > low.cve


# -- anti-farming ----------------------------------------------------------


def test_repeated_scan_pays_once(engine: RewardEngine) -> None:
    first = engine.score(scan_event())
    second = engine.score(scan_event())
    assert first.tactic == 1.0
    assert second.tactic == 0.0
    assert second.total == 0.0


def test_scan_with_no_new_information_pays_nothing(engine: RewardEngine) -> None:
    assert engine.score(scan_event(found=0)).total == 0.0


def test_second_exploit_on_same_host_pays_nothing(engine: RewardEngine) -> None:
    """Several exploits may apply to one host; only the first compromise pays."""
    first = engine.score(exploit_event(name="e_a", access=AccessLevel.USER))
    second = engine.score(exploit_event(name="e_b", access=AccessLevel.USER))
    assert first.paid and not second.paid
    assert second.cve == 0.0 and second.tactic == 0.0


def test_privesc_after_exploit_pays_because_access_rises(engine: RewardEngine) -> None:
    engine.score(exploit_event(access=AccessLevel.USER))
    privesc = AttackEvent(
        step=1, kind=ActionKind.PRIVESC, action_name="pe_0", target=(1, 0),
        success=True, access_gained=AccessLevel.ROOT, cvss_base=7.8,
    )
    assert engine.score(privesc).paid


def test_scan_farming_is_bounded(engine: RewardEngine) -> None:
    """1000 repeated scans must not out-earn a crown jewel."""
    total = sum(engine.score(scan_event()).total for _ in range(1000))
    assert total == 1.0


def test_reset_clears_state(engine: RewardEngine) -> None:
    engine.score(scan_event())
    engine.reset()
    assert engine.score(scan_event()).tactic == 1.0


# -- sparse baseline -------------------------------------------------------


def test_sparse_returns_only_zero_or_one() -> None:
    engine = RewardEngine(RewardConfig(mode=RewardMode.SPARSE))
    assert engine.score(exploit_event(crown=True, cvss=9.8)).total == 1.0
    assert engine.score(exploit_event(crown=False, cvss=9.8)).total == 0.0
    assert engine.score(exploit_event(success=False)).total == 0.0


def test_sparse_ignores_cvss_entirely() -> None:
    engine = RewardEngine(RewardConfig(mode=RewardMode.SPARSE))
    low = engine.score(exploit_event(cvss=3.7, crown=True))
    engine.reset()
    high = engine.score(exploit_event(cvss=10.0, crown=True))
    assert low.total == high.total


def test_sparse_uses_same_trigger_as_shaped_crown_jewel() -> None:
    """Both arms must pay on the same event, differing only in magnitude."""
    event = exploit_event(crown=True)
    sparse = RewardEngine(RewardConfig(mode=RewardMode.SPARSE)).score(event)
    shaped = RewardEngine(RewardConfig(mode=RewardMode.SHAPED)).score(event)
    assert sparse.paid and shaped.crown_jewel == 100.0
    assert sparse.total == 1.0


def test_sparse_pays_nothing_for_a_failed_crown_jewel_attempt() -> None:
    engine = RewardEngine(RewardConfig(mode=RewardMode.SPARSE))
    assert engine.score(exploit_event(crown=True, success=False)).total == 0.0


@pytest.mark.parametrize("mode", [RewardMode.SPARSE, RewardMode.SHAPED])
def test_crown_jewel_never_pays_again_without_new_access(mode: RewardMode) -> None:
    engine = RewardEngine(RewardConfig(mode=mode))
    first = engine.score(exploit_event(crown=True, access=AccessLevel.ROOT))
    repeated = engine.score(exploit_event(crown=True, access=AccessLevel.ROOT))
    later_scan = AttackEvent(
        step=2,
        kind=ActionKind.PROCESS_SCAN,
        action_name="process_scan",
        target=(1, 0),
        success=True,
        access_gained=AccessLevel.ROOT,
        newly_discovered=1,
        is_crown_jewel=True,
    )
    scanned = engine.score(later_scan)

    assert first.crown_jewel > 0
    assert repeated.crown_jewel == 0
    assert scanned.crown_jewel == 0


# -- native passthrough ----------------------------------------------------


def test_native_mode_is_exact_passthrough() -> None:
    """Proves the native arm is a real baseline, not a reimplementation."""
    engine = RewardEngine(RewardConfig(mode=RewardMode.NATIVE))
    events = [exploit_event(native=v, step=i) for i, v in enumerate([9.0, -1.0, 97.0])]
    assert sum(engine.score(e).total for e in events) == pytest.approx(105.0)


def test_native_reward_always_logged_in_every_mode() -> None:
    for mode in RewardMode:
        engine = RewardEngine(RewardConfig(mode=mode))
        assert engine.score(exploit_event(native=42.0)).native == 42.0


# -- goal dominance --------------------------------------------------------


def test_goal_dominance_holds_for_shipped_config(engine: RewardEngine) -> None:
    engine.assert_goal_dominance(num_hosts=8, num_crown_jewels=2)


def test_goal_dominance_raises_when_bonuses_too_large() -> None:
    engine = RewardEngine(RewardConfig(cve_scale=50.0))
    with pytest.raises(GoalDominanceError, match="farming"):
        engine.assert_goal_dominance(num_hosts=8, num_crown_jewels=2)


def test_goal_dominance_not_enforced_for_sparse() -> None:
    engine = RewardEngine(RewardConfig(mode=RewardMode.SPARSE, cve_scale=999.0))
    engine.assert_goal_dominance(num_hosts=8, num_crown_jewels=2)


# -- determinism and config ------------------------------------------------


def test_scoring_is_deterministic() -> None:
    events = [exploit_event(step=i, target=(1, i)) for i in range(5)]
    runs = []
    for _ in range(2):
        engine = RewardEngine(RewardConfig())
        runs.append([engine.score(e).total for e in events])
    assert runs[0] == runs[1]


def test_configs_differ_only_in_mode() -> None:
    """The ablation is valid only if nothing else varies between arms."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    shaped = RewardConfig.from_yaml(root / "shaped.yaml")
    sparse = RewardConfig.from_yaml(root / "sparse.yaml")

    assert shaped.mode is RewardMode.SHAPED
    assert sparse.mode is RewardMode.SPARSE
    assert shaped.cve_scale == sparse.cve_scale
    assert shaped.tactic_bonuses == sparse.tactic_bonuses
    assert shaped.crown_jewel == sparse.crown_jewel
    assert shaped.failed_action == sparse.failed_action
    assert shaped.weight == sparse.weight


def test_exfil_bonus_is_defined_but_unreachable() -> None:
    """NASim has no exfiltration action; the entry exists pending sign-off."""
    from rlredteam.reward import _KIND_TO_TACTIC

    config = RewardConfig()
    assert "exfil" in config.tactic_bonuses
    assert "exfil" not in _KIND_TO_TACTIC.values()
