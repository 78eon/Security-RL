"""The Essential-tier reward engine.

Implements the locked reward specification:

    CVE/CVSS x weight + tactic bonus, crown-jewel +100, failed action -5

with a sparse baseline toggle (1.0 on crown-jewel, else 0) so the shaped-vs-sparse
ablation differs in nothing but the reward.

Depends only on :mod:`rlredteam.events` and :mod:`rlredteam.cvss` -- no nasim,
gymnasium or torch -- so it is fully unit-testable without an environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from rlredteam.cvss import WeightMode, WeightParams, severity_weight
from rlredteam.events import AccessLevel, ActionKind, AttackEvent


class RewardMode(StrEnum):
    SHAPED = "shaped"
    SPARSE = "sparse"
    NATIVE = "native"


# Locked tactic bonuses. Exfil is defined but UNREACHABLE on NASim topologies --
# NASim has no exfiltration action (its action set is exploit, privesc and four
# scan types). It is carried here so the table matches the agreed specification
# and so its zero occurrence count is visible in results, pending the
# supervisor decision on whether to drop it. See docs/BUILD_PLAN.md.
DEFAULT_TACTIC_BONUSES: dict[str, float] = {
    "recon": 1.0,
    "exploit": 0.5,
    "privesc": 2.0,
    "exfil": 1.5,
}

_KIND_TO_TACTIC: dict[ActionKind, str] = {
    ActionKind.SERVICE_SCAN: "recon",
    ActionKind.OS_SCAN: "recon",
    ActionKind.SUBNET_SCAN: "recon",
    ActionKind.PROCESS_SCAN: "recon",
    ActionKind.EXPLOIT: "exploit",
    ActionKind.PRIVESC: "privesc",
}

# MITRE ATT&CK technique per action kind. Recorded on every scored event so the
# Important-tier mapper has its data from day one; not yet used in the reward.
KIND_TO_TECHNIQUE: dict[ActionKind, str] = {
    ActionKind.SERVICE_SCAN: "T1046",  # Network Service Discovery
    ActionKind.OS_SCAN: "T1082",  # System Information Discovery
    ActionKind.SUBNET_SCAN: "T1018",  # Remote System Discovery
    ActionKind.PROCESS_SCAN: "T1057",  # Process Discovery
    ActionKind.EXPLOIT: "T1210",  # Exploitation of Remote Services
    ActionKind.PRIVESC: "T1068",  # Exploitation for Privilege Escalation
}


class GoalDominanceError(AssertionError):
    """Raised when shaping could out-earn reaching the goal."""


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Per-step attribution. Every term is logged, not just the total."""

    total: float
    native: float
    cve: float = 0.0
    tactic: float = 0.0
    crown_jewel: float = 0.0
    penalty: float = 0.0
    weight: float = 0.0
    cve_id: str | None = None
    technique_id: str | None = None
    tactic_name: str | None = None
    paid: bool = False

    def as_row(self) -> dict[str, object]:
        return {
            "reward": self.total,
            "native_reward": self.native,
            "cve_term": self.cve,
            "tactic_term": self.tactic,
            "crown_jewel_term": self.crown_jewel,
            "penalty_term": self.penalty,
            "cvss_weight": self.weight,
            "cve_id": self.cve_id,
            "technique_id": self.technique_id,
            "tactic": self.tactic_name,
        }


@dataclass(frozen=True)
class RewardConfig:
    mode: RewardMode = RewardMode.SHAPED

    # Coefficient on the normalised CVSS weight -- the "x weight" of the locked
    # spec. Default 2.5 is not arbitrary: it is the largest round value for
    # which assert_goal_dominance() holds on an 8-host topology with two
    # crown jewels (see that method for the arithmetic). Raising it makes
    # sweeping high-CVSS hosts out-earn reaching the goal.
    cve_scale: float = 2.5

    tactic_bonuses: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TACTIC_BONUSES)
    )
    crown_jewel: float = 100.0
    failed_action: float = -5.0
    sparse_goal_reward: float = 1.0

    weight: WeightParams = field(default_factory=WeightParams)

    # Anti-farming. NASim scans always succeed and can be repeated, and an
    # already-compromised host can be re-exploited, so paying on every success
    # makes farming optimal. See RewardEngine._is_payable.
    first_success_only: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> RewardConfig:
        raw = yaml.safe_load(Path(path).read_text())["reward"]
        weight_raw = raw.pop("weight", {}) or {}
        if "mode" in weight_raw:
            weight_raw["mode"] = WeightMode(weight_raw["mode"])
        return cls(
            mode=RewardMode(raw.pop("mode")),
            weight=WeightParams(**weight_raw),
            **raw,
        )


class RewardEngine:
    """Scores :class:`AttackEvent` streams. Deterministic; holds per-episode state."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._paid_scans: set[tuple[str, tuple[int, int] | None]] = set()
        self._best_access: dict[tuple[int, int] | None, AccessLevel] = {}

    def reset(self) -> None:
        """Clear per-episode anti-farming state. Call on every env reset."""
        self._paid_scans.clear()
        self._best_access.clear()

    # -- payment eligibility --------------------------------------------

    def _is_payable(self, event: AttackEvent) -> bool:
        """Whether this event earns shaping, and record that it has.

        Scans pay once per (action, target) and only when they reveal something
        new. Exploits and privescs pay only when they *raise* the access level
        held on a host -- so re-exploiting a compromised host, or landing a
        second applicable exploit on it, earns nothing. That caps shaping at two
        payments per host (NONE->USER->ROOT) however many exploits apply to it,
        which is what keeps the budget in assert_goal_dominance finite.
        """
        if not event.is_informative:
            return False
        if not self.config.first_success_only:
            return True

        if event.kind.is_scan:
            key = event.dedup_key
            if key in self._paid_scans:
                return False
            self._paid_scans.add(key)
            return True

        previous = self._best_access.get(event.target, AccessLevel.NONE)
        if event.access_gained <= previous:
            return False
        self._best_access[event.target] = event.access_gained
        return True

    # -- scoring ---------------------------------------------------------

    def score(self, event: AttackEvent) -> RewardBreakdown:
        mode = self.config.mode

        if mode is RewardMode.NATIVE:
            return RewardBreakdown(total=event.native_reward, native=event.native_reward)

        if mode is RewardMode.SPARSE:
            # Triggers on the same condition as the shaped crown-jewel term, so
            # the two arms differ only in what is ADDED on top of it. Keying
            # sparse on goal_reached instead would change when the signal
            # arrives as well as its magnitude, confounding the ablation.
            earned = event.success and event.is_crown_jewel
            goal = self.config.sparse_goal_reward if earned else 0.0
            return RewardBreakdown(
                total=goal,
                native=event.native_reward,
                crown_jewel=goal,
                paid=earned,
            )

        return self._score_shaped(event)

    def _score_shaped(self, event: AttackEvent) -> RewardBreakdown:
        config = self.config
        technique = KIND_TO_TECHNIQUE.get(event.kind)
        tactic_name = _KIND_TO_TACTIC.get(event.kind)

        # A failed action pays the flat penalty and nothing else.
        if not event.success:
            return RewardBreakdown(
                total=config.failed_action,
                native=event.native_reward,
                penalty=config.failed_action,
                technique_id=technique,
                tactic_name=tactic_name,
            )

        crown = config.crown_jewel if event.is_crown_jewel else 0.0
        paid = self._is_payable(event)

        cve_term = 0.0
        weight = 0.0
        if paid and event.cvss_base is not None:
            weight = severity_weight(event.cvss_base, config.weight)
            cve_term = config.cve_scale * weight

        tactic_term = (
            config.tactic_bonuses.get(tactic_name, 0.0) if (paid and tactic_name) else 0.0
        )

        return RewardBreakdown(
            total=cve_term + tactic_term + crown,
            native=event.native_reward,
            cve=cve_term,
            tactic=tactic_term,
            crown_jewel=crown,
            weight=weight,
            cve_id=event.cve_id,
            technique_id=technique,
            tactic_name=tactic_name,
            paid=paid,
        )

    # -- solvability guard ------------------------------------------------

    def shaping_budget(self, num_hosts: int, scans_per_host: int = 4) -> float:
        """Maximum shaping obtainable in one episode.

        Per host: at most two access-raising payments (user then root), each
        earning ``cve_scale * w_max`` plus its tactic bonus, and at most
        ``scans_per_host`` recon payments.
        """
        config = self.config
        w_max = config.weight.w_max
        exploit_pay = config.cve_scale * w_max + config.tactic_bonuses.get("exploit", 0.0)
        privesc_pay = config.cve_scale * w_max + config.tactic_bonuses.get("privesc", 0.0)
        # Scans carry no CVE, so recon earns its tactic bonus only.
        recon_pay = scans_per_host * config.tactic_bonuses.get("recon", 0.0)
        return num_hosts * (exploit_pay + privesc_pay + recon_pay)

    def assert_goal_dominance(
        self, num_hosts: int, num_crown_jewels: int, scans_per_host: int = 4
    ) -> None:
        """Raise unless reaching the goal beats farming shaping.

        Enforces ``budget < 0.5 * total goal value``. Without this an agent can
        out-earn the crown jewels by sweeping high-CVSS hosts and never
        terminating -- the episode then ends only at the step limit, and the
        study reports on an agent that learned to farm the bonus.
        """
        if self.config.mode is not RewardMode.SHAPED:
            return
        budget = self.shaping_budget(num_hosts, scans_per_host)
        goal_value = num_crown_jewels * self.config.crown_jewel
        if budget >= 0.5 * goal_value:
            raise GoalDominanceError(
                f"shaping budget {budget:.1f} >= half the goal value "
                f"({0.5 * goal_value:.1f}) for {num_hosts} hosts / "
                f"{num_crown_jewels} crown jewels. The agent could out-earn the "
                f"goal by farming. Lower cve_scale (currently "
                f"{self.config.cve_scale}) or the tactic bonuses."
            )
