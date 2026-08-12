"""The plain-data event type that the reward engine scores.

Deliberately has zero dependencies -- no nasim, gymnasium, torch or I/O. The
reward core scores :class:`AttackEvent`, so it is fully unit-testable before the
environment exists, and swapping simulator would mean rewriting only the adapter
(:mod:`rlredteam.nasim_adapter`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class ActionKind(StrEnum):
    """Action categories. Mirrors NASim's action classes one-for-one."""

    EXPLOIT = "exploit"
    PRIVESC = "privesc"
    SERVICE_SCAN = "service_scan"
    OS_SCAN = "os_scan"
    SUBNET_SCAN = "subnet_scan"
    PROCESS_SCAN = "process_scan"
    NOOP = "noop"

    @property
    def is_scan(self) -> bool:
        return self in _SCAN_KINDS


_SCAN_KINDS = frozenset(
    {
        ActionKind.SERVICE_SCAN,
        ActionKind.OS_SCAN,
        ActionKind.SUBNET_SCAN,
        ActionKind.PROCESS_SCAN,
    }
)


class AccessLevel(IntEnum):
    """Matches nasim.envs.utils.AccessLevel."""

    NONE = 0
    USER = 1
    ROOT = 2


@dataclass(frozen=True, slots=True)
class AttackEvent:
    """One environment step, reduced to what the reward engine needs.

    Carries more than the current reward uses (``native_reward``, ``cost``,
    ``service``) so that later reward terms need no signature change.
    """

    step: int
    kind: ActionKind
    action_name: str
    target: tuple[int, int] | None
    success: bool

    # Taken verbatim from the environment, always logged so results can also be
    # reported in native NASim units even when training on the shaped reward.
    native_reward: float = 0.0
    cost: float = 0.0

    cve_id: str | None = None
    cvss_base: float | None = None
    service: str | None = None
    access_gained: AccessLevel = AccessLevel.NONE
    newly_discovered: int = 0

    is_crown_jewel: bool = False
    goal_reached: bool = False
    terminal: bool = False
    error: str | None = None

    @property
    def is_informative(self) -> bool:
        """True when the action actually advanced the agent's position.

        This is the anti-farming predicate. NASim scans always "succeed" and can
        be repeated indefinitely, so paying a flat recon bonus on every
        successful scan makes scan-spam optimal (up to +1000 over a 1000-step
        episode, against a +100 crown jewel). Requiring *new* information -- and
        first-time-only payment in the engine -- bounds the shaping total.
        """
        if not self.success:
            return False
        if self.kind.is_scan:
            return self.newly_discovered > 0
        if self.kind in (ActionKind.EXPLOIT, ActionKind.PRIVESC):
            return self.access_gained > AccessLevel.NONE
        return False

    @property
    def dedup_key(self) -> tuple[str, tuple[int, int] | None]:
        """Identity used for first-informative-success accounting."""
        return (self.action_name, self.target)
