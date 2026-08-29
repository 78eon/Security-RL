"""Plain dataclasses passed between the data layer and the views.

No Qt import anywhere in this module or in repository.py -- the data layer is
tested headlessly, and workers hand these across thread boundaries where a
QWidget reference would be a crash waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One row of the Runs table."""

    experiment_id: int
    name: str
    reward_mode: str
    seeds: list[int]
    episode_count: int
    created_at: datetime | None
    topology_config_hash: str
    cve_manifest_sha256: str
    config_hash: str
    git_sha: str | None
    mean_native_reward: float | None
    success_rate: float | None
    mean_length: float | None
    has_steps: bool

    @property
    def status(self) -> str:
        """Derived, since the schema has no status column.

        A run with no episodes has not produced anything yet; everything else is
        reported complete. A crashed run is indistinguishable from a short one
        at the database level -- the honest label is 'complete' for what landed.
        """
        return "empty" if self.episode_count == 0 else "complete"

    @property
    def seed_label(self) -> str:
        if not self.seeds:
            return "—"
        if len(self.seeds) == 1:
            return str(self.seeds[0])
        return f"{min(self.seeds)}–{max(self.seeds)}"


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    experiment_id: int
    run_name: str
    reward_mode: str
    seed: int
    topology_seed: int
    episode_idx: int
    total_reward: float
    native_reward: float
    length: int
    terminal_state: str
    goal_reached: bool
    exploited_hosts: list
    mean_cvss_exploited: float | None
    episode_id: int | None = None


@dataclass(frozen=True, slots=True)
class StepRow:
    step_idx: int
    action_name: str
    action_kind: str
    tactic: str | None
    technique_id: str | None
    target_subnet: int | None
    target_host: int | None
    success: bool
    reward: float
    native_reward: float
    cve_id: str | None
    cvss_base: float | None
    target_entity: str | None = None
    state_changed: bool = False
    prerequisites: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)

    @property
    def target(self) -> str | tuple[int, int] | None:
        if self.target_entity:
            return self.target_entity
        if self.target_subnet is None or self.target_host is None:
            return None
        return (self.target_subnet, self.target_host)

    @property
    def severity(self) -> str:
        """CVSS v3.1 qualitative band for this step's vulnerability."""
        score = self.cvss_base
        if score is None:
            return "NONE"
        if score >= 9.0:
            return "CRITICAL"
        if score >= 7.0:
            return "HIGH"
        if score >= 4.0:
            return "MEDIUM"
        if score >= 0.1:
            return "LOW"
        return "NONE"


@dataclass(frozen=True, slots=True)
class Comparison:
    """Whether two runs may legitimately be plotted against each other.

    The app's central integrity guard. Two runs are comparable only when their
    topology and CVE-catalogue fingerprints match; anything else means the
    ablation is not controlled and a chart of the pair would be misleading.
    """

    left: RunSummary
    right: RunSummary

    @property
    def topology_matches(self) -> bool:
        return self.left.topology_config_hash == self.right.topology_config_hash

    @property
    def catalogue_matches(self) -> bool:
        return self.left.cve_manifest_sha256 == self.right.cve_manifest_sha256

    @property
    def reward_differs(self) -> bool:
        return self.left.reward_mode != self.right.reward_mode

    @property
    def comparable(self) -> bool:
        return self.topology_matches and self.catalogue_matches

    def reasons(self) -> list[str]:
        """Human-readable reasons a comparison is invalid."""
        problems = []
        if not self.topology_matches:
            problems.append("different topology configuration")
        if not self.catalogue_matches:
            problems.append("different CVE catalogue")
        if self.comparable and not self.reward_differs:
            problems.append(
                f"both runs use the same reward ({self.left.reward_mode}) — "
                "this compares a condition against itself"
            )
        return problems


@dataclass(slots=True)
class TopologyView:
    """Network structure for the replay canvas.

    ``subnets`` and ``adjacency`` are absent from runs created before the
    topology-persistence fix; the canvas degrades to a grouped layout without
    edges rather than failing.
    """

    num_hosts: int = 0
    num_subnets: int = 0
    subnets: list[int] = field(default_factory=list)
    adjacency: list[list[int]] = field(default_factory=list)
    sensitive_hosts: dict[str, float] = field(default_factory=dict)
    hosts: list[tuple[int, int]] = field(default_factory=list)

    @property
    def has_structure(self) -> bool:
        return bool(self.subnets) and bool(self.adjacency)

    def crown_jewels(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for key in self.sensitive_hosts:
            digits = [int(p) for p in key.strip("() ").split(",") if p.strip().isdigit()]
            if len(digits) == 2:
                out.add((digits[0], digits[1]))
        return out
