"""Plain dataclasses passed between the data layer and the views.

No Qt import anywhere in this module or in repository.py -- the data layer is
tested headlessly, and workers hand these across thread boundaries where a
QWidget reference would be a crash waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    title: str = "Not yet available"
    simulator: str = "Not yet available"
    topology_seed: int | None = None
    reward_mode: str = "Not yet available"
    training_seed: int | None = None
    host_count: int | None = None
    subnet_count: int | None = None
    crown_jewels: tuple[str, ...] = ()
    catalogue: str = "Not yet available"
    provenance: tuple[tuple[str, str], ...] = ()
    # Analyst-only DTO. Never accepted by knowledge/observation conversion.
    true_nodes: tuple[dict, ...] = ()
    true_edges: tuple[dict, ...] = ()
    status: str = "Not yet available"


@dataclass(frozen=True, slots=True)
class AgentKnowledgeView:
    nodes: tuple[dict, ...] = ()
    edges: tuple[dict, ...] = ()
    services: tuple[str, ...] = ()
    vulnerabilities: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()
    available_actions: tuple[str, ...] = ()
    status: str = "Not yet available: no recorded knowledge snapshot"


@dataclass(frozen=True, slots=True)
class CurrentActionView:
    step: int = 0
    kind: str = "Unknown"
    name: str = "Unknown"
    target: str = "Unknown"
    cve: str = "Unknown"
    cvss: float | None = None
    mitre: tuple[str, ...] = ()
    native_reward: float | None = None
    shaped_reward: float | None = None
    recorded_reward: float | None = None
    breakdown: tuple[tuple[str, float], ...] = ()
    success: bool | None = None
    access_gained: str = "Unknown"
    newly_discovered: int | None = None
    knowledge_delta: str = "Unknown: not recorded"
    source: str = "Not yet available"


@dataclass(frozen=True, slots=True)
class AttackTimelineEntry:
    action: CurrentActionView = field(default_factory=CurrentActionView)
    before: AgentKnowledgeView = field(default_factory=AgentKnowledgeView)
    after: AgentKnowledgeView = field(default_factory=AgentKnowledgeView)
    crown_jewel: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    title: str = "Not yet available"
    outcome: str = "Not yet available"
    scope: str = "Simulation report — no live-network evidence"
    metrics: tuple[tuple[str, str], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    limitation: str = "Discovery-derived baseline metrics are withheld from scientific comparison."


@dataclass(frozen=True, slots=True)
class AttackTrajectorySummary:
    title: str = "No episode selected"
    entries: tuple[AttackTimelineEntry, ...] = ()
    report: ExperimentReport = field(default_factory=ExperimentReport)
    status: str = "Not yet available"
    # Replay contains only event-derived data, never a true-topology field.


@dataclass(frozen=True, slots=True)
class ExperimentComparisonRow:
    metric: str
    experiment_a: str
    experiment_b: str
    value_a: float | None
    value_b: float | None
    difference: float | None


@dataclass(frozen=True, slots=True)
class ComparisonOption:
    label: str
    cohort: str
    metrics: tuple[tuple[str, float | None], ...]
    basis: str


@dataclass(frozen=True, slots=True)
class ResearchEvidenceSummary:
    status: str = "Not yet available"
    diagnostics: tuple[tuple[int, tuple[dict, ...]], ...] = ()
    convergence: tuple[tuple[str, str, str], ...] = ()
    images: tuple[tuple[str, bytes], ...] = ()
    criterion: str = "Not yet available"


@dataclass(frozen=True, slots=True)
class WorkspaceEvidence:
    scenario: ScenarioSummary = field(default_factory=ScenarioSummary)
    research: ResearchEvidenceSummary = field(default_factory=ResearchEvidenceSummary)
    episode_choices: tuple[tuple[str, str, int, int], ...] = ()
    comparisons: tuple[ComparisonOption, ...] = ()
    status: str = "Not yet available"


@dataclass(frozen=True, slots=True)
class DefectAuditSummary:
    affected_scan_count: int | None = None
    affected_reward: float | None = None
    total_positive_reward: float | None = None
    affected_share_pct: float | None = None
    tactic_share_pct: float | None = None
    interpretation: str = "Audit evidence not yet available."
    evidence_available: bool = False
    source: str = ""


@dataclass(frozen=True, slots=True)
class ConfirmatoryComparisonRow:
    arm: str
    metric: str
    frozen: float | None
    corrected: float | None
    corrected_minus_frozen: float | None


@dataclass(frozen=True, slots=True)
class ConfirmatoryStudySummary:
    study_id: str = "Not yet available"
    stage: str = "Not yet available"
    training_seeds: tuple[int, ...] = ()
    evaluation_seeds: tuple[int, ...] = ()
    total_timesteps: int | None = None
    registered: bool = False
    corrected_commit: str = "Not yet available"
    parent_commit: str = "Not yet available"
    completed_training_seeds: tuple[int, ...] = ()
    convergence_status: str = "Not yet available"
    evaluation_status: str = "Not yet available"
    comparison_available: bool = False
    comparison_path: str = ""
    status_message: str = "Confirmatory evidence not yet available."
    comparisons: tuple[ConfirmatoryComparisonRow, ...] = ()
    audit: DefectAuditSummary = field(default_factory=DefectAuditSummary)


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
    rl_action_index: int | None = None
    simulator_action: str | None = None
    framework_mappings: list[dict[str, str]] = field(default_factory=list)
    mapping_persisted: bool = False

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


@dataclass(frozen=True, slots=True)
class StudyMetric:
    """One paired comparison from a canonical research analysis."""

    name: str
    arm_a_mean: float | None
    arm_b_mean: float | None
    difference: float | None
    p_value: float | None
    p_adjusted: float | None
    effect_size: float | None
    significant: bool
    primary: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class StudySummary:
    """Small, presentation-safe view of one local canonical study package."""

    phase: int
    study_id: str
    title: str
    arm_a: str
    arm_b: str
    complete: bool
    outcome: str
    code_commit: str
    config_hash: str
    result_path: str
    training_seeds: int
    evaluation_episodes: int
    metrics: list[StudyMetric] = field(default_factory=list)

    @property
    def primary_metrics(self) -> list[StudyMetric]:
        return [metric for metric in self.metrics if metric.primary]
