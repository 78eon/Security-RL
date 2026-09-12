"""Backend-owned data models and loaders for the native desktop console."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gui.data.models import StudySummary, TopologyView
from gui.data.repository import Repository, RepositoryError
from gui.data.runs import (
    CATALOGUE_DB,
    RUNS_DIR,
    list_run_folders,
    load_snapshot,
    load_summary,
    load_topology,
    read_episode_csv,
)
from gui.data.studies import load_studies

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CampaignData:
    name: str
    reward_mode: str
    status: str
    episodes: int
    progress: int | None
    seed: str
    mean_reward: float | None
    success_rate: float | None


@dataclass(frozen=True, slots=True)
class AgentData:
    algorithm: str = "Unavailable"
    environment: str = "Unavailable"
    learning_rate: float | None = None
    gamma: float | None = None
    batch_size: int | None = None
    specification: str = "No stored agent configuration is available."


@dataclass(frozen=True, slots=True)
class EventData:
    sequence: str
    level: str
    source: str
    message: str
    context: str


@dataclass(frozen=True, slots=True)
class DatasetData:
    title: str
    count: str
    detail: str
    integrity: str


@dataclass(frozen=True, slots=True)
class SimulationData:
    profile: str
    topology_seed: int
    topology_hash: str
    topology_name: str
    nodes: list[dict]
    edges: list[dict]
    trajectory: list[dict]
    goal_reached: bool
    episode_steps: int
    total_reward: float
    discovery_coverage: float
    agent: str = "Knowledge-only deterministic feasibility agent"


@dataclass(frozen=True, slots=True)
class DashboardData:
    source_status: str
    run_name: str = "No stored runs"
    reward_mode: str = "—"
    seed: str = "—"
    episodes: int = 0
    progress: int | None = None
    success_rate: float | None = None
    mean_steps: float | None = None
    mean_cvss: float | None = None
    tactic_count: int = 0
    campaigns: list[CampaignData] = field(default_factory=list)
    paths: list[dict] = field(default_factory=list)
    events: list[EventData] = field(default_factory=list)
    datasets: list[DatasetData] = field(default_factory=list)
    experiments: list[dict] = field(default_factory=list)
    agent: AgentData = field(default_factory=AgentData)
    topology: TopologyView = field(default_factory=TopologyView)
    database_label: str = "Not configured"
    studies: list[StudySummary] = field(default_factory=list)


class BackendPort(Protocol):
    def load_dashboard(self) -> DashboardData: ...
    def refresh_paths(self) -> list[dict]: ...
    def simulation_profiles(self) -> list[dict]: ...
    def run_simulation(self, profile: str, topology_seed: int) -> SimulationData: ...
    def export_report(self) -> str: ...


class ApplicationBackend:
    """Read UI state from PostgreSQL and persisted experiment artefacts."""

    def __init__(self, repository: Repository | None = None) -> None:
        self.repository = repository or Repository()

    def pause_campaign(self, campaign_id: str) -> None:
        raise RuntimeError("no campaign scheduler is configured")

    def resume_campaign(self, campaign_id: str) -> None:
        raise RuntimeError("no campaign scheduler is configured")

    def save_agent_config(self, config: dict) -> None:
        raise RuntimeError("run snapshots are read-only; no configuration service is configured")

    def simulation_profiles(self) -> list[dict]:
        """Return backend-supported simulation profiles, never nearby networks."""
        from rlredteam.enterprise.profiles import DeploymentProfile

        labels = {
            DeploymentProfile.ON_PREMISES: "On-premises",
            DeploymentProfile.LEGACY: "Legacy estate",
            DeploymentProfile.CLOUD: "Cloud estate",
            DeploymentProfile.HYBRID: "Hybrid estate",
        }
        return [
            {"id": profile.value, "label": labels[profile]}
            for profile in DeploymentProfile
        ]

    def run_simulation(self, profile: str, topology_seed: int) -> SimulationData:
        """Execute one offline graph episode for native GUI replay.

        This is a deterministic feasibility demonstration, not a research
        evaluation and not a live network scan. The blocking call is invoked
        through the GUI's worker pool.
        """
        from rlredteam.enterprise.environment import EnterpriseCyberEnv
        from rlredteam.enterprise.onprem import knowledge_policy_action, topology_digest
        from rlredteam.enterprise.profiles import (
            DeploymentProfile,
            EnterpriseProfileConfig,
            generate_profile_topology,
        )
        from rlredteam.enterprise.trajectory import reconstruct_attack_path

        if isinstance(topology_seed, bool) or not 0 <= int(topology_seed) <= 2_147_483_647:
            raise ValueError("topology seed must be an integer between 0 and 2147483647")
        selected = DeploymentProfile(profile)
        config = EnterpriseProfileConfig.from_yaml()
        topology = generate_profile_topology(selected, int(topology_seed), config)
        env = EnterpriseCyberEnv(
            topology,
            max_steps=config.max_steps,
            max_nodes=config.max_nodes,
            max_vulnerabilities=config.max_vulnerabilities,
        )
        env.reset(seed=int(topology_seed))
        terminated = truncated = False
        reward_total = 0.0
        rows: list[dict] = []
        while not (terminated or truncated):
            action = knowledge_policy_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            event = info["event"]
            rows.append(
                {
                    "step": event.step,
                    "action": event.action.name,
                    "action_kind": event.action.type.value,
                    "target_entity": event.action.target,
                    "target": event.action.target,
                    "success": event.success,
                    "state_changed": event.state_changed,
                    "reward": event.reward,
                    "prerequisites": list(event.prerequisites),
                    "outcomes": list(event.outcomes),
                    "goal_reached": event.goal_reached,
                }
            )
        causal = reconstruct_attack_path(rows)
        nodes = [
            {
                "id": item.id,
                "type": item.type.value,
                "name": item.name,
                "attributes": dict(item.attributes),
            }
            for item in sorted(topology.nodes.values(), key=lambda item: item.id)
        ]
        edges = [
            {"source": edge.source, "target": edge.target, "type": edge.type.value}
            for edge in topology.edges
        ]
        return SimulationData(
            profile=selected.value,
            topology_seed=int(topology_seed),
            topology_hash=topology_digest(topology),
            topology_name=topology.name,
            nodes=nodes,
            edges=edges,
            trajectory=causal,
            goal_reached=bool(terminated and not truncated),
            episode_steps=len(rows),
            total_reward=reward_total,
            discovery_coverage=len(env.knowledge.discovered) / len(topology.nodes),
        )

    def refresh_paths(self) -> list[dict]:
        for run in self.repository.list_runs():
            if not run.has_steps:
                continue
            paths = []
            for episode in self.repository.replayable_episodes(run.experiment_id)[-12:]:
                if episode.episode_id is None:
                    continue
                steps = self.repository.steps(episode.episode_id)
                if not steps:
                    continue
                successful = [step for step in steps if step.success]
                causal = _causal_steps(steps)
                target = (successful[-1] if successful else steps[-1]).target
                max_cvss = max(
                    (step.cvss_base for step in steps if step.cvss_base is not None), default=0.0
                )
                risk = (
                    "Critical"
                    if max_cvss >= 9
                    else "High"
                    if max_cvss >= 7
                    else "Medium"
                    if max_cvss >= 4
                    else "Unknown"
                )
                paths.append(
                    {
                        "id": f"AP-{episode.episode_idx:03d}",
                        "target": str(target or "environment"),
                        "risk": risk,
                        "steps": len(steps),
                        "detection": "—",
                        "confidence": f"{round(100 * len(successful) / len(steps))}%",
                        "trajectory": [
                            {
                                "step": step.step_idx,
                                "action": step.action_kind,
                                "target": str(step.target or "environment"),
                                "outcomes": list(getattr(step, "outcomes", []) or []),
                            }
                            for step in causal
                        ],
                    }
                )
            return paths
        return []

    def load_dashboard(self) -> DashboardData:
        folders = [name for name in list_run_folders() if not name.startswith("_")]
        folders.sort(key=lambda name: (RUNS_DIR / name).stat().st_mtime, reverse=True)
        artifact_campaigns = self._campaigns(folders)
        db_runs, latest_episodes, db_steps, paths = [], [], [], []
        status = "PostgreSQL connected"
        try:
            db_runs = self.repository.list_runs()
            paths = self.refresh_paths()
            if db_runs:
                latest_episodes = self.repository.episodes([db_runs[0].experiment_id])
                replayable = self.repository.replayable_episodes(db_runs[0].experiment_id)
                if replayable and replayable[-1].episode_id is not None:
                    db_steps = self.repository.steps(replayable[-1].episode_id)
        except (RepositoryError, AttributeError) as exc:
            status = f"Artefact mode · {getattr(exc, 'message', str(exc))}"
        campaigns = self._merge_campaigns(db_runs, artifact_campaigns)
        latest = campaigns[0].name if campaigns else ""
        summary, snapshot = load_summary(latest), load_snapshot(latest)
        csv_rows = read_episode_csv(latest)
        campaign = campaigns[0] if campaigns else None
        latest_db = db_runs[0] if db_runs else None
        successful_lengths = [episode.length for episode in latest_episodes if episode.goal_reached]
        observed_cvss = [
            episode.mean_cvss_exploited
            for episode in latest_episodes
            if episode.mean_cvss_exploited is not None
        ]
        success_rate = _number(summary.get("success_rate_overall"))
        if success_rate is None and latest_db is not None:
            success_rate = latest_db.success_rate
        mean_steps = _number(summary.get("mean_steps_to_goal"))
        if mean_steps is None and successful_lengths:
            mean_steps = sum(successful_lengths) / len(successful_lengths)
        mean_cvss = _number(summary.get("mean_cvss_exploited_last_10pct"))
        if mean_cvss is None and observed_cvss:
            mean_cvss = sum(observed_cvss) / len(observed_cvss)
        tactics = {step.tactic for step in db_steps if step.tactic}
        studies = load_studies()
        if studies:
            status = f"{status} · Phase {studies[0].phase} evidence loaded"
        database_label = getattr(
            getattr(self.repository, "settings", None), "label", "Configured backend"
        )
        return DashboardData(
            source_status=status,
            run_name=latest or "No stored runs",
            reward_mode=str(snapshot.get("reward_mode", campaign.reward_mode if campaign else "—")),
            seed=str(snapshot.get("training_seed", campaign.seed if campaign else "—")),
            episodes=int(
                summary.get(
                    "episodes",
                    latest_db.episode_count if latest_db is not None else len(csv_rows),
                )
            ),
            progress=campaign.progress if campaign else None,
            success_rate=success_rate,
            mean_steps=mean_steps,
            mean_cvss=mean_cvss,
            tactic_count=len(tactics),
            campaigns=campaigns,
            paths=paths,
            events=self._events(db_steps, latest, csv_rows),
            datasets=self._datasets(
                folders,
                db_runs,
                sum(int(run.episode_count) for run in db_runs),
            ),
            experiments=[],
            agent=self._agent(snapshot),
            topology=load_topology(latest) if latest else TopologyView(),
            database_label=database_label,
            studies=studies,
        )

    @staticmethod
    def _merge_campaigns(db_runs, artifact_campaigns) -> list[CampaignData]:
        """Prefer database truth and append artifact-only runs without duplicates."""
        output = [
            CampaignData(
                run.name,
                run.reward_mode,
                run.status,
                run.episode_count,
                None,
                run.seed_label,
                run.mean_native_reward,
                run.success_rate,
            )
            for run in db_runs
        ]
        names = {campaign.name for campaign in output}
        output.extend(
            campaign for campaign in artifact_campaigns if campaign.name not in names
        )
        return output

    @staticmethod
    def _campaigns(folders: list[str]) -> list[CampaignData]:
        output = []
        for name in folders:
            summary, snapshot, rows = (
                load_summary(name),
                load_snapshot(name),
                read_episode_csv(name),
            )
            if not (summary or snapshot or rows):
                continue
            budget = snapshot.get("training_budget")
            timesteps = int(rows[-1].get("timesteps", 0)) if rows else 0
            progress = min(100, round(100 * timesteps / budget)) if budget else None
            output.append(
                CampaignData(
                    name,
                    str(snapshot.get("reward_mode", "unknown")),
                    "complete" if summary else "partial artefact",
                    int(summary.get("episodes", len(rows))),
                    progress,
                    str(snapshot.get("training_seed", "—")),
                    _number(summary.get("mean_native_return_overall")),
                    _number(summary.get("success_rate_overall")),
                )
            )
        return output

    @staticmethod
    def _agent(snapshot: dict) -> AgentData:
        ppo, topology = snapshot.get("ppo_config", {}), snapshot.get("topology", {})
        if not ppo:
            return AgentData()
        spec = {
            "algorithm": "PPO",
            "policy": ppo.get("policy"),
            "environment": "NASim",
            "topology": topology.get("name"),
            "hosts": topology.get("num_hosts"),
            "step_limit": topology.get("step_limit"),
            "reward_mode": snapshot.get("reward_mode"),
            "training_seed": snapshot.get("training_seed"),
        }
        return AgentData(
            "PPO",
            "NASim",
            _number(ppo.get("learning_rate")),
            _number(ppo.get("gamma")),
            ppo.get("batch_size"),
            json.dumps(spec, indent=2),
        )

    @staticmethod
    def _events(steps, run_name: str, rows) -> list[EventData]:
        if steps:
            return [
                EventData(
                    str(s.step_idx),
                    "SUCCESS" if s.success else "INFO",
                    s.action_kind,
                    " · ".join(v for v in (s.action_name, s.technique_id, s.cve_id) if v),
                    str(s.target or run_name),
                )
                for s in reversed(steps[-100:])
            ]
        return [
            EventData(
                str(r.get("episode_idx", "—")),
                "SUCCESS" if r.get("goal_reached", "").lower() == "true" else "INFO",
                "runs.episodes",
                f"{r.get('terminal_state', 'unknown')} after {r.get('length', '—')} steps",
                run_name,
            )
            for r in reversed(rows[-100:])
        ]

    @staticmethod
    def _datasets(folders, db_runs, db_episode_count: int) -> list[DatasetData]:
        cves = 0
        try:
            with sqlite3.connect(f"file:{CATALOGUE_DB}?mode=ro", uri=True) as conn:
                cves = int(conn.execute("SELECT count(*) FROM cves").fetchone()[0])
        except sqlite3.Error:
            pass
        episode_rows = sum(len(read_episode_csv(name)) for name in folders)
        return [
            DatasetData("Frozen CVE catalogue", f"{cves:,} CVEs", str(CATALOGUE_DB), "READ ONLY"),
            DatasetData("Run artefacts", f"{len(folders):,} runs", str(RUNS_DIR), "LOCAL"),
            DatasetData(
                "Episode summaries",
                f"{episode_rows:,} rows",
                "Persisted CSV episode records",
                "LOCAL",
            ),
            DatasetData(
                "PostgreSQL records",
                f"{db_episode_count:,} episodes",
                f"{len(db_runs):,} experiment records",
                "CONNECTED" if db_runs else "EMPTY / OFFLINE",
            ),
        ]

    def export_report(self) -> str:
        studies = load_studies()
        if studies:
            report = Path(studies[0].result_path) / "tables" / "statistics.csv"
            if report.is_file():
                return str(report)
        report = REPO_ROOT / "runs" / "_analysis" / "results_table.txt"
        if not report.is_file():
            raise FileNotFoundError("no analysis report is available")
        return str(report)


def _number(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalise_evidence_atom(atom: str) -> set[str]:
    prefix, separator, entity = atom.partition(":")
    if not separator:
        return {atom}
    atoms = {atom}
    if prefix.startswith("known_"):
        atoms.add(f"known:{entity}")
    if prefix in {"authenticated", "pivoted", "root_access", "user_access", "read_access"}:
        atoms.add(f"access:{entity}")
    if prefix == "discovered":
        atoms.update({f"known:{entity}", f"reachable:{entity}"})
    if prefix in {"access_target", "network"}:
        atoms.add(f"reachable:{entity}")
    if prefix == "vulnerability":
        atoms.add(f"known_vulnerability:{entity}")
    if prefix == "credential_source":
        atoms.add(f"known:{entity}")
    return atoms


def _causal_steps(steps):
    """Back-chain a displayed route from persisted evidence, never topology."""
    progress = [
        step
        for step in steps
        if step.success and getattr(step, "state_changed", True)
    ]
    if not progress:
        return []
    final = progress[-1]
    selected = [final]
    required = {
        atom
        for item in (getattr(final, "prerequisites", []) or [])
        for atom in _normalise_evidence_atom(str(item))
    }
    if not required:
        return progress
    for step in reversed(progress[:-1]):
        produced = {
            atom
            for item in (getattr(step, "outcomes", []) or [])
            for atom in _normalise_evidence_atom(str(item))
        }
        if not produced & required:
            continue
        selected.append(step)
        required -= produced
        required.update(
            atom
            for item in (getattr(step, "prerequisites", []) or [])
            for atom in _normalise_evidence_atom(str(item))
        )
    return list(reversed(selected))
