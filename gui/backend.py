"""Backend-owned data models and loaders for the native desktop console."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gui.data.models import TopologyView
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


class BackendPort(Protocol):
    def load_dashboard(self) -> DashboardData: ...
    def refresh_paths(self) -> list[dict]: ...
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
                    }
                )
            return paths
        return []

    def load_dashboard(self) -> DashboardData:
        folders = [name for name in list_run_folders() if not name.startswith("_")]
        folders.sort(key=lambda name: (RUNS_DIR / name).stat().st_mtime, reverse=True)
        campaigns = self._campaigns(folders)
        latest = campaigns[0].name if campaigns else ""
        summary, snapshot = load_summary(latest), load_snapshot(latest)
        csv_rows = read_episode_csv(latest)
        db_runs, db_episodes, db_steps, paths = [], [], [], []
        status = "PostgreSQL connected"
        try:
            db_runs = self.repository.list_runs()
            paths = self.refresh_paths()
            if db_runs:
                db_episodes = self.repository.episodes(
                    [run.experiment_id for run in db_runs]
                )
                replayable = self.repository.replayable_episodes(db_runs[0].experiment_id)
                if replayable and replayable[-1].episode_id is not None:
                    db_steps = self.repository.steps(replayable[-1].episode_id)
        except (RepositoryError, AttributeError) as exc:
            status = f"Artefact mode · {getattr(exc, 'message', str(exc))}"
        if not campaigns and db_runs:
            campaigns = [
                CampaignData(
                    r.name,
                    r.reward_mode,
                    r.status,
                    r.episode_count,
                    None,
                    r.seed_label,
                    r.mean_native_reward,
                    r.success_rate,
                )
                for r in db_runs
            ]
            latest = campaigns[0].name
        campaign = campaigns[0] if campaigns else None
        tactics = {step.tactic for step in db_steps if step.tactic}
        return DashboardData(
            source_status=status,
            run_name=latest or "No stored runs",
            reward_mode=str(snapshot.get("reward_mode", campaign.reward_mode if campaign else "—")),
            seed=str(snapshot.get("training_seed", campaign.seed if campaign else "—")),
            episodes=int(summary.get("episodes", len(csv_rows))),
            progress=campaign.progress if campaign else None,
            success_rate=_number(summary.get("success_rate_overall")),
            mean_steps=_number(summary.get("mean_steps_to_goal")),
            mean_cvss=_number(summary.get("mean_cvss_exploited_last_10pct")),
            tactic_count=len(tactics),
            campaigns=campaigns,
            paths=paths,
            events=self._events(db_steps, latest, csv_rows),
            datasets=self._datasets(folders, db_runs, db_episodes),
            experiments=self._experiments(),
            agent=self._agent(snapshot),
            topology=load_topology(latest) if latest else TopologyView(),
            database_label=self.repository.settings.label,
        )

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
    def _datasets(folders, db_runs, db_episodes) -> list[DatasetData]:
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
                f"{len(db_episodes):,} episodes",
                f"{len(db_runs):,} experiment records",
                "CONNECTED" if db_runs else "EMPTY / OFFLINE",
            ),
        ]

    @staticmethod
    def _experiments() -> list[dict]:
        try:
            return list(
                json.loads((RUNS_DIR / "_analysis" / "analysis.json").read_text()).get("runs", [])
            )
        except (OSError, json.JSONDecodeError):
            return []

    def export_report(self) -> str:
        report = REPO_ROOT / "runs" / "_analysis" / "results_table.txt"
        if not report.is_file():
            raise FileNotFoundError("no analysis report is available")
        return str(report)


def _number(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
