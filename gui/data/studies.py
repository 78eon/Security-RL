"""Read canonical Phase 7--13 result packages without loading large traces.

The generated result tree is local and gitignored.  This module intentionally
reads only metadata, analysis JSON and the small episode CSV files; model
checkpoints and trajectory JSONL never enter the GUI process.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from gui.data.models import StudyMetric, StudySummary

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True, slots=True)
class _StudySpec:
    phase: int
    study_id: str
    title: str
    result_dir: str
    metadata: str
    analysis: str


STUDIES = (
    _StudySpec(
        7,
        "experiment_01_fixed_budget_200k",
        "Fixed-budget reward ablation",
        "experiment_01_fixed_budget_200k",
        "metadata/experiment.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        8,
        "advanced-rl-recurrent-v1",
        "Knowledge-masked recurrent policy",
        "advanced-rl-recurrent-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        9,
        "advanced-rl-curriculum-v1",
        "Curriculum learning",
        "advanced-rl-curriculum-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        10,
        "advanced-rl-graph-policy-v1",
        "Graph policy",
        "advanced-rl-graph-policy-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        11,
        "advanced-rl-transfer-v1",
        "Transfer learning",
        "advanced-rl-transfer-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        12,
        "advanced-rl-hierarchical-policy-v1",
        "Hierarchical graph policy",
        "advanced-rl-hierarchical-policy-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
    _StudySpec(
        13,
        "advanced-rl-multiagent-defense-v1",
        "Multi-agent red/blue defence",
        "advanced-rl-multiagent-defense-v1/test",
        "metadata/study.json",
        "summaries/analysis.json",
    ),
)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _episode_count(result_dir: Path) -> int:
    count = 0
    episode_files = list(result_dir.glob("raw/*/episodes.csv"))
    episode_files.extend(result_dir.glob("raw/*/evaluation.csv"))
    for episode_file in episode_files:
        try:
            with episode_file.open(newline="") as handle:
                count += sum(1 for _ in csv.DictReader(handle))
        except OSError:
            continue
    return count


def _seed_count(analysis: dict) -> int:
    expected = analysis.get("expected_training_seeds") or analysis.get("expected_seeds")
    if isinstance(expected, list):
        return len(expected)
    arms = analysis.get("arms")
    if isinstance(arms, dict) and arms:
        values = [value for value in arms.values() if isinstance(value, int)]
        return min(values) if values else 0
    return 0


def _hash(metadata: dict) -> str:
    for key in (
        "study_config_hash",
        "scientific_config_hash",
        "experiment_config_sha256",
        "config_hash",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    frozen = metadata.get("frozen_inputs")
    if isinstance(frozen, dict):
        for key in ("experiment_config_sha256", "study_config_hash"):
            value = frozen.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _metric_rows(analysis: dict) -> list[StudyMetric]:
    primary = set(
        analysis.get("primary_metrics")
        or (analysis.get("protocol") or {}).get("primary_metrics")
        or []
    )
    rows: list[StudyMetric] = []
    for comparison in analysis.get("comparisons", []):
        if not isinstance(comparison, dict) or not comparison.get("metric"):
            continue
        name = str(comparison["metric"])
        rows.append(
            StudyMetric(
                name=name,
                arm_a_mean=_number(comparison.get("mean_a")),
                arm_b_mean=_number(comparison.get("mean_b")),
                difference=_number(comparison.get("difference")),
                p_value=_number(comparison.get("p_value")),
                p_adjusted=_number(comparison.get("p_bonferroni")),
                effect_size=_number(comparison.get("cohens_d")),
                significant=bool(comparison.get("significant", False)),
                primary=name in primary,
                warning=(
                    str(comparison["assumption_warning"])
                    if comparison.get("assumption_warning")
                    else None
                ),
            )
        )
    return rows


def load_studies(results_dir: Path | None = None) -> list[StudySummary]:
    """Return complete and partial canonical studies ordered newest first."""
    root = results_dir or RESULTS_DIR
    output: list[StudySummary] = []
    for spec in STUDIES:
        result_dir = root / spec.result_dir
        metadata = _json(result_dir / spec.metadata)
        analysis = _json(result_dir / spec.analysis)
        if not metadata and not analysis:
            continue
        metrics = _metric_rows(analysis)
        training_seeds = _seed_count(analysis)
        evaluation_episodes = _episode_count(result_dir)
        code_commit = str(metadata.get("code_commit") or "")
        config_hash = _hash(metadata)
        provenance = metadata.get("frozen_inputs")
        complete = bool(
            metadata.get("complete")
            and analysis.get("complete")
            and metrics
            and training_seeds
            and evaluation_episodes
            and code_commit
            and config_hash
            and isinstance(provenance, dict)
            and provenance
        )
        primary = [metric for metric in metrics if metric.primary]
        significant = [metric for metric in primary if metric.significant]
        if not complete:
            outcome = "Incomplete evidence package"
        elif significant:
            names = ", ".join(metric.name.replace("_", " ") for metric in significant)
            outcome = f"Significant primary result: {names}"
        else:
            outcome = "No significant primary effect"
        arm_a = str(analysis.get("arm_a") or "sparse")
        arm_b = str(analysis.get("arm_b") or "shaped")
        output.append(
            StudySummary(
                phase=spec.phase,
                study_id=spec.study_id,
                title=spec.title,
                arm_a=arm_a,
                arm_b=arm_b,
                complete=complete,
                outcome=outcome,
                code_commit=code_commit,
                config_hash=config_hash,
                result_path=str(result_dir),
                training_seeds=training_seeds,
                evaluation_episodes=evaluation_episodes,
                metrics=metrics,
            )
        )
    return sorted(output, key=lambda study: study.phase, reverse=True)


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
