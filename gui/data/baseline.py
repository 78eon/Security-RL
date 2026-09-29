"""Read the reviewed fixed-topology baseline; never run scientific code.

Manifest checks establish consistency of the mounted report, not a fresh
training/DB audit or authenticity against a separately trusted signature.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = Path("results/convergence_v2/final_baseline_evaluation")
WITHHELD = ("newly_discovered", "discovered_hosts", "attack_path_length")
DOCUMENTS = (
    ("Supervisor Word showcase", "docs/supervisor_office/Security_RL_Supervisor_Showcase.docx"),
    (
        "Excel evidence and log record",
        "docs/supervisor_office/Security_RL_Evidence_and_Log_Record.xlsx",
    ),
    ("Supervisor evidence bundle", "docs/SUPERVISOR_EVIDENCE_BUNDLE.md"),
    ("Supervisor archive", "docs/SECURITY_RL_SUPERVISOR_EVIDENCE_BUNDLE.tar.gz"),
    ("Discovery-defect audit", "docs/DISCOVERY_COUNTING_DEFECT_AUDIT.md"),
    ("Proposed correction plan", "docs/DISCOVERY_COUNTING_CORRECTION_PLAN.md"),
    ("Dissertation wording", "docs/BASELINE_DISSERTATION_WORDING.md"),
    ("Supervisor email — unsent draft", "docs/BASELINE_SUPERVISOR_EMAIL.md"),
)


@dataclass(frozen=True, slots=True)
class BaselineData:
    state: str = "UNAVAILABLE"
    detail: str = "No reviewed baseline package is mounted."
    candidate: str = ""
    interpretation: str = ""
    warning: str = ""
    policy_pairs: int = 0
    total_episodes: int = 0
    summary: dict = field(default_factory=dict)
    comparisons: list[dict] = field(default_factory=list)
    training: list[tuple] = field(default_factory=list)
    provenance: list[tuple[str, str]] = field(default_factory=list)
    documents: list[tuple[str, str, bool]] = field(default_factory=list)
    statistics_path: str = ""


def _path(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("unsafe evidence path")
    path = (root / part).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence path leaves its mounted root")
    return path


def _read(path: Path) -> bytes:
    # Small report files only. Checkpoints and trajectory streams are never read.
    with path.open("rb") as handle:
        data = handle.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("report file exceeds the desktop size limit")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked(path: Path, expected: str) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
        raise ValueError("invalid evidence hash")
    data = _read(path)
    if _sha(data) != expected:
        raise ValueError(f"evidence hash mismatch: {path.name}")
    return data


def _number(value) -> bool:
    return isinstance(value, float | int) and not isinstance(value, bool) and math.isfinite(value)


def load_baseline(repo_root: Path) -> BaselineData:
    """Fail closed on missing/corrupt reviewed evidence; no raw-result fallback."""
    documents = []
    for title, relative in DOCUMENTS:
        try:
            available = _path(repo_root, relative).is_file()
        except (OSError, ValueError):
            available = False
        documents.append((title, relative, available))
    try:
        package = _path(repo_root, str(PACKAGE))
        if not package.is_dir():
            return BaselineData(documents=documents)
        report = _path(package, "report")
        manifest = json.loads(_read(_path(report, "reviewed_manifest.json")))
        entries = manifest["files"]
        if not isinstance(entries, dict) or not 1 <= len(entries) <= 256:
            raise ValueError("invalid reviewed manifest")
        required = {"reviewed_results.json", "metric_quality_addendum.json",
                    "results.json", "primary_statistics.csv"}
        if not required <= entries.keys():
            raise ValueError("incomplete reviewed manifest")
        blobs = {}
        total_bytes = 0
        for name, digest in entries.items():
            data = _checked(_path(report, name), digest)
            total_bytes += len(data)
            if total_bytes > 64 * 1024 * 1024:
                raise ValueError("report exceeds the desktop memory budget")
            blobs[name] = data
        reviewed = json.loads(blobs["reviewed_results.json"])
        quality = json.loads(blobs["metric_quality_addendum.json"])
        # Bind the reviewed addendum to the report and its preserved original.
        if (
            _sha(blobs["metric_quality_addendum.json"])
            != reviewed["metric_quality_addendum_sha256"]
            or _sha(blobs["results.json"]) != reviewed["original_results_sha256"]
        ):
            raise ValueError("reviewed report references do not match")
        protocol_bytes = _read(_path(package, "protocol.json"))
        protocol = json.loads(protocol_bytes)
        sparse = json.loads(_read(_path(package, "sparse_frozen.json")))
        if sparse["protocol_sha256"] != _sha(protocol_bytes):
            raise ValueError("sparse freeze uses a different protocol")
        frozen = json.loads(
            _checked(
                _path(repo_root, "results/convergence_v2/first_stable.json"),
                protocol["frozen_shaped_sha256"],
            )
        )
        candidate = frozen["candidate"]
        if not isinstance(candidate, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
            raise ValueError("invalid candidate identifier")
        candidate_root = _path(repo_root, f"results/convergence_v2/{candidate}")
        registration = json.loads(
            _checked(
                _path(candidate_root, "registration.json"),
                protocol["registration_sha256"],
            )
        )
        assessment = json.loads(
            _checked(
                _path(candidate_root, "assessment.json"),
                frozen["assessment_sha256"],
            )
        )
        pairs = reviewed["policy_pairs"]
        episodes = reviewed["total_episodes"]
        seeds = protocol["training_seeds"]
        if (
            type(pairs) is not int
            or pairs <= 0
            or pairs != len(seeds)
            or type(episodes) is not int
            or episodes != 2 * pairs * reviewed["episodes_per_policy"]
        ):
            raise ValueError("inconsistent paired evaluation counts")
        comparisons = reviewed["primary_statistics"]
        names = [row["metric"] for row in comparisons]
        if len(names) != len(set(names)) or set(names) != set(protocol["primary_metrics"]):
            raise ValueError("primary metric family does not match protocol")
        for row in comparisons:
            if row["n_pairs"] != pairs or type(row["significant"]) is not bool:
                raise ValueError("invalid paired comparison")
            for key in (
                "mean_a",
                "mean_b",
                "difference",
                "p_value",
                "p_bonferroni",
                "ci_low",
                "ci_high",
            ):
                if not _number(row[key]):
                    raise ValueError("non-finite or missing primary statistic")
        summary = reviewed["summary"]
        for arm in ("sparse", "shaped"):
            if any(summary[arm].get(key) is not None for key in WITHHELD):
                raise ValueError("reviewed report contains unwithheld discovery metrics")
            for key in ("success_rate", "steps_to_goal", "native_return", "failed_actions"):
                if not _number(summary[arm][key]):
                    raise ValueError("invalid evaluation summary")
        training = []
        for arm, checkpoints in (
            ("shaped", protocol["shaped_checkpoints"]),
            ("sparse", sparse["checkpoints"]),
        ):
            if set(checkpoints) != {str(seed) for seed in seeds}:
                raise ValueError("incomplete frozen checkpoint metadata")
            if any(row["status"] != "complete" for row in checkpoints.values()):
                raise ValueError("unfinished checkpoint metadata")
            timesteps = sorted({row["actual_timesteps"] for row in checkpoints.values()})
            passed = (
                assessment["passing_seeds"]
                if arm == "shaped"
                else sum(
                    sparse["native_training_stability"][str(seed)]["native_return_stability_passed"]
                    is True
                    for seed in seeds
                )
            )
            training.append(
                (
                    arm,
                    ", ".join(map(str, seeds)),
                    ", ".join(f"{value:,}" for value in timesteps),
                    f"{passed}/{pairs}",
                    "Shaped convergence gate"
                    if arm == "shaped"
                    else "Separate native-return stability diagnostic",
                )
            )
        provenance = [
            ("Recorded code commit", protocol["git_commit"]),
            ("Experiment config SHA-256", registration["inputs"]["config_sha256"]),
            ("Topology YAML SHA-256", protocol["literal_topology_yaml_sha256"]),
            ("Topology instance SHA-256", protocol["reconstructed_topology_sha256"]),
            ("CVE snapshot SHA-256", protocol["cve_catalogue_sha256"]),
            ("Protocol SHA-256", _sha(protocol_bytes)),
            ("Reviewed report SHA-256", _sha(blobs["reviewed_results.json"])),
            ("Evaluation seeds", ", ".join(map(str, protocol["evaluation_episode_seeds"]))),
            ("Action selection", protocol["evaluation"]["action_selection"]),
            ("Policy updates", str(protocol["evaluation"]["policy_updates"])),
            ("Observation/reward normalization", str(protocol["normalization"]["enabled"])),
            ("PPO advantage normalization", str(protocol["ppo"]["normalize_advantage"])),
        ]
        return BaselineData(
            state="REVIEWED",
            detail="Mounted report hashes match; not a fresh training or DB audit.",
            candidate=candidate,
            interpretation=reviewed["interpretation"],
            warning=f"{quality['reward_impact']} {quality['caution']}",
            policy_pairs=pairs,
            total_episodes=episodes,
            summary=summary,
            comparisons=comparisons,
            training=training,
            provenance=provenance,
            documents=documents,
            statistics_path=str(_path(report, "primary_statistics.csv")),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        # Do not display partial numbers or raw, unreviewed alternatives.
        return BaselineData(
            state="INVALID", detail=f"Reviewed evidence unavailable: {exc}", documents=documents
        )
