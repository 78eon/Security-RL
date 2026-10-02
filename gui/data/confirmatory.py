"""Read persisted confirmatory evidence without importing experiment runners."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import yaml

from gui.data.baseline import _checked, _number, _path, _read, _sha
from gui.data.models import (
    ConfirmatoryComparisonRow,
    ConfirmatoryStudySummary,
    DefectAuditSummary,
)

CONFIG = "configs/experiments/experiment_01_discovery_corrected_v3.yaml"
PROVENANCE = "configs/experiments/experiment_01_discovery_corrected_v3.provenance.json"
RESULTS = "results/confirmatory_discovery_v3"
AUDIT = "gui/data/evidence/discovery_count_audit.json"
METRICS = frozenset(
    {
        "success_rate",
        "steps_to_success",
        "native_return",
        "failed_actions",
        "mean_cvss_exploited",
    }
)
ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, yaml.YAMLError)


def _json(path: Path) -> dict:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError("expected an evidence object")
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _seeds(value) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError("missing or excessive seed set")
    if any(type(seed) is not int or seed < 0 for seed in value) or len(set(value)) != len(value):
        raise ValueError("invalid seed set")
    return tuple(value)


def _commit(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", value):
        raise ValueError("missing recorded code commit")
    return value


def load_defect_audit(root: Path) -> DefectAuditSummary:
    try:
        path = _path(root, AUDIT)
        if not path.exists():
            return DefectAuditSummary()
        raw = _json(path)
        if raw["schema"] != "security-rl-discovery-audit-summary-v1":
            raise ValueError("unsupported audit schema")
        keys = (
            "affected_reward",
            "total_positive_reward",
            "affected_share_pct",
            "tactic_share_pct",
        )
        if type(raw["affected_scan_count"]) is not int or raw["affected_scan_count"] < 0:
            raise ValueError("invalid affected scan count")
        if any(not _number(raw[key]) or raw[key] < 0 for key in keys):
            raise ValueError("invalid audit magnitude")
        if raw["total_positive_reward"] <= 0 or any(
            raw[key] > 100 for key in ("affected_share_pct", "tactic_share_pct")
        ):
            raise ValueError("invalid audit share")
        if not all(
            isinstance(raw[key], str) and raw[key].strip() for key in ("interpretation", "source")
        ):
            raise ValueError("missing audit attribution")
        return DefectAuditSummary(
            affected_scan_count=raw["affected_scan_count"],
            **{key: raw[key] for key in keys},
            interpretation=raw["interpretation"],
            evidence_available=True,
            source=f"{AUDIT} — {raw['source']}",
        )
    except ERRORS as exc:
        return DefectAuditSummary(interpretation=f"Audit evidence unavailable: {exc}")


class EvidenceReader:
    """Bound one desktop refresh; never deserialize policies or run analysis."""

    def __init__(self):
        self.bytes = 0

    def checked(self, path: Path, expected: str) -> bytes:
        data = _checked(path, expected)
        self.bytes += len(data)
        if self.bytes > 128 * 1024 * 1024:
            raise ValueError("evidence exceeds desktop refresh budget")
        return data


def _completed(reader, directory, config, seed, commit, arm="shaped"):
    complete = _json(_path(directory, "complete.json"))
    rollout = config["ppo"]["n_steps"]
    expected_steps = math.ceil(config["total_timesteps"] / rollout) * rollout
    if (
        complete["status"] != "complete"
        or complete["seed"] != seed
        or complete["reward_mode"] != arm
        or complete["actual_timesteps"] != expected_steps
        or complete["config_sha256"] != _digest(config)
    ):
        raise ValueError(f"inconsistent completion metadata for seed {seed}")
    files = complete["files"]
    if set(files) != {"model.zip", "manifest.json", "episodes.csv", "diagnostics.csv"}:
        raise ValueError("incomplete training evidence file set")
    contents = {
        name: reader.checked(_path(directory, name), expected) for name, expected in files.items()
    }
    manifest = json.loads(contents["manifest.json"])
    if (
        manifest["training_seed"] != seed
        or manifest["git_commit"] != commit
        or manifest["git_dirty"] is not False
    ):
        raise ValueError("training source provenance mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", complete["policy_sha256"]):
        raise ValueError("missing recorded policy digest")
    return complete


def load_confirmatory(root: Path) -> ConfirmatoryStudySummary:
    result = ConfirmatoryStudySummary(audit=load_defect_audit(root))
    try:
        config_path = _path(root, CONFIG)
        if not config_path.exists():
            return result
        config_bytes = _read(config_path)
        config = yaml.safe_load(config_bytes)
        study_id = config["id"]
        if not isinstance(study_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,100}", study_id):
            raise ValueError("invalid study identifier")
        training, evaluation = _seeds(config["training_seeds"]), _seeds(config["evaluation_seeds"])
        if set(training) & set(evaluation):
            raise ValueError("training and evaluation seed sets overlap")
        if type(config["total_timesteps"]) is not int or config["total_timesteps"] <= 0:
            raise ValueError("invalid training budget")
        if type(config["ppo"]["n_steps"]) is not int or config["ppo"]["n_steps"] < 1:
            raise ValueError("invalid rollout size")
        provenance_path = _path(root, PROVENANCE)
        provenance = _json(provenance_path)
        result = replace(
            result,
            study_id=study_id,
            stage="Prepared",
            training_seeds=training,
            evaluation_seeds=evaluation,
            total_timesteps=config["total_timesteps"],
            parent_commit=_commit(provenance["parent"]["git_commit"]),
            status_message="Configuration available. Registration not yet available.",
        )
        package = _path(root, f"{RESULTS}/{study_id}")
        paths = [_path(package, "registration.json"), _path(root, f"{RESULTS}/registration.json")]
        present = [path for path in paths if path.exists()]
        if not present:
            return result
        registration = _json(present[0])
        if any(_json(path) != registration for path in present[1:]):
            raise ValueError("conflicting registration locations")
        if (
            registration["config"] != config
            or registration["inputs"]["config_sha256"] != _sha(config_bytes)
            or registration["confirmatory"]
            != provenance | {"protocol_sha256": _sha(_read(provenance_path))}
        ):
            raise ValueError("registration does not match mounted configuration/provenance")
        commit = _commit(registration["inputs"]["git_commit"])
        result = replace(
            result,
            registered=True,
            corrected_commit=commit,
            stage="Registered",
            status_message="Registration recorded; training outputs not yet available.",
        )
        reader = EvidenceReader()
        completed, notices = [], []
        for seed in training:
            directory = _path(package, f"shaped-{seed}")
            if not _path(directory, "complete.json").exists():
                continue
            try:
                _completed(reader, directory, config, seed, commit)
                completed.append(seed)
            except ERRORS as exc:
                notices.append(f"Seed {seed} evidence unavailable: {exc}")
        result = replace(
            result,
            completed_training_seeds=tuple(completed),
            stage="Training outputs available" if completed else "Registered",
            status_message=(
                f"{len(completed)}/{len(training)} shaped seed completions verified. "
                "File evidence only; no new training or database audit. " + " ".join(notices)
            ),
        )
        assessment_path = _path(package, "assessment.json")
        if assessment_path.exists():
            assessment = _json(assessment_path)
            if (
                len(completed) != len(training)
                or type(assessment["passed"]) is not bool
                or assessment["registration_sha256"] != _sha(_read(present[0]))
                or set(assessment["per_seed"]) != {str(seed) for seed in training}
            ):
                raise ValueError("assessment is not bound to complete registered runs")
            for seed in training:
                reader.checked(
                    _path(package, f"shaped-{seed}/complete.json"),
                    assessment["run_evidence"][str(seed)],
                )
            result = replace(
                result,
                convergence_status="Recorded PASS" if assessment["passed"] else "Recorded FAIL",
            )
        matched_path = _path(package, "final-evaluation/matched_outcomes.json")
        if matched_path.exists():
            if result.convergence_status != "Recorded PASS":
                raise ValueError("evaluation lacks recorded passing convergence evidence")
            matched = _json(matched_path)
            expected = {
                f"{arm}-{seed}/{name}"
                for arm in ("shaped", "sparse")
                for seed in training
                for name in ("evaluation.csv", "steps.jsonl", "evaluation_metadata.json")
            }
            if set(matched["files"]) != expected:
                raise ValueError("incomplete evaluation evidence")
            frozen = json.loads(
                reader.checked(
                    _path(root, f"{RESULTS}/first_stable.json"), matched["frozen_sha256"]
                )
            )
            if (
                frozen["registration_sha256"] != _sha(_read(present[0]))
                or frozen["assessment_sha256"] != _sha(_read(assessment_path))
                or frozen["config"] != config
            ):
                raise ValueError("frozen candidate provenance mismatch")
            keys = {
                (row["reward_mode"], row["training_seed"], row["evaluation_seed"])
                for row in matched["rows"]
            }
            expected_keys = {
                (arm, seed, episode)
                for arm in ("shaped", "sparse")
                for seed in training
                for episode in evaluation
            }
            if keys != expected_keys or len(matched["rows"]) != len(expected_keys):
                raise ValueError("incomplete or duplicate matched evaluation outcomes")
            for name, expected_hash in matched["files"].items():
                blob = reader.checked(_path(package, f"final-evaluation/{name}"), expected_hash)
                if name.endswith("evaluation_metadata.json"):
                    meta = json.loads(blob)
                    arm, seed_text = name.split("/")[0].split("-")
                    trained = _completed(
                        reader,
                        _path(package, f"{arm}-{seed_text}"),
                        config,
                        int(seed_text),
                        commit,
                        arm,
                    )
                    if (
                        meta["gradient_updates"] is not False
                        or meta["normalization_updates"] is not False
                        or meta["policy_sha256_before"] != meta["policy_sha256_after"]
                        or meta["policy_sha256_before"] != trained["policy_sha256"]
                        or meta["checkpoint_sha256"] != trained["files"]["model.zip"]
                        or meta["training_seed"] != int(seed_text)
                        or meta["registration"] != registration
                        or meta["evaluation_seeds"] != list(evaluation)
                        or meta["action_selection"] != config["evaluation"]["action_selection"]
                    ):
                        raise ValueError("evaluation policy or seed protocol mismatch")
            result = replace(
                result,
                stage="Evaluation complete",
                evaluation_status="Evaluation complete (recorded evidence)",
            )
        comparison_path = _path(package, "comparison.json")
        if comparison_path.exists():
            if not matched_path.exists():
                raise ValueError("comparison has no evaluation evidence")
            comparison = _json(comparison_path)
            if comparison["schema"] != "security-rl-discovery-confirmatory-comparison-v1":
                raise ValueError("unsupported comparison schema")
            refs = comparison["provenance"]
            if (
                refs["registration_sha256"] != _sha(_read(present[0]))
                or refs["code_commit"] != commit
                or refs["corrected_matched_outcomes_sha256"] != _sha(_read(matched_path))
            ):
                raise ValueError("comparison provenance mismatch")
            baseline = "results/convergence_v2/final_baseline_evaluation/report"
            for name in ("reviewed_manifest", "reviewed_results"):
                expected_hash = refs[f"baseline_{name}_sha256"]
                if expected_hash != provenance["parent"][f"{name}_sha256"]:
                    raise ValueError("comparison frozen baseline reference mismatch")
                reader.checked(_path(root, f"{baseline}/{name}.json"), expected_hash)
            rows = []
            for row in comparison["table"]:
                if row["metric"] not in METRICS:
                    continue
                if row["arm"] not in {"shaped", "sparse"}:
                    raise ValueError("unknown comparison arm")
                values = [row[key] for key in ("frozen", "corrected", "corrected_minus_frozen")]
                if any(value is not None and not _number(value) for value in values):
                    raise ValueError("non-finite comparison value")
                a, b, delta = values
                if a is None or b is None:
                    if delta is not None:
                        raise ValueError("difference supplied for unavailable outcome")
                elif delta is None or not math.isclose(delta, b - a, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("comparison difference does not match outcomes")
                rows.append(ConfirmatoryComparisonRow(row["arm"], row["metric"], a, b, delta))
            if len(rows) != 2 * len(METRICS) or {(row.arm, row.metric) for row in rows} != {
                (arm, metric) for arm in ("shaped", "sparse") for metric in METRICS
            }:
                raise ValueError("incomplete or duplicate native comparison metrics")
            result = replace(
                result,
                stage="Comparison available",
                comparisons=tuple(rows),
                comparison_available=True,
                comparison_path=str(comparison_path),
            )
        return result
    except ERRORS as exc:
        return replace(
            result,
            comparisons=(),
            comparison_available=False,
            comparison_path="",
            status_message=f"Evidence unavailable: {exc}",
        )
