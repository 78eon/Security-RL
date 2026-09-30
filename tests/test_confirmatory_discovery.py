"""Confirmatory tooling stays descriptive and scoped to native outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_discovery_confirmatory as comparison  # noqa: E402

METRICS = comparison.METRICS
_summary = comparison._summary


def test_comparison_uses_success_only_steps_and_native_outcomes():
    rows = [
        {"goal_reached": True, "length": 10, "native_return": 30,
         "failed_actions": 2, "mean_cvss_exploited": 8.0},
        {"goal_reached": False, "length": 100, "native_return": -20,
         "failed_actions": 20, "mean_cvss_exploited": None},
    ]
    summary = _summary(rows)
    assert summary == {
        "success_rate": 0.5,
        "steps_to_success": 10,
        "native_return": 5,
        "failed_actions": 11,
        "mean_cvss_exploited": 8,
    }
    assert "discovered_hosts" not in METRICS
    assert "attack_path_length" not in METRICS


def test_no_success_has_unknown_steps_to_success():
    rows = [{"goal_reached": False, "length": 100, "native_return": -1,
             "failed_actions": 4, "mean_cvss_exploited": ""}]
    assert _summary(rows)["steps_to_success"] is None


def test_comparison_joins_only_matched_native_evaluation_rows(tmp_path, monkeypatch):
    report = tmp_path / "baseline"
    report.mkdir()
    root = tmp_path / "corrected"
    out = root / "candidate"
    directory = out / "final-evaluation"
    directory.mkdir(parents=True)
    config = {"id": "candidate", "training_seeds": [42],
              "evaluation_seeds": [1001, 1002]}
    monkeypatch.setattr(comparison, "BASELINE", report)
    monkeypatch.setattr(comparison, "load_config", lambda _: config)
    monkeypatch.setattr(
        comparison, "verify_registration",
        lambda *_: ({"inputs": {"git_commit": "corrected-commit"}}, out),
    )
    monkeypatch.setattr(comparison, "verify_evaluation", lambda *_: None)

    baseline_hashes = {}
    for arm in ("shaped", "sparse"):
        baseline_csv = report / f"{arm}_evaluation.csv"
        with baseline_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "training_seed", "evaluation_seed", "goal_reached", "length",
                "native_return", "failed_actions", "mean_cvss_exploited",
            ])
            writer.writeheader()
            for seed in (1001, 1002):
                writer.writerow({"training_seed": 42, "evaluation_seed": seed,
                                 "goal_reached": True, "length": 20,
                                 "native_return": 10, "failed_actions": 3,
                                 "mean_cvss_exploited": 8})
        baseline_hashes[baseline_csv.name] = hashlib.sha256(
            baseline_csv.read_bytes()
        ).hexdigest()
        episode_dir = directory / f"{arm}-42"
        episode_dir.mkdir()
        with (episode_dir / "evaluation.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "training_seed", "evaluation_seed", "goal_reached", "length",
                "native_return", "mean_cvss_exploited",
            ])
            writer.writeheader()
            for seed in (1001, 1002):
                writer.writerow({"training_seed": 42, "evaluation_seed": seed,
                                 "goal_reached": True, "length": 15,
                                 "native_return": 15, "mean_cvss_exploited": 9})
        (episode_dir / "steps.jsonl").write_text(
            json.dumps({"evaluation_seed": 1001, "success": False}) + "\n"
        )
    (report / "reviewed_results.json").write_text("{}")
    baseline_hashes["reviewed_results.json"] = hashlib.sha256(
        (report / "reviewed_results.json").read_bytes()
    ).hexdigest()
    (report / "reviewed_manifest.json").write_text(json.dumps({"files": baseline_hashes}))
    (out / "registration.json").write_text("{}")
    (directory / "matched_outcomes.json").write_text("{}")
    result = comparison.compare(tmp_path / "config.yaml", root)
    assert len(result["table"]) == 2 * len(METRICS)
    shaped = {row["metric"]: row for row in result["table"] if row["arm"] == "shaped"}
    assert shaped["success_rate"]["corrected_minus_frozen"] == 0
    assert shaped["steps_to_success"]["corrected_minus_frozen"] == -5
    assert shaped["failed_actions"]["corrected"] == 0.5
    assert "discovered_hosts" not in shaped
