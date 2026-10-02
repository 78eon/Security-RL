"""Synthetic frontend evidence only; no training, database or scientific reruns."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gui.data.confirmatory import (
    AUDIT,
    CONFIG,
    METRICS,
    PROVENANCE,
    RESULTS,
    _digest,
    load_confirmatory,
    load_defect_audit,
)
from gui.data.models import ConfirmatoryComparisonRow, ConfirmatoryStudySummary


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return sha(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    config = {
        "id": "synthetic_study",
        "training_seeds": [42, 43],
        "evaluation_seeds": [1001, 1002],
        "total_timesteps": 8,
        "ppo": {"n_steps": 4},
        "evaluation": {"action_selection": "stochastic"},
    }
    put(tmp_path / CONFIG, config)
    baseline = tmp_path / "results/convergence_v2/final_baseline_evaluation/report"
    parent = {"git_commit": "a" * 40}
    for name in ("reviewed_manifest", "reviewed_results"):
        parent[name + "_sha256"] = put(baseline / (name + ".json"), {"fixture": name})
    provenance = {"parent": parent}
    put(tmp_path / PROVENANCE, provenance)
    put(tmp_path / AUDIT, json.loads((Path(__file__).parents[1] / AUDIT).read_text()))
    return tmp_path, config, provenance


def register(evidence, flat=False):
    root, config, provenance = evidence
    package = root / RESULTS / config["id"]
    registration = {
        "config": config,
        "inputs": {"config_sha256": sha(root / CONFIG), "git_commit": "b" * 40},
        "confirmatory": provenance | {"protocol_sha256": sha(root / PROVENANCE)},
    }
    path = (root / RESULTS if flat else package) / "registration.json"
    put(path, registration)
    return package, registration, path


def complete(package, config, seed, arm="shaped"):
    directory = package / f"{arm}-{seed}"
    files = {
        name: put(directory / name, {"fixture": name})
        for name in ("model.zip", "episodes.csv", "diagnostics.csv")
    }
    files["manifest.json"] = put(
        directory / "manifest.json",
        {"training_seed": seed, "git_commit": "b" * 40, "git_dirty": False},
    )
    return put(
        directory / "complete.json",
        {
            "status": "complete",
            "seed": seed,
            "reward_mode": arm,
            "actual_timesteps": 8,
            "config_sha256": _digest(config),
            "policy_sha256": "c" * 64,
            "files": files,
        },
    )


def comparison_fixture(evidence):
    root, config, provenance = evidence
    package, registration, reg_path = register(evidence)
    runs = {str(s): complete(package, config, s) for s in config["training_seeds"]}
    assessment = put(
        package / "assessment.json",
        {
            "passed": True,
            "registration_sha256": sha(reg_path),
            "run_evidence": runs,
            "per_seed": {s: {"passed": True} for s in runs},
        },
    )
    frozen = put(
        root / RESULTS / "first_stable.json",
        {"registration_sha256": sha(reg_path), "assessment_sha256": assessment, "config": config},
    )
    files, rows = {}, []
    for arm in ("shaped", "sparse"):
        for seed in config["training_seeds"]:
            complete(package, config, seed, arm)
            prefix = f"{arm}-{seed}"
            for name in ("evaluation.csv", "steps.jsonl"):
                files[f"{prefix}/{name}"] = put(package / "final-evaluation" / prefix / name, {})
            files[f"{prefix}/evaluation_metadata.json"] = put(
                package / "final-evaluation" / prefix / "evaluation_metadata.json",
                {
                    "gradient_updates": False,
                    "normalization_updates": False,
                    "policy_sha256_before": "c" * 64,
                    "policy_sha256_after": "c" * 64,
                    "checkpoint_sha256": sha(package / prefix / "model.zip"),
                    "registration": registration,
                    "training_seed": seed,
                    "evaluation_seeds": config["evaluation_seeds"],
                    "action_selection": "stochastic",
                },
            )
            rows.extend(
                {"reward_mode": arm, "training_seed": seed, "evaluation_seed": s}
                for s in config["evaluation_seeds"]
            )
    matched = put(
        package / "final-evaluation/matched_outcomes.json",
        {"files": files, "rows": rows, "frozen_sha256": frozen},
    )
    comparison = {
        "schema": "security-rl-discovery-confirmatory-comparison-v1",
        "provenance": {
            "registration_sha256": sha(reg_path),
            "code_commit": "b" * 40,
            "corrected_matched_outcomes_sha256": matched,
            **{
                f"baseline_{name}_sha256": provenance["parent"][f"{name}_sha256"]
                for name in ("reviewed_manifest", "reviewed_results")
            },
        },
        "table": [
            {"arm": arm, "metric": metric, "frozen": 1, "corrected": 2, "corrected_minus_frozen": 1}
            for arm in ("shaped", "sparse")
            for metric in sorted(METRICS)
        ],
    }
    put(package / "comparison.json", comparison)
    return package, comparison


def test_audit_and_prepared_configuration(evidence):
    root, _, _ = evidence
    audit = load_defect_audit(root)
    assert audit.evidence_available and audit.affected_scan_count == 16598
    assert audit.total_positive_reward == 81866671.368
    assert audit.affected_share_pct == 0.0203 and audit.tactic_share_pct == 1.334
    data = load_confirmatory(root)
    assert data.stage == "Prepared" and not data.registered
    assert data.training_seeds == (42, 43) and data.evaluation_seeds == (1001, 1002)
    assert data.completed_training_seeds == () and not data.comparison_available
    assert data.corrected_commit == "Not yet available"


@pytest.mark.parametrize("flat", [True, False])
def test_progress_requires_persisted_hashed_outputs(evidence, flat):
    root, config, _ = evidence
    package, _, _ = register(evidence, flat)
    complete(package, config, 42)
    put(package / "shaped-43/complete.json", {"status": "complete"})
    data = load_confirmatory(root)
    assert data.registered and data.completed_training_seeds == (42,)
    assert data.convergence_status == "Not yet available"
    (package / "shaped-42/model.zip").write_text("tampered")
    assert load_confirmatory(root).completed_training_seeds == ()


@pytest.mark.parametrize("path", [CONFIG, PROVENANCE, AUDIT])
def test_malformed_evidence_fails_safely(evidence, path):
    root, _, _ = evidence
    (root / path).write_text("[broken")
    data = load_confirmatory(root)
    assert not data.comparison_available
    if path == AUDIT:
        assert not data.audit.evidence_available
    else:
        assert "unavailable" in data.status_message


def test_comparison_loads_only_native_metrics_read_only(evidence):
    root, _, _ = evidence
    package, comparison = comparison_fixture(evidence)
    comparison["table"].append({"metric": "discovered_hosts", "arm": "shaped", "frozen": 999})
    put(package / "comparison.json", comparison)
    before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    data = load_confirmatory(root)
    assert data.comparison_available, data.status_message
    assert len(data.comparisons) == 10
    assert {row.metric for row in data.comparisons} == METRICS
    assert data.convergence_status == "Recorded PASS"
    assert data.evaluation_status.startswith("Evaluation complete")
    assert before == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "target",
    [
        "final-evaluation/shaped-42/evaluation_metadata.json",
        "shaped-42/model.zip",
        "assessment.json",
    ],
)
def test_tampered_chain_withholds_comparison(evidence, target):
    root, _, _ = evidence
    package, _ = comparison_fixture(evidence)
    (package / target).write_text("{}")
    assert not load_confirmatory(root).comparison_available


def test_escaping_source_is_rejected(evidence, tmp_path):
    root, _, _ = evidence
    source = root / PROVENANCE
    source.unlink()
    source.symlink_to(root.parent / "outside.json")
    assert "unavailable" in load_confirmatory(root).status_message


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_page_missing_and_completed_comparison_and_stale_reset(app, evidence):
    from gui.views.confirmatory import ConfirmatoryPage

    root, _, _ = evidence
    page = ConfirmatoryPage()
    page.apply_confirmatory(load_confirmatory(root))
    assert page.comparison_note.text() == "Confirmatory comparison not yet available."
    assert page.progress.value() == 0
    comparison_fixture(evidence)
    data = load_confirmatory(root)
    extra = ConfirmatoryComparisonRow("shaped", "discovered_hosts", 99, 999, 900)
    page.apply_confirmatory(replace(data, comparisons=data.comparisons + (extra,)))
    assert page.comparison.rowCount() == 10
    assert {page.comparison.item(i, 1).text() for i in range(10)} == METRICS
    assert page.comparison.item(0, 2).text() != page.comparison.item(0, 3).text()
    page.apply_confirmatory(ConfirmatoryStudySummary())
    assert page.comparison.rowCount() == 0 and page.audit.rowCount() == 0
    page.close()


def test_console_file_evidence_survives_database_failure(app, evidence, monkeypatch):
    from gui.backend import ApplicationBackend
    from gui.data.baseline import BaselineData
    from gui.views import research_console as console
    from tests.test_gui_console import FakeBackend

    root, _, _ = evidence
    monkeypatch.setattr("gui.backend.REPO_ROOT", root)
    assert ApplicationBackend(repository=object()).load_confirmatory().stage == "Prepared"

    def synchronous(fn, success, failure):
        try:
            result = fn()
        except Exception as exc:
            failure(str(exc), "test")
        else:
            success(result)

    class Offline(FakeBackend):
        def load_confirmatory(self):
            return load_confirmatory(root)

        def load_dashboard(self):
            raise ConnectionError("PostgreSQL offline")

    monkeypatch.setattr(console, "run_async", synchronous)
    window = console.MainWindow(Offline())
    assert window.pages[9].data.stage == "Prepared"
    assert window.pages[8].data.summary == {}
    assert "PostgreSQL offline" in window.statusBar().currentMessage()
    baseline = BaselineData(
        state="REVIEWED",
        summary={"shaped": {"native_return": 111.49}, "sparse": {"native_return": -9.36}},
    )
    window.apply_baseline(baseline)
    before = [window.pages[8].outcomes.item(i, 2).text()
              for i in range(window.pages[8].outcomes.rowCount())]
    comparison_fixture(evidence)
    window.pages[9].apply_confirmatory(load_confirmatory(root))
    assert window.pages[9].comparison.rowCount() == 10
    assert window.pages[8].data is baseline
    assert before == [window.pages[8].outcomes.item(i, 2).text()
                      for i in range(window.pages[8].outcomes.rowCount())]
    window.close()
