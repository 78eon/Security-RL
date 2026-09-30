"""Synthetic GUI evidence fixtures only; never train or change frozen output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gui.data.baseline import PACKAGE, BaselineData, load_baseline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return digest(path)


def reseal(root):
    report = root / PACKAGE / "report"
    put(
        report / "reviewed_manifest.json",
        {
            "files": {
                p.name: digest(p) for p in report.iterdir() if p.name != "reviewed_manifest.json"
            }
        },
    )


@pytest.fixture
def evidence(tmp_path):
    package = tmp_path / PACKAGE
    report = package / "report"
    quality = put(
        report / "metric_quality_addendum.json",
        {
            "reward_impact": "Training reward eligibility was affected.",
            "caution": "Discovery counts are withheld.",
        },
    )
    original = put(report / "results.json", {"legacy_discovered_hosts": 99999})
    names = [
        "native_return",
        "native_return_std",
        "success_rate",
        "steps_to_goal",
        "mean_cvss_exploited",
    ]
    summary = {
        arm: {
            "success_rate": 0.5,
            "steps_to_goal": 23,
            "native_return": 17,
            "failed_actions": 8,
            "discovered_hosts": None,
            "attack_path_length": None,
        }
        for arm in ("sparse", "shaped")
    }
    put(
        report / "reviewed_results.json",
        {
            "policy_pairs": 2,
            "total_episodes": 12,
            "episodes_per_policy": 3,
            "summary": summary,
            "interpretation": "Recorded policy outcomes only.",
            "metric_quality_addendum_sha256": quality,
            "original_results_sha256": original,
            "primary_statistics": [
                {
                    "metric": name,
                    "mean_a": 17,
                    "mean_b": 18,
                    "difference": 1,
                    "p_value": 0.2,
                    "p_bonferroni": 1.0,
                    "ci_low": -2,
                    "ci_high": 4,
                    "n_pairs": 2,
                    "significant": False,
                }
                for name in names
            ],
        },
    )
    (report / "primary_statistics.csv").write_text("metric,p_value\nnative_return,0.2\n")
    reseal(tmp_path)
    candidate = tmp_path / "results/convergence_v2/example"
    registration_hash = put(
        candidate / "registration.json", {"inputs": {"config_sha256": "c" * 64}}
    )
    assessment_hash = put(candidate / "assessment.json", {"passing_seeds": 2})
    frozen_hash = put(
        tmp_path / "results/convergence_v2/first_stable.json",
        {
            "candidate": "example",
            "assessment_sha256": assessment_hash,
        },
    )
    checkpoints = {str(seed): {"status": "complete", "actual_timesteps": 2048} for seed in (7, 8)}
    protocol_hash = put(
        package / "protocol.json",
        {
            "training_seeds": [7, 8],
            "primary_metrics": names,
            "frozen_shaped_sha256": frozen_hash,
            "registration_sha256": registration_hash,
            "shaped_checkpoints": checkpoints,
            "git_commit": "example-commit",
            "literal_topology_yaml_sha256": "a" * 64,
            "reconstructed_topology_sha256": "b" * 64,
            "cve_catalogue_sha256": "d" * 64,
            "evaluation_episode_seeds": [81, 82, 83],
            "evaluation": {"action_selection": "stochastic", "policy_updates": False},
            "normalization": {"enabled": False},
            "ppo": {"normalize_advantage": True},
        },
    )
    put(
        package / "sparse_frozen.json",
        {
            "protocol_sha256": protocol_hash,
            "checkpoints": checkpoints,
            "native_training_stability": {
                str(seed): {"native_return_stability_passed": True} for seed in (7, 8)
            },
        },
    )
    return tmp_path


def test_loader_reads_reviewed_data_without_fixed_outcomes_or_mutations(evidence):
    before = {str(p): digest(p) for p in evidence.rglob("*") if p.is_file()}
    data = load_baseline(evidence)
    assert data.state == "REVIEWED", data.detail
    assert data.policy_pairs == 2 and data.total_episodes == 12
    assert data.summary["shaped"]["native_return"] == 17
    assert data.training[0][3] == "2/2"
    assert data.summary["shaped"]["discovered_hosts"] is None
    assert "99999" not in repr(data)
    assert {str(p): digest(p) for p in evidence.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "relative",
    [
        "report/reviewed_results.json",
        "report/primary_statistics.csv",
        "protocol.json",
        "report/metric_quality_addendum.json",
    ],
)
def test_tamper_does_not_fallback_to_original(evidence, relative):
    (evidence / PACKAGE / relative).write_text("{}")
    data = load_baseline(evidence)
    assert data.state == "INVALID"
    assert not data.summary and not data.comparisons and not data.training


def test_missing_package_is_unavailable(tmp_path):
    assert load_baseline(tmp_path).state == "UNAVAILABLE"


def test_unbound_statistics_file_is_not_exported(evidence):
    path = evidence / PACKAGE / "report/reviewed_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["primary_statistics.csv"]
    put(path, manifest)
    assert load_baseline(evidence).state == "INVALID"


@pytest.mark.parametrize("mode", ["unwithheld", "nonfinite", "duplicate", "wrong_n"])
def test_invalid_semantics_even_with_matching_manifest(evidence, mode):
    path = evidence / PACKAGE / "report/reviewed_results.json"
    value = json.loads(path.read_text())
    if mode == "unwithheld":
        value["summary"]["shaped"]["discovered_hosts"] = 123
    elif mode == "nonfinite":
        value["primary_statistics"][0]["p_value"] = float("nan")
    elif mode == "duplicate":
        value["primary_statistics"][1] = value["primary_statistics"][0]
    else:
        value["primary_statistics"][0]["n_pairs"] = 12
    put(path, value)
    reseal(evidence)
    assert load_baseline(evidence).state == "INVALID"


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/outside.json"])
def test_manifest_path_traversal_rejected(evidence, name):
    path = evidence / PACKAGE / "report/reviewed_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = "a" * 64
    put(path, manifest)
    data = load_baseline(evidence)
    assert data.state == "INVALID" and "unsafe evidence path" in data.detail


def test_document_symlink_outside_root_not_available(evidence, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "private.md"
    outside.write_text("must not read")
    (evidence / "docs").mkdir()
    (evidence / "docs/SUPERVISOR_EVIDENCE_BUNDLE.md").symlink_to(outside)
    data = load_baseline(evidence)
    assert next(row for row in data.documents if row[0] == "Supervisor evidence bundle")[2] is False


def test_backend_prefers_reviewed_statistics_and_rejects_invalid(evidence, monkeypatch):
    from gui.backend import ApplicationBackend

    monkeypatch.setattr("gui.backend.REPO_ROOT", evidence)
    backend = ApplicationBackend()
    assert Path(backend.export_report()) == evidence / PACKAGE / "report/primary_statistics.csv"
    (evidence / PACKAGE / "report/reviewed_results.json").write_text("{}")
    with pytest.raises(ValueError, match="unavailable"):
        backend.export_report()


def test_native_page_withholds_metrics_and_clears_stale_snapshot(evidence):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.views.baseline import BaselinePage

    QApplication.instance() or QApplication([])
    page = BaselinePage(lambda message: None)
    page.apply_baseline(load_baseline(evidence))
    assert page.pairs.value.text() == "2"
    assert page.episodes.value.text() == "12"
    assert page.outcomes.item(7, 1).text() == "WITHHELD"
    assert page.statistics.rowCount() == 5
    assert "Training reward" in page.warning.text()
    page.apply_baseline(BaselineData(state="INVALID", detail="missing evidence"))
    assert page.outcomes.rowCount() == page.statistics.rowCount() == page.training.rowCount() == 0
    assert page.pairs.value.text() == "—"
    assert page.provenance.toPlainText() == ""
    assert not page.copy_button.isEnabled()
    page.close()


def test_supervisor_path_copies_host_path_without_execution(evidence, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.views.baseline import BaselinePage

    app = QApplication.instance() or QApplication([])
    path = evidence / "docs/SUPERVISOR_EVIDENCE_BUNDLE.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("local document")
    monkeypatch.setenv("RLREDTEAM_HOST_REPO", "/host/project with spaces")
    page = BaselinePage(lambda message: None)
    page.apply_baseline(load_baseline(evidence))
    page.documents.selectRow(2)
    assert page.copy_button.isEnabled()
    page.copy_button.click()
    assert app.clipboard().text() == "/host/project with spaces/docs/SUPERVISOR_EVIDENCE_BUNDLE.md"
    page.documents.selectRow(0)
    assert not page.copy_button.isEnabled()
    page.close()


def test_baseline_worker_loads_independently_and_failure_clears_old_values(evidence):
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    from gui.views.main_window import MainWindow

    class LocalOnlyBackend:
        def load_baseline(self):
            return load_baseline(evidence)

    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=LocalOnlyBackend())
    deadline = time.monotonic() + 3
    while window.pages[8].data.state != "REVIEWED" and time.monotonic() < deadline:
        app.processEvents()
    assert window.pages[8].pairs.value.text() == "2"
    assert "2 policy pairs" in window.pages[0].baseline_banner.message.text()
    window._baseline_failed("evidence removed", "")
    assert window.pages[8].statistics.rowCount() == 0
    assert "evidence removed" in window.pages[0].baseline_banner.message.text()
    QThreadPool.globalInstance().waitForDone()
    window.close()


def test_latest_statistics_path_rebased_to_host(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.views.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=object())
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RLREDTEAM_HOST_REPO", "/host/evidence")
    window._statistics_ready(str(root / PACKAGE / "report/primary_statistics.csv"))
    assert app.clipboard().text() == f"/host/evidence/{PACKAGE}/report/primary_statistics.csv"
    window.close()


def test_baseline_does_not_wait_for_database_dashboard(evidence):
    pytest.importorskip("PySide6")
    import threading
    import time

    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    from gui.backend import DashboardData
    from gui.views.main_window import MainWindow

    release = threading.Event()

    class BlockedDashboard:
        def load_baseline(self):
            return load_baseline(evidence)

        def load_dashboard(self):
            release.wait(timeout=5)
            return DashboardData(source_status="test", baseline=self.load_baseline())

    app = QApplication.instance() or QApplication([])
    window = MainWindow(backend=BlockedDashboard())
    try:
        deadline = time.monotonic() + 3
        while window.pages[8].data.state != "REVIEWED" and time.monotonic() < deadline:
            app.processEvents()
        assert window.pages[8].data.state == "REVIEWED"
        assert not window.refresh_button.isEnabled()
    finally:
        release.set()
        QThreadPool.globalInstance().waitForDone()
        app.processEvents()
        window.close()
