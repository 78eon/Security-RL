from __future__ import annotations

from types import SimpleNamespace

import pytest

from gui.backend import ApplicationBackend


class FakeRepository:
    def list_runs(self):
        return [SimpleNamespace(experiment_id=7, has_steps=True)]

    def replayable_episodes(self, experiment_id: int):
        assert experiment_id == 7
        return [SimpleNamespace(episode_idx=42, episode_id=9)]

    def steps(self, episode_id: int):
        assert episode_id == 9
        return [
            SimpleNamespace(
                target=(2, 3),
                action_name="simulated-service-action",
                technique_id="T1190",
                cve_id="CVE-2024-3400",
                success=True,
                cvss_base=10.0,
            )
        ]


def test_backend_adapts_stored_steps_to_path_data() -> None:
    backend = ApplicationBackend(repository=FakeRepository())

    assert backend.refresh_paths() == [
        {
            "id": "AP-042",
            "target": "(2, 3)",
            "risk": "Critical",
            "steps": 1,
            "detection": "—",
            "confidence": "100%",
        }
    ]


def test_backend_does_not_fake_unsupported_controls() -> None:
    backend = ApplicationBackend(repository=FakeRepository())

    with pytest.raises(RuntimeError, match="scheduler"):
        backend.pause_campaign("EXP-09")
    with pytest.raises(RuntimeError, match="read-only"):
        backend.save_agent_config({"algorithm": "PPO"})


def test_backend_rejects_configuration_without_a_real_service() -> None:
    backend = ApplicationBackend(repository=FakeRepository())

    with pytest.raises(RuntimeError, match="configuration service"):
        backend.save_agent_config({"algorithm": "live-agent"})


def test_backend_exports_existing_analysis_artifact() -> None:
    path = ApplicationBackend(repository=FakeRepository()).export_report()

    assert path.endswith("runs/_analysis/results_table.txt")
