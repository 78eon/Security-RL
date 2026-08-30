from __future__ import annotations

import socket
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
                step_idx=0,
                target=(2, 3),
                action_name="simulated-service-action",
                action_kind="exploit",
                technique_id="T1190",
                cve_id="CVE-2024-3400",
                success=True,
                state_changed=True,
                prerequisites=["known_vulnerability:CVE-2024-3400"],
                outcomes=["root_access:(2, 3)"],
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
            "trajectory": [
                {
                    "step": 0,
                    "action": "exploit",
                    "target": "(2, 3)",
                    "outcomes": ["root_access:(2, 3)"],
                }
            ],
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


def test_backend_profiles_come_from_enterprise_model() -> None:
    profiles = ApplicationBackend(repository=FakeRepository()).simulation_profiles()

    assert [item["id"] for item in profiles] == [
        "on_premises",
        "legacy",
        "cloud",
        "hybrid",
    ]


def test_backend_runs_typed_graph_simulation_without_network(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("desktop simulation attempted network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = ApplicationBackend(repository=FakeRepository()).run_simulation("hybrid", 2001)

    assert result.profile == "hybrid"
    assert result.topology_seed == 2001
    assert result.goal_reached
    assert len(result.topology_hash) == 64
    assert {node["type"] for node in result.nodes} >= {
        "legacy_host",
        "cloud_workload",
        "cloud_network",
        "network_segment",
    }
    assert result.trajectory[-1]["action_kind"] == "access_asset"
    assert len(result.trajectory) < result.episode_steps
