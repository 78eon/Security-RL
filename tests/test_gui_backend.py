from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from gui.backend import ApplicationBackend
from gui.data.repository import RepositoryError


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
                rl_action_index=17,
                target=(2, 3),
                action_name="simulated-service-action",
                simulator_action="simulated-service-action",
                action_kind="exploit",
                framework_mappings=[
                    {
                        "framework": "attack-enterprise",
                        "technique_id": "T1210",
                        "technique_name": "Exploitation of Remote Services",
                    }
                ],
                mapping_persisted=True,
                technique_id="T1190",
                cve_id="CVE-2024-3400",
                success=True,
                state_changed=True,
                prerequisites=["known_vulnerability:CVE-2024-3400"],
                outcomes=["root_access:(2, 3)"],
                cvss_base=10.0,
            )
        ]


class FailingRepository:
    settings = SimpleNamespace(label="offline@localhost:5433")

    def list_runs(self):
        raise RepositoryError("Cannot reach PostgreSQL", "connection refused")


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
                        "rl_action_index": 17,
                        "simulator_action": "simulated-service-action",
                        "framework_mappings": [
                            {
                                "framework": "attack-enterprise",
                                "technique_id": "T1210",
                                "technique_name": "Exploitation of Remote Services",
                            }
                        ],
                        "mapping_persisted": True,
                        "target": "(2, 3)",
                        "success": True,
                        "state_changed": True,
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


def test_backend_exports_latest_canonical_analysis_artifact(tmp_path, monkeypatch) -> None:
    table = tmp_path / "tables" / "statistics.csv"
    table.parent.mkdir()
    table.write_text("metric,p_value\n")
    monkeypatch.setattr(
        "gui.backend.load_studies",
        lambda: [SimpleNamespace(result_path=str(tmp_path))],
    )

    path = ApplicationBackend(repository=FakeRepository()).export_report()

    assert path == str(table)


def test_database_campaigns_take_precedence_over_duplicate_artifacts() -> None:
    database = SimpleNamespace(
        name="same-run",
        reward_mode="adaptive",
        status="complete",
        episode_count=60,
        seed_label="601",
        mean_native_reward=155.0,
        success_rate=1.0,
    )
    artifact = SimpleNamespace(name="same-run")
    artifact_only = SimpleNamespace(name="local-only")

    campaigns = ApplicationBackend._merge_campaigns(
        [database], [artifact, artifact_only]
    )

    assert [campaign.name for campaign in campaigns] == ["same-run", "local-only"]
    assert campaigns[0].reward_mode == "adaptive"
    assert campaigns[0].episodes == 60


def test_dashboard_explicitly_reports_database_degradation(monkeypatch) -> None:
    monkeypatch.setattr("gui.backend.list_run_folders", lambda: [])
    monkeypatch.setattr("gui.backend.load_studies", lambda: [])

    dashboard = ApplicationBackend(repository=FailingRepository()).load_dashboard()

    assert dashboard.source_status == "Artefact mode · Cannot reach PostgreSQL"
    assert dashboard.database_label == "offline@localhost:5433"
    assert dashboard.campaigns == []


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
    assert all(event["rl_action_index"] >= 0 for event in result.events)
    assert all(event["simulator_action"] == event["action"] for event in result.events)
    assert all(
        mapping["framework"] != "atlas"
        for event in result.events
        for mapping in event["framework_mappings"]
    )
    unsupported = next(
        event for event in result.events if event["action_kind"] == "obtain_credential"
    )
    assert unsupported["framework_mappings"] == []
