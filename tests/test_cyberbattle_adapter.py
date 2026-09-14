from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cyberbattle")

from rlredteam.attack_path_report import generate_report  # noqa: E402
from rlredteam.cyberbattle_adapter import (  # noqa: E402
    CYBERBATTLE_SOURCE_REVISION,
    CyberBattleAdapter,
    CyberBattleScenario,
)
from rlredteam.cyberbattle_study import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    protected_artifact_hash,
    scripted_smoke,
)
from rlredteam.evaluation import policy_digest  # noqa: E402
from rlredteam.frameworks import map_simulator_behavior  # noqa: E402
from rlredteam.simulator_adapter import SimulatorAdapter  # noqa: E402


@pytest.fixture
def adapter():
    value = CyberBattleAdapter(CyberBattleScenario())
    value.reset(seed=42)
    yield value
    value.close()


def test_adapter_contract_and_partially_observed_knowledge(adapter) -> None:
    assert isinstance(adapter, SimulatorAdapter)
    assert adapter.agent_knowledge.discovery_order == ["start"]
    assert adapter.observation_space.contains(
        adapter.convert_observation(adapter._native_observation)
    )
    assert adapter.space_summary()["policy_observation_source"].startswith(
        "AgentKnowledge"
    )


def test_hidden_native_fields_cannot_change_policy_observation(adapter) -> None:
    original = adapter.convert_observation(adapter._native_observation)
    poisoned = dict(adapter._native_observation)
    poisoned["hidden_topology"] = {"secret-node": ["all-vulnerabilities"]}
    poisoned["true_nodes"] = ["crown-jewel"]
    converted = adapter.convert_observation(poisoned)
    np.testing.assert_array_equal(original, converted)


def test_observation_and_action_conversion_are_deterministic(adapter) -> None:
    left_observation = adapter.convert_observation(adapter._native_observation)
    right_observation = adapter.convert_observation(adapter._native_observation)
    np.testing.assert_array_equal(left_observation, right_observation)
    action_index = next(
        item.index
        for item in adapter.action_catalogue()
        if item.simulator_vulnerability_id == "ScanExplorerRecentFiles"
    )
    left = adapter.to_native_action(action_index)
    right = adapter.to_native_action(action_index)
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_native_vulnerability_is_retained_without_inventing_cve(adapter) -> None:
    action_index = next(
        item.index
        for item in adapter.action_catalogue()
        if item.simulator_vulnerability_id == "ScanExplorerRecentFiles"
    )
    _, _, _, _, info = adapter.step(action_index)
    transition = info["simulator_transition"]
    event = transition.to_trajectory_event()
    assert event["simulator_vulnerability_id"] == "ScanExplorerRecentFiles"
    assert event["cve_id"] is None
    assert event["framework_mappings"] == []
    assert "1_LinuxNode" in adapter.agent_knowledge.discovered
    assert "SSH" in adapter.agent_knowledge.known_services["1_LinuxNode"]


def test_mitre_mapping_is_catalogue_derived_after_semantic_conversion(adapter) -> None:
    local = next(
        item.index
        for item in adapter.action_catalogue()
        if item.simulator_vulnerability_id == "ScanExplorerRecentFiles"
    )
    connect = next(
        item.index for item in adapter.action_catalogue() if item.native_kind == "connect"
    )
    adapter.step(local)
    _, _, _, _, info = adapter.step(connect)
    event = info["simulator_transition"].to_trajectory_event()
    expected = [item.as_dict() for item in map_simulator_behavior("pivot")]
    assert event["action_kind"] == "pivot"
    assert event["framework_mappings"] == expected
    assert event["framework_mappings"][0]["technique_id"] == "T1021"


def test_phase14_report_accepts_cyberbattle_trajectory(tmp_path: Path) -> None:
    before = protected_artifact_hash()
    summary = scripted_smoke(result_dir=tmp_path)
    assert protected_artifact_hash() == before
    trajectory = json.loads((tmp_path / "example_trajectory.json").read_text())
    assert trajectory["provenance"]["cyberbattle_source_revision"] == (
        CYBERBATTLE_SOURCE_REVISION
    )
    report = generate_report(
        tmp_path / "example_trajectory.json",
        mode="simulation_report",
        experiment_id="test-cyberbattle",
    )
    assert summary["goal_reached"] is True
    assert report.facts
    episode = trajectory["trajectories"][0]
    assert len(episode["agent_knowledge_graph"]["nodes"]) == 4
    assert len(episode["agent_knowledge_graph"]["edges"]) == 3
    assert len(episode["reconstructed_attack_path"]) == 6
    assert {fact.technique_id for fact in report.facts if fact.technique_id} == {"T1021"}
    assert all(fact.cve_id is None for fact in report.facts)


def test_deterministic_policy_evaluation_does_not_update_weights(adapter) -> None:
    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        adapter,
        seed=77,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        device="cpu",
        verbose=0,
    )
    before = policy_digest(model)
    observation, _ = adapter.reset(seed=2101)
    for _ in range(8):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = adapter.step(int(action))
        if terminated or truncated:
            break
    assert policy_digest(model) == before


def test_fixed_held_out_protocol_is_disjoint_and_versioned() -> None:
    raw = load_config(DEFAULT_CONFIG)
    assert not set(raw["train_seeds"]) & set(raw["evaluation_seeds"])
    assert raw["train_scenario"] != raw["held_out_scenario"]
    assert raw["simulator_source_revision"] == CYBERBATTLE_SOURCE_REVISION
