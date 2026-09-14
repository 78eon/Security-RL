from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("CybORG")

from rlredteam.attack_path_report import generate_report  # noqa: E402
from rlredteam.cyborg_adapter import (  # noqa: E402
    CYBORG_SOURCE_REVISION,
    CybORGAdapter,
    CybORGScenario,
)
from rlredteam.cyborg_study import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    protected_artifact_hash,
    scripted_actions,
    scripted_smoke,
)
from rlredteam.evaluation import policy_digest  # noqa: E402
from rlredteam.frameworks import map_simulator_behavior  # noqa: E402
from rlredteam.simulator_adapter import SimulatorAdapter  # noqa: E402


@pytest.fixture
def adapter():
    value = CybORGAdapter(CybORGScenario())
    value.reset(seed=42)
    yield value
    value.close()


def test_adapter_contract_and_initial_observation_is_partial(adapter) -> None:
    assert isinstance(adapter, SimulatorAdapter)
    assert len(adapter.agent_knowledge.discovered) == 2
    assert len(adapter.agent_knowledge.discovered) < adapter.scenario.maximum_node_count
    assert adapter.observation_space.contains(adapter.convert_observation({}))
    assert adapter.space_summary()["policy_observation_source"].startswith(
        "AgentKnowledge"
    )


def test_hidden_native_fields_cannot_change_policy_observation(adapter) -> None:
    original = adapter.convert_observation(adapter._native_observation)
    poisoned = dict(adapter._native_observation)
    poisoned["hidden_topology"] = {"Internal": ["all-vulnerabilities"]}
    poisoned["true_state"] = {"Defender": "owned"}
    converted = adapter.convert_observation(poisoned)
    np.testing.assert_array_equal(original, converted)


def test_source_contains_no_true_state_access() -> None:
    source = Path("src/rlredteam/cyborg_adapter.py").read_text()
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "get_true_state" not in attributes
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "state"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environment_controller"
        for node in ast.walk(tree)
    )


def test_observation_and_action_conversion_are_deterministic(adapter) -> None:
    np.testing.assert_array_equal(
        adapter.convert_observation(adapter._native_observation),
        adapter.convert_observation(adapter._native_observation),
    )
    left = adapter.to_native_action(1)
    right = adapter.to_native_action(1)
    assert type(left) is type(right)
    assert left.get_params() == right.get_params()


def test_native_exploit_identity_is_not_promoted_to_cve(adapter) -> None:
    for action in scripted_actions():
        _, _, terminated, truncated, info = adapter.step(action)
        event = info["simulator_transition"].to_trajectory_event()
        assert event["cve_id"] is None
        if event["simulator_vulnerability_id"]:
            assert event["simulator_vulnerability_id"] in {
                "SSHLoginExploit",
                "UpgradeToMeterpreter",
                "MS17_010_PSExec",
            }
        if terminated or truncated:
            break


def test_mitre_mapping_is_catalogue_derived_after_semantic_conversion(adapter) -> None:
    _, _, _, _, info = adapter.step(1)
    event = info["simulator_transition"].to_trajectory_event()
    assert event["action_kind"] == "enumerate_service"
    assert event["framework_mappings"] == [
        item.as_dict() for item in map_simulator_behavior("enumerate_service")
    ]
    assert event["framework_mappings"][0]["technique_id"] == "T1046"


def test_phase14_report_accepts_cyborg_trajectory(tmp_path: Path) -> None:
    before = protected_artifact_hash()
    summary = scripted_smoke(result_dir=tmp_path)
    assert protected_artifact_hash() == before
    trajectory = json.loads((tmp_path / "example_trajectory.json").read_text())
    assert trajectory["provenance"]["cyborg_source_revision"] == CYBORG_SOURCE_REVISION
    report = generate_report(
        tmp_path / "example_trajectory.json",
        mode="simulation_report",
        experiment_id="test-cyborg",
    )
    episode = trajectory["trajectories"][0]
    assert summary["goal_reached"] is True
    assert report.facts
    assert episode["reconstructed_attack_path"]
    assert len(episode["agent_knowledge_graph"]["nodes"]) == 3
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
    observation, _ = adapter.reset(seed=4101)
    for _ in range(8):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = adapter.step(int(action))
        if terminated or truncated:
            break
    assert policy_digest(model) == before


def test_fixed_train_and_evaluation_protocol_is_disjoint() -> None:
    raw = load_config(DEFAULT_CONFIG)
    assert not set(raw["train_seeds"]) & set(raw["evaluation_seeds"])
    assert raw["simulator_source_revision"] == CYBORG_SOURCE_REVISION
    assert raw["scenario"]["source_file"] == "Scenario1.yaml"
