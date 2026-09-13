from __future__ import annotations

import pytest

from rlredteam.frameworks import (
    AIBehavior,
    Framework,
    event_framework_fields,
    map_simulator_behavior,
    technique_progress,
)


@pytest.mark.parametrize(
    ("action_kind", "technique_id", "tactic_id"),
    [
        ("discover_network", "T1018", "TA0007"),
        ("enumerate_service", "T1046", "TA0007"),
        ("exploit", "T1210", "TA0008"),
        ("pivot", "T1021", "TA0008"),
        ("escalate_privilege", "T1068", "TA0004"),
        ("access_asset", "T1005", "TA0009"),
    ],
)
def test_enterprise_behaviour_maps_to_exact_attack_semantics(
    action_kind: str, technique_id: str, tactic_id: str
) -> None:
    mappings = map_simulator_behavior(action_kind)

    assert len(mappings) == 1
    assert mappings[0].framework is Framework.ATTACK_ENTERPRISE
    assert mappings[0].technique_id == technique_id
    assert mappings[0].tactic_id == tactic_id


@pytest.mark.parametrize("action_kind", ["obtain_credential", "noop", "unknown"])
def test_unsupported_or_abstract_actions_remain_unmapped(action_kind: str) -> None:
    assert map_simulator_behavior(action_kind) == ()


def test_rl_control_does_not_turn_enterprise_action_into_atlas_behavior() -> None:
    mappings = map_simulator_behavior("exploit")

    assert all(mapping.framework is not Framework.ATLAS for mapping in mappings)


def test_explicit_ai_behavior_can_add_atlas_semantics() -> None:
    mappings = map_simulator_behavior(
        "enumerate_application",
        ai_behavior=AIBehavior.DISCOVER_MODEL_ONTOLOGY,
    )

    assert [mapping.technique_id for mapping in mappings] == ["T1518", "AML.T0013"]
    assert mappings[1].framework is Framework.ATLAS


def test_event_fields_keep_policy_simulator_and_framework_layers_separate() -> None:
    fields = event_framework_fields(
        rl_action_index=321,
        simulator_action="pivot:host_data",
        action_kind="pivot",
    )

    assert fields["rl_action_index"] == 321
    assert fields["simulator_action"] == "pivot:host_data"
    assert fields["framework_mappings"][0]["technique_id"] == "T1021"
    assert "target_entity" not in fields["framework_mappings"][0]


def test_negative_rl_action_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        event_framework_fields(
            rl_action_index=-1,
            simulator_action="noop",
            action_kind="noop",
        )


def test_matrix_progress_uses_event_mappings_not_hidden_topology() -> None:
    event = {
        **event_framework_fields(
            rl_action_index=1,
            simulator_action="discover_network:segment_edge",
            action_kind="discover_network",
        ),
        "success": True,
        "state_changed": True,
    }
    first = technique_progress([event], Framework.ATTACK_ENTERPRISE)
    second = technique_progress(
        [{**event, "unrelated_hidden_topology": ["secret_host", "secret_database"]}],
        Framework.ATTACK_ENTERPRISE,
    )

    assert first == second
    assert first["T1018"].progressed == 1
    assert technique_progress([event], Framework.ATLAS) == {}
