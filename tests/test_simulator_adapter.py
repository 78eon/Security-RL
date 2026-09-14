from __future__ import annotations

import numpy as np

from rlredteam.events import AccessLevel, ActionKind, AttackEvent
from rlredteam.nasim_adapter import NASimSimulatorAdapter
from rlredteam.simulator_adapter import (
    KnowledgeDelta,
    SemanticAction,
    SimulatorAdapter,
    SimulatorTransition,
)


def test_nasim_exposes_the_common_semantic_contract() -> None:
    assert issubclass(NASimSimulatorAdapter, SimulatorAdapter)


def test_framework_mapping_uses_semantics_not_raw_action_index() -> None:
    action = SemanticAction(
        index=999,
        native_kind="connect",
        simulator_action="native-connect",
        semantic_behavior="pivot",
    )
    transition = SimulatorTransition(
        observation=np.zeros(2, dtype=np.float32),
        reward=1.0,
        terminated=False,
        truncated=False,
        event=AttackEvent(
            step=1,
            kind=ActionKind.EXPLOIT,
            action_name="native-connect",
            target="node-1",
            success=True,
            rl_action_index=999,
            access_gained=AccessLevel.USER,
        ),
        action=action,
        semantic_behavior="pivot",
        simulator_action="native-connect",
        simulator_vulnerability_id=None,
        knowledge_delta=KnowledgeDelta(
            observed_edges=(("node-0", "node-1", "pivots_to"),)
        ),
    )
    event = transition.to_trajectory_event()
    assert event["rl_action_index"] == 999
    assert [row["technique_id"] for row in event["framework_mappings"]] == ["T1021"]


def test_native_vulnerability_does_not_create_a_cve() -> None:
    action = SemanticAction(
        index=0,
        native_kind="local_vulnerability",
        simulator_action="local_vulnerability:native-only",
        semantic_behavior="local_exploit",
        simulator_vulnerability_id="native-only",
    )
    assert action.simulator_vulnerability_id == "native-only"
    assert action.cve_id is None
