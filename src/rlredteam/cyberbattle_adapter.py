"""CyberBattleSim adapter with an AgentKnowledge-only PPO boundary.

CyberBattleSim is optional and imported only when this adapter is constructed.
The dedicated Podman image pins the upstream revision and applies a narrow
construction-time NumPy compatibility shim, leaving the frozen NASim image
and its dependency stack unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rlredteam.enterprise.model import EdgeType, NodeType
from rlredteam.enterprise.state import AgentKnowledge, Observation
from rlredteam.events import AccessLevel, ActionKind, AttackEvent
from rlredteam.simulator_adapter import (
    KnowledgeDelta,
    SemanticAction,
    SimulatorAdapter,
    SimulatorTransition,
)

CYBERBATTLE_SOURCE_REVISION = "aebf185ab0a7515f983ee696cb7b71d3fdcd107f"
ADAPTER_SCHEMA_VERSION = "security-rl-cyberbattle-adapter-v1"


class CyberBattleAdapterError(RuntimeError):
    """The native simulator cannot satisfy the adapter contract."""


@dataclass(frozen=True, slots=True)
class CyberBattleScenario:
    name: str = "CyberBattleChain-v0"
    size: int = 2
    maximum_node_count: int = 6
    maximum_total_credentials: int = 6
    max_steps: int = 64
    failure_penalty: float = -1.0

    def __post_init__(self) -> None:
        if self.name != "CyberBattleChain-v0":
            raise ValueError("the first integration supports only CyberBattleChain-v0")
        if self.size < 2 or self.size % 2:
            raise ValueError("CyberBattle chain size must be a positive even number")
        if self.maximum_node_count < self.size + 2:
            raise ValueError("maximum_node_count cannot truncate the chain scenario")
        if self.maximum_total_credentials < self.size + 1:
            raise ValueError("credential capacity cannot truncate the chain scenario")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.failure_penalty > 0:
            raise ValueError("failure_penalty cannot reward failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "maximum_node_count": self.maximum_node_count,
            "maximum_total_credentials": self.maximum_total_credentials,
            "max_steps": self.max_steps,
            "failure_penalty": self.failure_penalty,
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class CyberBattleObservationTranslator:
    """Encode only accumulated AgentKnowledge and the native visible mask."""

    def __init__(
        self,
        *,
        maximum_node_count: int,
        properties: tuple[str, ...],
        services: tuple[str, ...],
        action_count: int,
        max_steps: int,
    ) -> None:
        self.maximum_node_count = maximum_node_count
        self.properties = properties
        self.services = services
        self.action_count = action_count
        self.max_steps = max_steps
        base = (
            maximum_node_count * Observation.feature_count(tuple(NodeType))
            + maximum_node_count * maximum_node_count
            + 1
        )
        self.shape = (
            base
            + maximum_node_count * len(properties)
            + maximum_node_count * len(services)
            + action_count,
        )

    def encode(
        self,
        knowledge: AgentKnowledge,
        *,
        step: int,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        base = Observation.from_knowledge(
            knowledge,
            max_nodes=self.maximum_node_count,
            step=step,
            max_steps=self.max_steps,
        ).as_array()
        property_values: list[float] = []
        service_values: list[float] = []
        for slot in range(self.maximum_node_count):
            node = (
                knowledge.discovery_order[slot]
                if slot < len(knowledge.discovery_order)
                else None
            )
            known_properties = knowledge.known_properties.get(node, set()) if node else set()
            known_services = knowledge.known_services.get(node, set()) if node else set()
            property_values.extend(float(item in known_properties) for item in self.properties)
            service_values.extend(float(item in known_services) for item in self.services)
        output = np.concatenate(
            (
                base,
                np.asarray(property_values, dtype=np.float32),
                np.asarray(service_values, dtype=np.float32),
                np.asarray(action_mask, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        if output.shape != self.shape:
            raise CyberBattleAdapterError(
                f"observation shape drift: expected {self.shape}, got {output.shape}"
            )
        return output.copy()


@dataclass(frozen=True, slots=True)
class _ResolvedAction:
    native_action: Mapping[str, np.ndarray]
    source: str | None
    target: str | None
    service: str | None
    simulator_vulnerability_id: str | None


class CyberBattleAdapter(gym.Env, SimulatorAdapter):
    """Discrete semantic projection of a small native CyberBattle environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: CyberBattleScenario | None = None,
        *,
        reset_seeds: tuple[int, ...] = (),
    ) -> None:
        self.scenario = scenario or CyberBattleScenario()
        self._native_env = self._make_native_env()
        self._knowledge = AgentKnowledge()
        self._native_observation: Mapping[str, Any] | None = None
        self._step = 0
        self._reset_seeds = tuple(int(item) for item in reset_seeds)
        self._reset_index = 0

        identifiers = self._native_env.identifiers
        self._properties = tuple(map(str, identifiers.properties))
        self._services = tuple(map(str, identifiers.ports))
        self._local_vulnerabilities = tuple(map(str, identifiers.local_vulnerabilities))
        self._remote_vulnerabilities = tuple(map(str, identifiers.remote_vulnerabilities))
        self._catalogue = self._build_action_catalogue()
        self.action_space = gym.spaces.Discrete(len(self._catalogue))
        self._translator = CyberBattleObservationTranslator(
            maximum_node_count=self.scenario.maximum_node_count,
            properties=self._properties,
            services=self._services,
            action_count=len(self._catalogue),
            max_steps=self.scenario.max_steps,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self._translator.shape, dtype=np.float32
        )

    def _make_native_env(self):
        try:
            import cyberbattle  # noqa: F401 - import registers Gym environments
            from cyberbattle._env.cyberbattle_env import AttackerGoal
        except ImportError as exc:
            raise CyberBattleAdapterError(
                "CyberBattleSim is unavailable; use the dedicated Podman image"
            ) from exc
        # Upstream 0.1.0 calls np.can_cast with Python scalar values, an API
        # removed by NumPy 2.  Constrain the compatibility shim to construction
        # instead of downgrading Security-RL's frozen numerical environment.
        original_can_cast = np.can_cast

        def compatible_can_cast(value, dtype, *args, **kwargs):
            try:
                return original_can_cast(value, dtype, *args, **kwargs)
            except TypeError:
                target = np.dtype(dtype)
                if isinstance(value, int | np.integer) and np.issubdtype(
                    target, np.integer
                ):
                    limits = np.iinfo(target)
                    return limits.min <= int(value) <= limits.max
                return original_can_cast(np.asarray(value).dtype, dtype, *args, **kwargs)

        np.can_cast = compatible_can_cast  # type: ignore[assignment]
        try:
            return gym.make(
                self.scenario.name,
                size=self.scenario.size,
                maximum_node_count=np.int32(self.scenario.maximum_node_count),
                maximum_total_credentials=np.int32(
                    self.scenario.maximum_total_credentials
                ),
                maximum_discoverable_credentials_per_action=np.int32(
                    self.scenario.maximum_total_credentials
                ),
                attacker_goal=AttackerGoal(own_atleast_percent=1.0),
                defender_agent=None,
                throws_on_invalid_actions=False,
            ).unwrapped
        finally:
            np.can_cast = original_can_cast  # type: ignore[assignment]

    @property
    def agent_knowledge(self) -> AgentKnowledge:
        return self._knowledge

    @property
    def package_version(self) -> str:
        try:
            return version("cyberbattlesim")
        except PackageNotFoundError:
            return "unknown"

    def _build_action_catalogue(self) -> tuple[SemanticAction, ...]:
        rows: list[SemanticAction] = []
        for vulnerability_index, vulnerability_id in enumerate(self._local_vulnerabilities):
            rows.append(
                SemanticAction(
                    index=len(rows),
                    native_kind="local_vulnerability",
                    simulator_action=f"local_vulnerability:{vulnerability_id}",
                    semantic_behavior="local_exploit",
                    native_parameters=(vulnerability_index,),
                    simulator_vulnerability_id=vulnerability_id,
                )
            )
        for vulnerability_index, vulnerability_id in enumerate(self._remote_vulnerabilities):
            rows.append(
                SemanticAction(
                    index=len(rows),
                    native_kind="remote_vulnerability",
                    simulator_action=f"remote_vulnerability:{vulnerability_id}",
                    semantic_behavior="exploit",
                    native_parameters=(vulnerability_index,),
                    simulator_vulnerability_id=vulnerability_id,
                )
            )
        rows.append(
            SemanticAction(
                index=len(rows),
                native_kind="connect",
                simulator_action="connect:latest_credential",
                semantic_behavior="authenticate",
            )
        )
        return tuple(rows)

    def action_catalogue(self) -> tuple[SemanticAction, ...]:
        return self._catalogue

    def _discovered_nodes(self, observation: Mapping[str, Any] | None = None) -> list[str]:
        selected = observation if observation is not None else self._native_observation
        if not selected:
            return []
        return [str(item) for item in selected.get("_discovered_nodes", ())]

    def _active_source_slot(self, observation: Mapping[str, Any]) -> int | None:
        discovered_count = int(observation.get("discovered_node_count", 0))
        privileges = np.asarray(observation.get("nodes_privilegelevel", ()), dtype=int)
        owned = [slot for slot in range(discovered_count) if int(privileges[slot]) > 0]
        return owned[-1] if owned else None

    def _remote_target_slot(self, observation: Mapping[str, Any]) -> int | None:
        discovered_count = int(observation.get("discovered_node_count", 0))
        if not discovered_count:
            return None
        privileges = np.asarray(observation.get("nodes_privilegelevel", ()), dtype=int)
        unowned = [slot for slot in range(discovered_count) if int(privileges[slot]) == 0]
        return unowned[-1] if unowned else discovered_count - 1

    def _resolve_action(self, action: int) -> _ResolvedAction:
        if self._native_observation is None:
            raise CyberBattleAdapterError("reset must be called before resolving an action")
        index = int(action)
        if not 0 <= index < len(self._catalogue):
            raise ValueError(f"action {index} outside [0, {len(self._catalogue)})")
        specification = self._catalogue[index]
        observation = self._native_observation
        nodes = self._discovered_nodes(observation)
        source_slot = self._active_source_slot(observation)
        source = nodes[source_slot] if source_slot is not None else None

        if specification.native_kind == "local_vulnerability":
            vulnerability_index = specification.native_parameters[0]
            slot = source_slot if source_slot is not None else 0
            return _ResolvedAction(
                native_action={
                    "local_vulnerability": np.asarray(
                        [slot, vulnerability_index], dtype=np.int32
                    )
                },
                source=source,
                target=source,
                service=None,
                simulator_vulnerability_id=specification.simulator_vulnerability_id,
            )

        if specification.native_kind == "remote_vulnerability":
            vulnerability_index = specification.native_parameters[0]
            target_slot = self._remote_target_slot(observation)
            source_value = source_slot if source_slot is not None else 0
            target_value = target_slot if target_slot is not None else 0
            target = nodes[target_value] if target_value < len(nodes) else None
            return _ResolvedAction(
                native_action={
                    "remote_vulnerability": np.asarray(
                        [source_value, target_value, vulnerability_index], dtype=np.int32
                    )
                },
                source=source,
                target=target,
                service=None,
                simulator_vulnerability_id=specification.simulator_vulnerability_id,
            )

        credential_count = int(observation.get("credential_cache_length", 0))
        credential_slot = max(credential_count - 1, 0)
        credentials = tuple(observation.get("credential_cache_matrix", ()))
        if credential_count and credential_slot < len(credentials):
            target_slot, port_slot = map(int, np.asarray(credentials[credential_slot]))
        else:
            target_slot = port_slot = 0
        source_value = source_slot if source_slot is not None else 0
        target = nodes[target_slot] if target_slot < len(nodes) else None
        service = self._services[port_slot] if port_slot < len(self._services) else None
        return _ResolvedAction(
            native_action={
                "connect": np.asarray(
                    [source_value, target_slot, port_slot, credential_slot], dtype=np.int32
                )
            },
            source=source,
            target=target,
            service=service,
            simulator_vulnerability_id=None,
        )

    def to_native_action(self, action: int) -> Mapping[str, np.ndarray]:
        """Deterministically resolve a policy action from visible state only."""
        resolved = self._resolve_action(action)
        return {key: value.copy() for key, value in resolved.native_action.items()}

    def action_mask(self) -> np.ndarray:
        if self._native_observation is None:
            return np.zeros(len(self._catalogue), dtype=np.int8)
        native_mask = self._native_observation.get("action_mask", {})
        output = np.zeros(len(self._catalogue), dtype=np.int8)
        for specification in self._catalogue:
            resolved = self._resolve_action(specification.index)
            coordinates = next(iter(resolved.native_action.values()))
            field = native_mask.get(specification.native_kind)
            if field is not None:
                try:
                    output[specification.index] = int(bool(np.asarray(field)[tuple(coordinates)]))
                except IndexError:
                    output[specification.index] = 0
        return output

    def convert_observation(self, native_observation: Any) -> np.ndarray:
        if not isinstance(native_observation, Mapping):
            raise CyberBattleAdapterError("CyberBattle observation must be a mapping")
        # The translator accepts no environment/topology object.  In particular,
        # arbitrary hidden keys added to the native mapping cannot cross to PPO.
        return self._translator.encode(
            self._knowledge,
            step=self._step,
            action_mask=self.action_mask(),
        )

    @staticmethod
    def _fact_pairs(values: Mapping[str, set[str]]) -> set[tuple[str, str]]:
        return {(node, fact) for node, facts in values.items() for fact in facts}

    def update_agent_knowledge(
        self, native_observation: Any, *, native_info=None
    ) -> KnowledgeDelta:
        if not isinstance(native_observation, Mapping):
            raise CyberBattleAdapterError("CyberBattle observation must be a mapping")
        context = dict(native_info or {})
        before_nodes = set(self._knowledge.discovered)
        before_properties = self._fact_pairs(self._knowledge.known_properties)
        before_services = self._fact_pairs(self._knowledge.known_services)
        before_access = dict(self._knowledge.access)
        before_edges = set(self._knowledge.known_edges)
        before_credentials = set(self._knowledge.credentials)

        nodes = self._discovered_nodes(native_observation)
        for node in nodes:
            self._knowledge.discover(node, NodeType.HOST, reachable=True)

        property_matrix = np.asarray(native_observation.get("discovered_nodes_properties", ()))
        for node_slot, node in enumerate(nodes):
            if node_slot >= len(property_matrix):
                continue
            for property_slot, flag in enumerate(property_matrix[node_slot]):
                if int(flag) == 1 and property_slot < len(self._properties):
                    self._knowledge.learn_property(node, self._properties[property_slot])

        privileges = np.asarray(native_observation.get("nodes_privilegelevel", ()), dtype=int)
        for node_slot, node in enumerate(nodes):
            if node_slot >= len(privileges):
                continue
            level = int(privileges[node_slot])
            if level > 0:
                self._knowledge.access[node] = "root" if level > 1 else "user"

        credential_count = int(native_observation.get("credential_cache_length", 0))
        credentials = tuple(native_observation.get("credential_cache_matrix", ()))
        for target_slot, port_slot in (
            map(int, np.asarray(credentials[index]))
            for index in range(min(credential_count, len(credentials)))
        ):
            if target_slot < len(nodes):
                node = nodes[target_slot]
                self._knowledge.credentials.add(node)
                if port_slot < len(self._services):
                    self._knowledge.learn_service(node, self._services[port_slot])

        native_mask = native_observation.get("action_mask", {})
        local_mask = np.asarray(native_mask.get("local_vulnerability", ()))
        if local_mask.ndim == 2:
            for node_slot, node in enumerate(nodes):
                if node_slot >= local_mask.shape[0] or self._knowledge.access_level(node) == 0:
                    continue
                for vulnerability_slot, flag in enumerate(local_mask[node_slot]):
                    if int(flag) and vulnerability_slot < len(self._local_vulnerabilities):
                        self._knowledge.learn_native_vulnerability(
                            node, self._local_vulnerabilities[vulnerability_slot]
                        )

        resolved = context.get("resolved_action")
        if (
            isinstance(resolved, _ResolvedAction)
            and int(native_observation.get("lateral_move", 0)) == 1
            and resolved.source in self._knowledge.discovered
            and resolved.target in self._knowledge.discovered
        ):
            self._knowledge.learn_edge(
                str(resolved.source), str(resolved.target), EdgeType.PIVOTS_TO
            )

        after_properties = self._fact_pairs(self._knowledge.known_properties)
        after_services = self._fact_pairs(self._knowledge.known_services)
        new_edges = self._knowledge.known_edges - before_edges
        return KnowledgeDelta(
            discovered_nodes=tuple(sorted(self._knowledge.discovered - before_nodes)),
            discovered_properties=tuple(sorted(after_properties - before_properties)),
            discovered_services=tuple(sorted(after_services - before_services)),
            access_changes=tuple(
                sorted(
                    (node, level)
                    for node, level in self._knowledge.access.items()
                    if before_access.get(node) != level
                )
            ),
            observed_edges=tuple(
                sorted((source, target, edge.value) for source, target, edge in new_edges)
            ),
            credential_targets=tuple(sorted(self._knowledge.credentials - before_credentials)),
        )

    @staticmethod
    def _access_level(raw: int) -> AccessLevel:
        if raw <= 0:
            return AccessLevel.NONE
        return AccessLevel.ROOT if raw > 1 else AccessLevel.USER

    def native_result_to_event(self, native_result):
        specification = native_result["specification"]
        resolved = native_result["resolved_action"]
        observation = native_result["observation"]
        delta = native_result["knowledge_delta"]
        native_reward = float(native_result["native_reward"])
        terminated = bool(native_result["terminated"])
        truncated = bool(native_result["truncated"])
        valid = bool(native_result["valid"])

        escalation = int(observation.get("escalation", 0))
        lateral_move = int(observation.get("lateral_move", 0)) == 1
        customer_data = int(observation.get("customer_data_found", 0)) == 1
        probe_success = int(observation.get("probe_result", 0)) == 2
        success = bool(delta.changed or native_reward > 0 or lateral_move or probe_success)
        behavior = specification.semantic_behavior
        kind = ActionKind.EXPLOIT
        if specification.native_kind == "local_vulnerability" and escalation > 0:
            behavior, kind = "escalate_privilege", ActionKind.PRIVESC
        elif customer_data:
            behavior = "access_asset"
        elif specification.native_kind == "connect":
            behavior = "pivot" if lateral_move else "authenticate"

        target = resolved.target
        access = AccessLevel.NONE
        if target in self._knowledge.discovery_order:
            target_slot = self._knowledge.node_slot(str(target))
            privileges = np.asarray(observation.get("nodes_privilegelevel", ()), dtype=int)
            if target_slot < len(privileges):
                access = self._access_level(int(privileges[target_slot]))
        error = None
        if not valid:
            error = "masked_invalid"
        elif not success:
            error = "simulator_failure"
        return (
            AttackEvent(
                step=self._step,
                kind=kind,
                action_name=specification.simulator_action,
                target=target,
                success=success,
                rl_action_index=specification.index,
                native_reward=native_reward,
                cost=0.0,
                cve_id=None,
                cvss_base=None,
                service=resolved.service,
                access_gained=access,
                newly_discovered=len(delta.discovered_nodes),
                is_crown_jewel=bool(terminated and success),
                goal_reached=terminated,
                terminal=bool(terminated or truncated),
                error=error,
            ),
            behavior,
        )

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if seed is None and self._reset_seeds:
            seed = self._reset_seeds[self._reset_index % len(self._reset_seeds)]
            self._reset_index += 1
        self._knowledge = AgentKnowledge()
        self._step = 0
        native_observation, native_info = self._native_env.reset(seed=seed)
        self._native_observation = native_observation
        delta = self.update_agent_knowledge(native_observation)
        observation = self.convert_observation(native_observation)
        return observation, {
            "simulator": "CyberBattleSim",
            "seed": seed,
            "knowledge_delta": delta,
            "native_step_count": int(native_info.get("step_count", 0)),
        }

    def step(self, action: int):
        if self._native_observation is None:
            raise CyberBattleAdapterError("reset must be called before step")
        index = int(action)
        if not 0 <= index < len(self._catalogue):
            raise ValueError(f"action {index} outside [0, {len(self._catalogue)})")
        specification = self._catalogue[index]
        resolved = self._resolve_action(index)
        mask = self.action_mask()
        valid = bool(mask[index])
        native_observation, native_reward, terminated, native_truncated, native_info = (
            self._native_env.step(resolved.native_action)
        )
        self._step += 1
        self._native_observation = native_observation
        delta = self.update_agent_knowledge(
            native_observation, native_info={"resolved_action": resolved}
        )
        truncated = bool(
            native_truncated
            or (self._step >= self.scenario.max_steps and not terminated)
        )
        event, behavior = self.native_result_to_event(
            {
                "specification": specification,
                "resolved_action": resolved,
                "observation": native_observation,
                "knowledge_delta": delta,
                "native_reward": native_reward,
                "terminated": terminated,
                "truncated": truncated,
                "valid": valid,
            }
        )
        # Keep native CyberBattle rewards in the event, but train on a bounded
        # knowledge-progress signal.  This avoids a 5000-point terminal reward
        # overwhelming PPO while still rewarding only agent-visible progress.
        if event.goal_reached and event.success:
            policy_reward = 100.0
        elif delta.changed:
            policy_reward = float(
                5
                + 5 * len(delta.discovered_nodes)
                + 10 * len(delta.access_changes)
            )
        elif event.success:
            policy_reward = 0.0
        else:
            policy_reward = float(self.scenario.failure_penalty)
        transition = SimulatorTransition(
            observation=self.convert_observation(native_observation),
            reward=policy_reward,
            terminated=bool(terminated),
            truncated=truncated,
            event=event,
            action=specification,
            semantic_behavior=behavior,
            simulator_action=specification.simulator_action,
            simulator_vulnerability_id=resolved.simulator_vulnerability_id,
            knowledge_delta=delta,
        )
        return (
            transition.observation,
            transition.reward,
            transition.terminated,
            transition.truncated,
            {
                "simulator": "CyberBattleSim",
                "simulator_transition": transition,
                "native_reward": float(native_reward),
                "native_step_count": int(native_info.get("step_count", self._step)),
                "network_availability": float(
                    native_info.get("network_availability", 1.0)
                ),
            },
        )

    def space_summary(self) -> dict[str, Any]:
        return {
            "action_space": f"Discrete({self.action_space.n})",
            "observation_space": f"Box(0.0, 1.0, {self.observation_space.shape})",
            "native_action_kinds": ["local_vulnerability", "remote_vulnerability", "connect"],
            "action_mask": True,
            "policy_observation_source": "AgentKnowledge_and_visible_action_mask_only",
            "policy_reward": "bounded_agent_knowledge_progress",
        }

    def agent_knowledge_graph(self) -> dict[str, list[dict[str, Any]]]:
        """Return the observed graph only; this method has no native env access."""
        nodes = []
        for node in self._knowledge.discovery_order:
            nodes.append(
                {
                    "id": node,
                    "type": self._knowledge.known_node_types[node].value,
                    "properties": sorted(self._knowledge.known_properties.get(node, set())),
                    "services": sorted(self._knowledge.known_services.get(node, set())),
                    "access": self._knowledge.access.get(node, "none"),
                }
            )
        edges = [
            {"source": source, "target": target, "type": edge.value}
            for source, target, edge in sorted(
                self._knowledge.known_edges,
                key=lambda item: (item[0], item[1], item[2].value),
            )
        ]
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self._native_env.close()


def cyberbattle_adapter_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
