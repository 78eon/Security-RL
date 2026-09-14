"""CybORG adapter with an AgentKnowledge-only policy boundary.

The adapter uses CybORG's external Red-agent API. It never calls
``get_true_state`` or reads ``environment_controller.state``. Native actions
are resolved from facts present in Red's observation, converted to
Security-RL semantics, and only then mapped by the reporting catalogue.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from ipaddress import IPv4Address, IPv4Network
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

CYBORG_SOURCE_REVISION = "2742b5e0ce4330c9b14006b38acd3b5ebe00d6fd"
ADAPTER_SCHEMA_VERSION = "security-rl-cyborg-adapter-v1"
_SERVICE_LIMIT = 16.0
_PROPERTY_LIMIT = 16.0


class CybORGAdapterError(RuntimeError):
    """The native CybORG environment cannot satisfy the adapter contract."""


@dataclass(frozen=True, slots=True)
class CybORGScenario:
    name: str = "Scenario1"
    source_file: str = "Scenario1.yaml"
    agent: str = "Red"
    maximum_node_count: int = 8
    maximum_service_count: int = 16
    max_steps: int = 96
    objective_native_reward: float = 10.0
    failure_penalty: float = -1.0

    def __post_init__(self) -> None:
        if self.name != "Scenario1" or self.source_file != "Scenario1.yaml":
            raise ValueError("the first CybORG integration supports only Scenario1.yaml")
        if self.agent != "Red":
            raise ValueError("the first CybORG integration supports the external Red agent")
        if self.maximum_node_count < 4:
            raise ValueError("Scenario1 requires capacity for at least four hosts")
        if self.maximum_service_count < 1 or self.max_steps < 1:
            raise ValueError("service and step capacities must be positive")
        if self.objective_native_reward <= 0:
            raise ValueError("objective_native_reward must be positive")
        if self.failure_penalty > 0:
            raise ValueError("failure_penalty cannot reward failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self.source_file,
            "agent": self.agent,
            "maximum_node_count": self.maximum_node_count,
            "maximum_service_count": self.maximum_service_count,
            "max_steps": self.max_steps,
            "objective_native_reward": self.objective_native_reward,
            "failure_penalty": self.failure_penalty,
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class CybORGObservationTranslator:
    """Encode accumulated observable knowledge, never a native state object."""

    def __init__(self, *, maximum_node_count: int, action_count: int, max_steps: int):
        self.maximum_node_count = maximum_node_count
        self.action_count = action_count
        self.max_steps = max_steps
        base = (
            maximum_node_count * Observation.feature_count(tuple(NodeType))
            + maximum_node_count * maximum_node_count
            + 1
        )
        # Per node: property count, service count, native-vulnerability flag,
        # and credential flag. Exact strings and secrets never enter PPO.
        self.shape = (base + maximum_node_count * 4 + action_count,)

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
        summaries: list[float] = []
        for slot in range(self.maximum_node_count):
            node = (
                knowledge.discovery_order[slot]
                if slot < len(knowledge.discovery_order)
                else None
            )
            if node is None:
                summaries.extend((0.0, 0.0, 0.0, 0.0))
                continue
            summaries.extend(
                (
                    min(len(knowledge.known_properties.get(node, ())) / _PROPERTY_LIMIT, 1.0),
                    min(len(knowledge.known_services.get(node, ())) / _SERVICE_LIMIT, 1.0),
                    float(bool(knowledge.known_native_vulnerabilities.get(node))),
                    float(node in knowledge.credentials),
                )
            )
        output = np.concatenate(
            (
                base,
                np.asarray(summaries, dtype=np.float32),
                np.asarray(action_mask, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        if output.shape != self.shape:
            raise CybORGAdapterError(
                f"observation shape drift: expected {self.shape}, got {output.shape}"
            )
        return output.copy()


@dataclass(frozen=True, slots=True)
class _ResolvedAction:
    native_action: Any
    source: str | None
    target: str | None
    simulator_vulnerability_id: str | None
    target_session: int | None = None


def _enum_text(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _success(value: Any) -> bool:
    try:
        return int(getattr(value, "value", value)) == 1
    except (TypeError, ValueError):
        return value is True


class CybORGAdapter(gym.Env, SimulatorAdapter):
    """Fixed semantic action projection over CybORG's small Scenario1."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: CybORGScenario | None = None,
        *,
        reset_seeds: tuple[int, ...] = (),
    ) -> None:
        self.scenario = scenario or CybORGScenario()
        self._classes = self._load_native_classes()
        self._native_env = self._make_native_env(seed=None)
        self._knowledge = AgentKnowledge()
        self._native_observation: Mapping[str, Any] | None = None
        self._visible_action_space: Mapping[str, Any] = {}
        self._step = 0
        self._reset_seeds = tuple(map(int, reset_seeds))
        self._reset_index = 0
        self._node_by_ip: dict[str, str] = {}
        self._subnets: dict[str, IPv4Network] = {}
        self._sessions: dict[int, dict[str, Any]] = {}
        self._credentials: dict[str, list[tuple[str, str]]] = {}
        self._routed_subnets: set[str] = set()
        self._upgraded_sessions: set[int] = set()
        self._configured_sessions: set[int] = set()
        self._autorouted_sessions: set[int] = set()
        self._scanned_subnets: set[str] = set()
        self._catalogue = self._build_action_catalogue()
        self.action_space = gym.spaces.Discrete(len(self._catalogue))
        self._translator = CybORGObservationTranslator(
            maximum_node_count=self.scenario.maximum_node_count,
            action_count=len(self._catalogue),
            max_steps=self.scenario.max_steps,
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self._translator.shape, dtype=np.float32
        )

    @staticmethod
    def _load_native_classes() -> dict[str, Any]:
        try:
            from CybORG import CybORG
            from CybORG.Simulator.Actions import (
                MeterpreterIPConfig,
                MS17_010_PSExec,
                MSFAutoroute,
                MSFPingsweep,
                MSFPortscan,
                Sleep,
                SSHLoginExploit,
                UpgradeToMeterpreter,
            )
            from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import (
                FileReaderScenarioGenerator,
            )
        except ImportError as exc:
            raise CybORGAdapterError(
                "CybORG is unavailable; use the dedicated Podman image"
            ) from exc
        return {
            "CybORG": CybORG,
            "FileReaderScenarioGenerator": FileReaderScenarioGenerator,
            "Sleep": Sleep,
            "MSFPortscan": MSFPortscan,
            "SSHLoginExploit": SSHLoginExploit,
            "UpgradeToMeterpreter": UpgradeToMeterpreter,
            "MeterpreterIPConfig": MeterpreterIPConfig,
            "MSFAutoroute": MSFAutoroute,
            "MSFPingsweep": MSFPingsweep,
            "MS17_010_PSExec": MS17_010_PSExec,
        }

    def _scenario_path(self) -> Path:
        root = Path("/opt/cyborg/CybORG/Simulator/Scenarios/scenario_files")
        path = root / self.scenario.source_file
        if not path.is_file():
            raise CybORGAdapterError(f"pinned CybORG scenario is missing: {path}")
        return path

    def _make_native_env(self, *, seed: int | None):
        generator = self._classes["FileReaderScenarioGenerator"](
            str(self._scenario_path())
        )
        return self._classes["CybORG"](
            generator, "sim", seed=seed
        )

    @property
    def agent_knowledge(self) -> AgentKnowledge:
        return self._knowledge

    @property
    def package_version(self) -> str:
        try:
            return version("CybORG")
        except PackageNotFoundError:
            return "unknown"

    def _build_action_catalogue(self) -> tuple[SemanticAction, ...]:
        rows = (
            ("Sleep", "noop", None),
            ("MSFPortscan", "enumerate_service", None),
            ("SSHLoginExploit", "authenticate", "SSHLoginExploit"),
            ("UpgradeToMeterpreter", "local_exploit", "UpgradeToMeterpreter"),
            ("MeterpreterIPConfig", "enumerate_host", None),
            ("MSFAutoroute", "pivot", None),
            ("MSFPingsweep", "discover_network", None),
            ("MS17_010_PSExec", "exploit", "MS17_010_PSExec"),
        )
        return tuple(
            SemanticAction(
                index=index,
                native_kind=name,
                simulator_action=f"cyborg:{name}",
                semantic_behavior=behavior,
                simulator_vulnerability_id=vulnerability,
            )
            for index, (name, behavior, vulnerability) in enumerate(rows)
        )

    def action_catalogue(self) -> tuple[SemanticAction, ...]:
        return self._catalogue

    def _candidate_nodes(self, *, service: str | None = None) -> list[str]:
        nodes = list(self._knowledge.discovery_order)
        if service is not None:
            nodes = [
                node
                for node in nodes
                if service in self._knowledge.known_services.get(node, set())
            ]
        return [node for node in nodes if node in self._node_by_ip.values()]

    def _latest_session(self, *types: str) -> tuple[int, dict[str, Any]] | None:
        allowed = {item.upper() for item in types}
        candidates = [
            (identifier, data)
            for identifier, data in self._sessions.items()
            if not allowed or str(data.get("type", "")).upper() in allowed
        ]
        return max(candidates, default=None, key=lambda item: item[0])

    def _latest_unprocessed_session(
        self, session_type: str, processed: set[int]
    ) -> tuple[int, dict[str, Any]] | None:
        candidates = [
            (identifier, data)
            for identifier, data in self._sessions.items()
            if str(data.get("type", "")).upper() == session_type
            and identifier not in processed
        ]
        return max(candidates, default=None, key=lambda item: item[0])

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self._catalogue), dtype=np.int8)
        mask[0] = 1
        nodes = self._candidate_nodes()
        mask[1] = int(any(node not in self._knowledge.enumerated for node in nodes))
        mask[2] = int(
            any(
                self._knowledge.access_level(node) == 0
                for node in self._candidate_nodes(service="22")
            )
        )
        meterpreter = self._latest_session("METERPRETER")
        shell = self._latest_unprocessed_session("MSF_SHELL", self._upgraded_sessions)
        mask[3] = int(shell is not None)
        mask[4] = int(
            self._latest_unprocessed_session("METERPRETER", self._configured_sessions)
            is not None
        )
        mask[5] = int(
            self._latest_unprocessed_session("METERPRETER", self._autorouted_sessions)
            is not None
        )
        mask[6] = int(
            meterpreter is not None
            and bool(set(self._routed_subnets) - self._scanned_subnets)
        )
        mask[7] = int(
            any(
                self._knowledge.access_level(node) < 2 and node in self._knowledge.credentials
                for node in self._candidate_nodes(service="139")
            )
        )
        return mask

    def action_masks(self) -> np.ndarray:
        """SB3-Contrib MaskablePPO compatibility without widening the contract."""
        return self.action_mask().astype(bool, copy=False)

    def convert_observation(self, native_observation: Any) -> np.ndarray:
        # native_observation is intentionally unused. Tests poison it with
        # hidden-state keys to enforce the AgentKnowledge-only boundary.
        del native_observation
        return self._translator.encode(
            self._knowledge, step=self._step, action_mask=self.action_mask()
        )

    def _node_for_record(self, key: Any, record: Mapping[str, Any]) -> str:
        interfaces = record.get("Interface", ())
        for interface in interfaces if isinstance(interfaces, list | tuple) else ():
            if isinstance(interface, Mapping) and interface.get("IP Address") is not None:
                return str(interface["IP Address"])
        system = record.get("System info")
        if isinstance(system, Mapping) and system.get("Hostname"):
            return str(system["Hostname"])
        try:
            session = self._sessions.get(int(key))
        except (TypeError, ValueError):
            session = None
        if session is not None:
            return str(session["node"])
        return str(key)

    def _visible_records(
        self, observation: Mapping[str, Any]
    ) -> list[tuple[str, Mapping[str, Any]]]:
        output: list[tuple[str, Mapping[str, Any]]] = []
        for key, value in observation.items():
            if str(key) in {"success", "message", "raw"} or not isinstance(value, Mapping):
                continue
            output.append((self._node_for_record(key, value), value))
        return output

    def update_agent_knowledge(
        self,
        native_observation: Any,
        *,
        native_info: Mapping[str, Any] | None = None,
    ) -> KnowledgeDelta:
        if not isinstance(native_observation, Mapping):
            raise CybORGAdapterError("CybORG observation must be a mapping")
        before_nodes = set(self._knowledge.discovered)
        before_properties = {
            node: set(values) for node, values in self._knowledge.known_properties.items()
        }
        before_services = {
            node: set(values) for node, values in self._knowledge.known_services.items()
        }
        before_access = dict(self._knowledge.access)
        before_edges = set(self._knowledge.known_edges)
        before_credentials = set(self._knowledge.credentials)
        credential_targets: set[str] = set()

        records = self._visible_records(native_observation)
        for node, record in records:
            self._knowledge.discover(node, NodeType.HOST, reachable=True)
            interfaces = record.get("Interface", ())
            for interface in interfaces if isinstance(interfaces, list | tuple) else ():
                if not isinstance(interface, Mapping):
                    continue
                address = interface.get("IP Address")
                subnet = interface.get("Subnet")
                if address is not None:
                    self._node_by_ip[str(address)] = node
                if isinstance(subnet, IPv4Network):
                    self._subnets[str(subnet)] = subnet
                    self._knowledge.learn_property(node, f"subnet:{subnet}")

            system = record.get("System info")
            if isinstance(system, Mapping):
                for key, value in sorted(system.items(), key=lambda item: str(item[0])):
                    if key == "position" or value is None:
                        continue
                    self._knowledge.learn_property(node, f"{key}:{_enum_text(value)}")

            processes = record.get("Processes", ())
            for process in processes if isinstance(processes, list | tuple) else ():
                if not isinstance(process, Mapping):
                    continue
                connections = process.get("Connections", ())
                for connection in connections if isinstance(connections, list | tuple) else ():
                    if not isinstance(connection, Mapping):
                        continue
                    port = connection.get("local_port")
                    if port is not None and connection.get("remote_address") is None:
                        self._knowledge.learn_service(node, str(port))

            sessions = record.get("Sessions", ())
            for session in sessions if isinstance(sessions, list | tuple) else ():
                if not isinstance(session, Mapping) or session.get("ID") is None:
                    continue
                identifier = int(session["ID"])
                username = str(session.get("Username") or "")
                session_type = _enum_text(session.get("Type")).upper()
                self._sessions[identifier] = {
                    "node": node,
                    "type": session_type,
                    "username": username,
                }
                level = "root" if username.lower() in {"root", "system"} else "user"
                self._knowledge.access[node] = level
                self._knowledge.learn_property(node, f"session:{session_type}")
                if identifier == 0:
                    self._knowledge.enumerated.add(node)

            users = record.get("User Info", ())
            for user in users if isinstance(users, list | tuple) else ():
                if not isinstance(user, Mapping):
                    continue
                username, password = user.get("Username"), user.get("Password")
                if username is not None and password is not None:
                    pair = (str(username), str(password))
                    values = self._credentials.setdefault(node, [])
                    if pair not in values:
                        values.append(pair)
                    self._knowledge.credentials.add(node)
                    if node not in before_credentials:
                        credential_targets.add(node)

        info = dict(native_info or {})
        resolved = info.get("resolved_action")
        specification = info.get("specification")
        succeeded = bool(info.get("success"))
        native_reward = float(info.get("native_reward", 0.0))
        if isinstance(resolved, _ResolvedAction) and succeeded:
            target = resolved.target
            if target and target in self._knowledge.discovered:
                if isinstance(specification, SemanticAction):
                    if specification.semantic_behavior in {
                        "enumerate_service",
                        "enumerate_host",
                    }:
                        self._knowledge.enumerated.add(target)
                    if specification.simulator_vulnerability_id:
                        self._knowledge.learn_native_vulnerability(
                            target, specification.simulator_vulnerability_id
                        )
                    if specification.native_kind == "MS17_010_PSExec":
                        self._knowledge.access[target] = "root"
                    elif specification.native_kind in {
                        "SSHLoginExploit",
                        "UpgradeToMeterpreter",
                    } and self._knowledge.access_level(target) == 0:
                        self._knowledge.access[target] = "user"
                    if resolved.target_session is not None:
                        if specification.native_kind == "UpgradeToMeterpreter":
                            self._upgraded_sessions.add(resolved.target_session)
                        elif specification.native_kind == "MeterpreterIPConfig":
                            self._configured_sessions.add(resolved.target_session)
                        elif specification.native_kind == "MSFAutoroute":
                            self._autorouted_sessions.add(resolved.target_session)
                        elif specification.native_kind == "MSFPingsweep":
                            self._scanned_subnets.update(self._routed_subnets)
                if native_reward >= self.scenario.objective_native_reward:
                    self._knowledge.accessed_assets.add(target)
            if (
                resolved.source
                and target
                and resolved.source in self._knowledge.discovered
                and target in self._knowledge.discovered
                and resolved.source != target
                and isinstance(specification, SemanticAction)
                and specification.semantic_behavior in {"authenticate", "exploit", "pivot"}
            ):
                self._knowledge.learn_edge(resolved.source, target, EdgeType.PIVOTS_TO)
            if (
                isinstance(specification, SemanticAction)
                and specification.native_kind == "MSFAutoroute"
            ):
                self._routed_subnets.update(self._subnets)

        new_nodes = tuple(
            node for node in self._knowledge.discovery_order if node not in before_nodes
        )
        new_properties = tuple(
            sorted(
                (node, value)
                for node, values in self._knowledge.known_properties.items()
                for value in values - before_properties.get(node, set())
            )
        )
        new_services = tuple(
            sorted(
                (node, value)
                for node, values in self._knowledge.known_services.items()
                for value in values - before_services.get(node, set())
            )
        )
        access_changes = tuple(
            sorted(
                (node, value)
                for node, value in self._knowledge.access.items()
                if before_access.get(node) != value
            )
        )
        observed_edges = tuple(
            sorted(
                (source, target, edge.value)
                for source, target, edge in self._knowledge.known_edges - before_edges
            )
        )
        return KnowledgeDelta(
            discovered_nodes=new_nodes,
            discovered_properties=new_properties,
            discovered_services=new_services,
            access_changes=access_changes,
            observed_edges=observed_edges,
            credential_targets=tuple(sorted(credential_targets)),
        )

    def _main_session(self) -> int:
        if 0 in self._sessions:
            return 0
        return min(self._sessions, default=0)

    def _ip_for_node(self, node: str) -> IPv4Address:
        for address, selected in self._node_by_ip.items():
            if selected == node:
                return IPv4Address(address)
        raise CybORGAdapterError(f"no observed IP address for {node}")

    def _scan_target(self) -> str:
        nodes = self._candidate_nodes()
        if not nodes:
            raise CybORGAdapterError("no observed target is available")
        unenumerated = [node for node in nodes if node not in self._knowledge.enumerated]
        return (unenumerated or nodes)[-1]

    def _service_target(self, service: str) -> str:
        nodes = self._candidate_nodes(service=service)
        if not nodes:
            raise CybORGAdapterError(f"no observed target exposes service {service}")
        unowned = [node for node in nodes if self._knowledge.access_level(node) == 0]
        return (unowned or nodes)[-1]

    def _source_for_session(self, session: int) -> str | None:
        data = self._sessions.get(session)
        return str(data["node"]) if data else None

    def _observed_pivot_source(self, target: str) -> str | None:
        selected = self._latest_session("METERPRETER")
        if selected is not None:
            node = str(selected[1]["node"])
            if node != target:
                return node
        return self._source_for_session(self._main_session())

    def _resolve_action(self, action: int) -> _ResolvedAction:
        index = int(action)
        if not 0 <= index < len(self._catalogue):
            raise ValueError(f"action {index} outside [0, {len(self._catalogue)})")
        specification = self._catalogue[index]
        name = specification.native_kind
        agent = self.scenario.agent
        main_session = self._main_session()
        main_source = self._source_for_session(main_session)

        if name == "Sleep":
            return _ResolvedAction(
                self._classes[name](), main_source, main_source, None
            )
        if name == "MSFPortscan":
            target = self._scan_target()
            native = self._classes[name](
                ip_address=self._ip_for_node(target), session=main_session, agent=agent
            )
            return _ResolvedAction(native, main_source, target, None)
        if name == "SSHLoginExploit":
            target = self._service_target("22")
            native = self._classes[name](
                ip_address=self._ip_for_node(target),
                agent=agent,
                session=main_session,
                port=22,
            )
            return _ResolvedAction(native, self._observed_pivot_source(target), target, name)
        if name in {"UpgradeToMeterpreter", "MeterpreterIPConfig", "MSFAutoroute"}:
            session_type = "MSF_SHELL" if name == "UpgradeToMeterpreter" else "METERPRETER"
            selected = self._latest_session(session_type)
            if selected is None:
                raise CybORGAdapterError(f"no observed {session_type} session is available")
            target_session, data = selected
            native = self._classes[name](
                session=main_session, agent=agent, target_session=target_session
            )
            vulnerability = name if name == "UpgradeToMeterpreter" else None
            return _ResolvedAction(
                native,
                main_source,
                str(data["node"]),
                vulnerability,
                target_session,
            )
        if name == "MSFPingsweep":
            selected = self._latest_session("METERPRETER")
            if selected is None or not self._subnets:
                raise CybORGAdapterError("no observed pivot session/subnet is available")
            target_session, data = selected
            node = str(data["node"])
            node_subnets = [
                value
                for key, value in self._subnets.items()
                if f"subnet:{key}" in self._knowledge.known_properties.get(node, set())
            ]
            subnet = (node_subnets or list(self._subnets.values()))[-1]
            native = self._classes[name](
                subnet=subnet,
                session=main_session,
                agent=agent,
                target_session=target_session,
            )
            return _ResolvedAction(native, node, node, None, target_session)
        if name == "MS17_010_PSExec":
            target = self._service_target("139")
            observed = self._credentials.get(target, [])
            username, password = observed[-1] if observed else ("", "")
            native = self._classes[name](
                ip_address=self._ip_for_node(target),
                session=main_session,
                agent=agent,
                username=username,
                password=password,
            )
            return _ResolvedAction(native, self._observed_pivot_source(target), target, name)
        raise CybORGAdapterError(f"unsupported catalogue action {name}")

    def to_native_action(self, action: int) -> Any:
        return self._resolve_action(action).native_action

    def native_result_to_event(
        self, native_result: Mapping[str, Any]
    ) -> tuple[AttackEvent, str]:
        specification = native_result["specification"]
        resolved = native_result["resolved_action"]
        if not isinstance(specification, SemanticAction) or not isinstance(
            resolved, _ResolvedAction
        ):
            raise CybORGAdapterError("native result lacks resolved semantic action")
        succeeded = bool(native_result["success"])
        reward = float(native_result["native_reward"])
        goal = succeeded and reward >= self.scenario.objective_native_reward
        kind_by_behavior = {
            "noop": ActionKind.NOOP,
            "enumerate_service": ActionKind.SERVICE_SCAN,
            "authenticate": ActionKind.EXPLOIT,
            "local_exploit": ActionKind.EXPLOIT,
            "enumerate_host": ActionKind.OS_SCAN,
            "pivot": ActionKind.EXPLOIT,
            "discover_network": ActionKind.SUBNET_SCAN,
            "exploit": ActionKind.EXPLOIT,
        }
        level = AccessLevel.NONE
        if succeeded and specification.semantic_behavior in {"authenticate", "local_exploit"}:
            level = AccessLevel.USER
        if succeeded and specification.native_kind == "MS17_010_PSExec":
            level = AccessLevel.ROOT
        event = AttackEvent(
            step=self._step,
            kind=kind_by_behavior[specification.semantic_behavior],
            action_name=specification.simulator_action,
            target=resolved.target,
            success=succeeded,
            rl_action_index=specification.index,
            native_reward=reward,
            cve_id=None,
            access_gained=level,
            newly_discovered=len(native_result["knowledge_delta"].discovered_nodes),
            is_crown_jewel=goal,
            goal_reached=goal,
            terminal=goal or bool(native_result["native_done"]),
            error=native_result.get("error"),
        )
        return event, specification.semantic_behavior

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and self._reset_seeds:
            seed = self._reset_seeds[self._reset_index % len(self._reset_seeds)]
            self._reset_index += 1
        self._knowledge = AgentKnowledge()
        self._native_observation = None
        self._visible_action_space = {}
        self._step = 0
        self._node_by_ip.clear()
        self._subnets.clear()
        self._sessions.clear()
        self._credentials.clear()
        self._routed_subnets.clear()
        self._upgraded_sessions.clear()
        self._configured_sessions.clear()
        self._autorouted_sessions.clear()
        self._scanned_subnets.clear()
        result = self._native_env.reset(agent=self.scenario.agent, seed=seed)
        self._native_observation = dict(result.observation)
        self._visible_action_space = dict(result.action_space or {})
        delta = self.update_agent_knowledge(self._native_observation)
        return self.convert_observation(self._native_observation), {
            "simulator": "CybORG",
            "seed": seed,
            "initial_knowledge_delta": delta,
        }

    def step(self, action: int):
        index = int(action)
        valid = bool(self.action_mask()[index]) if 0 <= index < len(self._catalogue) else False
        specification = self._catalogue[index]
        if valid:
            resolved = self._resolve_action(index)
            result = self._native_env.step(
                agent=self.scenario.agent, action=resolved.native_action
            )
            observation = dict(result.observation)
            succeeded = _success(observation.get("success")) and not result.has_error()
            native_reward = float(result.reward or 0.0)
            native_done = bool(result.done)
            error = str(result.error_msg or result.error) if result.has_error() else None
            self._visible_action_space = dict(result.action_space or {})
        else:
            resolved = _ResolvedAction(None, None, None, specification.simulator_vulnerability_id)
            observation = {"success": False}
            succeeded = False
            native_reward = 0.0
            native_done = False
            error = "action masked by observable-state precondition"
        self._step += 1
        self._native_observation = observation
        provisional = {
            "resolved_action": resolved,
            "specification": specification,
            "success": succeeded,
            "native_reward": native_reward,
        }
        delta = self.update_agent_knowledge(observation, native_info=provisional)
        goal = succeeded and native_reward >= self.scenario.objective_native_reward
        truncated = self._step >= self.scenario.max_steps and not (goal or native_done)
        event, behavior = self.native_result_to_event(
            {
                **provisional,
                "knowledge_delta": delta,
                "native_done": native_done,
                "error": error,
            }
        )
        if event.goal_reached:
            policy_reward = 100.0
        elif delta.changed:
            policy_reward = float(
                5
                + 5 * len(delta.discovered_nodes)
                + 2 * len(delta.discovered_services)
                + 10 * len(delta.access_changes)
            )
        elif succeeded:
            policy_reward = 0.0
        else:
            policy_reward = float(self.scenario.failure_penalty)
        transition = SimulatorTransition(
            observation=self.convert_observation(observation),
            reward=policy_reward,
            terminated=bool(goal or native_done),
            truncated=bool(truncated),
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
                "simulator": "CybORG",
                "simulator_transition": transition,
                "native_reward": native_reward,
                "action_valid": valid,
            },
        )

    def agent_knowledge_graph(self) -> dict[str, list[dict[str, Any]]]:
        """Return recorded observable knowledge; never query the native simulator."""
        nodes = [
            {
                "id": node,
                "type": self._knowledge.known_node_types[node].value,
                "properties": sorted(self._knowledge.known_properties.get(node, set())),
                "services": sorted(self._knowledge.known_services.get(node, set())),
                "access": self._knowledge.access.get(node, "none"),
            }
            for node in self._knowledge.discovery_order
        ]
        edges = [
            {"source": source, "target": target, "type": edge.value}
            for source, target, edge in sorted(
                self._knowledge.known_edges,
                key=lambda item: (item[0], item[1], item[2].value),
            )
        ]
        return {"nodes": nodes, "edges": edges}

    def space_summary(self) -> dict[str, Any]:
        return {
            "action_space": f"Discrete({self.action_space.n})",
            "observation_space": f"Box(0.0, 1.0, {self.observation_space.shape})",
            "action_mask": True,
            "native_action_kinds": [item.native_kind for item in self._catalogue],
            "policy_observation_source": "AgentKnowledge_and_visible_action_mask_only",
            "policy_reward": "bounded_agent_knowledge_progress",
        }

    def close(self) -> None:
        self._native_env.shutdown()


def cyborg_adapter_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
