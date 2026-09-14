"""Bridge between NASim and the environment-agnostic reward core.

This is the only module that knows about both. It converts a NASim step into an
:class:`~rlredteam.events.AttackEvent` and applies the reward engine on top.

Two NASim facts drive the design, both verified against nasim 0.12.0:

* ``step()`` returns ``(obs, reward, done, step_limit_reached, info)``.
* ``info`` carries no action and no target, so the acting action must be
  resolved separately via ``action_space.get_action(idx)``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from nasim.envs.action import (
    Exploit,
    NoOp,
    OSScan,
    PrivilegeEscalation,
    ProcessScan,
    ServiceScan,
    SubnetScan,
)

from rlredteam.assign import Assignment, assign_cves
from rlredteam.catalogue import CVECatalogue
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.state import AgentKnowledge
from rlredteam.events import AccessLevel, ActionKind, AttackEvent
from rlredteam.reward import RewardBreakdown, RewardConfig, RewardEngine
from rlredteam.simulator_adapter import (
    KnowledgeDelta,
    SemanticAction,
    SimulatorAdapter,
    SimulatorTransition,
)

_ACTION_KIND: dict[type, ActionKind] = {
    Exploit: ActionKind.EXPLOIT,
    PrivilegeEscalation: ActionKind.PRIVESC,
    ServiceScan: ActionKind.SERVICE_SCAN,
    OSScan: ActionKind.OS_SCAN,
    SubnetScan: ActionKind.SUBNET_SCAN,
    ProcessScan: ActionKind.PROCESS_SCAN,
    NoOp: ActionKind.NOOP,
}


class AdapterError(RuntimeError):
    pass


def _access_level(
    raw: object, target: tuple[int, int] | None = None
) -> AccessLevel:
    """Normalise ``info["access"]`` to an AccessLevel.

    NASim's ActionResult docstring documents ``access`` as a dict of
    address -> level, but the exploit and privesc path in host_vector.py
    actually passes ``access=action.access``, a bare int. Both shapes are
    handled so a NASim change in either direction cannot silently zero the
    access level -- which would make every event uninformative and disable
    all shaping.
    """
    if not raw:
        return AccessLevel.NONE
    if isinstance(raw, dict):
        if target is not None:
            # Access belongs to the action's target. Taking max(raw.values())
            # can incorrectly report ROOT on B merely because A was already
            # root-compromised, corrupting reward and path reconstruction.
            return AccessLevel(int(raw.get(target, AccessLevel.NONE)))
        if len(raw) == 1:
            return AccessLevel(int(next(iter(raw.values()))))
        if len(raw) > 1:
            raise AdapterError("target is required for a multi-host access mapping")
        return AccessLevel.NONE
    return AccessLevel(int(raw))


def _count(raw: object) -> int:
    """Length of a dict/collection field, tolerating a bare count or None."""
    if not raw:
        return 0
    if isinstance(raw, int):
        return raw
    return len(raw)


class NASimEventAdapter:
    """Turns a NASim step into an :class:`AttackEvent`."""

    def __init__(
        self,
        env,
        catalogue: CVECatalogue,
        topology_seed: int,
        require_cve: bool = True,
    ) -> None:
        self.env = env
        self.topology_seed = topology_seed
        scenario = env.scenario

        exploit_names = list(scenario.exploits)
        privesc_names = list(scenario.privescs)
        self.assignment: Assignment = assign_cves(
            exploit_names, privesc_names, catalogue, topology_seed
        )

        if require_cve:
            missing = [
                name
                for name in exploit_names + privesc_names
                if self.assignment.cve_for(name) is None
            ]
            if missing:
                # Never degrade silently: an unmapped exploit would score zero
                # for its CVE term and quietly turn a shaped run into a
                # partially-unshaped one.
                raise AdapterError(f"no CVE assigned for actions: {missing}")

        self._sensitive: set[tuple[int, int]] = set(scenario.sensitive_hosts)

    def kind_of(self, action) -> ActionKind:
        try:
            return _ACTION_KIND[type(action)]
        except KeyError:
            raise AdapterError(f"unknown NASim action type {type(action)}") from None

    def build(
        self,
        action_idx: int,
        reward: float,
        done: bool,
        step_limit_reached: bool,
        info: dict,
        step: int,
    ) -> AttackEvent:
        action = self.env.action_space.get_action(int(action_idx))
        kind = self.kind_of(action)
        target = tuple(action.target) if action.target is not None else None

        record = self.assignment.cve_for(action.name)
        success = bool(info.get("success", False))

        access = _access_level(info.get("access"), target)

        error = "mitigation_disabled" if info.get("mitigation_blocked") else None
        for flag, label in (
            ("connection_error", "connection"),
            ("permission_error", "permission"),
            ("undefined_error", "undefined"),
        ):
            if error is None and info.get(flag):
                error = label
                break

        return AttackEvent(
            step=step,
            kind=kind,
            action_name=action.name,
            target=target,
            success=success,
            rl_action_index=int(action_idx),
            native_reward=float(reward),
            cost=float(action.cost),
            cve_id=record.cve_id if record else None,
            cvss_base=record.base_score if record else None,
            service=getattr(action, "service", None),
            access_gained=access,
            newly_discovered=_count(info.get("newly_discovered")),
            is_crown_jewel=bool(success and target in self._sensitive and access > 0),
            goal_reached=bool(done),
            terminal=bool(done or step_limit_reached),
            error=error,
        )


class RewardWrapper(gym.Wrapper):
    """Replaces the NASim reward with the configured reward engine.

    The shaped reward *replaces* rather than adds to the native reward: the
    locked spec already defines the terminal (+100) and failure (-5) cases, and
    NASim natively pays ``value - cost``, so stacking would double-count every
    crown jewel. The native value is preserved in ``info["native_reward"]`` and
    is what evaluation reports, since comparing raw shaped returns across arms
    would measure each with its own ruler.
    """

    def __init__(
        self,
        env,
        catalogue: CVECatalogue,
        topology_seed: int,
        reward_config: RewardConfig | None = None,
        check_goal_dominance: bool = True,
    ) -> None:
        super().__init__(env)
        self.adapter = NASimEventAdapter(env, catalogue, topology_seed)
        self.engine = RewardEngine(reward_config or RewardConfig())
        self._step = 0
        self.last_breakdown: RewardBreakdown | None = None

        if check_goal_dominance:
            self.engine.assert_goal_dominance(
                num_hosts=len(env.scenario.hosts),
                num_crown_jewels=len(env.scenario.sensitive_hosts),
            )

    def reset(self, **kwargs):
        self.engine.reset()
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # NASim's FlatActionSpace.get_action asserts isinstance(idx, int), and
        # SB3 hands down a numpy int64, which fails that check. Coerce here so
        # any array-based caller works rather than pushing the burden upstream.
        action = int(action)
        # CP-13. NASim indexes a Python list, so a negative index silently
        # aliases a valid action from the other end -- action -1 executes the
        # LAST action rather than failing. A bad index must be rejected, not
        # quietly turned into a different move.
        if not 0 <= action < int(self.action_space.n):
            raise ValueError(
                f"action {action} outside [0, {int(self.action_space.n)}) — "
                "refusing to execute; a negative index would alias a valid action"
            )
        obs, reward, done, step_limit_reached, info = self.env.step(action)
        event = self.adapter.build(
            action, reward, done, step_limit_reached, info, self._step
        )
        breakdown = self.engine.score(event)
        self._step += 1
        self.last_breakdown = breakdown

        info = dict(info)
        info["native_reward"] = float(reward)
        info["reward_breakdown"] = breakdown
        info["attack_event"] = event
        return obs, breakdown.total, done, step_limit_reached, info


class NASimSimulatorAdapter(SimulatorAdapter):
    """Additive common-contract facade over the frozen NASim reward wrapper.

    Existing training and evaluation continue to use :class:`RewardWrapper`
    directly.  This facade proves NASim and other simulators can expose the
    same semantic boundary without altering the frozen NASim execution path.
    """

    def __init__(self, environment: RewardWrapper) -> None:
        self.environment = environment
        self._knowledge = AgentKnowledge()
        self._catalogue = self._build_action_catalogue()

    @property
    def agent_knowledge(self) -> AgentKnowledge:
        return self._knowledge

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def observation_space(self):
        return self.environment.observation_space

    def _build_action_catalogue(self) -> tuple[SemanticAction, ...]:
        rows: list[SemanticAction] = []
        for index in range(int(self.environment.action_space.n)):
            action = self.environment.action_space.get_action(index)
            kind = self.environment.adapter.kind_of(action)
            rows.append(
                SemanticAction(
                    index=index,
                    native_kind=type(action).__name__,
                    simulator_action=str(action.name),
                    semantic_behavior=str(kind),
                )
            )
        return tuple(rows)

    def action_catalogue(self) -> tuple[SemanticAction, ...]:
        return self._catalogue

    def action_mask(self) -> np.ndarray | None:
        return None

    def convert_observation(self, native_observation) -> np.ndarray:
        return np.asarray(native_observation, dtype=np.float32).reshape(-1).copy()

    def reset(self, *, seed: int | None = None):
        self._knowledge = AgentKnowledge()
        observation, info = self.environment.reset(seed=seed)
        return self.convert_observation(observation), dict(info)

    def update_agent_knowledge(
        self, native_observation, *, native_info=None
    ) -> KnowledgeDelta:
        del native_observation
        info = dict(native_info or {})
        event = info.get("attack_event")
        if not isinstance(event, AttackEvent):
            return KnowledgeDelta()

        before_nodes = set(self._knowledge.discovered)
        before_access = dict(self._knowledge.access)
        target = event.target
        if isinstance(target, tuple) and event.success:
            node_id = f"nasim:{target[0]}.{target[1]}"
            self._knowledge.discover(node_id, NodeType.HOST, reachable=True)
            if event.kind.is_scan:
                self._knowledge.enumerated.add(node_id)
            if event.access_gained > AccessLevel.NONE:
                self._knowledge.access[node_id] = (
                    "root" if event.access_gained is AccessLevel.ROOT else "user"
                )
            if event.cve_id:
                self._knowledge.learn_vulnerability(
                    event.cve_id,
                    node_id,
                    grants_access_to=node_id,
                    privilege=self._knowledge.access.get(node_id, "user"),
                )

        new_nodes = tuple(sorted(self._knowledge.discovered - before_nodes))
        changed_access = tuple(
            sorted(
                (node, level)
                for node, level in self._knowledge.access.items()
                if before_access.get(node) != level
            )
        )
        return KnowledgeDelta(discovered_nodes=new_nodes, access_changes=changed_access)

    def native_result_to_event(self, native_result):
        event = native_result.get("attack_event")
        if not isinstance(event, AttackEvent):
            raise AdapterError("NASim result does not contain a normalised attack event")
        return event, str(event.kind)

    def step(self, action: int):
        index = int(action)
        if not 0 <= index < len(self._catalogue):
            raise ValueError(f"action {index} outside common NASim catalogue")
        observation, reward, terminated, truncated, info = self.environment.step(index)
        event, behavior = self.native_result_to_event(info)
        delta = self.update_agent_knowledge(observation, native_info=info)
        transition = SimulatorTransition(
            observation=self.convert_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            event=event,
            action=self._catalogue[index],
            semantic_behavior=behavior,
            simulator_action=event.action_name,
            simulator_vulnerability_id=None,
            knowledge_delta=delta,
        )
        output_info = dict(info)
        output_info["simulator_transition"] = transition
        return (
            transition.observation,
            transition.reward,
            transition.terminated,
            transition.truncated,
            output_info,
        )
