"""Deterministic reference episode shared by the CLI and analyst GUI."""

from __future__ import annotations

from rlredteam.enterprise.environment import (
    EnterpriseActionType,
    EnterpriseCyberEnv,
    EnterpriseEvent,
)
from rlredteam.enterprise.generator import (
    generate_enterprise,
    load_reference_enterprise_catalogue,
)

_REFERENCE = load_reference_enterprise_catalogue()
DEMO_PATH = tuple(
    (EnterpriseActionType(action), str(target))
    for action, target in _REFERENCE["demo_path"]
)


def run_demo(seed: int = 42) -> tuple[EnterpriseCyberEnv, tuple[EnterpriseEvent, ...], float]:
    """Execute the known-feasible path and return its actual recorded events."""
    env = EnterpriseCyberEnv(
        generate_enterprise(seed),
        max_steps=int(_REFERENCE["demo_max_steps"]),
        render_mode="ansi",
    )
    env.reset(seed=seed)
    total_reward = 0.0
    for action_type, target in DEMO_PATH:
        _, reward, terminated, truncated, _ = env.step(env.action_index(action_type, target))
        total_reward += reward
        if terminated or truncated:
            break
    return env, env.attack_path(), total_reward
