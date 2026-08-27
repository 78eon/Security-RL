"""Deterministic reference episode shared by the CLI and analyst GUI."""

from __future__ import annotations

from rlredteam.enterprise.environment import (
    EnterpriseActionType,
    EnterpriseCyberEnv,
    EnterpriseEvent,
)
from rlredteam.enterprise.generator import generate_enterprise

DEMO_PATH = (
    (EnterpriseActionType.DISCOVER_NETWORK, "seg_dmz"),
    (EnterpriseActionType.ENUMERATE_HOST, "web_host"),
    (EnterpriseActionType.ENUMERATE_SERVICE, "http"),
    (EnterpriseActionType.ENUMERATE_APPLICATION, "portal"),
    (EnterpriseActionType.ASSESS_VULNERABILITY, "http"),
    (EnterpriseActionType.EXPLOIT, "CVE-2021-42013"),
    (EnterpriseActionType.OBTAIN_CREDENTIAL, "svc_orders"),
    (EnterpriseActionType.AUTHENTICATE, "app_host"),
    (EnterpriseActionType.ENUMERATE_HOST, "app_host"),
    (EnterpriseActionType.PIVOT, "db_host"),
    (EnterpriseActionType.ENUMERATE_HOST, "db_host"),
    (EnterpriseActionType.ENUMERATE_APPLICATION, "customer_db"),
    (EnterpriseActionType.AUTHENTICATE, "customer_db"),
    (EnterpriseActionType.ACCESS_ASSET, "customer_records"),
)


def run_demo(seed: int = 42) -> tuple[EnterpriseCyberEnv, tuple[EnterpriseEvent, ...], float]:
    """Execute the known-feasible path and return its actual recorded events."""
    env = EnterpriseCyberEnv(generate_enterprise(seed), max_steps=50, render_mode="ansi")
    env.reset(seed=seed)
    total_reward = 0.0
    for action_type, target in DEMO_PATH:
        _, reward, terminated, truncated, _ = env.step(env.action_index(action_type, target))
        total_reward += reward
        if terminated or truncated:
            break
    return env, env.attack_path(), total_reward
