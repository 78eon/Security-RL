"""Phase 8 recurrent-policy transport, masking and memory invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlredteam.enterprise.profiles import DeploymentProfile, InfrastructureCurriculumEnv
from rlredteam.enterprise.recurrent import (
    KnowledgeActionGuard,
    KnowledgeFeatureExtractor,
    KnowledgeMaskedRecurrentPolicy,
    KnowledgeMaskObservationWrapper,
    RecurrentResearchConfig,
)

PROFILES = (
    DeploymentProfile.LEGACY,
    DeploymentProfile.CLOUD,
    DeploymentProfile.HYBRID,
)


def wrapped_env(seed: int = 1) -> KnowledgeMaskObservationWrapper:
    return KnowledgeMaskObservationWrapper(
        KnowledgeActionGuard(InfrastructureCurriculumEnv((seed,), PROFILES))
    )


def make_model(env: KnowledgeMaskObservationWrapper):
    from sb3_contrib import RecurrentPPO

    return RecurrentPPO(
        KnowledgeMaskedRecurrentPolicy,
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=7,
        device="cpu",
        policy_kwargs={
            "feature_dim": 16,
            "lstm_hidden_size": 16,
            "n_lstm_layers": 1,
            "net_arch": [16],
        },
    )


def test_phase8_configuration_pins_matched_design() -> None:
    config = RecurrentResearchConfig.from_yaml()
    assert config.arms == (
        "maskable_ppo",
        "knowledge_masked_recurrent_ppo",
    )
    assert config.training_seeds == tuple(range(101, 111))
    assert config.development_seed not in config.training_seeds
    assert config.total_timesteps == 50176
    assert config.total_timesteps % config.common_ppo["n_steps"] == 0
    assert config.primary_metrics == (
        "success_rate",
        "penalized_steps",
        "total_reward",
    )
    assert len(config.digest()) == 64


def test_wrapper_transports_exact_knowledge_mask_without_learning_it() -> None:
    env = wrapped_env()
    observation, _ = env.reset(
        seed=11,
        options={"profile": DeploymentProfile.HYBRID.value, "topology_seed": 1},
    )
    policy_observation, transported_mask = env.split_observation(observation)
    assert env.policy_observation_space.contains(policy_observation)
    assert np.array_equal(transported_mask.astype(bool), env.action_masks())
    assert transported_mask.sum() > 0

    extractor = KnowledgeFeatureExtractor(
        env.observation_space,
        policy_observation_size=env.policy_observation_size,
        feature_dim=16,
    )
    first = torch.as_tensor(observation).unsqueeze(0)
    changed_mask = first.clone()
    changed_mask[:, env.policy_observation_size :] = 1 - changed_mask[
        :, env.policy_observation_size :
    ]
    assert torch.equal(extractor(first), extractor(changed_mask))
    env.close()


def test_wrapper_mask_cannot_read_ground_truth() -> None:
    env = wrapped_env()
    observation, _ = env.reset(
        seed=12,
        options={"profile": DeploymentProfile.LEGACY.value, "topology_seed": 1},
    )
    expected = env.action_masks().copy()

    class ForbiddenTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"mask transport read ground truth: {name}")

    env.unwrapped._env.true_topology = ForbiddenTruth()
    assert np.array_equal(env.action_masks(), expected)
    assert np.array_equal(env.split_observation(observation)[1].astype(bool), expected)
    env.close()


def test_action_guard_rejects_a_masked_action_before_environment_step() -> None:
    guard = KnowledgeActionGuard(InfrastructureCurriculumEnv((1,), PROFILES))
    guard.reset(
        seed=12,
        options={"profile": DeploymentProfile.LEGACY.value, "topology_seed": 1},
    )
    invalid = int(np.flatnonzero(~guard.action_masks())[0])
    with pytest.raises(RuntimeError, match="outside AgentKnowledge mask"):
        guard.step(invalid)
    assert guard.invalid_action_selections == 1
    guard.close()


def test_recurrent_policy_never_selects_a_masked_action() -> None:
    env = wrapped_env()
    model = make_model(env)
    observation, _ = env.reset(
        seed=13,
        options={"profile": DeploymentProfile.CLOUD.value, "topology_seed": 1},
    )
    state = None
    episode_start = np.ones((1,), dtype=bool)
    for _ in range(20):
        mask = env.action_masks().copy()
        action, state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=False,
        )
        action = int(np.asarray(action).item())
        assert mask[action]
        observation, _, terminated, truncated, _ = env.step(action)
        episode_start = np.asarray([terminated or truncated], dtype=bool)
        if terminated or truncated:
            break
    env.close()


def test_recurrent_state_resets_at_episode_boundary() -> None:
    env = wrapped_env()
    model = make_model(env)
    options = {"profile": DeploymentProfile.HYBRID.value, "topology_seed": 1}
    observation, _ = env.reset(seed=14, options=options)
    start = np.ones((1,), dtype=bool)

    first_action, first_state = model.predict(
        observation, state=None, episode_start=start, deterministic=True
    )
    continued_action, continued_state = model.predict(
        observation,
        state=first_state,
        episode_start=np.zeros((1,), dtype=bool),
        deterministic=True,
    )
    reset_action, reset_state = model.predict(
        observation,
        state=continued_state,
        episode_start=start,
        deterministic=True,
    )
    fresh_action, fresh_state = model.predict(
        observation, state=None, episode_start=start, deterministic=True
    )

    assert np.array_equal(reset_action, fresh_action)
    assert all(np.allclose(a, b) for a, b in zip(reset_state, fresh_state, strict=True))
    assert any(
        not np.allclose(a, b)
        for a, b in zip(first_state, continued_state, strict=True)
    )
    assert env.action_masks()[int(np.asarray(continued_action).item())]
    env.close()


def test_recurrent_training_path_applies_masks_during_updates() -> None:
    env = wrapped_env()
    model = make_model(env)
    model.learn(total_timesteps=8)
    assert model.num_timesteps == 8
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    env.close()


def test_policy_fails_closed_when_no_action_is_valid() -> None:
    env = wrapped_env()
    model = make_model(env)
    observation, _ = env.reset(
        seed=15,
        options={"profile": DeploymentProfile.LEGACY.value, "topology_seed": 1},
    )
    observation[env.policy_observation_size :] = 0.0
    with pytest.raises(RuntimeError, match="no valid AgentKnowledge action"):
        model.predict(observation, deterministic=True)
    env.close()
