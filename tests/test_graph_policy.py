from pathlib import Path

import numpy as np
import pytest
import torch as th

from rlredteam.enterprise.graph_policy import (
    PHASE10_ARMS,
    AgentKnowledgeGraphExtractor,
    GraphPolicyResearchConfig,
    phase10_policy_kwargs,
)
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment() -> InfrastructureCurriculumEnv:
    return InfrastructureCurriculumEnv((3001,), (DeploymentProfile.HYBRID,))


def _permute_observation(
    observation: np.ndarray,
    permutation: np.ndarray,
    *,
    max_nodes: int,
    feature_count: int,
) -> np.ndarray:
    node_values = max_nodes * feature_count
    nodes = observation[:node_values].reshape(max_nodes, feature_count)
    adjacency = observation[node_values:-1].reshape(max_nodes, max_nodes)
    return np.concatenate(
        (
            nodes[permutation].reshape(-1),
            adjacency[permutation][:, permutation].reshape(-1),
            observation[-1:],
        )
    ).astype(np.float32)


def test_phase10_config_is_prospective_disjoint_and_fixed() -> None:
    config = GraphPolicyResearchConfig.from_yaml(
        REPO_ROOT / "configs/experiments/graph_policy.yaml"
    )
    assert config.arms == PHASE10_ARMS
    assert config.development_seed not in config.training_seeds
    assert len(config.training_seeds) == 10
    assert config.total_timesteps == 50_176
    assert config.topology_splits["train"] == tuple(range(3001, 3061))
    assert config.topology_splits["validation"] == tuple(range(4001, 4021))
    assert config.topology_splits["test"] == tuple(range(5001, 5021))
    train = set(config.topology_splits["train"])
    validation = set(config.topology_splits["validation"])
    test = set(config.topology_splits["test"])
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_graph_extractor_parses_agent_knowledge_observation_and_backpropagates() -> None:
    env = _environment()
    observation, _ = env.reset(seed=9301)
    profile = EnterpriseProfileConfig.from_yaml()
    extractor = AgentKnowledgeGraphExtractor(
        env.observation_space,
        max_nodes=profile.max_nodes,
        node_hidden_dim=16,
        message_passing_steps=2,
        feature_dim=24,
    )
    batch = th.as_tensor(np.stack((observation, observation))).requires_grad_()
    output = extractor(batch)
    assert output.shape == (2, 24)
    assert th.isfinite(output).all()
    output.sum().backward()
    assert batch.grad is not None
    assert th.isfinite(batch.grad).all()


def test_graph_extractor_is_invariant_to_consistent_slot_permutation() -> None:
    env = _environment()
    observation, _ = env.reset(seed=9301)
    profile = EnterpriseProfileConfig.from_yaml()
    feature_count = (len(observation) - profile.max_nodes**2 - 1) // profile.max_nodes
    permutation = np.arange(profile.max_nodes)
    permutation[:8] = np.asarray([3, 5, 0, 7, 1, 6, 2, 4])
    permuted = _permute_observation(
        observation,
        permutation,
        max_nodes=profile.max_nodes,
        feature_count=feature_count,
    )
    th.manual_seed(7)
    extractor = AgentKnowledgeGraphExtractor(
        env.observation_space,
        max_nodes=profile.max_nodes,
        node_hidden_dim=16,
        message_passing_steps=2,
        feature_dim=24,
    )
    with th.no_grad():
        original_embedding = extractor(th.as_tensor(observation).unsqueeze(0))
        permuted_embedding = extractor(th.as_tensor(permuted).unsqueeze(0))
    assert th.allclose(original_embedding, permuted_embedding, atol=1e-6, rtol=1e-6)


def test_graph_policy_configuration_uses_no_environment_or_topology_reference() -> None:
    config = GraphPolicyResearchConfig.from_yaml()
    kwargs = phase10_policy_kwargs(config, "knowledge_graph_gnn")
    assert kwargs["features_extractor_class"] is AgentKnowledgeGraphExtractor
    assert set(kwargs["features_extractor_kwargs"]) == {
        "max_nodes",
        "node_hidden_dim",
        "message_passing_steps",
        "feature_dim",
    }
    with pytest.raises(ValueError, match="unknown Phase 10 arm"):
        phase10_policy_kwargs(config, "oracle_graph")


@pytest.mark.slow
def test_maskable_ppo_can_train_and_predict_with_graph_extractor() -> None:
    from sb3_contrib import MaskablePPO

    config = GraphPolicyResearchConfig.from_yaml()
    env = _environment()
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=35,
        device="cpu",
        verbose=0,
        policy_kwargs=phase10_policy_kwargs(config, "knowledge_graph_gnn"),
        **config.common_ppo,
    )
    model.learn(total_timesteps=256)
    observation, _ = env.reset(seed=9301)
    mask = env.action_masks().copy()
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(np.asarray(action).item())]
