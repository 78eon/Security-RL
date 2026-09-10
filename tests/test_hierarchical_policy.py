from pathlib import Path

import numpy as np
import pytest
import torch

from rlredteam.enterprise.hierarchical_policy import (
    OBJECTIVES,
    PHASE12_ARMS,
    HierarchicalResearchConfig,
    _factorized_logits,
    action_to_objective,
    fixed_action_catalogue,
    phase12_policy,
    phase12_policy_kwargs,
)
from rlredteam.enterprise.profiles import DeploymentProfile, InfrastructureCurriculumEnv

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phase12_config_is_prospective_fresh_disjoint_and_fixed() -> None:
    config = HierarchicalResearchConfig.from_yaml(
        REPO_ROOT / "configs/experiments/hierarchical_policy.yaml"
    )
    assert config.arms == PHASE12_ARMS
    assert config.development_seed not in config.training_seeds
    assert len(config.training_seeds) == 10
    assert config.total_timesteps == 25_600
    assert config.topology_splits["train"] == tuple(range(10001, 10061))
    assert config.topology_splits["validation"] == tuple(range(11001, 11021))
    assert config.topology_splits["test"] == tuple(range(12001, 12021))
    splits = [set(config.topology_splits[name]) for name in ("train", "validation", "test")]
    assert all(not splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3))


def test_objective_mapping_is_topology_free_and_partitions_public_catalogue() -> None:
    config = HierarchicalResearchConfig.from_yaml()
    mapping = action_to_objective(config)
    catalogue = fixed_action_catalogue()
    assert len(mapping) == len(catalogue) == 968
    assert set(mapping) == set(range(len(OBJECTIVES)))
    assert not any("CVE-" in item for item in catalogue)
    assert all("slot_" in item for item in catalogue)
    expected_objective = {
        action: option
        for option, objective in enumerate(OBJECTIVES)
        for action in config.hierarchy[objective]
    }
    for index, action in enumerate(catalogue):
        assert mapping[index] == expected_objective[action.split(":", 1)[0]]
    assert catalogue[-1].startswith("exploit:")
    assert mapping[-1] == OBJECTIVES.index("gain_access")
    environment = InfrastructureCurriculumEnv((10001,), (DeploymentProfile.HYBRID,))
    environment.reset(seed=9501)
    actual_catalogue = tuple(action.name for action in environment._env.actions)
    assert catalogue == actual_catalogue
    kwargs = phase12_policy_kwargs(config, "hierarchical_graph")
    assert set(kwargs) == {
        "features_extractor_class",
        "features_extractor_kwargs",
        "net_arch",
        "action_to_option",
        "option_dim",
    }
    assert not any("topology" in key or "environment" in key for key in kwargs)
    with pytest.raises(ValueError, match="unknown Phase 12 arm"):
        phase12_policy(config, "oracle")


def test_factorized_distribution_matches_manual_probability_and_backpropagates() -> None:
    option_logits = torch.tensor([[0.3, -0.1]], requires_grad=True)
    action_logits = torch.tensor([[0.2, 0.7, -0.4, 0.1]], requires_grad=True)
    mapping = torch.tensor([0, 0, 1, 1])
    masks = np.asarray([[True, False, True, True]])
    logits, valid = _factorized_logits(option_logits, action_logits, mapping, masks)
    probabilities = torch.softmax(logits, dim=1)
    option_probabilities = torch.softmax(option_logits, dim=1)
    second_conditional = torch.softmax(action_logits[:, 2:], dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert probabilities[0, 1] == 0
    assert torch.allclose(probabilities[0, 0], option_probabilities[0, 0])
    assert torch.allclose(
        probabilities[0, 2:],
        option_probabilities[0, 1] * second_conditional[0],
    )
    assert valid.tolist() == [[True, False, True, True]]
    (-torch.log(probabilities[0, 2])).backward()
    assert option_logits.grad is not None
    assert action_logits.grad is not None
    assert torch.isfinite(option_logits.grad).all()
    assert torch.isfinite(action_logits.grad).all()


def test_mask_eliminates_an_entire_invalid_objective_and_renormalizes() -> None:
    option_logits = torch.tensor([[10.0, -10.0]])
    action_logits = torch.zeros((1, 4))
    mapping = torch.tensor([0, 0, 1, 1])
    logits, _ = _factorized_logits(
        option_logits,
        action_logits,
        mapping,
        np.asarray([[False, False, True, True]]),
    )
    probabilities = torch.softmax(logits, dim=1)
    assert torch.equal(probabilities[:, :2], torch.zeros((1, 2)))
    assert torch.allclose(probabilities[:, 2:], torch.tensor([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="no valid action"):
        _factorized_logits(
            option_logits,
            action_logits,
            mapping,
            np.zeros((1, 4), dtype=bool),
        )


@pytest.mark.slow
def test_hierarchical_maskable_ppo_trains_saves_loads_and_selects_valid_action(
    tmp_path: Path,
) -> None:
    from sb3_contrib import MaskablePPO

    from rlredteam.enterprise.profiles import DeploymentProfile, InfrastructureCurriculumEnv

    config = HierarchicalResearchConfig.from_yaml()
    env = InfrastructureCurriculumEnv((10001,), (DeploymentProfile.HYBRID,))
    model = MaskablePPO(
        phase12_policy(config, "hierarchical_graph"),
        env,
        seed=config.development_seed,
        device="cpu",
        verbose=0,
        policy_kwargs=phase12_policy_kwargs(config, "hierarchical_graph"),
        **config.common_ppo,
    )
    model.learn(total_timesteps=256)
    observation, _ = env.reset(seed=9501)
    mask = env.action_masks().copy()
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(np.asarray(action).item())]
    checkpoint = tmp_path / "hierarchical"
    model.save(checkpoint)
    loaded = MaskablePPO.load(checkpoint.with_suffix(".zip"), device="cpu")
    loaded_action, _ = loaded.predict(observation, action_masks=mask, deterministic=True)
    assert int(np.asarray(loaded_action).item()) == int(np.asarray(action).item())
