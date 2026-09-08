import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlredteam.enterprise.hierarchical_policy import HierarchicalResearchConfig
from rlredteam.enterprise.hierarchical_study import (
    HierarchicalStudyError,
    TrainingDistributionAudit,
    current_input_manifest,
    evaluate_arm,
    train_arm,
    validate_paired_training_isolation,
)
from rlredteam.enterprise.profiles import InfrastructureCurriculumEnv
from scripts.run_hierarchical_study import (
    project_parallel_training_minutes,
    validate_development_gate,
)


class HighestValidPolicy:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1, bias=False)

    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation, *, action_masks, deterministic: bool):
        del observation, deterministic
        return int(np.flatnonzero(action_masks)[-1]), None


def test_phase12_input_manifest_hashes_hierarchy_and_research_boundary() -> None:
    config = HierarchicalResearchConfig.from_yaml()
    manifest = current_input_manifest(config)
    assert manifest["experiment_id"] == config.experiment_id
    assert manifest["observation_schema"]["source"] == "AgentKnowledge"
    assert manifest["observation_schema"]["hidden_topology_fields"] == 0
    assert manifest["hierarchy"]["source"] == "static_public_slot_catalogue"
    assert manifest["hierarchy"]["transition_semantics"] == (
        "one_concrete_action_per_environment_step"
    )
    assert len(manifest["hierarchy"]["action_to_objective"]) == 968
    assert len(manifest["distribution_sha256"]) == 64
    assert len(manifest["vulnerability_snapshot_sha256"]) == 64
    assert "src/rlredteam/enterprise/hierarchical_policy.py" in manifest["source_sha256"]
    assert "scripts/run_hierarchical_study.py" in manifest["source_sha256"]


def test_phase12_training_distribution_reset_sequence_is_deterministic() -> None:
    config = HierarchicalResearchConfig.from_yaml()
    environments = [
        TrainingDistributionAudit(
            InfrastructureCurriculumEnv(config.topology_splits["train"], config.train_profiles)
        )
        for _ in range(2)
    ]
    for environment in environments:
        environment.reset(seed=3701)
        for _ in range(15):
            environment.reset()
    assert environments[0].records == environments[1].records


def test_phase12_frozen_evaluation_is_masked_distinct_and_immutable() -> None:
    model = HighestValidPolicy()
    before = model.policy.weight.detach().clone()
    config = HierarchicalResearchConfig.from_yaml()
    episodes, steps, integrity = evaluate_arm(
        model,
        arm="hierarchical_graph",
        training_seed=501,
        config=config,
        topology_seeds=(11001,),
    )
    assert len(episodes) == 3
    assert {row["profile"] for row in episodes} == {"legacy", "cloud", "hybrid"}
    assert all(row["goal_reached"] for row in episodes)
    assert all(row["invalid_mask_selections"] == 0 for row in episodes)
    assert integrity["evaluation_reset_count"] == 3
    assert integrity["knowledge_mask_checks"] == len(steps)
    assert integrity["environment_steps"] == integrity["option_decisions"] == len(steps)
    assert integrity["policy_sha256_before"] == integrity["policy_sha256_after"]
    assert torch.equal(before, model.policy.weight)


@pytest.mark.slow
def test_phase12_both_arms_train_exact_rollouts_and_are_isolated(tmp_path: Path) -> None:
    config = HierarchicalResearchConfig.from_yaml()
    manifests = [
        train_arm(
            tmp_path / arm,
            arm=arm,
            training_seed=config.development_seed,
            config=config,
            timesteps=256,
            development=True,
            allow_dirty=True,
        )
        for arm in config.arms
    ]
    for manifest in manifests:
        assert manifest["actual_training_timesteps"] == 256
        assert manifest["training_reset_count"] > 0
        assert manifest["action_mask_source"] == "AgentKnowledge"
        assert manifest["observation_source"] == "AgentKnowledge"
        assert manifest["transition_semantics"] == "one_concrete_action_per_environment_step"
        assert Path(manifest["checkpoint"]).is_file()
    assert manifests[0]["parameter_count"] != manifests[1]["parameter_count"]
    validate_paired_training_isolation(*manifests)
    poisoned = dict(manifests[1])
    poisoned["ppo"] = {**poisoned["ppo"], "gamma": 0.5}
    with pytest.raises(HierarchicalStudyError, match="outside hierarchy"):
        validate_paired_training_isolation(manifests[0], poisoned)


def test_phase12_runtime_projection_and_development_gate(tmp_path: Path) -> None:
    config = HierarchicalResearchConfig.from_yaml()
    elapsed = {config.arms[0]: 900.0, config.arms[1]: 950.0}
    projected = project_parallel_training_minutes(
        elapsed,
        seeds=len(config.training_seeds),
        arms=config.arms,
        workers=config.parallel_training_workers,
    )
    assert projected < config.runtime_cap_minutes
    metadata = {
        "phase": "development",
        "complete": True,
        "scientific_config_hash": config.scientific_digest(),
        "training_timesteps": config.total_timesteps,
        "topology_seeds": list(config.topology_splits["validation"]),
        "training_elapsed_seconds_by_arm": elapsed,
        "training_manifests": [
            {
                "arm": arm,
                "training_seed": config.development_seed,
                "development": True,
            }
            for arm in config.arms
        ],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(metadata))
    validated = validate_development_gate(path, config)
    assert validated["parallel_projected_canonical_minutes"] == projected
