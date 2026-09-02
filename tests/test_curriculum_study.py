import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlredteam.enterprise.curriculum import CurriculumResearchConfig
from rlredteam.enterprise.curriculum_study import (
    CurriculumStudyError,
    aggregate_seed_metrics,
    analyse_seed_metrics,
    current_input_manifest,
    evaluate_arm,
    validate_finite_evidence,
)
from rlredteam.enterprise.profiles import DeploymentProfile
from scripts.run_curriculum_study import (
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


def test_phase9_input_manifest_hashes_every_research_boundary() -> None:
    config = CurriculumResearchConfig.from_yaml()
    manifest = current_input_manifest(config)
    assert manifest["experiment_id"] == config.experiment_id
    assert len(manifest["schedule_distribution_sha256"]) == 64
    assert len(manifest["base_distribution_sha256"]) == 64
    assert set(manifest["source_sha256"]) == {
        "scripts/run_curriculum_study.py",
        "src/rlredteam/analyse.py",
        "src/rlredteam/train.py",
        "src/rlredteam/enterprise/curriculum.py",
        "src/rlredteam/enterprise/curriculum_study.py",
        "src/rlredteam/enterprise/environment.py",
        "src/rlredteam/enterprise/generalisation.py",
        "src/rlredteam/enterprise/infrastructure_generalisation.py",
        "src/rlredteam/enterprise/model.py",
        "src/rlredteam/enterprise/onprem.py",
        "src/rlredteam/enterprise/profiles.py",
        "src/rlredteam/enterprise/recurrent.py",
        "src/rlredteam/enterprise/state.py",
        "src/rlredteam/storage/postgres_logger.py",
        "src/rlredteam/storage/schema.sql",
    }


def test_phase9_frozen_evaluation_is_masked_distinct_and_immutable() -> None:
    model = HighestValidPolicy()
    before = model.policy.weight.detach().clone()
    profiles = (
        DeploymentProfile.LEGACY,
        DeploymentProfile.CLOUD,
        DeploymentProfile.HYBRID,
    )
    episodes, steps, integrity = evaluate_arm(
        model,
        arm="direct_full_distribution",
        training_seed=201,
        profiles=profiles,
        topology_seeds=(1001,),
        evaluation_episode_seeds=(9201,),
    )
    assert len(episodes) == 3
    assert {row["profile"] for row in episodes} == {
        "legacy",
        "cloud",
        "hybrid",
    }
    assert all(row["goal_reached"] for row in episodes)
    assert all(row["invalid_mask_selections"] == 0 for row in episodes)
    assert all(row["arm"] == "direct_full_distribution" for row in episodes + steps)
    assert integrity["evaluation_reset_count"] == 3
    assert integrity["knowledge_mask_checks"] == len(steps)
    assert integrity["policy_sha256_before"] == integrity["policy_sha256_after"]
    assert torch.equal(before, model.policy.weight)


def _episode(arm: str, seed: int, value: float) -> dict:
    return {
        "arm": arm,
        "training_seed": seed,
        "profile": "legacy",
        "topology_seed": 2001,
        "evaluation_seed": 9201,
        "goal_reached": True,
        "steps_to_goal": int(value),
        "total_reward": value,
        "discovery_coverage": value / 100,
        "failed_actions": 0,
    }


def test_phase9_aggregation_and_statistics_use_matched_training_seed() -> None:
    config = CurriculumResearchConfig.from_yaml()
    config = CurriculumResearchConfig(
        **{
            **{field: getattr(config, field) for field in config.__dataclass_fields__},
            "train_profiles": (DeploymentProfile.LEGACY,),
            "evaluation_episode_seeds": (9201,),
        }
    )
    rows = []
    for seed in config.training_seeds:
        rows.extend(
            [
                _episode(config.arms[0], seed, float(seed % 7 + 20)),
                _episode(config.arms[1], seed, float(seed % 7 + 18)),
            ]
        )
    metrics = aggregate_seed_metrics(rows, config, expected_topology_seeds=(2001,))
    report = analyse_seed_metrics(metrics, config)
    assert report["complete"] is True
    assert all(item["n_pairs"] == 10 for item in report["comparisons"])
    assert all(item["difference"] <= 0 for item in report["comparisons"])
    json.dumps(report, allow_nan=False)


def test_phase9_runtime_projection_and_development_gate(tmp_path: Path) -> None:
    config = CurriculumResearchConfig.from_yaml()
    elapsed = {config.arms[0]: 1200.0, config.arms[1]: 1250.0}
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


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_phase9_runtime_projection_rejects_invalid_values(bad: float) -> None:
    with pytest.raises(CurriculumStudyError, match="finite and positive"):
        project_parallel_training_minutes(
            {"direct_full_distribution": bad},
            seeds=1,
            arms=("direct_full_distribution",),
            workers=1,
        )


def test_phase9_evidence_rejects_nonfinite_values() -> None:
    with pytest.raises(CurriculumStudyError, match="non-finite"):
        validate_finite_evidence([{"reward": float("nan")}], [])
