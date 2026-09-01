"""Phase 8 study provenance, evaluation and paired-analysis tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rlredteam.enterprise.profiles import DeploymentProfile
from rlredteam.enterprise.recurrent import RecurrentResearchConfig
from rlredteam.enterprise.recurrent_study import (
    RecurrentStudyError,
    aggregate_seed_metrics,
    analyse_seed_metrics,
    current_input_manifest,
    evaluate_arm,
    validate_finite_evidence,
    validate_frozen_inputs,
    write_study_summary,
)
from scripts.run_recurrent_study import (
    project_parallel_training_minutes,
    validate_development_gate,
)


class HighestValidPolicy:
    def __init__(self, *, recurrent: bool) -> None:
        self.policy = torch.nn.Linear(1, 1, bias=False)
        self.recurrent = recurrent

    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def predict(
        self,
        observation,
        *,
        deterministic: bool,
        action_masks=None,
        state=None,
        episode_start=None,
    ):
        del deterministic
        if self.recurrent:
            assert action_masks is None
            assert episode_start is not None
            mask = np.asarray(observation[-968:], dtype=bool)
            counter = 0 if state is None or bool(episode_start[0]) else int(state[0][0])
            next_state = (np.asarray([counter + 1]), np.asarray([counter + 1]))
            return int(np.flatnonzero(mask)[-1]), next_state
        assert state is None and episode_start is None
        return int(np.flatnonzero(action_masks)[-1]), None


def test_input_manifest_hashes_every_research_boundary() -> None:
    manifest = current_input_manifest(RecurrentResearchConfig.from_yaml())
    assert manifest["schema_version"] == 1
    assert len(manifest["experiment_config_sha256"]) == 64
    assert len(manifest["profile_config_sha256"]) == 64
    assert len(manifest["distribution_sha256"]) == 64
    assert len(manifest["synthetic_vulnerability_manifest_sha256"]) == 64
    assert set(manifest["source_sha256"]) == {
        "src/rlredteam/enterprise/recurrent.py",
        "src/rlredteam/enterprise/recurrent_study.py",
        "src/rlredteam/enterprise/infrastructure_generalisation.py",
        "src/rlredteam/enterprise/environment.py",
        "src/rlredteam/enterprise/state.py",
        "scripts/run_recurrent_study.py",
    }


def test_six_worker_projection_uses_excluded_timing_and_fits_cap(tmp_path) -> None:
    config = RecurrentResearchConfig.from_yaml()
    elapsed = {
        "maskable_ppo": 1232.54067283,
        "knowledge_masked_recurrent_ppo": 1544.529278333,
    }
    projected = project_parallel_training_minutes(
        elapsed,
        seeds=len(config.training_seeds),
        arms=config.arms,
        workers=config.parallel_training_workers,
    )
    assert projected == pytest.approx(92.57, abs=0.1)
    assert projected < config.runtime_cap_minutes

    metadata = {
        "phase": "development",
        "complete": True,
        # The development run predates the scheduling-only config field.
        "study_config_hash": config.scientific_digest(),
        "training_timesteps": config.total_timesteps,
        "topology_seeds": list(config.topology_splits["validation"]),
        "training_elapsed_seconds_by_arm": elapsed,
        "training_manifests": [
            {"arm": arm, "training_seed": config.development_seed, "development": True}
            for arm in config.arms
        ],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(metadata))
    validated = validate_development_gate(path, config)
    assert validated["parallel_projected_canonical_minutes"] == pytest.approx(
        projected
    )


@pytest.mark.parametrize("bad_timing", [0.0, -1.0, float("nan"), float("inf")])
def test_parallel_projection_rejects_invalid_timing(bad_timing) -> None:
    with pytest.raises(RecurrentStudyError, match="finite and positive"):
        project_parallel_training_minutes(
            {"maskable_ppo": bad_timing},
            seeds=1,
            arms=("maskable_ppo",),
            workers=1,
        )


def test_frozen_inputs_fail_closed_on_drift() -> None:
    config = RecurrentResearchConfig.from_yaml()
    frozen = current_input_manifest(config)
    frozen["distribution_sha256"] = "0" * 64
    with pytest.raises(RecurrentStudyError, match="distribution_sha256"):
        validate_frozen_inputs(frozen, config)


@pytest.mark.parametrize(
    ("arm", "recurrent"),
    [
        ("maskable_ppo", False),
        ("knowledge_masked_recurrent_ppo", True),
    ],
)
def test_frozen_evaluation_uses_same_cases_and_no_invalid_actions(
    arm: str, recurrent: bool
) -> None:
    episodes, steps, integrity = evaluate_arm(
        HighestValidPolicy(recurrent=recurrent),
        arm=arm,
        training_seed=101,
        profiles=(
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        ),
        topology_seeds=(2001,),
        evaluation_episode_seeds=(9101,),
    )
    assert len(episodes) == 3
    assert all(row["goal_reached"] for row in episodes)
    assert all(row["invalid_mask_selections"] == 0 for row in episodes)
    assert {row["profile"] for row in episodes} == {"legacy", "cloud", "hybrid"}
    assert steps[-1]["goal_reached"]
    assert integrity["policy_sha256_before"] == integrity["policy_sha256_after"]
    if recurrent:
        assert integrity["state_reset_count"] == 3
        assert integrity["recurrent_predictions"] > 3
        assert integrity["mask_transport_checks"] == integrity["recurrent_predictions"]
    else:
        assert integrity["state_reset_count"] == 0
        assert integrity["recurrent_predictions"] == 0


def study_episodes(config: RecurrentResearchConfig) -> list[dict]:
    rows = []
    for arm in config.arms:
        recurrent = arm == "knowledge_masked_recurrent_ppo"
        for seed in config.training_seeds:
            for profile in config.train_profiles:
                rows.append(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "profile": profile.value,
                        "topology_seed": 2001,
                        "evaluation_seed": 9101,
                        "goal_reached": True,
                        "steps_to_goal": 80 - (5 if recurrent else 0) + seed % 3,
                        "total_reward": 120 + (10 if recurrent else 0) + seed % 4,
                        "discovery_coverage": 0.8 + (0.01 if recurrent else 0),
                        "failed_actions": 0,
                    }
                )
    return rows


def test_analysis_pairs_on_training_seed_and_is_strict_json(tmp_path) -> None:
    config = RecurrentResearchConfig.from_yaml()
    seed_metrics = aggregate_seed_metrics(
        study_episodes(config), config, expected_topology_seeds=(2001,)
    )
    report = analyse_seed_metrics(seed_metrics, config)
    assert report["complete"]
    assert all(item["n_pairs"] == 10 for item in report["comparisons"])
    success = next(
        item for item in report["comparisons"] if item["metric"] == "success_rate"
    )
    assert success["difference"] == 0.0
    assert success["p_value"] == 1.0
    json.dumps(report, allow_nan=False)

    write_study_summary(
        tmp_path,
        seed_metrics=seed_metrics,
        report=report,
        metadata={"complete": True},
    )
    assert json.loads((tmp_path / "summaries" / "analysis.json").read_text())[
        "complete"
    ]
    assert (tmp_path / "tables" / "statistics.csv").is_file()


def test_aggregate_rejects_missing_or_duplicate_cases() -> None:
    config = RecurrentResearchConfig.from_yaml()
    episodes = study_episodes(config)
    episodes.pop()
    with pytest.raises(RecurrentStudyError, match="expected"):
        aggregate_seed_metrics(episodes, config, expected_topology_seeds=(2001,))


def test_non_finite_evidence_is_rejected() -> None:
    with pytest.raises(RecurrentStudyError, match="non-finite"):
        validate_finite_evidence([{"total_reward": float("nan")}], [])
