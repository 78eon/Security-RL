from dataclasses import replace

import numpy as np

from rlredteam.enterprise.curriculum import (
    BOUND_FIELDS,
    CurriculumResearchConfig,
    StageAuditEnv,
    resolve_schedule,
    stage_distribution_manifest,
)
from rlredteam.enterprise.curriculum_study import (
    train_arm,
    validate_paired_training_isolation,
)
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
)


def tiny_config() -> CurriculumResearchConfig:
    config = CurriculumResearchConfig.from_yaml()
    stages = tuple(replace(stage, rollouts=1) for stage in config.stages)
    return replace(
        config,
        total_timesteps=32,
        common_ppo={
            **config.common_ppo,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
        },
        policy={"policy_layers": [16]},
        stages=stages,
        topology_splits={
            "train": (1, 2, 3),
            "validation": (1001,),
            "test": (2001,),
        },
    )


def test_phase9_configuration_pins_matched_design() -> None:
    config = CurriculumResearchConfig.from_yaml()
    assert config.arms == ("direct_full_distribution", "staged_curriculum")
    assert config.training_seeds == tuple(range(201, 211))
    assert config.development_seed not in config.training_seeds
    assert config.total_timesteps == 50176
    assert config.stage_timesteps == 12544
    assert config.parallel_training_workers == 6
    assert sum(stage.rollouts for stage in config.stages) == 196
    assert set(config.stages[-1].bounds) == set(BOUND_FIELDS)
    assert config.stages[-1].profiles == config.train_profiles


def test_direct_arm_controls_segment_lifecycle_while_curriculum_changes_distribution() -> None:
    config = CurriculumResearchConfig.from_yaml()
    direct = resolve_schedule(config, config.arms[0])
    curriculum = resolve_schedule(config, config.arms[1])
    base = EnterpriseProfileConfig.from_yaml()

    assert [stage.timesteps for stage in direct] == [12544] * 4
    assert [stage.timesteps for stage in curriculum] == [12544] * 4
    assert all(stage.profiles == config.train_profiles for stage in direct)
    assert all(stage.profile_config.digest() == base.digest() for stage in direct)
    assert curriculum[0].profiles == (DeploymentProfile.LEGACY,)
    assert curriculum[1].profiles == (
        DeploymentProfile.LEGACY,
        DeploymentProfile.CLOUD,
    )
    assert curriculum[2].profiles == (DeploymentProfile.HYBRID,)
    assert curriculum[3].profiles == config.train_profiles
    assert curriculum[3].profile_config.digest() == base.digest()


def test_stage_distribution_manifest_is_deterministic_and_arm_specific() -> None:
    config = tiny_config()
    direct = stage_distribution_manifest(config, config.arms[0])
    repeated = stage_distribution_manifest(config, config.arms[0])
    curriculum = stage_distribution_manifest(config, config.arms[1])
    assert direct == repeated
    assert direct != curriculum
    assert len(direct) == len(curriculum) == 4
    assert all(len(item["topology_distribution_sha256"]) == 64 for item in direct)


def test_stage_audit_records_selected_distribution_without_truth_payload() -> None:
    base = InfrastructureCurriculumEnv(
        (1,),
        (DeploymentProfile.LEGACY,),
        config=EnterpriseProfileConfig.from_yaml(),
    )
    env = StageAuditEnv(base, stage_name="audit")
    observation, info = env.reset(seed=77)
    assert observation.shape == env.observation_space.shape
    assert info["profile"] == "legacy"
    assert env.reset_records == [
        {
            "stage": "audit",
            "profile": "legacy",
            "topology_seed": 1,
            "topology_hash": info["topology_hash"],
            "profile_config_hash": info["profile_config_hash"],
        }
    ]
    assert "nodes" not in env.reset_records[0]
    assert env.exposure_counts() == {"legacy:1": 1}
    assert len(env.trace_digest()) == 64
    env.close()


def test_phase9_training_mask_cannot_read_hidden_topology() -> None:
    base = InfrastructureCurriculumEnv(
        (2,),
        (DeploymentProfile.HYBRID,),
        config=EnterpriseProfileConfig.from_yaml(),
    )
    env = StageAuditEnv(base, stage_name="truth-isolation")
    observation, _ = env.reset(seed=78)
    expected_mask = env.action_masks().copy()

    class ForbiddenTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"Phase 9 policy path read hidden topology: {name}")

    env.unwrapped._env.true_topology = ForbiddenTruth()
    assert env.observation_space.contains(observation)
    assert env.reset_records[0].keys() == {
        "stage",
        "profile",
        "topology_seed",
        "topology_hash",
        "profile_config_hash",
    }
    assert np.array_equal(env.action_masks(), expected_mask)
    env.close()


def test_stage_resets_are_deterministic_for_identical_seeds() -> None:
    environments = [
        StageAuditEnv(
            InfrastructureCurriculumEnv(
                (1, 2, 3),
                (DeploymentProfile.LEGACY, DeploymentProfile.CLOUD),
                config=EnterpriseProfileConfig.from_yaml(),
            ),
            stage_name="deterministic-reset",
        )
        for _ in range(2)
    ]
    first_observation, first_info = environments[0].reset(seed=99)
    second_observation, second_info = environments[1].reset(seed=99)

    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(environments[0].action_masks(), environments[1].action_masks())
    assert first_info.keys() == second_info.keys()
    for key, first_value in first_info.items():
        second_value = second_info[key]
        if isinstance(first_value, np.ndarray):
            assert np.array_equal(first_value, second_value)
        else:
            assert first_value == second_value
    assert environments[0].reset_records == environments[1].reset_records
    assert environments[0].trace_digest() == environments[1].trace_digest()
    for environment in environments:
        environment.close()


def test_tiny_matched_training_has_exact_stage_budgets_and_isolation(tmp_path) -> None:
    config = tiny_config()
    manifests = {}
    for arm in config.arms:
        manifests[arm] = train_arm(
            tmp_path / arm,
            arm=arm,
            training_seed=config.development_seed,
            config=config,
            timesteps=config.total_timesteps,
            development=True,
            allow_dirty=True,
        )
        assert manifests[arm]["actual_training_timesteps"] == 32
        assert [item["actual_stage_timesteps"] for item in manifests[arm]["stage_audit"]] == [
            8,
            8,
            8,
            8,
        ]
        assert [item["cumulative_timesteps"] for item in manifests[arm]["stage_audit"]] == [
            8,
            16,
            24,
            32,
        ]
        assert all(item["reset_count"] >= 1 for item in manifests[arm]["stage_audit"])
    validate_paired_training_isolation(manifests[config.arms[0]], manifests[config.arms[1]])
