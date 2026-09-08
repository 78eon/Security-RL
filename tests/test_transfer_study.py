import copy
import json
from pathlib import Path

import pytest

from rlredteam.enterprise.transfer_learning import TransferResearchConfig
from rlredteam.enterprise.transfer_study import (
    TransferStudyError,
    current_input_manifest,
    evaluate_arm,
    load_model,
    persist_and_write_run_evaluation,
    train_arm,
    validate_finite_evidence,
    validate_frozen_inputs,
    validate_paired_target_isolation,
    validate_training_manifest,
)


def test_current_phase11_inputs_cover_source_and_safe_observation() -> None:
    config = TransferResearchConfig.from_yaml()
    manifest = current_input_manifest(config)
    assert manifest["experiment_config_sha256"] == config.digest()
    assert manifest["observation_schema"]["source"] == "AgentKnowledge"
    assert manifest["observation_schema"]["hidden_topology_fields"] == 0
    assert "src/rlredteam/enterprise/transfer_study.py" in manifest["source_sha256"]
    assert "scripts/run_transfer_study.py" in manifest["source_sha256"]


def test_frozen_phase11_inputs_fail_closed_on_drift() -> None:
    config = TransferResearchConfig.from_yaml()
    frozen = current_input_manifest(config)
    frozen["protocol_commit"] = "test"
    validate_frozen_inputs(frozen, config)
    frozen["distribution_sha256"] = "0" * 64
    with pytest.raises(TransferStudyError, match="distribution_sha256"):
        validate_frozen_inputs(frozen, config)


def test_paired_target_isolation_rejects_target_control_drift() -> None:
    common = {
        "study_id": "study",
        "description": "test",
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "representation": "agent_knowledge_message_passing_graph",
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": {"hidden_topology_fields": 0},
        "development": True,
        "training_seed": 36,
        "target_adaptation_timesteps": 256,
        "actual_target_adaptation_timesteps": 256,
        "target_profiles": ["hybrid"],
        "source_topology_seeds": [6001],
        "target_train_topology_seeds": [7001],
        "validation_topology_seeds": [8001],
        "test_topology_seeds": [9001],
        "study_config_hash": "study",
        "scientific_config_hash": "science",
        "base_profile_config_hash": "profile",
        "dependency_lock_hash": "lock",
        "git_commit": "commit",
        "git_dirty": True,
        "ppo": {"n_steps": 256},
        "policy_config": {"pooling": "mean_max"},
        "parameter_count": 10,
        "torch_num_threads": 1,
        "base_distribution": {},
        "base_vulnerability_snapshot_sha256": "vulnerability",
        "frozen_inputs_sha256": None,
    }
    scratch = {
        **common,
        "initialization_type": "random",
        "source_pretraining": None,
        "pre_adaptation_policy_sha256": "scratch",
    }
    source = {"policy_sha256": "source"}
    transfer = {
        **common,
        "initialization_type": "legacy_cloud_source_checkpoint",
        "source_pretraining": source,
        "pre_adaptation_policy_sha256": "source",
    }
    validate_paired_target_isolation(scratch, transfer)
    drifted = copy.deepcopy(transfer)
    drifted["target_adaptation_timesteps"] = 512
    with pytest.raises(TransferStudyError, match="outside initialization"):
        validate_paired_target_isolation(scratch, drifted)


def test_short_transfer_pipeline_uses_source_checkpoint_and_frozen_eval(tmp_path: Path) -> None:
    config = TransferResearchConfig.from_yaml()
    manifests = {}
    for arm in config.arms:
        output = tmp_path / "runs" / arm
        manifest = train_arm(
            output,
            arm=arm,
            training_seed=config.development_seed,
            config=config,
            source_timesteps=256,
            target_timesteps=256,
            development=True,
            allow_dirty=True,
        )
        checkpoint = output / "model.zip"
        validate_training_manifest(
            manifest,
            checkpoint,
            arm=arm,
            training_seed=config.development_seed,
            config=config,
            frozen_inputs=None,
            development=True,
            allow_unverifiable=True,
        )
        assert manifest["actual_target_adaptation_timesteps"] == 256
        assert manifest["target_training_audit"]["reset_count"] > 0
        assert set(manifest["target_training_audit"]["observed_profile_counts"]) == {"hybrid"}
        manifests[arm] = manifest

    validate_paired_target_isolation(manifests[config.arms[0]], manifests[config.arms[1]])
    transfer = manifests["transfer_legacy_cloud_to_hybrid"]
    source = transfer["source_pretraining"]
    assert source["actual_training_timesteps"] == 256
    assert set(source["observed_profile_counts"]) <= {"legacy", "cloud"}
    assert source["policy_sha256"] == transfer["pre_adaptation_policy_sha256"]
    assert Path(source["checkpoint"]).is_file()

    arm = "transfer_legacy_cloud_to_hybrid"
    checkpoint = tmp_path / "runs" / arm / "model.zip"
    model = load_model(checkpoint)
    episodes, steps, integrity = evaluate_arm(
        model,
        arm=arm,
        training_seed=config.development_seed,
        config=config,
        topology_seeds=(config.topology_splits["validation"][0],),
    )
    validate_finite_evidence(episodes, steps)
    assert len(episodes) == 1
    assert episodes[0]["profile"] == "hybrid"
    assert integrity["policy_sha256_before"] == integrity["policy_sha256_after"]
    assert integrity["invalid_mask_selections"] == 0
    metadata = persist_and_write_run_evaluation(
        tmp_path / "results",
        episodes=episodes,
        steps=steps,
        integrity=integrity,
        training_manifest=transfer,
        checkpoint=checkpoint,
        split_name="validation",
        postgres=False,
    )
    assert metadata["phase"] == "phase11_frozen_transfer_comparison"
    assert (
        json.loads((tmp_path / "results" / "evaluation_metadata.json").read_text())[
            "gradient_updates"
        ]
        is False
    )
    paths = json.loads((tmp_path / "results" / "attack_paths.json").read_text())
    assert len(paths) == 1
