from dataclasses import replace

import pytest

from rlredteam.enterprise.profiles import DeploymentProfile, EnterpriseProfileConfig
from rlredteam.enterprise.transfer_learning import (
    PHASE11_ARMS,
    TransferResearchConfig,
    observation_schema,
    transfer_distribution_manifest,
    transfer_vulnerability_manifest,
)


def test_phase11_protocol_is_fresh_disjoint_and_exact() -> None:
    config = TransferResearchConfig.from_yaml()
    assert config.arms == PHASE11_ARMS
    assert config.source_profiles == (DeploymentProfile.LEGACY, DeploymentProfile.CLOUD)
    assert config.target_profiles == (DeploymentProfile.HYBRID,)
    sets = [set(values) for values in config.topology_splits.values()]
    assert all(
        sets[left].isdisjoint(sets[right]) for left in range(4) for right in range(left + 1, 4)
    )
    assert config.source_pretrain_timesteps == 25_600
    assert config.target_adaptation_timesteps == 12_800
    assert config.digest() == TransferResearchConfig.from_yaml().digest()


def test_phase11_rejects_target_profile_or_split_leakage() -> None:
    config = TransferResearchConfig.from_yaml()
    with pytest.raises(ValueError, match="target profile"):
        replace(config, target_profiles=(DeploymentProfile.CLOUD,)).validate()
    leaked = dict(config.topology_splits)
    leaked["test"] = leaked["target_train"]
    with pytest.raises(ValueError, match="overlap"):
        replace(config, topology_splits=leaked).validate()


def test_transfer_distribution_has_only_declared_profiles_and_hashes() -> None:
    config = TransferResearchConfig.from_yaml()
    manifest = transfer_distribution_manifest(config)
    assert set(manifest) == set(config.topology_splits)
    assert set(manifest["source_train"]) == {"legacy", "cloud"}
    assert set(manifest["target_train"]) == {"hybrid"}
    assert set(manifest["validation"]) == {"hybrid"}
    assert set(manifest["test"]) == {"hybrid"}
    assert all(
        len(value) == 64
        for split in manifest.values()
        for values in split.values()
        for value in values.values()
    )
    assert len(transfer_vulnerability_manifest(config)) == 64


def test_transfer_observation_schema_exposes_no_hidden_topology() -> None:
    schema = observation_schema()
    profile = EnterpriseProfileConfig.from_yaml()
    assert schema["source"] == "AgentKnowledge"
    assert schema["hidden_topology_fields"] == 0
    assert schema["observation_values"] == profile.max_nodes * 27 + profile.max_nodes**2 + 1
