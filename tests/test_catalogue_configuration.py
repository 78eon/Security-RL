from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rlredteam.catalogues import load_mitre_catalogue, load_world_catalogue
from rlredteam.constant_audit import classify_python_constants
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.profiles import (
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
    generate_profile_topology,
)
from rlredteam.protocols import protocol_versions

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_new_environment_profile_requires_configuration_only(tmp_path: Path) -> None:
    document = yaml.safe_load((REPO_ROOT / "configs/enterprise_profiles.yaml").read_text())
    profiles = document["enterprise_profiles"]["profiles"]
    profiles["edge_research_lab"] = {
        **profiles["hybrid"],
        "network_types": ["network_segment", "cloud_network"],
        "host_types": ["host", "cloud_workload"],
        "identity_types": ["identity", "iam_role"],
        "store_types": ["storage"],
        "control": "research_boundary",
        "boundary": "edge_cloud_trust",
    }
    config_path = tmp_path / "profiles.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False))

    config = EnterpriseProfileConfig.from_yaml(config_path)
    topology = generate_profile_topology("edge_research_lab", 731, config)

    assert "edge_research_lab" in config.profiles
    assert topology.name == "enterprise-edge_research_lab-v1-seed-731"
    assert NodeType.CLOUD_WORKLOAD in {node.type for node in topology.nodes.values()}
    assert topology.crown_jewels == ("asset_crown",)

    curriculum = InfrastructureCurriculumEnv(
        [731], ["edge_research_lab"], config=config
    )
    observation, info = curriculum.reset(
        seed=731,
        options={"profile": "edge_research_lab", "topology_seed": 731},
    )
    assert curriculum.observation_space.contains(observation)
    assert info["profile"] == "edge_research_lab"


def test_catalogues_are_versioned_validated_and_complete() -> None:
    world = load_world_catalogue()
    mitre = load_mitre_catalogue()

    assert world.version == "security-rl-world-v1"
    assert world.service_pool("profile_entry")
    assert world.os_pool("enterprise")
    assert mitre["version"] == "security-rl-mitre-v1"
    assert mitre["simulator_action_mappings"]["exploit"] == "T1210"
    assert protocol_versions()["enterprise_action_catalogue"] == 1


def test_invalid_catalogue_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("catalogue:\n  version: mutable\n")
    with pytest.raises(ValueError, match="frozen"):
        load_world_catalogue(invalid)


def test_every_symbolic_python_constant_has_one_allowed_classification() -> None:
    findings = classify_python_constants(REPO_ROOT)
    assert findings
    assert {item.category for item in findings} == {
        "protocol", "model_schema", "deployment", "simulated_world"
    }
    assert len({(item.path, item.line, item.name) for item in findings}) == len(findings)


def test_simulated_world_and_mitre_records_are_not_embedded_in_generators() -> None:
    sources = "\n".join(
        (REPO_ROOT / path).read_text()
        for path in (
            "src/rlredteam/frameworks.py",
            "src/rlredteam/enterprise/generator.py",
            "src/rlredteam/enterprise/demo.py",
            "src/rlredteam/enterprise/hybrid.py",
            "src/rlredteam/enterprise/onprem.py",
            "src/rlredteam/enterprise/profiles.py",
        )
    )
    for forbidden in (
        "CVE-2021-42013",
        "synthetic_https",
        '"windows"',
        '"T1210"',
        '"AML.T0040"',
    ):
        assert forbidden not in sources
