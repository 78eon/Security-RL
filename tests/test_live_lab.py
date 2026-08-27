"""Safety gates and evidence parsing for the isolated real-network backend."""

from __future__ import annotations

import subprocess

import pytest

from rlredteam.enterprise.live import (
    LabExecutionError,
    LabScope,
    LiveKnowledgeGraph,
    NmapLabRunner,
    ScanProfile,
    ScopeViolation,
    parse_greenbone_xml,
    parse_nmap_xml,
)

NMAP_XML = b"""<?xml version="1.0"?>
<nmaprun scanner="nmap" args="nmap -sT -sV 10.250.0.10" version="7.95">
  <host>
    <status state="up" reason="conn-refused"/>
    <address addr="10.250.0.10" addrtype="ipv4"/>
    <hostnames><hostname name="legacy-app.lab" type="PTR"/></hostnames>
    <ports>
      <port protocol="tcp" portid="22">
        <state state="open" reason="syn-ack"/>
        <service name="ssh" product="OpenSSH" version="9.2p1" method="probed" conf="10">
          <cpe>cpe:/a:openbsd:openssh:9.2p1</cpe>
        </service>
      </port>
      <port protocol="tcp" portid="80"><state state="closed"/></port>
    </ports>
    <os><osmatch name="Linux 5.x" accuracy="92"/></os>
  </host>
</nmaprun>
"""

GREENBONE_XML = b"""<get_reports_response>
  <report><results><result id="finding-1">
    <name>OpenSSH vulnerability</name><host>10.250.0.10</host><port>22/tcp</port>
    <severity>8.1</severity>
    <nvt oid="1.3.6.1.4.1.25623.1"><name>OpenSSH check</name><refs>
      <ref type="cve" id="CVE-2024-6387"/>
    </refs></nvt>
  </result></results></report>
</get_reports_response>"""


@pytest.fixture
def scope() -> LabScope:
    return LabScope.from_strings(
        authorization_id="LAB-2026-001",
        allowed_networks=["10.250.0.0/24"],
        max_addresses=256,
    )


def test_scope_accepts_only_literal_allowlisted_private_targets(scope: LabScope) -> None:
    assert str(scope.validate_target("10.250.0.10")) == "10.250.0.10/32"
    assert str(scope.validate_target("10.250.0.0/25")) == "10.250.0.0/25"
    for target in ("example.com", "8.8.8.8", "10.251.0.1", "10.250.0.0/16"):
        with pytest.raises(ScopeViolation):
            scope.validate_target(target)


def test_plan_is_conservative_and_has_machine_readable_output(scope: LabScope) -> None:
    runner = NmapLabRunner(scope)
    command = runner.plan("10.250.0.10", ScanProfile.SERVICE_INVENTORY)
    assert command[0] == "nmap"
    assert "-sT" in command and "-sV" in command and "--version-light" in command
    assert "-oX" in command and "-" in command
    assert "--script" not in command and "-A" not in command and "-O" not in command


def test_execution_requires_config_and_per_action_approval(scope: LabScope) -> None:
    runner = NmapLabRunner(scope)
    with pytest.raises(LabExecutionError, match="disabled"):
        runner.run("10.250.0.10", ScanProfile.HOST_DISCOVERY, operator_approved=True)

    enabled = LabScope.from_strings(
        authorization_id="LAB-2026-001",
        allowed_networks=["10.250.0.0/24"],
        execution_enabled=True,
    )
    with pytest.raises(LabExecutionError, match="operator approval"):
        NmapLabRunner(enabled).run("10.250.0.10", ScanProfile.HOST_DISCOVERY)


def test_out_of_scope_target_never_starts_a_process(monkeypatch, scope: LabScope) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("subprocess must not start for an out-of-scope target")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(ScopeViolation):
        NmapLabRunner(scope).run("8.8.8.8", ScanProfile.HOST_DISCOVERY, operator_approved=True)


def test_nmap_xml_becomes_host_service_and_version_evidence() -> None:
    evidence = parse_nmap_xml(NMAP_XML)
    assert evidence.scanner_version == "7.95"
    assert len(evidence.hosts) == 1
    host = evidence.hosts[0]
    assert host.address == "10.250.0.10"
    assert host.hostnames == ("legacy-app.lab",)
    assert host.os_match == "Linux 5.x"
    assert len(host.services) == 1
    assert host.services[0].product == "OpenSSH"
    assert host.services[0].version == "9.2p1"
    assert host.services[0].confidence == 10

    graph = LiveKnowledgeGraph()
    graph.ingest_nmap(evidence)
    assert "host:10.250.0.10" in graph.nodes
    assert "service:10.250.0.10:tcp:22" in graph.nodes
    assert graph.nodes["service:10.250.0.10:tcp:22"]["version"] == "9.2p1"


def test_greenbone_report_links_cve_finding_to_observed_host() -> None:
    findings = parse_greenbone_xml(GREENBONE_XML)
    assert len(findings) == 1
    assert findings[0].cves == ("CVE-2024-6387",)
    assert findings[0].severity == pytest.approx(8.1)

    graph = LiveKnowledgeGraph()
    graph.ingest_greenbone(findings)
    assert graph.edges == [
        {
            "source": "host:10.250.0.10",
            "target": "finding:finding-1",
            "type": "has_finding",
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><nmaprun/>',
        b'<!ENTITY xxe "boom"><nmaprun/>',
    ],
)
def test_scanner_xml_rejects_dtd_and_entities(payload: bytes) -> None:
    with pytest.raises(ValueError, match="DTD/entity"):
        parse_nmap_xml(payload)
