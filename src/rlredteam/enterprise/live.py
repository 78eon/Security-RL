"""Authorized isolated-lab discovery and vulnerability evidence ingestion.

This module deliberately contains no exploit runner. It turns conservative
Nmap XML and operator-exported Greenbone reports into evidence for the same
knowledge-graph pipeline used by simulation.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ScopeViolation(ValueError):
    pass


class LabExecutionError(RuntimeError):
    pass


class ScanProfile(StrEnum):
    HOST_DISCOVERY = "host_discovery"
    SERVICE_INVENTORY = "service_inventory"


@dataclass(frozen=True, slots=True)
class LabScope:
    authorization_id: str
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    max_addresses: int = 256
    execution_enabled: bool = False

    @classmethod
    def from_strings(
        cls,
        *,
        authorization_id: str,
        allowed_networks: list[str] | tuple[str, ...],
        max_addresses: int = 256,
        execution_enabled: bool = False,
    ) -> LabScope:
        if not authorization_id.strip():
            raise ValueError("authorization_id is required")
        parsed = tuple(ipaddress.ip_network(item, strict=False) for item in allowed_networks)
        if not parsed:
            raise ValueError("at least one allowed network is required")
        for network in parsed:
            if not network.is_private or network.is_loopback or network.is_link_local:
                raise ValueError(f"lab allowlist must contain private routed ranges: {network}")
        if max_addresses <= 0:
            raise ValueError("max_addresses must be positive")
        return cls(authorization_id, parsed, max_addresses, execution_enabled)

    def validate_target(self, target: str):
        """Return a parsed target or reject it before any process is started."""
        try:
            network = ipaddress.ip_network(target, strict=False)
        except ValueError as exc:
            raise ScopeViolation(
                "targets must be literal IP addresses or CIDRs; DNS is disabled"
            ) from exc
        if network.num_addresses > self.max_addresses:
            raise ScopeViolation(
                f"target {network} contains {network.num_addresses} addresses; "
                f"limit is {self.max_addresses}"
            )
        if not any(
            network.version == allowed.version and network.subnet_of(allowed)
            for allowed in self.allowed_networks
        ):
            raise ScopeViolation(f"target {network} is outside the authorized lab allowlist")
        return network


@dataclass(frozen=True, slots=True)
class ObservedService:
    host: str
    port: int
    protocol: str
    state: str
    name: str | None = None
    product: str | None = None
    version: str | None = None
    cpes: tuple[str, ...] = ()
    confidence: int | None = None


@dataclass(frozen=True, slots=True)
class ObservedHost:
    address: str
    hostnames: tuple[str, ...] = ()
    status: str = "up"
    os_match: str | None = None
    services: tuple[ObservedService, ...] = ()


@dataclass(frozen=True, slots=True)
class NmapEvidence:
    scanner_version: str | None
    command: str | None
    hosts: tuple[ObservedHost, ...]


@dataclass(frozen=True, slots=True)
class GreenboneFinding:
    finding_id: str
    host: str
    port: str | None
    name: str
    severity: float | None
    cves: tuple[str, ...] = ()
    nvt_oid: str | None = None


@dataclass(slots=True)
class LiveKnowledgeGraph:
    """Append-only evidence graph built from authorized lab observations."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def ingest_nmap(self, observation: NmapEvidence) -> None:
        for host in observation.hosts:
            host_id = f"host:{host.address}"
            self.nodes[host_id] = {
                "type": "host",
                "address": host.address,
                "hostnames": list(host.hostnames),
                "status": host.status,
                "os_match": host.os_match,
            }
            for service in host.services:
                service_id = f"service:{host.address}:{service.protocol}:{service.port}"
                self.nodes[service_id] = {
                    "type": "service",
                    "port": service.port,
                    "protocol": service.protocol,
                    "state": service.state,
                    "name": service.name,
                    "product": service.product,
                    "version": service.version,
                    "cpes": list(service.cpes),
                    "confidence": service.confidence,
                }
                edge = {"source": host_id, "target": service_id, "type": "hosts"}
                if edge not in self.edges:
                    self.edges.append(edge)
        self.evidence.append(
            {
                "kind": "nmap",
                "scanner_version": observation.scanner_version,
                "host_count": len(observation.hosts),
            }
        )

    def ingest_greenbone(self, findings: tuple[GreenboneFinding, ...]) -> None:
        for finding in findings:
            finding_id = f"finding:{finding.finding_id}"
            self.nodes[finding_id] = {
                "type": "vulnerability_finding",
                "name": finding.name,
                "severity": finding.severity,
                "cves": list(finding.cves),
                "nvt_oid": finding.nvt_oid,
            }
            host_id = f"host:{finding.host}"
            self.nodes.setdefault(host_id, {"type": "host", "address": finding.host})
            edge = {"source": host_id, "target": finding_id, "type": "has_finding"}
            if edge not in self.edges:
                self.edges.append(edge)
        self.evidence.append({"kind": "greenbone", "finding_count": len(findings)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [{"id": key, **value} for key, value in sorted(self.nodes.items())],
            "edges": list(self.edges),
            "evidence": list(self.evidence),
        }


def _safe_xml_root(raw: bytes | str) -> ET.Element:
    data = raw.encode() if isinstance(raw, str) else raw
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("DTD/entity declarations are not accepted in scanner XML")
    return ET.fromstring(data)


def parse_nmap_xml(raw: bytes | str) -> NmapEvidence:
    root = _safe_xml_root(raw)
    if root.tag != "nmaprun":
        raise ValueError("not an Nmap XML document")
    hosts: list[ObservedHost] = []
    for host_element in root.findall("host"):
        status_element = host_element.find("status")
        status = status_element.get("state", "unknown") if status_element is not None else "unknown"
        if status != "up":
            continue
        addresses = [
            item.get("addr")
            for item in host_element.findall("address")
            if item.get("addrtype") in {"ipv4", "ipv6"} and item.get("addr")
        ]
        if not addresses:
            continue
        address = addresses[0]
        hostnames = tuple(
            item.get("name")
            for item in host_element.findall("./hostnames/hostname")
            if item.get("name")
        )
        os_match_element = host_element.find("./os/osmatch")
        os_match = os_match_element.get("name") if os_match_element is not None else None
        services: list[ObservedService] = []
        for port_element in host_element.findall("./ports/port"):
            state_element = port_element.find("state")
            state = (
                state_element.get("state", "unknown")
                if state_element is not None
                else "unknown"
            )
            if state not in {"open", "open|filtered"}:
                continue
            service_element = port_element.find("service")
            confidence_raw = service_element.get("conf") if service_element is not None else None
            services.append(
                ObservedService(
                    host=address,
                    port=int(port_element.get("portid", "0")),
                    protocol=port_element.get("protocol", "tcp"),
                    state=state,
                    name=service_element.get("name") if service_element is not None else None,
                    product=service_element.get("product") if service_element is not None else None,
                    version=service_element.get("version") if service_element is not None else None,
                    cpes=tuple(
                        item.text for item in port_element.findall("./service/cpe") if item.text
                    ),
                    confidence=int(confidence_raw) if confidence_raw else None,
                )
            )
        hosts.append(ObservedHost(address, hostnames, status, os_match, tuple(services)))
    return NmapEvidence(root.get("version"), root.get("args"), tuple(hosts))


def parse_greenbone_xml(raw: bytes | str) -> tuple[GreenboneFinding, ...]:
    root = _safe_xml_root(raw)
    findings: list[GreenboneFinding] = []
    for index, result in enumerate(root.findall(".//result")):
        host = (result.findtext("host") or "").strip()
        if not host:
            continue
        nvt = result.find("nvt")
        finding_id = result.get("id") or f"result-{index}"
        name = (result.findtext("name") or (nvt.findtext("name") if nvt is not None else ""))
        severity_text = result.findtext("severity")
        refs = [] if nvt is None else nvt.findall("./refs/ref")
        cves = tuple(
            ref.get("id")
            for ref in refs
            if ref.get("type", "").lower() == "cve" and ref.get("id")
        )
        findings.append(
            GreenboneFinding(
                finding_id=finding_id,
                host=host,
                port=(result.findtext("port") or "").strip() or None,
                name=name.strip() or "Unnamed Greenbone finding",
                severity=float(severity_text) if severity_text else None,
                cves=cves,
                nvt_oid=nvt.get("oid") if nvt is not None else None,
            )
        )
    return tuple(findings)


class NmapLabRunner:
    """Build and optionally execute conservative, scope-checked Nmap commands."""

    def __init__(self, scope: LabScope, *, binary: str = "nmap", timeout: int = 300) -> None:
        self.scope = scope
        self.binary = binary
        self.timeout = timeout

    def plan(self, target: str, profile: ScanProfile | str) -> tuple[str, ...]:
        network = self.scope.validate_target(target)
        selected = ScanProfile(profile)
        common = [
            self.binary,
            "-n",
            "--max-retries",
            "1",
            "--host-timeout",
            "60s",
            "-oX",
            "-",
        ]
        if selected == ScanProfile.HOST_DISCOVERY:
            options = ["-sn"]
        else:
            options = ["-sT", "-sV", "--version-light", "--top-ports", "100", "--open"]
        return tuple([*common, *options, str(network)])

    def run(
        self,
        target: str,
        profile: ScanProfile | str,
        *,
        operator_approved: bool = False,
    ) -> NmapEvidence:
        argv = self.plan(target, profile)
        if not self.scope.execution_enabled:
            raise LabExecutionError("execution is disabled in the lab-scope configuration")
        if not operator_approved:
            raise LabExecutionError("operator approval is required for each live scan")
        if shutil.which(self.binary) is None:
            raise LabExecutionError(f"scanner binary is unavailable: {self.binary}")
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                check=False,
                timeout=self.timeout,
                env={"PATH": os.environ.get("PATH", ""), "LANG": "C"},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LabExecutionError(f"Nmap execution failed: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()[-1000:]
            raise LabExecutionError(f"Nmap exited with {result.returncode}: {detail}")
        return parse_nmap_xml(result.stdout)


def load_greenbone_report(path: Path) -> tuple[GreenboneFinding, ...]:
    return parse_greenbone_xml(Path(path).read_bytes())
