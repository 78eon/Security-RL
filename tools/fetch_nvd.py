#!/usr/bin/env python3
"""One-shot, ONLINE tool: fetch CVE records by ID from the NVD API 2.0.

This is the only component that touches the network, and it is never imported by
training code. Training reads the frozen SQLite catalogue built from the raw JSON
this tool writes to data/provenance/.

Why freeze rather than query live: NVD data mutates (CVEs are re-scored, CNA and
NVD analysts disagree, entries migrate between cvssMetricV31 and cvssMetricV40).
A live lookup would make the experiment non-reproducible across time, which is a
graded criterion (charter §7).

Usage:
    python tools/fetch_nvd.py --fetch      # download raw JSON to data/provenance/
    python tools/fetch_nvd.py --verify     # diff committed catalogue vs live NVD

Requires NVD_API_KEY in .env (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_DIR = REPO_ROOT / "data" / "provenance"
NVD_ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# The CVEs backing the fixed `small` topology, one per NASim exploit/privesc.
# exploit_key MUST match the exploit name in data/scenarios/small_cve.yaml.
# The catalogue is a POOL, not a fixed exploit-name map.
#
# NASim's scenario generator names exploits positionally (e_srv_2_os_0), so
# there are no stable names to key on across random topologies. Instead the
# pool is assigned to a generated topology's exploits deterministically from
# the topology seed (see rlredteam.assign).
#
# Pool selection criterion is SEVERITY SPREAD. Real remote-code-execution CVEs
# cluster at 9.8-10.0; a pool drawn only from those has ~1.4x dynamic range in
# reward weight, which is below PPO's seed-to-seed noise at n=10 and would make
# the ablation undetectable. The pool therefore deliberately spans LOW through
# CRITICAL, mixing information disclosure, DoS, privilege escalation and RCE.
#
# `kind` records what the CVE is used to model: "exploit" (remote service
# compromise) or "privesc" (local privilege escalation).
TARGETS: dict[str, dict[str, str]] = {
    # --- CRITICAL (9.0-10.0) ---
    "CVE-2021-35211": {"kind": "exploit", "note": "SolarWinds Serv-U FTP RCE"},
    "CVE-2021-42013": {"kind": "exploit", "note": "Apache 2.4.50 traversal -> RCE"},
    "CVE-2019-0708": {"kind": "exploit", "note": "BlueKeep, RDP pre-auth RCE"},
    "CVE-2017-0144": {"kind": "exploit", "note": "EternalBlue, SMBv1 RCE"},
    "CVE-2014-0160": {"kind": "exploit", "note": "Heartbleed, OpenSSL"},
    # --- HIGH (7.0-8.9) ---
    "CVE-2024-6387": {"kind": "exploit", "note": "regreSSHion, OpenSSH RCE"},
    "CVE-2020-1472": {"kind": "privesc", "note": "Zerologon, Netlogon EoP"},
    "CVE-2016-5425": {"kind": "privesc", "note": "Tomcat RedHat pkg local root"},
    "CVE-2021-36934": {"kind": "privesc", "note": "HiveNightmare, Windows EoP"},
    "CVE-2021-4034": {"kind": "privesc", "note": "PwnKit, polkit local root"},
    "CVE-2016-5195": {"kind": "privesc", "note": "Dirty COW, Linux kernel EoP"},
    "CVE-2020-9484": {"kind": "exploit", "note": "Tomcat deserialization RCE"},
    # --- MEDIUM (4.0-6.9) ---
    "CVE-2018-15473": {"kind": "exploit", "note": "OpenSSH username enumeration"},
    "CVE-2016-2183": {"kind": "exploit", "note": "SWEET32, 3DES birthday attack"},
    "CVE-2015-4000": {"kind": "exploit", "note": "Logjam, weak DH export"},
    "CVE-2019-11510": {"kind": "exploit", "note": "Pulse Secure file read"},
    # --- LOW (0.1-3.9) ---
    "CVE-2005-3299": {"kind": "exploit", "note": "phpMyAdmin path disclosure"},
    "CVE-2007-6750": {"kind": "exploit", "note": "Slowloris, Apache DoS"},
}
# CVE-2016-1240 (Tomcat init-script local root) was the first pe_tomcat
# candidate but NVD carries no CVSS v3.1 metric for it -- only v2. Using it
# would force either an indefensible v2->v3 conversion or an invented score,
# so it was dropped. Any candidate lacking a v3.1 metric is rejected by
# extract_v31() rather than silently converted. This is a methodology decision.


def load_api_key() -> str | None:
    """Read NVD_API_KEY from the environment or .env. Absent is not fatal.

    Without a key NVD allows 5 requests / 30s, which is enough for 5 CVEs.
    """
    key = os.environ.get("NVD_API_KEY")
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("NVD_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip()
                return value or None
    return None


def fetch_one(cve_id: str, api_key: str | None) -> dict:
    """Fetch a single CVE. Raises on HTTP error or an empty result."""
    headers = {"apiKey": api_key} if api_key else {}
    response = requests.get(
        NVD_ENDPOINT, params={"cveId": cve_id}, headers=headers, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    vulns = payload.get("vulnerabilities", [])
    if not vulns:
        raise RuntimeError(f"{cve_id}: NVD returned no record")
    return payload


def extract_v31(payload: dict, cve_id: str) -> dict:
    """Pull the primary CVSS v3.1 base metric out of an NVD response.

    Prefers the NVD-assigned (Primary) score over a CNA-assigned (Secondary)
    one, so the catalogue records a single consistent authority. Raises if the
    CVE carries no v3.1 metric -- a v2-only entry would force either an
    indefensible v2->v3 conversion or an invented score.
    """
    cve = payload["vulnerabilities"][0]["cve"]
    metrics = cve.get("metrics", {}).get("cvssMetricV31", [])
    if not metrics:
        raise RuntimeError(f"{cve_id}: no CVSS v3.1 metric in NVD record")

    primary = next((m for m in metrics if m.get("type") == "Primary"), metrics[0])
    data = primary["cvssData"]

    cwe = None
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            if desc.get("value", "").startswith("CWE-"):
                cwe = desc["value"]
                break
        if cwe:
            break

    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    return {
        "cve_id": cve["id"],
        "cvss_version": data["version"],
        "base_score": float(data["baseScore"]),
        "base_severity": data["baseSeverity"],
        "vector": data["vectorString"],
        "metric_source": primary.get("type", "Unknown"),
        "cwe": cwe,
        "published": cve.get("published", "")[:10],
        "last_modified": cve.get("lastModified", "")[:10],
        "description": description,
        "source_url": f"https://nvd.nist.gov/vuln/detail/{cve['id']}",
    }


def do_fetch(api_key: str | None) -> int:
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, cve_id in enumerate(TARGETS):  # noqa: B007
        try:
            payload = fetch_one(cve_id, api_key)
            # Extract before writing: a v2-only record must not land in
            # provenance/ as though it were usable.
            summary = extract_v31(payload, cve_id)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL  {cve_id}: {exc}", file=sys.stderr)
            failures += 1
            continue

        out = PROVENANCE_DIR / f"{cve_id}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(
            f"  OK    {cve_id}  {summary['base_score']:>4}  "
            f"{summary['base_severity']:<8} {summary['metric_source']:<9} "
            f"{summary['vector']}"
        )
        # Rate limit: 50 req/30s with a key, 5 req/30s without.
        if index < len(TARGETS) - 1:
            time.sleep(0.7 if api_key else 6.5)
    return failures


def do_verify(api_key: str | None) -> int:
    """Diff the committed SQLite catalogue against live NVD."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from rlredteam.catalogue import CVECatalogue  # noqa: PLC0415

    catalogue = CVECatalogue.open_default()
    mismatches = 0
    for index, cve_id in enumerate(TARGETS):  # noqa: B007
        try:
            live = extract_v31(fetch_one(cve_id, api_key), cve_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {cve_id}: {exc}", file=sys.stderr)
            mismatches += 1
            continue

        stored = catalogue.lookup(cve_id)
        drift = []
        if abs(stored.base_score - live["base_score"]) > 1e-9:
            drift.append(f"score {stored.base_score} != {live['base_score']}")
        if stored.vector != live["vector"]:
            drift.append(f"vector {stored.vector} != {live['vector']}")
        if drift:
            mismatches += 1
            print(f"  DRIFT {cve_id}: {'; '.join(drift)}")
        else:
            print(f"  MATCH {cve_id}  {stored.base_score}  {stored.base_severity}")
        if index < len(TARGETS) - 1:
            time.sleep(0.7 if api_key else 6.5)
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true", help="download raw JSON")
    group.add_argument("--verify", action="store_true", help="diff catalogue vs live NVD")
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"NVD API key: {'present' if api_key else 'ABSENT (slow rate limit)'}")

    if args.fetch:
        failures = do_fetch(api_key)
        print(f"\nfetched {len(TARGETS) - failures}/{len(TARGETS)} into {PROVENANCE_DIR}")
        return 1 if failures else 0

    mismatches = do_verify(api_key)
    print(f"\n{len(TARGETS) - mismatches}/{len(TARGETS)} entries match live NVD")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
