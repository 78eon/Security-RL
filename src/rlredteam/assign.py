"""Deterministic assignment of catalogue CVEs to a generated topology's exploits.

NASim's scenario generator names exploits positionally (``e_srv_2_os_0``), so
there are no stable names to key a CVE map on across random topologies. Instead
each generated exploit is bound to a pool CVE by a seeded, reproducible draw.

Two properties matter and are both tested:

* **Deterministic** -- the same (exploit names, seed) always yields the same
  mapping, so a topology's CVE assignment is reproducible from its seed alone
  and needs no extra artefact to be checked into the repo.
* **Severity-spread** -- assignment stratifies across the pool's severity range
  rather than drawing uniformly, so a topology cannot by chance receive only
  CRITICAL CVEs and lose the reward contrast the ablation depends on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rlredteam.catalogue import CVECatalogue, CVERecord


@dataclass(frozen=True, slots=True)
class Assignment:
    """A frozen exploit-name -> CVE binding for one topology."""

    topology_seed: int
    mapping: dict[str, str]  # action_name -> cve_id
    records: dict[str, CVERecord]  # action_name -> record

    def cve_for(self, action_name: str) -> CVERecord | None:
        return self.records.get(action_name)

    def scores(self) -> list[float]:
        return sorted(r.base_score for r in self.records.values())


def _stratified_cycle(pool: list[CVERecord], count: int, rng: random.Random) -> list[CVERecord]:
    """Draw ``count`` records spread across the pool's severity range.

    Sorts the pool by score, splits it into ``count`` contiguous strata, and
    draws one record from each. This guarantees the drawn set spans the pool's
    full range: a uniform draw could return all-CRITICAL and collapse the
    reward contrast to ~1.0, making the ablation undetectable.
    """
    ordered = sorted(pool, key=lambda r: (r.base_score, r.cve_id))
    if count <= 0:
        return []
    if count >= len(ordered):
        # More exploits than CVEs: use every CVE, then cycle for the remainder.
        drawn = list(ordered)
        while len(drawn) < count:
            drawn.append(ordered[len(drawn) % len(ordered)])
        return drawn

    chosen: list[CVERecord] = []
    stride = len(ordered) / count
    for index in range(count):
        low = int(index * stride)
        high = max(low + 1, int((index + 1) * stride))
        chosen.append(rng.choice(ordered[low:high]))
    return chosen


def assign_cves(
    exploit_names: list[str],
    privesc_names: list[str],
    catalogue: CVECatalogue,
    topology_seed: int,
) -> Assignment:
    """Bind every exploit and privesc action of a topology to a pool CVE.

    Exploits draw from the ``exploit`` pool and privescs from the ``privesc``
    pool, so a remote service compromise is never scored by a local
    privilege-escalation CVE.
    """
    # Sorted names make the mapping independent of dict iteration order.
    exploit_names = sorted(exploit_names)
    privesc_names = sorted(privesc_names)

    rng = random.Random(topology_seed)
    mapping: dict[str, str] = {}
    records: dict[str, CVERecord] = {}

    for names, kind in ((exploit_names, "exploit"), (privesc_names, "privesc")):
        pool = catalogue.by_kind(kind)
        if not pool:
            raise ValueError(f"catalogue has no CVEs of kind {kind!r}")
        for name, record in zip(names, _stratified_cycle(pool, len(names), rng), strict=True):
            mapping[name] = record.cve_id
            records[name] = record

    return Assignment(topology_seed=topology_seed, mapping=mapping, records=records)
