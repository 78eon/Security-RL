"""Module 0 -- frozen CVE catalogue and its SHA-256 manifest."""

from __future__ import annotations

import pytest

from rlredteam import manifest
from rlredteam.assign import assign_cves
from rlredteam.catalogue import CatalogueError, CVECatalogue
from rlredteam.cvss import contrast_ratio, parse_vector, severity_band

# The floor below which the catalogue cannot support the ablation: too little
# spread in reward weight and any effect is drowned by PPO seed noise at n=10.
MIN_CONTRAST = 1.8


@pytest.fixture(scope="module")
def catalogue() -> CVECatalogue:
    return CVECatalogue.open_default()


def test_catalogue_is_populated(catalogue: CVECatalogue) -> None:
    assert len(catalogue) >= 10
    assert catalogue.by_kind("exploit")
    assert catalogue.by_kind("privesc")


def test_known_cve_returns_expected_cvss(catalogue: CVECatalogue) -> None:
    """The work plan's acceptance test: a known CVE ID returns its CVSS."""
    record = catalogue.lookup("CVE-2021-42013")
    assert record.base_score == pytest.approx(9.8)
    assert record.base_severity == "CRITICAL"
    assert record.cvss_version == "3.1"
    assert record.vector == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def test_missing_cve_raises_rather_than_returning_none(catalogue: CVECatalogue) -> None:
    # A silent None would degrade a shaped run into an unshaped one.
    with pytest.raises(CatalogueError):
        catalogue.lookup("CVE-0000-0000")


def test_every_severity_matches_its_score(catalogue: CVECatalogue) -> None:
    for record in catalogue.all_records():
        assert record.base_severity == severity_band(record.base_score).value


def test_every_vector_parses_and_matches_version(catalogue: CVECatalogue) -> None:
    for record in catalogue.all_records():
        parsed = parse_vector(record.vector)
        assert parsed.version == record.cvss_version


def test_recomputed_base_scores_match_committed(catalogue: CVECatalogue) -> None:
    """Cross-check every committed score against its own CVSS vector.

    Uses the test-only `cvss` library, never installed in the training image.
    Proves the catalogue's scores are consistent with their vectors rather
    than transcribed.
    """
    cvss_lib = pytest.importorskip("cvss")
    for record in catalogue.all_records():
        recomputed = cvss_lib.CVSS3(record.vector).base_score
        assert float(recomputed) == pytest.approx(record.base_score), record.cve_id


def test_source_url_refers_to_its_own_cve(catalogue: CVECatalogue) -> None:
    for record in catalogue.all_records():
        assert record.cve_id in record.source_url


def test_catalogue_spans_enough_severity_range(catalogue: CVECatalogue) -> None:
    """Design assertion: fails loudly if the pool is edited into uniformity."""
    assert contrast_ratio(catalogue.scores()) >= MIN_CONTRAST


# -- manifest --------------------------------------------------------------


def test_manifest_digest_stable_across_reloads() -> None:
    """The work plan's acceptance test: the hash is stable across reloads.

    Hashing the .sqlite bytes would FAIL here -- SQLite rewrites page headers
    and internal counters between opens. The manifest hashes a canonical row
    dump instead.
    """
    first = manifest.digest(CVECatalogue.open_default())
    second = manifest.digest(CVECatalogue.open_default())
    assert first == second
    assert len(first) == 64


def test_committed_manifest_matches_catalogue() -> None:
    assert manifest.verify(), "catalogue changed without regenerating the manifest"


def test_canonical_dump_is_order_independent(catalogue: CVECatalogue) -> None:
    records = catalogue.all_records()
    shuffled = CVECatalogue({r.cve_id: r for r in reversed(records)})
    assert manifest.canonical_dump(shuffled) == manifest.canonical_dump(catalogue)


# -- deterministic assignment ---------------------------------------------

EXPLOITS = [f"e_srv_{i}_os_{j}" for i in range(4) for j in range(2)]
PRIVESCS = [f"pe_proc_{i}" for i in range(3)]


def test_assignment_is_deterministic(catalogue: CVECatalogue) -> None:
    first = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    second = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    assert first.mapping == second.mapping


def test_assignment_varies_with_seed(catalogue: CVECatalogue) -> None:
    a = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    b = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=43)
    assert a.mapping != b.mapping


def test_assignment_is_independent_of_name_order(catalogue: CVECatalogue) -> None:
    a = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    b = assign_cves(list(reversed(EXPLOITS)), PRIVESCS, catalogue, topology_seed=42)
    assert a.mapping == b.mapping


def test_assignment_covers_every_action(catalogue: CVECatalogue) -> None:
    assignment = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    for name in EXPLOITS + PRIVESCS:
        assert assignment.cve_for(name) is not None


def test_privescs_never_get_exploit_cves(catalogue: CVECatalogue) -> None:
    assignment = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=42)
    for name in PRIVESCS:
        assert assignment.cve_for(name).kind == "privesc"
    for name in EXPLOITS:
        assert assignment.cve_for(name).kind == "exploit"


@pytest.mark.parametrize("seed", range(42, 52))
def test_every_evaluation_seed_keeps_enough_contrast(
    catalogue: CVECatalogue, seed: int
) -> None:
    """Stratified draw must preserve dynamic range on all of seeds 42-51.

    A uniform draw could hand a topology all-CRITICAL CVEs, collapsing the
    contrast to ~1.0 and making that seed uninformative.
    """
    assignment = assign_cves(EXPLOITS, PRIVESCS, catalogue, topology_seed=seed)
    assert contrast_ratio(assignment.scores()) >= MIN_CONTRAST
