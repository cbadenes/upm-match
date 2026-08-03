"""Tests for the researcher exclusion list (RGPD arts. 17 and 21).

These guard a legal obligation, not just a feature: if an excluded researcher
leaks back into a search result, a directory listing or a profile endpoint, the
system is failing to honour an exercised right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import exclusions


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Every test starts with no cache and no env floor."""
    monkeypatch.delenv("EXCLUDED_RESEARCHER_IDS", raising=False)
    exclusions.invalidate_cache()
    yield
    exclusions.invalidate_cache()


def _stub_minio(monkeypatch, ids: list[int] | None = None, fail: bool = False):
    """Make the module read a fake exclusions document instead of MinIO."""

    def fake_read():
        if fail:
            raise ConnectionError("MinIO unreachable")
        return {
            "version": 1,
            "entries": [{"researcher_id": rid} for rid in (ids or [])],
        }

    monkeypatch.setattr(exclusions, "_read_document", fake_read)


# normalise_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (305196, 305196),
        ("305196", 305196),
        ("pdi_305196", 305196),
        ("  pdi_305196  ", 305196),
        ("", None),
        (None, None),
        ("not-an-id", None),
        (0, None),
        (True, None),  # bool is an int subclass; must not become an id
    ],
)
def test_normalise_id(value, expected):
    assert exclusions.normalise_id(value) == expected


# Loading


def test_ids_come_from_minio(monkeypatch):
    _stub_minio(monkeypatch, [305196, 307844])
    assert exclusions.load_excluded_ids() == {305196, 307844}


def test_env_floor_is_unioned(monkeypatch):
    monkeypatch.setenv("EXCLUDED_RESEARCHER_IDS", "111, pdi_222 , , junk")
    _stub_minio(monkeypatch, [305196])
    assert exclusions.load_excluded_ids() == {305196, 111, 222}


def test_env_floor_survives_a_minio_outage(monkeypatch):
    """A storage failure must never resurrect a profile pinned in the env."""
    monkeypatch.setenv("EXCLUDED_RESEARCHER_IDS", "305196")
    _stub_minio(monkeypatch, fail=True)
    assert exclusions.load_excluded_ids() == {305196}


def test_outage_reuses_the_last_good_list(monkeypatch):
    _stub_minio(monkeypatch, [305196])
    assert exclusions.load_excluded_ids() == {305196}

    # MinIO goes down; the cached list must still apply after the TTL expires.
    _stub_minio(monkeypatch, fail=True)
    assert exclusions.load_excluded_ids(force=True) == {305196}


def test_cache_is_used_within_the_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_read():
        calls["n"] += 1
        return {"version": 1, "entries": [{"researcher_id": 305196}]}

    monkeypatch.setattr(exclusions, "_read_document", counting_read)
    exclusions.load_excluded_ids()
    exclusions.load_excluded_ids()
    assert calls["n"] == 1

    exclusions.load_excluded_ids(force=True)
    assert calls["n"] == 2


def test_is_excluded_accepts_both_id_forms(monkeypatch):
    _stub_minio(monkeypatch, [305196])
    assert exclusions.is_excluded(305196)
    assert exclusions.is_excluded("pdi_305196")
    assert not exclusions.is_excluded(999999)


def test_drop_excluded_preserves_order_and_form(monkeypatch):
    _stub_minio(monkeypatch, [305196])
    kept = exclusions.drop_excluded([999, "pdi_305196", 305196, "pdi_111"])
    assert kept == [999, "pdi_111"]


# Search integration — the leak that matters most


def test_search_results_drop_excluded_researchers(monkeypatch):
    from app import search as app_search
    from dbs.vector import SearchResult

    monkeypatch.setattr(app_search, "load_excluded_ids", lambda: {305196})

    raw = [
        SearchResult(id=1, score=0.9, payload={"researcher_id": "pdi_305196", "name": "A"}),
        SearchResult(id=2, score=0.8, payload={"researcher_id": "pdi_307844", "name": "B"}),
    ]
    results = app_search._to_pdi_results(raw, ["topic"])

    assert [r.researcher_id for r in results] == ["pdi_307844"]
    # Ranks stay contiguous after the removal.
    assert [r.rank for r in results] == [1]


def test_excluded_researcher_has_no_public_profile(monkeypatch):
    from app import evaluation

    monkeypatch.setattr(evaluation, "load_excluded_ids", lambda: {305196})
    # Fails loudly if the exclusion gate does not short-circuit before Qdrant.
    monkeypatch.setattr(
        evaluation._db,
        "get_payload",
        lambda *a, **k: pytest.fail("Qdrant must not be queried for an excluded PDI"),
    )
    assert evaluation.load_profile(305196) is None
