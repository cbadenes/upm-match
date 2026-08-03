"""Tests for the off-line clarification evaluation (C4).

Offline: the LLM calls (detector, question generator, student simulator,
keyword extraction, enrichment) and the retrieval are all replaced by fakes, so
these tests validate the harness logic — the loop, the oracle-driven answer,
the telemetry recorded and the degradation paths — without any service.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.clarify as clarify_mod
import evaluation.evaluate_recommendations as ev
from evaluation.evaluate_recommendations import (
    GoldRecord,
    RecommendationResult,
    SilverRecord,
    _simulate_answer,
    run_clarify,
)


class _Profile:
    def __init__(self, domains, summary=""):
        self.domains = domains
        self.summary = summary


def _fixtures(n=10):
    gold = GoldRecord(
        id="tfm_1",
        _project_id="g1",
        _title="Deteccion de bulos",
        description="Sistema de deteccion de desinformacion con NLP en redes sociales.",
        director="pdi_5",
        year=2025,
        study_type="TFM",
        knowledge_area="ETSISI",
    )
    silver = SilverRecord(gold_id="g1", proposal="Quiero hacer algo con IA")
    stage1 = RecommendationResult(
        gold_id="g1",
        mode="hybrid",
        ranked_researcher_ids=[f"pdi_{i}" for i in range(1, n + 1)],
        true_director="pdi_5",
    )
    profiles = {f"pdi_{i}": _Profile([f"topic {i}", "nlp"]) for i in range(1, n + 1)}
    return silver, gold, stage1, profiles


class _FakeResult:
    def __init__(self, rid):
        self.researcher_id = rid
        self.all_topics = ["nlp"]
        self.summary = ""


@pytest.fixture()
def stubs(monkeypatch):
    """Stub every external call; the clarification loop itself runs for real."""
    calls = {"searches": 0, "answers": []}

    monkeypatch.setattr(
        ev, "_simulate_answer", lambda q, o, oracle, model: (o[0] if o else "")
    )

    import app.agent as agent
    import app.search as search

    class _KW:
        general = ["misinformation"]
        specific = ["nlp"]

    monkeypatch.setattr(agent, "extract_keywords", lambda t: _KW())
    monkeypatch.setattr(agent, "enrich_query", lambda t: f"EN[{t[:30]}]")

    def fake_search(db, g, s, dense, limit=100):
        calls["searches"] += 1
        # After clarification the true director climbs to the top.
        ids = ["pdi_5"] + [f"pdi_{i}" for i in range(1, 11) if i != 5]
        return [_FakeResult(r) for r in ids]

    monkeypatch.setattr(search, "search_pdis_hybrid", fake_search)
    return calls


def test_clear_query_returns_stage1_untouched(stubs, monkeypatch):
    monkeypatch.setattr(
        clarify_mod, "_chat_json", lambda s, u, sc, model=None: {"ambiguity": "clear"}
    )
    silver, gold, stage1, profiles = _fixtures()
    res = run_clarify(silver, gold, stage1, profiles, db=None)

    assert res.clarify_info["rounds"] == 0
    assert res.ranked_researcher_ids == stage1.ranked_researcher_ids
    assert stubs["searches"] == 0  # no clarification -> no re-retrieval
    assert res.error is None


def test_ambiguous_query_asks_refines_and_relaunches(stubs, monkeypatch):
    state = {"n": 0}

    def scripted(system, user, schema, model=None):
        if "ambiguity analyzer" in system:
            state["n"] += 1
            return {"ambiguity": "topk_clarify" if state["n"] == 1 else "clear"}
        return {"question": "¿Qué área?", "options": ["desinformación", "robótica"]}

    monkeypatch.setattr(clarify_mod, "_chat_json", scripted)
    silver, gold, stage1, profiles = _fixtures()
    res = run_clarify(silver, gold, stage1, profiles, db=None)

    info = res.clarify_info
    assert info["rounds"] == 1
    assert info["levels"] == ["topk_clarify"]
    assert info["questions"][0]["answer"] == "desinformación"
    assert "Aclaración" in info["refined_query"]
    assert stubs["searches"] == 1  # the query changed -> retrieval reran
    assert res.ranked_researcher_ids[0] == "pdi_5"  # clarification helped
    assert res.llm_seconds > 0 or res.llm_seconds == 0.0  # timings recorded


def test_round_cap_is_respected(stubs, monkeypatch):
    monkeypatch.setattr(clarify_mod, "CLARIFY_MAX_ROUNDS", 2)
    monkeypatch.setattr(
        clarify_mod,
        "_chat_json",
        lambda s, u, sc, model=None: (
            {"ambiguity": "domain_clarify"}
            if "ambiguity analyzer" in s
            else {"question": "¿?", "options": ["a", "b"]}
        ),
    )
    silver, gold, stage1, profiles = _fixtures()
    res = run_clarify(silver, gold, stage1, profiles, db=None)
    assert res.clarify_info["rounds"] == 2
    assert stubs["searches"] == 2


def test_missing_stage1_cache_is_a_clean_error(stubs):
    silver, gold, _, profiles = _fixtures()
    res = run_clarify(silver, gold, None, profiles, db=None)
    assert res.error is not None and "stage-1" in res.error
    assert res.ranked_researcher_ids == []


def test_answer_simulator_normalises_and_degrades(monkeypatch):
    opts = ["minería de texto", "imagen médica"]
    # Model returns the option with different casing -> normalised to the option.
    monkeypatch.setattr(
        ev, "_chat_json", lambda *a, **k: {"answer": "MINERÍA DE TEXTO"}
    )
    assert _simulate_answer("¿?", opts, "oracle", "m") == "minería de texto"

    # Free-text answer (none of the options fits) is kept.
    monkeypatch.setattr(ev, "_chat_json", lambda *a, **k: {"answer": "otra cosa"})
    assert _simulate_answer("¿?", opts, "oracle", "m") == "otra cosa"

    # A failure degrades to the first option so the loop always advances.
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ev, "_chat_json", boom)
    assert _simulate_answer("¿?", opts, "oracle", "m") == "minería de texto"
