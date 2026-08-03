"""Integration tests of the smart-search clarification flow through /api/search.

The Flask app runs for real (test client); only the EXTERNAL services are
faked at the module boundary: keyword extraction / enrichment (Ollama agents),
the hybrid Qdrant search, the re-ranker and the clarifier's raw LLM call.
The clarification LOOP logic itself (app/clarify.py) runs unmodified, so these
tests exercise the real request/response contract the front-end depends on.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.clarify as clarify_mod
import app.routes as routes
from app.search import PDIResult


class _FakeKeywords:
    general = ["artificial intelligence"]
    specific = ["clinical text mining"]


def _fake_results(n):
    return [
        PDIResult(
            rank=i,
            researcher_id=f"pdi_{1000 + i}",
            name=f"PDI {i}",
            score=1.0 / i,
            score_pct=100 // i,
            all_topics=[f"topic {i}", "natural language processing"],
            summary=f"profile {i}",
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with the external services stubbed out."""
    calls = {"search_limits": [], "enrich_texts": [], "rerank": 0, "llm": []}

    monkeypatch.setattr(routes, "extract_keywords", lambda idea: _FakeKeywords())

    def fake_enrich(text):
        calls["enrich_texts"].append(text)
        return f"ENRICHED[{text[:60]}]"

    monkeypatch.setattr(routes, "enrich_query", fake_enrich)

    def fake_hybrid(db, general, specific, dense_text, limit=10):
        calls["search_limits"].append(limit)
        return _fake_results(limit)

    monkeypatch.setattr(routes, "search_pdis_hybrid", fake_hybrid)

    def fake_rerank(results, dense_text, use_kg):
        calls["rerank"] += 1
        reordered = list(reversed(results))
        for new_rank, r in enumerate(reordered, 1):
            r.stage1_rank = r.rank
            r.rank = new_rank
        info = {
            "variant": "kg" if use_kg else "text",
            "model": "stub",
            "pool": len(results),
            "windows": 2,
            "seconds": 1.0,
        }
        return reordered, info

    monkeypatch.setattr(routes, "rerank_pdi_results", fake_rerank)

    def scripted_llm(system, user, schema, model=None):
        calls["llm"].append((system, user))
        if "ambiguity analyzer" in system:
            # First round: ambiguous; once an exchange exists: clear.
            level = "clear" if "Clarification 1" in user else "topk_clarify"
            return {"ambiguity": level, "reason": "scripted"}
        return {
            "question": "¿Qué dirección te interesa más?",
            "options": ["minería de texto clínico", "imagen médica", "robótica"],
        }

    monkeypatch.setattr(clarify_mod, "_chat_json", scripted_llm)

    from app import create_app

    app = create_app()
    app.testing = True
    # Resolve the endpoint's concrete path from the url map: robust to
    # APP_BASE_PATH (loaded from .env) mounting the blueprint under a prefix.
    # (url_for without a real request would double the prefix here, because the
    # app sets APPLICATION_ROOT *and* mounts the blueprint under the base path;
    # real proxied requests carry the prefix in the path, so only requestless
    # contexts are affected.)
    search_path = next(
        r.rule for r in app.url_map.iter_rules() if r.endpoint == "main.search"
    )
    with app.test_client() as c:
        c.calls = calls
        c.search_path = search_path
        yield c


def _post(client, **body):
    payload = {"idea": "Quiero hacer algo con IA en medicina", "mode": "hybrid"}
    payload.update(body)
    return client.post(client.search_path, json=payload)


def test_first_smart_round_returns_question_without_results(client):
    resp = _post(client, rerank="kg", clarify=True)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["clarify"]["question"] == "¿Qué dirección te interesa más?"
    assert len(data["clarify"]["options"]) == 3
    assert data["clarify"]["round"] == 1
    assert "results" not in data
    # Intermediate rounds must not spend the re-ranker.
    assert client.calls["rerank"] == 0
    # The clarification loop retrieves the deep pool for domain-level aspects.
    assert client.calls["search_limits"] == [routes.CLARIFY_POOL]


def test_answer_refines_query_and_yields_reranked_results(client):
    history = [
        {"question": "¿Qué dirección te interesa más?", "answer": "minería de texto clínico"}
    ]
    resp = _post(client, rerank="kg", clarify=True, clarification=history)
    data = resp.get_json()
    assert data["clarify"] is None
    assert len(data["results"]) == 10  # default limit
    assert data["rerank"]["variant"] == "kg" and client.calls["rerank"] == 1
    # The refined query (idea + exchange) reached BOTH query-time agents.
    assert "Aclaración" in client.calls["enrich_texts"][0]
    assert "minería de texto clínico" in client.calls["enrich_texts"][0]
    # Results are the reranked head: stage1_rank present and order flipped.
    first = data["results"][0]
    assert first["stage1_rank"] == routes.RERANK_POOL


def test_skip_bypasses_the_detector_entirely(client):
    resp = _post(client, rerank="kg", clarify=True, clarify_skip=True)
    data = resp.get_json()
    assert data["clarify"] is None and len(data["results"]) == 10
    assert client.calls["llm"] == []  # detector never invoked


def test_round_cap_forces_results(client, monkeypatch):
    monkeypatch.setattr(clarify_mod, "CLARIFY_MAX_ROUNDS", 1)
    history = [{"question": "q1", "answer": "a1"}]
    # Detector scripted to stay ambiguous forever — the cap must win.
    monkeypatch.setattr(
        clarify_mod,
        "_chat_json",
        lambda s, u, sc, model=None: {"ambiguity": "domain_clarify"},
    )
    resp = _post(client, clarify=True, clarification=history)
    data = resp.get_json()
    assert data["clarify"] is None and "results" in data


def test_detector_failure_degrades_to_results(client, monkeypatch):
    def boom(system, user, schema, model=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(clarify_mod, "_chat_json", boom)
    resp = _post(client, rerank="kg", clarify=True)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["clarify"] is None and len(data["results"]) == 10


def test_plain_hybrid_untouched_by_clarify_machinery(client):
    resp = _post(client)
    data = resp.get_json()
    assert data["clarify"] is None and len(data["results"]) == 10
    assert client.calls["llm"] == []
    assert client.calls["search_limits"] == [10]  # no deep pool without clarify
