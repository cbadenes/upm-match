"""Tests for the app-side re-ranking glue (app/search.py: rerank_pdi_results).

Offline: the listwise LLM call and the Memgraph signal store are replaced with
fakes, so these tests validate the glue — pool reordering, rank renumbering,
stage1_rank bookkeeping, KG evidence attachment and the degradation paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.rerank as rerank_mod
import app.search as search_mod
from app.rerank import RerankOutcome
from app.search import PDIResult, rerank_pdi_results


def _results(n: int) -> list[PDIResult]:
    return [
        PDIResult(
            rank=i,
            researcher_id=f"pdi_{i}",
            name=f"PDI {i}",
            score=1.0 / i,
            score_pct=100 // i,
            all_topics=[f"topic{i}"],
            summary=f"summary {i}",
        )
        for i in range(1, n + 1)
    ]


def _fake_rerank(captured: dict):
    """Fake listwise reranker: reverses the candidate order and records inputs."""

    def fake(query, candidates, **kwargs):
        captured["query"] = query
        captured["candidates"] = candidates
        ids = [c.researcher_id for c in reversed(candidates)]
        components = {
            rid: {"stage1_rank": float(i + 1)} for i, rid in enumerate(ids)
        }
        return RerankOutcome(
            ranked_ids=ids, components=components, llm_seconds=1.5, windows=2
        )

    return fake


def test_text_variant_reorders_and_renumbers(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(rerank_mod, "rerank_listwise", _fake_rerank(captured))

    results = _results(3)
    reordered, info = rerank_pdi_results(results, "the enriched query", use_kg=False)

    assert [r.researcher_id for r in reordered] == ["pdi_3", "pdi_2", "pdi_1"]
    assert [r.rank for r in reordered] == [1, 2, 3]
    assert [r.stage1_rank for r in reordered] == [3, 2, 1]
    # Stage-1 affinity is preserved on each card.
    assert reordered[0].score == 1.0 / 3
    assert info["variant"] == "text"
    assert info["pool"] == 3 and info["windows"] == 2 and info["seconds"] == 1.5
    assert captured["query"] == "the enriched query"
    # Candidate text mirrors the indexed profile: summary + domains.
    assert captured["candidates"][0].summary == "summary 1"
    assert captured["candidates"][0].domains == ["topic1"]
    assert captured["candidates"][0].evidence == []


def test_kg_variant_attaches_graph_evidence(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(rerank_mod, "rerank_listwise", _fake_rerank(captured))

    class FakeStore:
        def evidence_lines(self, rid):
            return [f"Supervised student works for {rid}"]

    monkeypatch.setattr(search_mod, "_load_signal_store", lambda ids: FakeStore())

    reordered, info = rerank_pdi_results(_results(2), "q", use_kg=True)
    assert info["variant"] == "kg"
    assert "kg_degraded" not in info
    assert captured["candidates"][0].evidence == ["Supervised student works for pdi_1"]
    assert [r.researcher_id for r in reordered] == ["pdi_2", "pdi_1"]


def test_kg_variant_degrades_to_text_without_memgraph(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(rerank_mod, "rerank_listwise", _fake_rerank(captured))
    monkeypatch.setattr(search_mod, "_load_signal_store", lambda ids: None)

    _, info = rerank_pdi_results(_results(2), "q", use_kg=True)
    assert info["variant"] == "text"
    assert info["kg_degraded"] is True
    assert captured["candidates"][0].evidence == []


def test_signal_store_failure_cooldown(monkeypatch):
    """After a Memgraph failure the store is not re-loaded until the cooldown ends."""
    calls = {"n": 0}

    def failing_load(ids):
        calls["n"] += 1
        return None

    import app.graph_signals as gs

    monkeypatch.setattr(gs, "load_signal_store", failing_load)
    monkeypatch.setattr(search_mod, "_signal_fail_until", 0.0)

    assert search_mod._load_signal_store(["pdi_1"]) is None
    assert search_mod._load_signal_store(["pdi_1"]) is None
    assert calls["n"] == 1  # second call hit the cooldown, no reconnection attempt
