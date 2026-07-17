"""Tests for the listwise re-ranking logic (app/rerank.py).

Everything runs offline: the LLM window call is replaced by the `rank_fn` test
seam, so these tests validate the sliding-window mechanics, the permutation
parsing/repair and the per-candidate components — the parts that must never
corrupt a ranking regardless of what the model answers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rerank import RerankCandidate, _parse_ranking, rerank_listwise


def _cands(n: int) -> list[RerankCandidate]:
    return [RerankCandidate(researcher_id=f"pdi_{i}") for i in range(1, n + 1)]


def by_id_desc(_query, window):
    """Fake LLM: ranks a window by numeric id DESCENDING (worst-first input flips)."""
    order = sorted(
        range(len(window)),
        key=lambda i: -int(window[i].researcher_id.split("_")[1]),
    )
    return order


def identity(_query, window):
    return list(range(len(window)))


# --- Sliding-window mechanics ------------------------------------------------


def test_single_window_full_reorder():
    out = rerank_listwise("q", _cands(5), window=20, stride=10, rank_fn=by_id_desc)
    assert out.ranked_ids == ["pdi_5", "pdi_4", "pdi_3", "pdi_2", "pdi_1"]
    assert out.windows == 1


def test_sliding_window_bubbles_deep_candidate_to_top():
    # 30 candidates, window 10, stride 5: the fake LLM always prefers the highest
    # id, so pdi_30 (last in stage 1) must bubble across windows to rank 1.
    out = rerank_listwise("q", _cands(30), window=10, stride=5, rank_fn=by_id_desc)
    assert out.ranked_ids[0] == "pdi_30"
    # Full pass with stride 5 over 30 candidates: windows at 20,15,10,5,0.
    assert out.windows == 5
    # Every candidate is preserved exactly once.
    assert sorted(out.ranked_ids) == sorted(c.researcher_id for c in _cands(30))


def test_identity_rank_fn_keeps_stage1_order():
    out = rerank_listwise("q", _cands(25), window=10, stride=5, rank_fn=identity)
    assert out.ranked_ids == [c.researcher_id for c in _cands(25)]


def test_single_candidate_and_empty_need_no_calls():
    assert rerank_listwise("q", _cands(1), rank_fn=identity).windows == 0
    assert rerank_listwise("q", [], rank_fn=identity).ranked_ids == []


def test_failing_window_degrades_to_input_order():
    def boom(_q, _w):
        raise RuntimeError("ollama down")

    out = rerank_listwise("q", _cands(8), window=10, stride=5, rank_fn=boom)
    assert out.ranked_ids == [c.researcher_id for c in _cands(8)]


# --- Components for the sensitivity analysis ---------------------------------


def test_components_store_both_ranks():
    out = rerank_listwise("q", _cands(3), window=10, stride=5, rank_fn=by_id_desc)
    assert out.components["pdi_3"] == {"stage1_rank": 3.0, "listwise_rank": 1.0}
    assert out.components["pdi_1"] == {"stage1_rank": 1.0, "listwise_rank": 3.0}


# --- Permutation parsing/repair ----------------------------------------------


def test_parse_ranking_valid():
    assert _parse_ranking('{"ranking": [3, 1, 2]}', 3) == [3, 1, 2]


def test_parse_ranking_repairs_duplicates_out_of_range_and_missing():
    # 5 candidates; the model repeats 2, invents 9 and forgets 4 and 5.
    assert _parse_ranking('{"ranking": [2, 2, 9, 3, 1]}', 5) == [2, 3, 1, 4, 5]


def test_parse_ranking_json_wrapped_in_prose_and_think_tags():
    raw = '<think>let me see</think>Sure! {"ranking": [2, 1]} hope it helps'
    assert _parse_ranking(raw, 2) == [2, 1]


def test_parse_ranking_garbage_degrades_to_input_order():
    assert _parse_ranking("no json here", 4) == [1, 2, 3, 4]
