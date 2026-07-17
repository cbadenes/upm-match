"""Tests for the off-line late-fusion sensitivity sweep (analyze_results.py, T9).

Everything runs offline on synthetic `rerank_components`: no reports, no LLM, no
Memgraph. They check that the fusion re-composes rankings correctly, that a
text-only config reproduces the stored listwise order, and that a discriminative
graph signal both improves quality and is the one the grid search selects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.analyze_results import (
    _fused_order,
    _sweep_per_query,
    build_t9_sweep,
    SourceData,
)
from evaluation.evaluate_recommendations import RecommendationResult

IDS = [f"pdi_{i}" for i in range(5)]


def _synthetic_source(discriminative: str | None = "supervised_fit", runs_n: int = 2):
    """SourceData carrying a kg-mode run set where `discriminative` marks the truth."""
    runs = []
    for _ in range(runs_n):
        d = {}
        for q in range(6):
            true = IDS[q % 5]
            comps = {}
            for rank, pid in enumerate(IDS, 1):
                comp = {
                    "stage1_rank": float(rank),
                    "listwise_rank": float(rank),
                    "supervised_total": 0.0,
                    "supervised_fit": 0.0,
                    "teaching_fit": 0.0,
                    "coauthors_in_set": 0.0,
                }
                if discriminative and pid == true:
                    comp[discriminative] = 9.0
                comps[pid] = comp
            d[f"q{q}"] = RecommendationResult(
                gold_id=f"q{q}", mode="rerank_listwise_kg",
                ranked_researcher_ids=IDS, true_director=true, rerank_components=comps,
            )
        runs.append(d)
    sd = SourceData.__new__(SourceData)
    sd.source = "gold"
    sd.runs = []
    sd.by_mode = {"rerank_listwise_kg": runs}
    sd.modes = ["rerank_listwise_kg"]
    sd.common = [f"q{q}" for q in range(6)]
    return sd


def test_fused_order_text_only_reproduces_listwise():
    comp = {
        "pdi_a": {"listwise_rank": 1, "supervised_fit": 0.0},
        "pdi_b": {"listwise_rank": 2, "supervised_fit": 0.0},
        "pdi_c": {"listwise_rank": 3, "supervised_fit": 9.0},
    }
    assert _fused_order(comp, {"text": 1.0}) == ["pdi_a", "pdi_b", "pdi_c"]


def test_fused_order_strong_signal_lifts_tail_candidate():
    comp = {
        "pdi_a": {"stage1_rank": 1, "listwise_rank": 1, "supervised_fit": 0.0},
        "pdi_b": {"stage1_rank": 2, "listwise_rank": 2, "supervised_fit": 0.0},
        "pdi_c": {"stage1_rank": 3, "listwise_rank": 3, "supervised_fit": 9.0},
    }
    assert _fused_order(comp, {"text": 1.0, "supervised_fit": 5.0})[0] == "pdi_c"


def test_sweep_per_query_denoises_across_runs():
    sd = _synthetic_source()
    vals = _sweep_per_query(sd.by_mode["rerank_listwise_kg"], sd.common, {"text": 1.0}, "hit@5")
    # The true director is always inside the 5-candidate list, so Hit@5 == 1 everywhere.
    assert list(vals) == [1.0] * len(sd.common)


def test_discriminative_signal_improves_mrr_and_is_selected():
    sd = _synthetic_source(discriminative="supervised_fit")
    runs = sd.by_mode["rerank_listwise_kg"]
    base = _sweep_per_query(runs, sd.common, {"text": 1.0}, "mrr").mean()
    fused = _sweep_per_query(runs, sd.common, {"text": 1.0, "supervised_fit": 1.0}, "mrr").mean()
    assert fused > base
    md, _ = build_t9_sweep(sd, productivity=None)
    # The grid optimum row should credit the discriminative signal.
    assert "óptimo por MRR" in md and "ajuste de supervisión" in md


def test_build_t9_returns_none_without_kg_mode():
    sd = _synthetic_source()
    sd.by_mode = {}
    sd.modes = []
    assert build_t9_sweep(sd, productivity=None) is None


def test_equity_column_present_only_with_productivity():
    sd = _synthetic_source()
    prod = {pid: float(i) for i, pid in enumerate(IDS)}
    with_prod, _ = build_t9_sweep(sd, productivity=prod)
    without, _ = build_t9_sweep(sd, productivity=None)
    assert "brecha T3" in with_prod
    assert "brecha T3" not in without
