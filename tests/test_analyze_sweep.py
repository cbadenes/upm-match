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
    sd.extra_modes = {}
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


def test_loo_sweep_reads_the_extra_mode_and_is_skipped_when_absent():
    """The leave-one-out twin lives outside `by_mode` so it stays out of T1/T2/T3."""
    from evaluation.analyze_results import SWEEP_MODE_LOO

    sd = _synthetic_source()
    assert build_t9_sweep(sd, productivity=None, sweep_mode=SWEEP_MODE_LOO) is None

    sd.extra_modes = {SWEEP_MODE_LOO: sd.by_mode["rerank_listwise_kg"]}
    md, tex = build_t9_sweep(sd, productivity=None, sweep_mode=SWEEP_MODE_LOO)
    assert "leave-one-out" in tex
    assert "tab:rerank-sweep-gold-loo" in tex
    # And it must not have leaked into the search conditions.
    assert SWEEP_MODE_LOO not in sd.modes


def test_equity_column_present_only_with_productivity():
    sd = _synthetic_source()
    prod = {pid: float(i) for i, pid in enumerate(IDS)}
    with_prod, _ = build_t9_sweep(sd, productivity=prod)
    without, _ = build_t9_sweep(sd, productivity=None)
    assert "brecha T3" in with_prod
    assert "brecha T3" not in without


# --- Partial-run guard ---------------------------------------------------------


def test_partial_runs_do_not_shrink_the_paired_query_set(tmp_path, caplog):
    """A smoke test published alongside a campaign must not collapse `common`.

    `common` is an intersection across every (mode, run), so a 5-query run would
    otherwise reduce every table to 5 queries with no error raised.
    """
    import json
    from evaluation.analyze_results import SourceData

    def _write(run: str, mode: str, n: int) -> None:
        d = tmp_path / "silver" / "final" / run / "cache"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"recommendations_{mode}.json").write_text(
            json.dumps(
                [
                    {
                        "gold_id": f"q{i}",
                        "mode": mode,
                        "ranked_researcher_ids": ["pdi_1"],
                        "true_director": "pdi_1",
                    }
                    for i in range(n)
                ]
            ),
            encoding="utf-8",
        )

    _write("20260101_000000", "hybrid", 100)
    _write("20260102_000000", "rerank_listwise_kg_excl", 5)  # smoke test

    runs = [
        tmp_path / "silver" / "final" / "20260101_000000",
        tmp_path / "silver" / "final" / "20260102_000000",
    ]
    sd = SourceData("silver", runs)
    assert len(sd.common) == 100
    assert "rerank_listwise_kg_excl" not in sd.modes
