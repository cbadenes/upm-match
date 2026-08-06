"""Tests for the knowledge-graph re-ranking signals (app/graph_signals.py).

The SignalStore is built directly from fabricated data (no Memgraph), so these
validate the two consumer surfaces: readable evidence lines for the prompt and
numeric components for the late-fusion / sensitivity analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph_signals import PdiGraphInfo, SignalStore, SupervisedWork, _dedup


def _works() -> list[SupervisedWork]:
    """12 supervised works, newest first, as the graph loader returns them."""
    works = [
        SupervisedWork(
            work_id="tft_1",
            year=2025,
            type="TFG",
            keywords=["machine learning"],
            title="Un título de TFG",
        ),
        SupervisedWork(
            work_id="tft_2",
            year=2024,
            type="TFG",
            keywords=["visión por computador", "nlp"],
            title="Otro TFG",
        ),
    ]
    # Padding so the totals and the type histogram match the original fixture.
    works += [
        SupervisedWork(work_id=f"tft_pad_{i}", year=2023 - i, type="TFG")
        for i in range(6)
    ]
    works += [
        SupervisedWork(work_id=f"tft_tfm_{i}", year=2017 - i, type="TFM")
        for i in range(3)
    ]
    works.append(
        SupervisedWork(work_id="tft_phd", year=2014, type="Tesis Doctoral")
    )
    return works


def _store() -> SignalStore:
    info = {
        "pdi_1": PdiGraphInfo(
            works=_works(),
            subjects=["Aprendizaje Automático", "Programación I"],
        ),
        "pdi_2": PdiGraphInfo(),  # indexed but without graph activity
    }
    coauthors = {
        "pdi_1": {"pdi_2", "pdi_3", "pdi_9"},
        "pdi_2": {"pdi_1"},
        "pdi_3": {"pdi_1"},
        "pdi_9": {"pdi_1"},
    }
    return SignalStore(info, coauthors)


# --- Evidence lines (features-in-prompt mechanism) ----------------------------


def test_evidence_lines_summarise_supervision_and_teaching():
    lines = _store().evidence_lines("pdi_1")
    assert len(lines) == 2
    assert lines[0].startswith("Supervised student works: 12 (")
    assert "8 TFG" in lines[0] and "machine learning" in lines[0]
    assert lines[1].startswith("Teaches: Aprendizaje Automático")


def test_evidence_lines_empty_for_inactive_or_unknown():
    store = _store()
    assert store.evidence_lines("pdi_2") == []
    assert store.evidence_lines("pdi_unknown") == []


# --- Numeric components (late fusion / sensitivity) ---------------------------


def test_components_lexical_fit_is_diacritic_insensitive():
    store = _store()
    query = store.normalize_query(
        "This research applies machine learning and vision por computador to X."
    )
    comps = store.components("pdi_1", query, {"pdi_2", "pdi_3"})
    assert comps["supervised_total"] == 12.0
    # "machine learning" matches; "visión por computador" matches despite the
    # accent (normalisation); "nlp" is below the minimum term length.
    assert comps["supervised_fit"] == 2.0
    assert comps["teaching_fit"] == 0.0


def test_components_coauthor_degree_counts_only_inside_candidate_set():
    store = _store()
    q = store.normalize_query("anything")
    assert store.components("pdi_1", q, {"pdi_2", "pdi_3"})["coauthors_in_set"] == 2.0
    assert store.components("pdi_1", q, {"pdi_7"})["coauthors_in_set"] == 0.0
    assert store.components("pdi_2", q, {"pdi_1"})["coauthors_in_set"] == 1.0


def test_components_bilingual_query_concatenation_bridges_languages():
    # Recommended usage: normalise the ENGLISH enriched query concatenated with
    # the raw SPANISH proposal, so Spanish supervised topics can match too.
    store = _store()
    english_only = store.normalize_query("Research on computer vision for robots.")
    both = store.normalize_query(
        "Research on computer vision for robots. "
        "Quiero aplicar visión por computador a robots móviles."
    )
    assert store.components("pdi_1", english_only, set())["supervised_fit"] == 0.0
    assert store.components("pdi_1", both, set())["supervised_fit"] == 1.0


def test_components_unknown_candidate_all_zero():
    store = _store()
    comps = store.components("pdi_unknown", store.normalize_query("q"), set())
    assert all(v == 0.0 for v in comps.values())


# --- Leave-one-out masking ------------------------------------------------------


def test_excluding_a_work_removes_its_topic_and_decrements_the_total():
    store = _store()
    q = store.normalize_query("applies machine learning and vision por computador")
    full = store.components("pdi_1", q, set())
    masked = store.components("pdi_1", q, set(), exclude_work_id="tft_1")
    assert full["supervised_fit"] == 2.0
    assert masked["supervised_fit"] == 1.0  # "machine learning" is gone
    assert masked["supervised_total"] == full["supervised_total"] - 1


def test_excluding_a_work_leaves_the_clean_signals_untouched():
    """Teaching and co-authorship do not derive from supervised works."""
    store = _store()
    q = store.normalize_query("aprendizaje automático")
    full = store.components("pdi_1", q, {"pdi_2", "pdi_3"})
    masked = store.components("pdi_1", q, {"pdi_2", "pdi_3"}, exclude_work_id="tft_1")
    assert masked["teaching_fit"] == full["teaching_fit"] == 1.0
    assert masked["coauthors_in_set"] == full["coauthors_in_set"] == 2.0


def test_excluding_a_work_hides_it_from_the_evidence_lines():
    store = _store()
    assert "machine learning" in store.evidence_lines("pdi_1")[0]
    masked = store.evidence_lines("pdi_1", exclude_work_id="tft_1")
    assert "machine learning" not in masked[0]
    assert masked[0].startswith("Supervised student works: 11 (")
    assert masked[1].startswith("Teaches:")  # teaching survives


def test_excluding_an_unrelated_work_changes_nothing_but_the_count():
    store = _store()
    q = store.normalize_query("applies machine learning and vision por computador")
    masked = store.components("pdi_1", q, set(), exclude_work_id="tft_phd")
    assert masked["supervised_fit"] == 2.0


def test_masking_frees_a_slot_in_the_recency_window():
    """Dropping a work must let the next one into the 20-work window."""
    works = [
        SupervisedWork(work_id="tft_new", year=2025, keywords=["tema reciente"])
    ]
    works += [
        SupervisedWork(work_id=f"tft_{i}", year=2024 - i, keywords=[f"relleno {i}"])
        for i in range(19)
    ]
    works.append(SupervisedWork(work_id="tft_21", year=2000, keywords=["tema antiguo"]))
    store = SignalStore({"p": PdiGraphInfo(works=works)}, {})
    q = store.normalize_query("sobre tema antiguo")
    assert store.components("p", q, set())["supervised_fit"] == 0.0
    assert store.components("p", q, set(), exclude_work_id="tft_new")["supervised_fit"] == 1.0


# --- Helpers -------------------------------------------------------------------


def test_dedup_flattens_and_preserves_order():
    raw = [["ML", "nlp"], "ml", None, ["Grafos", ""], "NLP"]
    assert _dedup(raw) == ["ML", "nlp", "Grafos"]
