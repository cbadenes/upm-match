"""Tests for the knowledge-graph re-ranking signals (app/graph_signals.py).

The SignalStore is built directly from fabricated data (no Memgraph), so these
validate the two consumer surfaces: readable evidence lines for the prompt and
numeric components for the late-fusion / sensitivity analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph_signals import PdiGraphInfo, SignalStore, _dedup


def _store() -> SignalStore:
    info = {
        "pdi_1": PdiGraphInfo(
            supervised_total=12,
            supervised_types={"TFG": 8, "TFM": 3, "Tesis Doctoral": 1},
            supervised_topics=["machine learning", "visión por computador", "nlp"],
            supervised_titles=["Un título de TFG"],
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


# --- Helpers -------------------------------------------------------------------


def test_dedup_flattens_and_preserves_order():
    raw = [["ML", "nlp"], "ml", None, ["Grafos", ""], "NLP"]
    assert _dedup(raw) == ["ML", "nlp", "Grafos"]
