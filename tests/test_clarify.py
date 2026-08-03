"""Tests for the ASK-style clarification logic (app/clarify.py).

Everything runs offline: the LLM call is replaced by the `chat_fn` test seam,
so these tests validate the loop decisions, the aspect aggregation, the query
refinement and the degradation paths — the parts that must never block a
search regardless of what the model answers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.clarify as clarify
from app.clarify import (
    CLEAR,
    DOMAIN_CLARIFY,
    TOPK_CLARIFY,
    ClarifyTurn,
    _aspect_terms,
    detect_ambiguity,
    effective_idea,
    generate_question,
    next_clarification,
)


class _Cand:
    def __init__(self, topics):
        self.all_topics = topics
        self.summary = ""


def _cands(n, topics=None):
    return [_Cand(topics or [f"topic {i}", "machine learning"]) for i in range(n)]


def test_effective_idea_appends_exchanges():
    history = [ClarifyTurn(question="¿Qué área?", answer="visión artificial")]
    text = effective_idea("Quiero hacer algo de IA", history)
    assert text.startswith("Quiero hacer algo de IA")
    assert "¿Qué área?" in text and "visión artificial" in text


def test_effective_idea_without_history_is_the_idea():
    assert effective_idea("mi idea", []) == "mi idea"


def test_aspect_terms_order_by_frequency_and_dedupe():
    cands = [
        _Cand(["Machine Learning", "robotics"]),
        _Cand(["machine learning", "nlp"]),
        _Cand(["machine learning", "robotics", ""]),
    ]
    terms = _aspect_terms(cands)
    assert terms[0] == "Machine Learning"  # most frequent, first spelling kept
    assert terms[1] == "robotics"
    assert "nlp" in terms and "" not in terms


def test_detector_maps_classes_and_degrades_to_clear():
    ok = detect_ambiguity(
        "idea", [], _cands(5), chat_fn=lambda s, u, sc: {"ambiguity": "domain_clarify"}
    )
    assert ok.level == DOMAIN_CLARIFY

    bad_class = detect_ambiguity(
        "idea", [], _cands(5), chat_fn=lambda s, u, sc: {"ambiguity": "whatever"}
    )
    assert bad_class.level == CLEAR

    def boom(s, u, sc):
        raise RuntimeError("ollama down")

    down = detect_ambiguity("idea", [], _cands(5), chat_fn=boom)
    assert down.level == CLEAR


def test_generate_question_valid_and_capped():
    clar = generate_question(
        "idea",
        [],
        _cands(30),
        TOPK_CLARIFY,
        chat_fn=lambda s, u, sc: {
            "question": "¿Qué te interesa más?",
            "options": ["visión", "visión", "nlp", "robótica", "hri", "otra", "más"],
        },
    )
    assert clar is not None
    assert clar.question == "¿Qué te interesa más?"
    assert clar.options[:3] == ["visión", "nlp", "robótica"]  # deduped, order kept
    assert len(clar.options) <= 5


def test_generate_question_unusable_output_returns_none():
    too_few = generate_question(
        "idea", [], _cands(5), TOPK_CLARIFY,
        chat_fn=lambda s, u, sc: {"question": "¿?", "options": ["solo una"]},
    )
    assert too_few is None

    def boom(s, u, sc):
        raise RuntimeError("garbled")

    assert generate_question("idea", [], _cands(5), TOPK_CLARIFY, chat_fn=boom) is None


def test_domain_clarify_uses_the_whole_pool():
    seen = {}

    def capture(s, u, sc):
        seen["user"] = u
        return {"question": "¿?", "options": ["a", "b"]}

    # Deep tail candidates carry a distinctive domain: it must appear only in
    # the domain_clarify (whole pool) prompt, not in the shallow one.
    cands = _cands(30) + [_Cand(["quantum sensing"])] * 3
    generate_question("idea", [], cands, DOMAIN_CLARIFY, chat_fn=capture)
    assert "quantum sensing" in seen["user"]

    generate_question("idea", [], cands, TOPK_CLARIFY, chat_fn=capture)
    assert "quantum sensing" not in seen["user"]


def test_next_clarification_asks_when_ambiguous():
    def fake(s, u, sc):
        if "ambiguity analyzer" in s:
            return {"ambiguity": "topk_clarify"}
        return {"question": "¿Área?", "options": ["a", "b", "c"]}

    clar = next_clarification("idea", [], _cands(10), chat_fn=fake)
    assert clar is not None and clar.ambiguity == TOPK_CLARIFY


def test_next_clarification_respects_round_cap(monkeypatch):
    monkeypatch.setattr(clarify, "CLARIFY_MAX_ROUNDS", 2)
    history = [
        ClarifyTurn(question="q1", answer="a1"),
        ClarifyTurn(question="q2", answer="a2"),
    ]

    def always_ambiguous(s, u, sc):
        return {"ambiguity": "domain_clarify", "question": "¿?", "options": ["a", "b"]}

    assert next_clarification("idea", history, _cands(10), chat_fn=always_ambiguous) is None


def test_next_clarification_clear_means_answer_now():
    clar = next_clarification(
        "idea", [], _cands(10), chat_fn=lambda s, u, sc: {"ambiguity": "clear"}
    )
    assert clar is None
