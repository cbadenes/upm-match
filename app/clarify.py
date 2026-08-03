"""Aspect-based clarification for the smart search (fase C, adaptation of ASK).

Adapts ASK (Sahay et al., ACL 2025 Industry): before showing results, an LLM
analyses the student's idea together with the retrieved candidates and decides
whether a clarification question is needed. The registered design decisions
(docs/plan_siguientes_fases.md):

  - Detector: LLM-only (the trained Longformer of the paper is out of reach:
    the server allows no new model services). The LLM sees the idea, the
    conversation so far and a compact view of the top candidates, and emits one
    of the three ASK classes: clear / topk_clarify / domain_clarify.
  - Aspects: depth as a substitute for ASK's offline domain taxonomy. Medium
    ambiguity (topk_clarify) draws the question's aspects from the domains of
    the TOP candidates; high ambiguity (domain_clarify) from the domains of the
    WHOLE retrieved pool (~100), which plays the role of the paper's broad
    domain aspects without pre-curated artefacts.
  - Loop: up to CLARIFY_MAX_ROUNDS (default 3) with early stopping, ASK-style:
    each answer is appended to the query and the search relaunches; the
    detector decides whether to ask again.
  - The question is ONE, in Spanish, with 3-5 concrete options ("¿Te interesa
    más A, B o C?"); the UI adds the "none of these / free text" escape hatch.

Pure logic + Ollama calls; no retrieval code. The caller (the /api/search
smart flow) retrieves the candidates and passes them in.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import ollama
from loguru import logger

_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")
# Model for ambiguity analysis + question generation; defaults to the
# production instruct model (same as the re-ranker).
CLARIFY_MODEL = os.environ.get(
    "CLARIFY_MODEL", os.environ.get("RERANK_MODEL", "qwen2.5:7b-instruct")
)

# Loop geometry (registered decisions, fase C).
CLARIFY_MAX_ROUNDS = int(os.environ.get("CLARIFY_MAX_ROUNDS", "3"))
# Candidates the DETECTOR sees (compact context, the paper's top-k).
DETECT_TOP_N = int(os.environ.get("CLARIFY_DETECT_TOP_N", "10"))
# Aspect pools: medium ambiguity uses the head, high ambiguity the whole pool.
ASPECTS_SHALLOW_N = int(os.environ.get("CLARIFY_ASPECTS_SHALLOW_N", "25"))
# Cap of distinct domain terms fed to the question generator.
_MAX_ASPECT_TERMS = 40
_MAX_OPTIONS = 5

_SEED = 42

# The three ASK ambiguity classes (paper: show_result / topk_clarify /
# domain_clarify). "clear" means: answer now, no question.
CLEAR = "clear"
TOPK_CLARIFY = "topk_clarify"
DOMAIN_CLARIFY = "domain_clarify"
_CLASSES = (CLEAR, TOPK_CLARIFY, DOMAIN_CLARIFY)

_DETECT_SYSTEM = (
    "You are the ambiguity analyzer of an academic advisor matching system. "
    "Students describe a thesis idea and the system retrieves matching "
    "researchers. You decide whether the idea is specific enough to rank "
    "researchers reliably, or whether ONE clarifying question should be asked "
    "first."
)

_DETECT_INSTRUCTIONS = (
    "Classify the ambiguity of the student's thesis idea into exactly one of:\n"
    "  - 'clear': the idea names a concrete topic and approach; the retrieved "
    "researchers look thematically consistent. Show results now.\n"
    "  - 'topk_clarify': the idea is somewhat ambiguous — it maps to several "
    "distinct directions VISIBLE among the retrieved researchers; one question "
    "grounded in those directions would sharpen the ranking.\n"
    "  - 'domain_clarify': the idea is highly underspecified or vague (broad "
    "field only, no concrete problem or method); the retrieved head cannot be "
    "trusted and the question must explore the broader solution space.\n"
    "Rules:\n"
    "  1. Judge ONLY the idea and the clarification exchanges shown; the "
    "candidate list is evidence, not the object of judgement.\n"
    "  2. If earlier clarification answers already pinned the topic down, "
    "prefer 'clear': never re-ask what is answered.\n"
    "  3. Return a JSON object: {\"ambiguity\": <class>, \"reason\": <one "
    "short sentence>}."
)

_DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "ambiguity": {"type": "string", "enum": list(_CLASSES)},
        "reason": {"type": "string"},
    },
    "required": ["ambiguity"],
}

_QUESTION_SYSTEM = (
    "You help students sharpen a vague thesis idea. You ask ONE short "
    "clarifying question, in Spanish, with concrete answer options, so an "
    "academic advisor matching system can rank researchers better."
)

_QUESTION_INSTRUCTIONS = (
    "Write ONE clarifying question about the student's idea, with 3 to "
    f"{_MAX_OPTIONS} answer options.\n"
    "Rules:\n"
    "  1. The question and the options are in SPANISH (the student's "
    "language), even though the expertise areas below are mostly in English.\n"
    "  2. Ask about the SINGLE aspect that best splits the possible research "
    "directions (topic focus, application area, or method family). Never ask "
    "a generic 'can you give more details?'.\n"
    "  3. Each option is short (2-6 words), concrete, and grounded in the "
    "expertise areas provided; options must not overlap.\n"
    "  4. Never ask about something the idea or the previous answers already "
    "state.\n"
    "  5. Return a JSON object: {\"question\": <text>, \"options\": [<text>, "
    "...]}."
)

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "options"],
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ClarifyTurn:
    """One completed clarification round: the question asked and the answer."""

    question: str
    answer: str


@dataclass
class Ambiguity:
    """Detector verdict for one (idea, conversation, candidates) state."""

    level: str  # one of _CLASSES
    reason: str = ""


@dataclass
class Clarification:
    """A generated clarifying question with its options."""

    question: str
    options: list[str] = field(default_factory=list)
    ambiguity: str = TOPK_CLARIFY  # which class triggered it
    llm_seconds: float = 0.0


def effective_idea(idea: str, history: list[ClarifyTurn]) -> str:
    """The refined query: the idea plus each clarification exchange (ASK
    appends the answers; we keep the question too so a bare option label like
    'aprendizaje profundo' stays interpretable by the query pipeline)."""
    parts = [idea.strip()]
    for turn in history:
        parts.append(f"Aclaración ({turn.question.strip()}): {turn.answer.strip()}")
    return "\n".join(p for p in parts if p)


def _conversation_block(idea: str, history: list[ClarifyTurn]) -> str:
    lines = [f"Student idea: {idea.strip()}"]
    for i, turn in enumerate(history, 1):
        lines.append(f"Clarification {i} — asked: {turn.question.strip()}")
        lines.append(f"Clarification {i} — student answered: {turn.answer.strip()}")
    return "\n".join(lines)


def _chat_json(system: str, user: str, schema: dict, model: str) -> dict:
    """One structured-output Ollama call; defensive JSON extraction."""
    client = ollama.Client(host=_OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema,
        options={"temperature": 0.0, "seed": _SEED},
    )
    text = _THINK_RE.sub("", response["message"]["content"]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            return json.loads(match.group(0))
        raise


def detect_ambiguity(
    idea: str,
    history: list[ClarifyTurn],
    candidates: list,
    model: str = CLARIFY_MODEL,
    chat_fn=None,
) -> Ambiguity:
    """LLM-only ambiguity analysis (ASK-AA role, registered decision).

    :param candidates: head of the retrieved list (PDIResult-like: needs
        `.all_topics` and `.summary`); only the first DETECT_TOP_N are shown.
    :param chat_fn: test seam replacing the LLM call; receives (system, user,
        schema) and returns the parsed dict.
    :return: Ambiguity; on any failure degrades to CLEAR (never blocks the
        search behind a broken clarifier).
    """
    chat_fn = chat_fn or (lambda s, u, sc: _chat_json(s, u, sc, model))
    blocks = []
    for i, cand in enumerate(candidates[:DETECT_TOP_N], 1):
        domains = ", ".join((cand.all_topics or [])[:8])
        blocks.append(f"[{i}] {domains}" if domains else f"[{i}] (no domains)")
    user = (
        f"{_DETECT_INSTRUCTIONS}\n\n"
        f"--- Conversation ---\n{_conversation_block(idea, history)}\n\n"
        f"--- Expertise areas of the top retrieved researchers ---\n"
        + "\n".join(blocks)
    )
    try:
        data = chat_fn(_DETECT_SYSTEM, user, _DETECT_SCHEMA)
        level = str(data.get("ambiguity", "")).strip().lower()
        if level not in _CLASSES:
            raise ValueError(f"unknown ambiguity class {level!r}")
        return Ambiguity(level=level, reason=str(data.get("reason") or ""))
    except Exception as exc:  # noqa: BLE001 — degrade to "answer now"
        logger.warning("Ambiguity detection failed; treating as clear: {}", exc)
        return Ambiguity(level=CLEAR, reason="detector unavailable")


def _aspect_terms(candidates: list) -> list[str]:
    """Distinct candidate domains by descending frequency (query-independent).

    The frequency ordering keeps the terms that structure the retrieved space
    at the front, which is what the question generator needs to find the axis
    that best splits the directions.
    """
    counts: Counter[str] = Counter()
    label: dict[str, str] = {}
    for cand in candidates:
        for topic in cand.all_topics or []:
            text = str(topic).strip()
            key = text.lower()
            if not text:
                continue
            counts[key] += 1
            label.setdefault(key, text)
    return [label[k] for k, _ in counts.most_common(_MAX_ASPECT_TERMS)]


def generate_question(
    idea: str,
    history: list[ClarifyTurn],
    candidates: list,
    ambiguity: str,
    model: str = CLARIFY_MODEL,
    chat_fn=None,
) -> Clarification | None:
    """Generate ONE Spanish clarifying question with options (ASK-Clarify role).

    Depth as taxonomy substitute (registered decision): topk_clarify draws the
    aspect terms from the head (ASPECTS_SHALLOW_N candidates), domain_clarify
    from the WHOLE pool passed in (the caller retrieves ~100 deep).

    :return: Clarification, or None when generation fails or yields fewer than
        two options (the caller then shows results — never a broken question).
    """
    import time

    chat_fn = chat_fn or (lambda s, u, sc: _chat_json(s, u, sc, model))
    pool = candidates if ambiguity == DOMAIN_CLARIFY else candidates[:ASPECTS_SHALLOW_N]
    terms = _aspect_terms(pool)
    if not terms:
        return None
    user = (
        f"{_QUESTION_INSTRUCTIONS}\n\n"
        f"--- Conversation ---\n{_conversation_block(idea, history)}\n\n"
        f"--- Expertise areas ({'broad pool' if ambiguity == DOMAIN_CLARIFY else 'top matches'}) ---\n"
        + ", ".join(terms)
    )
    t0 = time.perf_counter()
    try:
        data = chat_fn(_QUESTION_SYSTEM, user, _QUESTION_SCHEMA)
        question = str(data.get("question") or "").strip()
        seen: set[str] = set()
        options: list[str] = []
        for raw in data.get("options") or []:
            text = str(raw).strip()[:80]
            if text and text.lower() not in seen:
                seen.add(text.lower())
                options.append(text)
            if len(options) >= _MAX_OPTIONS:
                break
        if not question or len(options) < 2:
            logger.warning(
                "Clarification generation returned unusable output "
                "(question={!r}, {} options)",
                question,
                len(options),
            )
            return None
        return Clarification(
            question=question,
            options=options,
            ambiguity=ambiguity,
            llm_seconds=round(time.perf_counter() - t0, 4),
        )
    except Exception as exc:  # noqa: BLE001 — degrade to showing results
        logger.warning("Clarification generation failed: {}", exc)
        return None


def next_clarification(
    idea: str,
    history: list[ClarifyTurn],
    candidates: list,
    model: str = CLARIFY_MODEL,
    chat_fn=None,
) -> Clarification | None:
    """One step of the ASK loop: decide whether to ask, and build the question.

    Returns None when the search should answer now: rounds exhausted, the
    detector says clear, or the generator could not produce a usable question
    (every failure path degrades to answering — the clarifier can only ever
    delay results, never block them).
    """
    if len(history) >= CLARIFY_MAX_ROUNDS:
        return None
    verdict = detect_ambiguity(idea, history, candidates, model=model, chat_fn=chat_fn)
    if verdict.level == CLEAR:
        return None
    return generate_question(
        idea, history, candidates, verdict.level, model=model, chat_fn=chat_fn
    )
