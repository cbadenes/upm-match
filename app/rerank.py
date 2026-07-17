"""Listwise LLM re-ranking of first-stage candidates (two-stage extension).

Implements the second stage of the retrieve-then-rerank pipeline (memoria
Sección sec:met-reranking): an LLM receives the student's (English, enriched)
query plus a numbered window of candidate researcher profiles and emits the
permutation ordered by relevance. Windows slide from the tail of the candidate
list towards the head (RankGPT/RankZephyr style), bubbling relevant candidates
upwards, so a list longer than one window can still be fully reordered.

Design decisions (see docs/plan_siguientes_fases.md, Registro de decisiones):
  - Backend: the already-deployed Ollama service (the server allows no new
    services; model configurable via RERANK_MODEL). vLLM/RankZephyr remains a
    later upgrade path if quality/latency disappoints.
  - Lightweight reimplementation of the sliding-window listwise loop instead of
    the rank_llm package (whose local backends are vLLM/SGLang/HF, not Ollama);
    RankLLM (Sharifymoghaddam et al., SIGIR'25) is the methodological reference.
  - Determinism: temperature 0 + fixed seed, so the re-ranking is reproducible
    given the same first-stage candidate list.

The module is pure logic + an Ollama call; it holds NO retrieval code. Callers
(the evaluation harness now, the app's smart search later) fetch the candidate
texts and pass them in.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import ollama
from loguru import logger

_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")
# Model used for the listwise re-ranking; defaults to the production instruct model.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "qwen2.5:7b-instruct")

# Sliding-window geometry (RankGPT-style): windows of WINDOW candidates advance
# from the tail of the list towards the head in steps of STRIDE, so every
# candidate can climb across window boundaries. WINDOW > STRIDE gives overlap.
RERANK_WINDOW = int(os.environ.get("RERANK_WINDOW", "20"))
RERANK_STRIDE = int(os.environ.get("RERANK_STRIDE", "10"))

# Per-candidate text budget inside the prompt (characters). Keeps a full window
# comfortably inside the model context while preserving the profile's substance.
_SUMMARY_CHARS = int(os.environ.get("RERANK_SUMMARY_CHARS", "600"))
_MAX_DOMAINS = 8

_SEED = 42

_SYSTEM = (
    "You are an expert academic advisor matching students with thesis supervisors. "
    "You rank researcher profiles by how well their expertise fits a student's "
    "thesis proposal."
)

_INSTRUCTIONS = (
    "Rank ALL the candidate researchers below by relevance to the student's proposal, "
    "from most to least relevant.\n"
    "Rules:\n"
    "  1. Judge ONLY by fit between the proposal and each researcher's evidence "
    "(profile, and supervision/teaching record when shown; those lines may mix "
    "English and Spanish).\n"
    "  2. Return the ranking as a JSON object with a single field 'ranking': the "
    "candidate numbers in descending relevance order.\n"
    "  3. Include EVERY candidate number exactly once."
)

_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking"],
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RerankCandidate:
    """One first-stage candidate as the re-ranker sees it.

    `evidence` carries optional extra lines shown verbatim under the profile
    (e.g. the knowledge-graph evidence of the kg-grounded mode: supervision
    record, subjects taught). The re-ranker itself stays source-agnostic.
    """

    researcher_id: str
    summary: str = ""
    domains: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class RerankOutcome:
    """Result of re-ranking one candidate list.

    :ranked_ids: researcher_ids in the new order (all input candidates).
    :components: researcher_id -> {"stage1_rank": r1, "listwise_rank": r2}, the
        decoupled per-candidate signals stored for the off-line sensitivity
        analysis (memoria sec:eval-estrategias-rerank). Graph signals (fase B3)
        will add further keys.
    :llm_seconds: total time spent inside the LLM calls.
    :windows: number of LLM window calls made.
    """

    ranked_ids: list[str]
    components: dict[str, dict[str, float]]
    llm_seconds: float
    windows: int


def _candidate_block(index: int, cand: RerankCandidate) -> str:
    """Render one candidate as a numbered block for the prompt."""
    summary = (cand.summary or "").strip()
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[:_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
    domains = ", ".join(cand.domains[:_MAX_DOMAINS])
    lines = [f"[{index}]"]
    if domains:
        lines.append(f"Expertise: {domains}")
    lines.append(summary or "(no profile summary available)")
    lines.extend(cand.evidence)
    return "\n".join(lines)


def _parse_ranking(content: str, n: int) -> list[int]:
    """Parse the model's ranking into a valid permutation of 1..n.

    Defensive: strips reasoning tags, extracts the first JSON object when the
    model wraps it in prose, drops out-of-range/duplicate numbers and appends any
    missing candidates in their original order, so the output is ALWAYS a
    complete permutation (a garbled answer degrades to the input order, never to
    a crash or a lost candidate).
    """
    text = _THINK_RE.sub("", content).strip()
    numbers: list = []
    try:
        numbers = json.loads(text).get("ranking", [])
    except (json.JSONDecodeError, AttributeError):
        match = _JSON_RE.search(text)
        if match:
            try:
                numbers = json.loads(match.group(0)).get("ranking", [])
            except (json.JSONDecodeError, AttributeError):
                numbers = []

    seen: set[int] = set()
    order: list[int] = []
    for num in numbers:
        if isinstance(num, int) and 1 <= num <= n and num not in seen:
            seen.add(num)
            order.append(num)
    order += [i for i in range(1, n + 1) if i not in seen]
    return order


def _rank_window(
    query: str, window: list[RerankCandidate], model: str
) -> list[int]:
    """Ask the LLM to order one window; returns 0-based positions into `window`."""
    blocks = "\n\n".join(_candidate_block(i + 1, c) for i, c in enumerate(window))
    user = (
        f"{_INSTRUCTIONS}\n\n"
        f"--- Student proposal ---\n{query}\n\n"
        f"--- Candidates ({len(window)}) ---\n{blocks}"
    )
    client = ollama.Client(host=_OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        format=_RANKING_SCHEMA,
        options={"temperature": 0.0, "seed": _SEED},
    )
    ranking = _parse_ranking(response["message"]["content"], len(window))
    return [num - 1 for num in ranking]


def rerank_listwise(
    query: str,
    candidates: list[RerankCandidate],
    model: str = RERANK_MODEL,
    window: int = RERANK_WINDOW,
    stride: int = RERANK_STRIDE,
    rank_fn=None,
) -> RerankOutcome:
    """Re-rank a first-stage candidate list with sliding-window listwise calls.

    Windows are processed from the TAIL of the list towards the head with the
    given stride, so a strong candidate buried deep can climb one window per
    step until it reaches the top (the RankGPT strategy). A list that fits in a
    single window needs exactly one call.

    :param query: the (English) query text the candidates were retrieved for.
    :param candidates: first-stage candidates, best-first (stage-1 order).
    :param rank_fn: test seam — replaces the LLM window call; receives
        (query, window_candidates) and returns 0-based positions.
    :return: RerankOutcome with the full permutation and per-candidate components.
    """
    rank_fn = rank_fn or (lambda q, w: _rank_window(q, w, model))
    n = len(candidates)
    order = list(range(n))
    llm_seconds = 0.0
    calls = 0

    if n > 1:
        start = max(0, n - window)
        while True:
            chunk = order[start : start + window]
            t0 = time.perf_counter()
            try:
                positions = rank_fn(query, [candidates[i] for i in chunk])
            except Exception as exc:  # noqa: BLE001 — keep the current order on failure
                logger.warning("Rerank window failed (start={}): {}", start, exc)
                positions = list(range(len(chunk)))
            llm_seconds += time.perf_counter() - t0
            calls += 1
            order[start : start + window] = [chunk[p] for p in positions]
            if start == 0:
                break
            start = max(0, start - stride)

    ranked_ids = [candidates[i].researcher_id for i in order]
    components = {
        cand.researcher_id: {"stage1_rank": float(i + 1)}
        for i, cand in enumerate(candidates)
    }
    for new_rank, rid in enumerate(ranked_ids, 1):
        components[rid]["listwise_rank"] = float(new_rank)
    return RerankOutcome(
        ranked_ids=ranked_ids,
        components=components,
        llm_seconds=round(llm_seconds, 4),
        windows=calls,
    )
