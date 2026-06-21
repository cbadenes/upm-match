"""
UPM Match — Recommendation quality evaluation harness.

METODOLOGY

1. GOLD STANDARD  (hand-made)
   500 real TFG/TFM records curated by hand, split in two files:
     - evaluation/gold_standard_tfg.json  (250 records)
     - evaluation/gold_standard_tfm.json  (250 records)
   Each record holds:
     - id: per-file string index, e.g. "gold_tfg_0001" (the stable key is `_project_id`)
     - description: title + abstract + keywords of the thesis work
     - year: academic year of the work
     - director: the actual supervisor, as a researcher_id (`pdi_<n>`)
     - study_type: "TFG" | "TFM"
     - knowledge_area: the school (ETSII / ETSIT / ETSISI / ETSIST)
     - _project_id: globally-unique thesis id (`tft_<n>`) — used as gold_id
     - _title/_keywords: debug fields

2. SILVER STANDARD  (synthetic, evaluation/silver_standard.json)
   For each gold record, 3 local LLMs each rewrite the thesis `description` into a *student-style proposal* (the free text a student would type into UPM Match when looking for a supervisor).
   The 3 candidates are then scored by the same 3 LLMs (LLM-as-judge); the highest-rated proposal becomes the silver record.
   Generation is reproducible (fixed seed) and resumable (cached to disk).

3. RECOMMENDATION RUN
   Every reference record is evaluated in five modes:
     - lexical_direct: control / ablation — the ORIGINAL thesis text goes straight to the sparse BM42 keyword index in Qdrant, with no LLM and no proposal rewriting (the most basic lexical baseline).
     - keywords: LLM keyword extraction -> sparse BM42 + RRF.
     - summary_direct: embed the silver proposal -> dense summary search.
     - summary_enriched: LLM-enriched proposal -> dense summary search.
     - hybrid: extracted keywords (sparse) + enriched text (dense), RRF.
   For each query the ranked researcher_ids are compared against the gold record's true director, and the wall-clock time is recorded, split into the generative-LLM portion and the retrieval portion, to report the cost of the LLM-in-the-loop.
   To keep the comparison fair, the query set is restricted to the supervisors present in BOTH indices (sparse keywords + dense summaries) so every mode can rank the true director (the "common universe" of the experimental protocol).

4. METRICS + ANALYSIS
   Single-relevant ranking metrics (Hit@k, Precision@k, Recall@k, MRR, MAP, NDCG@k, F1@k) aggregated per mode and broken down by year, knowledge area and study type, to determine which search approach performs best and where it fails.
   Per-query timings are aggregated alongside (mean / median / p90 total, plus the LLM and retrieval split), so the report carries everything the thesis needs.

RESUMABILITY
   Both stages are resumable and need not run in one go. 'build-silver' skips records already generated; 'evaluate' caches every mode's runs to disk (one file per mode) and skips queries already completed, retrying only the ones that errored.
   Re-run the same command to continue where it left off, so the full benchmark can be completed across several sessions; the report is rebuilt from cache each time.

USAGE

    # 1. Generate the silver standard from the gold standard (3 LLMs)
    uv run --module evaluation.evaluate_recommendations build-silver
    uv run --module evaluation.evaluate_recommendations build-silver --limit 10

    # 2. Evaluate every mode, compute metrics + timings + segment breakdowns, report.
    #    --source gold evaluates directly over the thesis descriptions (no silver
    #    needed, so it can run first); --source silver uses the generated proposals.
    #    --publish-minio uploads the report to MinIO for off-server retrieval.
    uv run --module evaluation.evaluate_recommendations evaluate --source gold
    uv run --module evaluation.evaluate_recommendations evaluate --source silver
    uv run --module evaluation.evaluate_recommendations evaluate --modes keywords hybrid

    # 3. Validate the metric math offline (no Qdrant / Ollama needed)
    uv run --module evaluation.evaluate_recommendations selftest

Requires Qdrant + Ollama reachable (same env vars as the app: QDRANT_*, OLLAMA_*).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ollama
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

# Configuration

GOLD_TFG_PATH = Path("evaluation/gold_standard_tfg.json")
GOLD_TFM_PATH = Path("evaluation/gold_standard_tfm.json")
SILVER_PATH = Path("evaluation/silver_standard.json")
CACHE_DIR = Path("evaluation/cache")
REPORTS_DIR = Path("evaluation/reports")

# The three local LLMs used both as proposal generators and as judges.
# Override with EVAL_LLMS="modelA,modelB,modelC". All must be pulled in Ollama.
_DEFAULT_MODELS = ["qwen2.5:7b-instruct", "llama3.1:latest", "qwen3:4b"]
GENERATOR_MODELS: list[str] = [
    m.strip()
    for m in os.environ.get("EVAL_LLMS", ",".join(_DEFAULT_MODELS)).split(",")
    if m.strip()
]

# Search modes compared by the harness. The first is a control (a pure lexical baseline with no LLM, used as an ablation).
# The other four are the strategies exposed by the web app (see app/routes.py).
# Order matches the memoria tables, with the control reported first.
SEARCH_MODES: list[str] = [
    "lexical_direct",
    "keywords",
    "summary_direct",
    "summary_enriched",
    "hybrid",
]

# Human-readable strategy names (match the labels used in the memoria tables).
MODE_LABELS: dict[str, str] = {
    "lexical_direct": "Léxica directa (control)",
    "keywords": "Léxica con LLM",
    "summary_direct": "Semántica directa",
    "summary_enriched": "Semántica enriquecida",
    "hybrid": "Híbrida",
}

# Cut-offs at which ranking metrics are reported.
K_VALUES: list[int] = [1, 3, 5, 10, 25]

# Fixed seed so the silver standard is reproducible across runs.
SEED = 42

# How often (in queries) the per-mode recommendation cache is flushed to disk.
# Kept small so an interrupted run loses at most this many queries on resume.
CACHE_FLUSH_EVERY = 10

OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{OLLAMA_PORT}")


# Data models
class GoldRecord(BaseModel):
    """One hand-curated TFG/TFM ground-truth record."""

    id: str
    description: str
    year: int
    director: str  # researcher_id known to the system, e.g. "pdi_308607"
    study_type: str  # "TFG" | "TFM"
    knowledge_area: str  # school abbreviation: ETSII / ETSIT / ETSISI / ETSIST
    project_id: str = Field(alias="_project_id")
    title: str = Field(default="", alias="_title")
    keywords: list[str] = Field(default_factory=list, alias="_keywords")

    model_config = {"populate_by_name": True}

    @property
    def gold_id(self) -> str:
        """Globally-unique, stable key for this record (the thesis id)."""
        return self.project_id


class ProposalCandidate(BaseModel):
    """A student-style proposal generated from a gold description by one LLM."""

    gold_id: str
    model: str
    proposal: str
    judge_score: float | None = None  # mean panel score, filled in during judging


class SilverRecord(BaseModel):
    """The best-rated synthetic proposal for a gold record."""

    gold_id: str
    proposal: str
    source_model: str
    judge_score: float
    candidates: list[ProposalCandidate] = Field(
        default_factory=list
    )  # kept for error analysis


class RecommendationResult(BaseModel):
    """Ranked recommendations produced for one query in one mode, with timings.

    Timings are split so the LLM-in-the-loop cost can be reported on its own:
      - llm_seconds:    generative LLM time (keyword extraction / query enrichment).
      - search_seconds: retrieval time (embedding + Qdrant query).
      - total_seconds:  llm_seconds + search_seconds.
    They are None for results loaded from a cache written before timing existed.
    """

    gold_id: str
    mode: str
    ranked_researcher_ids: list[str] = Field(default_factory=list)
    true_director: str = ""
    error: str | None = None
    llm_seconds: float | None = None
    search_seconds: float | None = None
    total_seconds: float | None = None


class MetricReport(BaseModel):
    """Aggregated metrics + timings for one search mode (optionally one segment)."""

    mode: str
    segment: str = "all"  # e.g. "all", "year=2023", "study_type=TFM"
    n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    timing: dict[str, float] = Field(default_factory=dict)


# LLM plumbing
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by some models."""
    return _THINK_RE.sub("", text).strip()


def _chat_json(
    model: str,
    system: str,
    user: str,
    schema: dict,
    temperature: float,
) -> dict:
    """Call an Ollama chat model and return a parsed JSON object matching `schema`.

    Uses Ollama structured outputs (the schema is passed as `format`) plus a fixed
    seed for reproducibility. Defensively strips reasoning tags and extracts the
    first JSON object if a model wraps its answer in prose.

    :raises RuntimeError: if the response cannot be parsed as JSON.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema,
        options={"seed": SEED, "temperature": temperature},
    )
    content = response["message"]["content"]
    content = _strip_think(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_RE.search(content)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"{model} returned non-JSON content: {content[:200]!r}")


# 1. Gold standard
def load_gold_standard(
    paths: tuple[Path, Path] = (GOLD_TFG_PATH, GOLD_TFM_PATH),
) -> list[GoldRecord]:
    """Load and validate the hand-made gold standard (TFG + TFM files combined).

    Records are keyed by their globally-unique `_project_id`; duplicates across
    files (which should not happen) are dropped with a warning.
    """
    records: list[GoldRecord] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Gold standard file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            record = GoldRecord.model_validate(item)
            if not record.description.strip():
                logger.warning("Empty description for {} — skipping", record.gold_id)
                continue
            if record.gold_id in seen:
                logger.warning("Duplicate gold_id {} — skipping", record.gold_id)
                continue
            seen.add(record.gold_id)
            records.append(record)
    logger.info("Loaded {} gold records ({} files)", len(records), len(paths))
    return records


def _scroll_researcher_ids(db, collection: str) -> set[str]:
    """Scroll a Qdrant collection (or alias) and return its researcher_ids.

    Returns an empty set if the collection is missing, so the universe computation
    degrades gracefully instead of aborting the whole run.
    """
    ids: set[str] = set()
    offset = None
    while True:
        try:
            points, offset = db._client.scroll(
                collection_name=collection,
                limit=512,
                offset=offset,
                with_payload=["researcher_id"],
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 — missing/unreachable collection
            logger.warning("Could not scroll collection {!r}: {}", collection, exc)
            return ids
        for p in points:
            rid = (p.payload or {}).get("researcher_id")
            if rid:
                ids.add(str(rid))
        if offset is None:
            break
    return ids


def common_indexed_universe(db) -> tuple[set[str], set[str], set[str]]:
    """Return (keywords_ids, summaries_ids, common) researcher_ids across both indices.

    The sparse keyword index (lexical modes) and the dense summary index (semantic
    and hybrid modes) cover slightly different researcher populations. Restricting
    the evaluation to supervisors present in *both* lets every strategy rank the
    true director, so the comparison is not biased by a mode searching a different
    universe (the experimental protocol's "common universe").
    """
    from app.search import COLLECTION_NAME, SUMMARY_COLLECTION

    keywords_ids = _scroll_researcher_ids(db, COLLECTION_NAME)
    summaries_ids = _scroll_researcher_ids(db, SUMMARY_COLLECTION)
    return keywords_ids, summaries_ids, keywords_ids & summaries_ids


def warn_unresolved_directors(gold: list[GoldRecord], common: set[str]) -> None:
    """Warn about gold directors that are absent from the common index universe."""
    missing = sorted({g.director for g in gold if g.director not in common})
    if missing:
        logger.warning(
            "{} gold directors are NOT in the common index universe (their queries "
            "can never hit and are dropped by the universe filter): {}",
            len(missing),
            missing,
        )
    else:
        logger.info(
            "All {} distinct gold directors are in the common index universe.",
            len({g.director for g in gold}),
        )


# 2. Silver standard

_PROPOSAL_SYSTEM = (
    "Eres un estudiante de la Universidad Politécnica de Madrid que busca director para su Trabajo de Fin de Grado o de Máster."
    "Escribes una breve propuesta con tu idea de proyecto para que el sistema te recomiende un director."
)
_PROPOSAL_INSTRUCTIONS = (
    "A partir de la descripción de un trabajo ya realizado, redacta la propuesta que habría escrito el estudiante ANTES de empezarlo."
    "Reglas:\n"
    "  1. Español, primera persona, tono natural de estudiante (no académico).\n"
    "  2. De 3 a 6 frases. Explica el interés, el problema y el enfoque técnico.\n"
    "  3. NO copies el título literal del trabajo.\n"
    "  4. NO menciones nombres de profesores, directores ni títulos de artículos.\n"
    "  5. Prosa continua, sin listas ni encabezados.\n"
    "Devuelve un JSON con el campo 'proposal'."
)


class _ProposalSchema(BaseModel):
    proposal: str


class _JudgeSchema(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


def generate_proposal_candidates(
    gold: GoldRecord,
    models: list[str] = GENERATOR_MODELS,
) -> list[ProposalCandidate]:
    """Rewrite a thesis description into student-style proposals (one per LLM).

    Models that fail (connection / parse error) are skipped with a warning so a
    single bad model does not abort the whole record.
    """
    schema = _ProposalSchema.model_json_schema()
    user = f"{_PROPOSAL_INSTRUCTIONS}\n\n--- Descripción del trabajo ---\n{gold.description}"

    candidates: list[ProposalCandidate] = []
    for model in models:
        try:
            data = _chat_json(model, _PROPOSAL_SYSTEM, user, schema, temperature=0.7)
            proposal = _ProposalSchema.model_validate(data).proposal.strip()
            if proposal:
                candidates.append(
                    ProposalCandidate(
                        gold_id=gold.gold_id, model=model, proposal=proposal
                    )
                )
        except Exception as exc:  # noqa: BLE001 — robustness across heterogeneous models
            logger.warning(
                "Proposal generation failed ({} / {}): {}", model, gold.gold_id, exc
            )
    return candidates


_JUDGE_SYSTEM = "Eres un evaluador imparcial de propuestas de estudiantes para un sistema de recomendación de directores de TFG/TFM."
_JUDGE_INSTRUCTIONS = (
    "Puntúa de 1 a 5 la propuesta del estudiante según estos criterios combinados:\n"
    "  - Fidelidad: refleja el mismo tema e idea técnica del trabajo original.\n"
    "  - Realismo: parece una consulta natural escrita por un estudiante.\n"
    "  - Limpieza: NO copia el título literal ni filtra el nombre del director.\n"
    "Devuelve un JSON con 'score' (entero 1-5) y 'reason' (breve)."
)


def judge_and_select(
    candidates: list[ProposalCandidate],
    gold: GoldRecord,
    judges: list[str] = GENERATOR_MODELS,
) -> SilverRecord | None:
    """Score candidate proposals with the LLM panel and keep the best.

    Each judge rates each candidate 1-5; scores are averaged.
    The top candidate wins, ties broken deterministically by generator order then model name.
    """
    if not candidates:
        return None

    schema = _JudgeSchema.model_json_schema()
    for cand in candidates:
        scores: list[int] = []
        user = (
            f"{_JUDGE_INSTRUCTIONS}\n\n"
            f"--- Trabajo original ---\n{gold.description}\n\n"
            f"--- Propuesta del estudiante ---\n{cand.proposal}"
        )
        for judge in judges:
            try:
                data = _chat_json(judge, _JUDGE_SYSTEM, user, schema, temperature=0.0)
                scores.append(int(_JudgeSchema.model_validate(data).score))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Judging failed ({} / {}): {}", judge, gold.gold_id, exc)
        cand.judge_score = sum(scores) / len(scores) if scores else 0.0

    # Deterministic selection: highest mean score, then generator order, then name.
    order = {m: i for i, m in enumerate(GENERATOR_MODELS)}
    best = max(
        candidates,
        key=lambda c: (c.judge_score or 0.0, -order.get(c.model, 99), c.model),
    )
    return SilverRecord(
        gold_id=gold.gold_id,
        proposal=best.proposal,
        source_model=best.model,
        judge_score=best.judge_score or 0.0,
        candidates=candidates,
    )


def build_silver_standard(
    gold_records: list[GoldRecord],
    output_path: Path = SILVER_PATH,
    limit: int | None = None,
) -> list[SilverRecord]:
    """Generate the silver standard from the gold standard and persist it.

    Resumable: records already present in `output_path` are skipped.
    The file is rewritten atomically after every new record so progress is never lost.
    """
    existing = (
        {s.gold_id: s for s in load_silver_standard(output_path)}
        if output_path.exists()
        else {}
    )
    pending = [g for g in gold_records if g.gold_id not in existing]
    if limit is not None:
        pending = pending[:limit]

    logger.info(
        "Silver build: {} existing, {} to generate (models={})",
        len(existing),
        len(pending),
        GENERATOR_MODELS,
    )

    silver = list(existing.values())
    for i, gold in enumerate(pending, 1):
        candidates = generate_proposal_candidates(gold)
        record = judge_and_select(candidates, gold)
        if record is None:
            logger.error("No candidates for {} — skipping", gold.gold_id)
            continue
        silver.append(record)
        existing[gold.gold_id] = record
        _atomic_write_json(output_path, [s.model_dump() for s in silver])
        logger.info(
            "[{}/{}] {} -> {} (score={:.2f})",
            i,
            len(pending),
            gold.gold_id,
            record.source_model,
            record.judge_score,
        )
    return silver


def load_silver_standard(path: Path = SILVER_PATH) -> list[SilverRecord]:
    """Load a previously generated silver standard."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SilverRecord.model_validate(s) for s in raw]


# 3. Recommendation run
def run_recommendations(
    silver: SilverRecord,
    gold: GoldRecord,
    mode: str,
    db,
    limit: int = max(K_VALUES),
) -> RecommendationResult:
    """Query UPM Match and capture the ranked directors plus per-query timings.

    Dispatches by mode against app.search (same code paths as the web /api/search
    route), except the control mode ``lexical_direct``, which bypasses the app and
    the LLM entirely: it sends the *original* thesis text (the gold description)
    straight to the sparse BM42 keyword index in Qdrant. A single sparse query, so
    no RRF fusion is needed; it is the most basic lexical baseline and isolates the
    contribution of the LLM in query interpretation.

    Wall-clock time is split into the generative-LLM portion (``llm_seconds`` —
    keyword extraction / query enrichment) and the retrieval portion
    (``search_seconds`` — embedding + Qdrant query). Dense embedding counts as
    retrieval; only the agent generation counts as LLM, so the split reports the
    cost of putting the LLM in the loop.

    Any failure is captured in ``error`` and yields an empty ranking (a miss) so a
    single flaky query never aborts the whole evaluation.
    """
    from app.agent import enrich_query, extract_keywords
    from app.search import (
        COLLECTION_NAME,
        SPARSE_FIELD,
        search_pdis,
        search_pdis_by_summary,
        search_pdis_hybrid,
    )

    proposal = silver.proposal
    llm_seconds = 0.0
    search_seconds = 0.0
    try:
        if mode == "lexical_direct":
            # Control: original thesis text -> sparse index, straight through the
            # Qdrant backend (no app layer, no LLM). researcher_ids come from the
            # payload of each hit; a single sparse query needs no RRF fusion.
            t0 = time.perf_counter()
            raw = db.search_by_keywords(
                collection=COLLECTION_NAME,
                sparse_field=SPARSE_FIELD,
                keywords=[gold.description],
                limit=limit,
            )
            search_seconds += time.perf_counter() - t0
            ranked = [
                rid
                for sr in raw
                if (rid := str((sr.payload or {}).get("researcher_id", "")))
            ]
            return RecommendationResult(
                gold_id=gold.gold_id,
                mode=mode,
                ranked_researcher_ids=ranked,
                true_director=gold.director,
                llm_seconds=0.0,
                search_seconds=round(search_seconds, 4),
                total_seconds=round(search_seconds, 4),
            )

        if mode == "keywords":
            t0 = time.perf_counter()
            kw = extract_keywords(proposal)
            llm_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            results = search_pdis(db, kw.general, kw.specific, limit=limit)
            search_seconds += time.perf_counter() - t0
        elif mode == "summary_direct":
            t0 = time.perf_counter()
            results = search_pdis_by_summary(db, proposal, limit=limit)
            search_seconds += time.perf_counter() - t0
        elif mode == "summary_enriched":
            t0 = time.perf_counter()
            enriched = enrich_query(proposal)
            llm_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            results = search_pdis_by_summary(db, enriched, limit=limit)
            search_seconds += time.perf_counter() - t0
        elif mode == "hybrid":
            t0 = time.perf_counter()
            kw = extract_keywords(proposal)
            try:
                dense_text = enrich_query(proposal)
            except Exception:  # noqa: BLE001 — fall back to raw idea, as the app does
                dense_text = proposal
            llm_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            results = search_pdis_hybrid(
                db, kw.general, kw.specific, dense_text, limit=limit
            )
            search_seconds += time.perf_counter() - t0
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Recommendation failed ({} / {}): {}", mode, gold.gold_id, exc)
        return RecommendationResult(
            gold_id=gold.gold_id, mode=mode, true_director=gold.director, error=str(exc)
        )

    return RecommendationResult(
        gold_id=gold.gold_id,
        mode=mode,
        ranked_researcher_ids=[r.researcher_id for r in results],
        true_director=gold.director,
        llm_seconds=round(llm_seconds, 4),
        search_seconds=round(search_seconds, 4),
        total_seconds=round(llm_seconds + search_seconds, 4),
    )


# 4. Metrics
# Every query has exactly one relevant item (the true director), which simplifies
# the single-relevant forms of the standard IR metrics.
def _rank_of(ranked: list[str], relevant: str) -> int | None:
    """1-based rank of `relevant` in `ranked`, or None if absent."""
    try:
        return ranked.index(relevant) + 1
    except ValueError:
        return None


def hit_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """1.0 if `relevant` appears in the top-k, else 0.0."""
    rank = _rank_of(ranked, relevant)
    return 1.0 if rank is not None and rank <= k else 0.0


def precision_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Precision@k for a single relevant item: hit/k."""
    return hit_at_k(ranked, relevant, k) / k


def recall_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Recall@k — with one relevant item this equals hit@k."""
    return hit_at_k(ranked, relevant, k)


def reciprocal_rank(ranked: list[str], relevant: str) -> float:
    """1/rank of the relevant item (0 if absent) — for MRR."""
    rank = _rank_of(ranked, relevant)
    return 1.0 / rank if rank is not None else 0.0


def average_precision(ranked: list[str], relevant: str) -> float:
    """Average precision for a single relevant item — equals 1/rank (for MAP)."""
    return reciprocal_rank(ranked, relevant)


def ndcg_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Normalised DCG@k. With one relevant item IDCG = 1, so NDCG = 1/log2(rank+1)."""
    rank = _rank_of(ranked, relevant)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of `values` (pct in 0..100). 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


# Timing keys reported per mode (seconds). Errors and untimed results are excluded.
_TIMING_KEYS = [
    "n_timed",
    "total_mean_s",
    "total_median_s",
    "total_p90_s",
    "llm_mean_s",
    "search_mean_s",
]


def _aggregate_timing(results: list[RecommendationResult]) -> dict[str, float]:
    """Aggregate per-query timings over the results that recorded one.

    Results that errored (no timing) or come from a pre-timing cache are skipped,
    so `n_timed` may be smaller than the query count.
    """
    timed = [r for r in results if r.total_seconds is not None]
    if not timed:
        return {}
    totals = [r.total_seconds for r in timed]
    llms = [r.llm_seconds or 0.0 for r in timed]
    searches = [r.search_seconds or 0.0 for r in timed]
    return {
        "n_timed": float(len(timed)),
        "total_mean_s": statistics.fmean(totals),
        "total_median_s": statistics.median(totals),
        "total_p90_s": _percentile(totals, 90),
        "llm_mean_s": statistics.fmean(llms),
        "search_mean_s": statistics.fmean(searches),
    }


def compute_metrics(
    results: list[RecommendationResult],
    segment: str = "all",
) -> MetricReport:
    """Aggregate per-query metrics and timings into a single report for a mode/segment."""
    mode = results[0].mode if results else ""
    n = len(results)
    metrics: dict[str, float] = {}
    if n == 0:
        return MetricReport(mode=mode, segment=segment, n=0, metrics=metrics)

    metrics["mrr"] = (
        sum(reciprocal_rank(r.ranked_researcher_ids, r.true_director) for r in results)
        / n
    )
    metrics["map"] = (
        sum(
            average_precision(r.ranked_researcher_ids, r.true_director) for r in results
        )
        / n
    )

    for k in K_VALUES:
        hit = (
            sum(hit_at_k(r.ranked_researcher_ids, r.true_director, k) for r in results)
            / n
        )
        prec = (
            sum(
                precision_at_k(r.ranked_researcher_ids, r.true_director, k)
                for r in results
            )
            / n
        )
        rec = (
            sum(
                recall_at_k(r.ranked_researcher_ids, r.true_director, k)
                for r in results
            )
            / n
        )
        ndcg = (
            sum(ndcg_at_k(r.ranked_researcher_ids, r.true_director, k) for r in results)
            / n
        )
        metrics[f"hit@{k}"] = hit
        metrics[f"precision@{k}"] = prec
        metrics[f"recall@{k}"] = rec
        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"f1@{k}"] = _f1(prec, rec)

    return MetricReport(
        mode=mode,
        segment=segment,
        n=n,
        metrics=metrics,
        timing=_aggregate_timing(results),
    )


# 5. Segmented analysis + reporting
def analyze_by_segment(
    results: list[RecommendationResult],
    gold_by_id: dict[str, GoldRecord],
    segment_key: str,
) -> list[MetricReport]:
    """Slice results by a gold attribute and compute metrics per slice.

    segment_key in {"year", "knowledge_area", "study_type"}.
    """
    groups: dict[str, list[RecommendationResult]] = defaultdict(list)
    for r in results:
        gold = gold_by_id.get(r.gold_id)
        if gold is None:
            continue
        groups[str(getattr(gold, segment_key))].append(r)

    reports = [
        compute_metrics(group, segment=f"{segment_key}={value}")
        for value, group in sorted(groups.items())
    ]
    return reports


_HEADLINE = ["mrr", "map", "hit@1", "hit@3", "hit@5", "hit@10", "ndcg@5", "ndcg@10"]


def _markdown_table(
    reports: list[MetricReport], row_label: str, keys: list[str]
) -> str:
    header = f"| {row_label} | n | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [header, sep]
    for rep in reports:
        label = rep.mode if row_label == "mode" else rep.segment
        cells = " | ".join(f"{rep.metrics.get(k, 0.0):.3f}" for k in keys)
        lines.append(f"| {label} | {rep.n} | {cells} |")
    return "\n".join(lines)


def _timing_table(reports: list[MetricReport]) -> str:
    """Per-mode table of query timings (seconds), for the LLM-cost discussion."""
    header = "| mode | " + " | ".join(_TIMING_KEYS) + " |"
    sep = "|" + "---|" * (len(_TIMING_KEYS) + 1)
    lines = [header, sep]
    for rep in reports:
        cells = " | ".join(
            f"{rep.timing.get(k, 0.0):.0f}"
            if k == "n_timed"
            else f"{rep.timing.get(k, 0.0):.3f}"
            for k in _TIMING_KEYS
        )
        lines.append(f"| {rep.mode} | {cells} |")
    return "\n".join(lines)


def _legend() -> str:
    """One-line mapping of internal mode keys to the memoria strategy labels."""
    return "Modos: " + " · ".join(
        f"`{m}` = {MODE_LABELS.get(m, m)}" for m in SEARCH_MODES
    )


def report(
    per_mode: list[MetricReport],
    per_segment: dict[str, dict[str, list[MetricReport]]] | None = None,
    out_dir: Path | None = None,
) -> None:
    """Render the comparison tables (per mode + per segment) and optionally save them."""
    blocks: list[str] = []
    blocks.append("## Comparativa por modo de búsqueda\n")
    blocks.append(_legend() + "\n")
    blocks.append(
        _markdown_table(
            sorted(per_mode, key=lambda r: -r.metrics.get("mrr", 0.0)),
            "mode",
            _HEADLINE,
        )
    )

    blocks.append("\n## Tiempo por consulta (segundos)\n")
    blocks.append(
        _timing_table(sorted(per_mode, key=lambda r: r.timing.get("total_mean_s", 0.0)))
    )

    if per_segment:
        for mode, seg_map in per_segment.items():
            for seg_key, reports in seg_map.items():
                blocks.append(f"\n## {mode} — desglose por {seg_key}\n")
                blocks.append(
                    _markdown_table(
                        reports,
                        "segment",
                        ["mrr", "hit@1", "hit@5", "hit@10", "ndcg@10"],
                    )
                )

    text = "\n".join(blocks)
    print("\n" + text + "\n")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        _atomic_write_json(
            out_dir / "metrics.json",
            {
                "per_mode": [r.model_dump() for r in per_mode],
                "per_segment": {
                    mode: {
                        seg: [r.model_dump() for r in reps]
                        for seg, reps in seg_map.items()
                    }
                    for mode, seg_map in (per_segment or {}).items()
                },
            },
        )
        logger.info("Report written to {}", out_dir)


# I/O helpers
def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically (temp file + os.replace) so partial writes never corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _recs_cache_path(mode: str, source: str) -> Path:
    """Cache path for one mode, namespaced by query source.

    Gold and silver runs are kept apart (``cache/gold/`` vs ``cache/silver/``) so
    re-running one never clobbers the other's cached rankings.
    """
    return CACHE_DIR / source / f"recommendations_{mode}.json"


def _load_recs_cache(mode: str, source: str) -> dict[str, RecommendationResult]:
    path = _recs_cache_path(mode, source)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {r["gold_id"]: RecommendationResult.model_validate(r) for r in raw}


def _save_recs_cache(
    mode: str, cache: dict[str, RecommendationResult], source: str
) -> None:
    _atomic_write_json(
        _recs_cache_path(mode, source), [r.model_dump() for r in cache.values()]
    )


def publish_to_minio(report_dir: Path, source: str, modes: list[str]) -> None:
    """Upload the report + raw per-mode rankings to MinIO for off-server retrieval.

    Writes two copies per artefact: a timestamped one under
    ``<source>/<timestamp>/`` (immutable history) and an overwriting
    ``<source>/latest/`` pointer (so consumers can always grab the newest run).
    The bucket (``MINIO_EVAL_BUCKET``, default ``tfm-evaluation``) is created on
    first use. Failures are logged but never abort the evaluation, whose results
    are already safe on disk.
    """
    try:
        from dbs import minio
    except Exception as exc:  # noqa: BLE001 — MinIO is optional for local runs
        logger.warning("MinIO not available, skipping publish: {}", exc)
        return

    bucket = os.environ.get("MINIO_EVAL_BUCKET", "tfm-evaluation")
    base = f"{source}/{report_dir.name}"
    latest = f"{source}/latest"

    uploads: list[tuple[Path, str]] = []
    for fname in ("report.md", "metrics.json"):
        fp = report_dir / fname
        if fp.exists():
            uploads.append((fp, fname))
    for mode in modes:
        cp = _recs_cache_path(mode, source)
        if cp.exists():
            uploads.append((cp, f"cache/{cp.name}"))

    ok = 0
    for fp, rel in uploads:
        if minio.upload_file(bucket, fp, f"{base}/{rel}") and minio.upload_file(
            bucket, fp, f"{latest}/{rel}"
        ):
            ok += 1
    logger.info(
        "Published {}/{} evaluation artefacts to MinIO bucket {!r} under {}/ (and {}/)",
        ok,
        len(uploads),
        bucket,
        base,
        latest,
    )


# Command implementations
def cmd_build_silver(args: argparse.Namespace) -> int:
    gold = load_gold_standard()
    build_silver_standard(gold, limit=args.limit)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from dbs.vector import get_vector_db

    gold = load_gold_standard()
    gold_by_id = {g.gold_id: g for g in gold}

    if args.source == "gold":
        # Evaluate directly over the real thesis descriptions as the query, with no
        # silver standard. Lets the benchmark run before the silver set exists; every
        # mode uses gold.description as its input text (already what the
        # lexical_direct control uses), so the pipeline is otherwise identical.
        silver = [
            SilverRecord(
                gold_id=g.gold_id,
                proposal=g.description,
                source_model="gold",
                judge_score=0.0,
            )
            for g in gold
        ]
        logger.info("Source=gold: querying with {} thesis descriptions.", len(silver))
    else:
        silver = load_silver_standard()
        if not silver:
            logger.error(
                "No silver standard found at {}. Run 'build-silver' first.", SILVER_PATH
            )
            return 1

    db = get_vector_db()
    keywords_ids, summaries_ids, common = common_indexed_universe(db)
    logger.info(
        "Index universe: keywords={}, summaries={}, common={}",
        len(keywords_ids),
        len(summaries_ids),
        len(common),
    )
    warn_unresolved_directors(gold, common)

    # Restrict to the common universe FIRST (before --limit) so every mode ranks the
    # true director over the same population and --limit yields evaluable queries.
    if not args.no_universe_filter:
        before = len(silver)
        silver = [
            s
            for s in silver
            if (g := gold_by_id.get(s.gold_id)) is not None and g.director in common
        ]
        logger.info(
            "Common-universe filter: {} -> {} queries ({} dropped).",
            before,
            len(silver),
            before - len(silver),
        )

    if args.limit is not None:
        silver = silver[: args.limit]

    logger.info(
        "Evaluating {} queries x {} mode(s): {}",
        len(silver),
        len(args.modes),
        ", ".join(args.modes),
    )

    per_mode: list[MetricReport] = []
    per_segment: dict[str, dict[str, list[MetricReport]]] = {}

    for mode in args.modes:
        cache = {} if args.no_cache else _load_recs_cache(mode, args.source)
        results: list[RecommendationResult] = []
        for i, s in enumerate(silver, 1):
            gold_rec = gold_by_id.get(s.gold_id)
            if gold_rec is None:
                continue
            cached = cache.get(s.gold_id)
            if cached is not None and cached.error is None:
                results.append(cached)
                continue
            res = run_recommendations(s, gold_rec, mode, db)
            cache[s.gold_id] = res
            results.append(res)
            if i % CACHE_FLUSH_EVERY == 0:
                _save_recs_cache(mode, cache, args.source)
                logger.info("[{}] {}/{} done", mode, i, len(silver))
        _save_recs_cache(mode, cache, args.source)

        per_mode.append(compute_metrics(results, segment="all"))
        per_segment[mode] = {
            key: analyze_by_segment(results, gold_by_id, key)
            for key in ("study_type", "knowledge_area", "year")
        }

    out_dir = REPORTS_DIR / args.source / datetime.now().strftime("%Y%m%d_%H%M%S")
    report(per_mode, per_segment, out_dir=out_dir)
    if args.publish_minio:
        publish_to_minio(out_dir, args.source, args.modes)
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """Validate the metric math offline (no Qdrant / Ollama needed)."""
    ranked = ["pdi_1", "pdi_2", "pdi_3", "pdi_4", "pdi_5"]

    assert hit_at_k(ranked, "pdi_1", 1) == 1.0
    assert hit_at_k(ranked, "pdi_3", 1) == 0.0
    assert hit_at_k(ranked, "pdi_3", 3) == 1.0
    assert hit_at_k(ranked, "pdi_9", 25) == 0.0
    assert precision_at_k(ranked, "pdi_3", 5) == 1.0 / 5
    assert recall_at_k(ranked, "pdi_3", 3) == 1.0
    assert reciprocal_rank(ranked, "pdi_2") == 0.5
    assert reciprocal_rank(ranked, "pdi_9") == 0.0
    assert average_precision(ranked, "pdi_2") == 0.5
    assert abs(ndcg_at_k(ranked, "pdi_1", 5) - 1.0) < 1e-9
    assert abs(ndcg_at_k(ranked, "pdi_3", 5) - 1.0 / math.log2(4)) < 1e-9
    assert ndcg_at_k(ranked, "pdi_3", 2) == 0.0

    rec = [
        RecommendationResult(
            gold_id="a",
            mode="keywords",
            ranked_researcher_ids=ranked,
            true_director="pdi_1",
        ),
        RecommendationResult(
            gold_id="b",
            mode="keywords",
            ranked_researcher_ids=ranked,
            true_director="pdi_5",
        ),
        RecommendationResult(
            gold_id="c",
            mode="keywords",
            ranked_researcher_ids=ranked,
            true_director="pdi_9",
        ),
    ]
    m = compute_metrics(rec).metrics
    assert abs(m["hit@5"] - 2 / 3) < 1e-9, m["hit@5"]
    assert abs(m["hit@1"] - 1 / 3) < 1e-9, m["hit@1"]
    assert abs(m["mrr"] - (1.0 + 0.2 + 0.0) / 3) < 1e-9, m["mrr"]

    # Percentile helper.
    assert _percentile([], 90) == 0.0
    assert _percentile([42], 90) == 42
    assert abs(_percentile([1, 2, 3, 4], 50) - 2.5) < 1e-9

    # Timing aggregation: only timed results count; mean/median/split are correct.
    timed = [
        RecommendationResult(
            gold_id="a",
            mode="m",
            ranked_researcher_ids=ranked,
            true_director="pdi_1",
            llm_seconds=1.0,
            search_seconds=0.5,
            total_seconds=1.5,
        ),
        RecommendationResult(
            gold_id="b",
            mode="m",
            ranked_researcher_ids=ranked,
            true_director="pdi_2",
            llm_seconds=3.0,
            search_seconds=0.5,
            total_seconds=3.5,
        ),
    ]
    t = compute_metrics(timed).timing
    assert t["n_timed"] == 2.0, t
    assert abs(t["total_mean_s"] - 2.5) < 1e-9, t
    assert abs(t["total_median_s"] - 2.5) < 1e-9, t
    assert abs(t["llm_mean_s"] - 2.0) < 1e-9, t
    assert abs(t["search_mean_s"] - 0.5) < 1e-9, t
    # Untimed / errored results contribute no timing at all.
    assert _aggregate_timing([RecommendationResult(gold_id="c", mode="m")]) == {}

    print("selftest OK — all metric and timing assertions passed.")
    return 0


# Entry point
def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        description="UPM Match recommendation quality evaluation harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_silver = sub.add_parser(
        "build-silver",
        help="Generate the silver standard from the gold standard (3 LLMs).",
    )
    p_silver.add_argument(
        "--limit", type=int, default=None, help="Only generate this many new records."
    )
    p_silver.set_defaults(func=cmd_build_silver)

    p_eval = sub.add_parser(
        "evaluate", help="Run recommendations for every mode and compute metrics."
    )
    p_eval.add_argument(
        "--source",
        choices=("gold", "silver"),
        default="silver",
        help="Query-text source: 'gold' uses the real thesis descriptions directly "
        "(no silver standard needed); 'silver' uses the generated student-style "
        "proposals (run 'build-silver' first). Cache + report are namespaced by "
        "source so the two never collide.",
    )
    p_eval.add_argument(
        "--modes",
        nargs="+",
        default=SEARCH_MODES,
        choices=SEARCH_MODES,
        help="Search modes to evaluate.",
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate this many records.",
    )
    p_eval.add_argument(
        "--publish-minio",
        action="store_true",
        help="After reporting, upload report.md + metrics.json + raw rankings to "
        "MinIO (bucket MINIO_EVAL_BUCKET, default 'tfm-evaluation') for "
        "off-server retrieval.",
    )
    p_eval.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached recommendation runs and recompute.",
    )
    p_eval.add_argument(
        "--no-universe-filter",
        action="store_true",
        help="Do not restrict queries to supervisors present in both indices "
        "(by default the comparison is limited to that common universe).",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_self = sub.add_parser(
        "selftest", help="Validate the metric math offline (no services needed)."
    )
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
