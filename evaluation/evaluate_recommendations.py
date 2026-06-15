"""
UPM Match — Recommendation quality evaluation harness.

==============================================================================
METHODOLOGY
==============================================================================

1. GOLD STANDARD  (hand-made)
   500 real TFG/TFM records curated by hand, split in two files:
     - evaluation/gold_standard_tfg.json  (250 records)
     - evaluation/gold_standard_tfm.json  (250 records)
   Each record holds:
     - id             : per-file integer index (the stable key is `_project_id`)
     - description    : title + abstract + keywords of the thesis work
     - year           : academic year of the work
     - director       : the actual supervisor, as a researcher_id (`pdi_<n>`)
     - study_type     : "TFG" | "TFM"
     - knowledge_area : the school (ETSII / ETSIT / ETSISI / ETSIST)
     - _project_id    : globally-unique thesis id (`tft_<n>`) — used as gold_id
     - _title/_keywords : debug fields

2. SILVER STANDARD  (synthetic, evaluation/silver_standard.json)
   For each gold record, 3 local LLMs each rewrite the thesis `description` into
   a *student-style proposal* (the free text a student would type into UPM Match
   when looking for a supervisor). The 3 candidates are then scored by the same
   3 LLMs (LLM-as-judge); the highest-rated proposal becomes the silver record.
   Generation is reproducible (fixed seed) and resumable (cached to disk).

3. RECOMMENDATION RUN
   Each silver proposal is fed to UPM Match in every search mode (keywords /
   summary_direct / summary_enriched / hybrid) to obtain a ranked list of
   recommended supervisors. The ranked researcher_ids are compared against the
   gold record's true director.

4. METRICS + ANALYSIS
   Single-relevant ranking metrics (Hit@k, Precision@k, Recall@k, MRR, MAP,
   NDCG@k, F1@k) aggregated per mode and broken down by year, knowledge area and
   study type, to determine which search approach performs best and where it fails.

==============================================================================
USAGE
==============================================================================
    # 1. Generate the silver standard from the gold standard (3 LLMs)
    uv run --module evaluation.evaluate_recommendations build-silver
    uv run --module evaluation.evaluate_recommendations build-silver --limit 10

    # 2. Evaluate every mode, compute metrics + segment breakdowns, write report
    uv run --module evaluation.evaluate_recommendations evaluate
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
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ollama as _ollama
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

# --- Configuration ----------------------------------------------------------

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

# Search modes exposed by the web app (see app/routes.py).
SEARCH_MODES: list[str] = ["keywords", "summary_direct", "summary_enriched", "hybrid"]

# Cut-offs at which ranking metrics are reported.
K_VALUES: list[int] = [1, 3, 5, 10, 25]

# Fixed seed so the silver standard is reproducible across runs.
SEED = 42

_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")


# --- Data models -------------------------------------------------------------


class GoldRecord(BaseModel):
    """One hand-curated TFG/TFM ground-truth record."""

    id: int
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
    """Ranked recommendations produced for one silver proposal in one mode."""

    gold_id: str
    mode: str
    ranked_researcher_ids: list[str] = Field(default_factory=list)
    true_director: str = ""
    error: str | None = None


class MetricReport(BaseModel):
    """Aggregated metrics for one search mode (optionally one segment)."""

    mode: str
    segment: str = "all"  # e.g. "all", "year=2023", "study_type=TFM"
    n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)


# --- LLM plumbing ------------------------------------------------------------

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
    client = _ollama.Client(host=_OLLAMA_HOST)
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


# --- 1. Gold standard --------------------------------------------------------


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


def indexed_researcher_ids(db) -> set[str]:
    """Scroll Qdrant and return the set of indexed researcher_ids (for validation)."""
    from app.search import COLLECTION_NAME

    ids: set[str] = set()
    offset = None
    while True:
        points, offset = db._client.scroll(
            collection_name=COLLECTION_NAME,
            limit=512,
            offset=offset,
            with_payload=["researcher_id"],
            with_vectors=False,
        )
        for p in points:
            rid = (p.payload or {}).get("researcher_id")
            if rid:
                ids.add(str(rid))
        if offset is None:
            break
    return ids


def warn_unresolved_directors(gold: list[GoldRecord], db) -> None:
    """Warn about gold directors that are not present in the Qdrant index."""
    indexed = indexed_researcher_ids(db)
    missing = sorted({g.director for g in gold if g.director not in indexed})
    if missing:
        logger.warning(
            "{} gold directors are NOT indexed in Qdrant (recommendations can never "
            "hit them): {}",
            len(missing),
            missing,
        )
    else:
        logger.info(
            "All {} distinct gold directors are indexed in Qdrant.",
            len({g.director for g in gold}),
        )


# --- 2. Silver standard ------------------------------------------------------

_PROPOSAL_SYSTEM = (
    "Eres un estudiante de la Universidad Politécnica de Madrid que busca director "
    "para su Trabajo de Fin de Grado o de Máster. Escribes una breve propuesta con "
    "tu idea de proyecto para que el sistema te recomiende un director."
)
_PROPOSAL_INSTRUCTIONS = (
    "A partir de la descripción de un trabajo ya realizado, redacta la propuesta que "
    "habría escrito el estudiante ANTES de empezarlo. Reglas:\n"
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


_JUDGE_SYSTEM = (
    "Eres un evaluador imparcial de propuestas de estudiantes para un sistema de "
    "recomendación de directores de TFG/TFM."
)
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

    Each judge rates each candidate 1-5; scores are averaged. The top candidate
    wins, ties broken deterministically by generator order then model name.
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

    Resumable: records already present in `output_path` are skipped. The file is
    rewritten atomically after every new record so progress is never lost.
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


# --- 3. Recommendation run ---------------------------------------------------


def run_recommendations(
    silver: SilverRecord,
    gold: GoldRecord,
    mode: str,
    db,
    limit: int = max(K_VALUES),
) -> RecommendationResult:
    """Query UPM Match with a silver proposal and capture the ranked directors.

    Dispatches by mode against app.search (same code paths as the web /api/search
    route). Any failure is captured in `error` and yields an empty ranking (a miss)
    so a single flaky query never aborts the whole evaluation.
    """
    from app.agent import enrich_query, extract_keywords
    from app.search import (
        search_pdis,
        search_pdis_by_summary,
        search_pdis_hybrid,
    )

    proposal = silver.proposal
    try:
        if mode == "keywords":
            kw = extract_keywords(proposal)
            results = search_pdis(db, kw.general, kw.specific, limit=limit)
        elif mode == "summary_direct":
            results = search_pdis_by_summary(db, proposal, limit=limit)
        elif mode == "summary_enriched":
            enriched = enrich_query(proposal)
            results = search_pdis_by_summary(db, enriched, limit=limit)
        elif mode == "hybrid":
            kw = extract_keywords(proposal)
            try:
                dense_text = enrich_query(proposal)
            except Exception:  # noqa: BLE001 — fall back to raw idea, as the app does
                dense_text = proposal
            results = search_pdis_hybrid(
                db, kw.general, kw.specific, dense_text, limit=limit
            )
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
    )


# --- 4. Metrics --------------------------------------------------------------
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


def compute_metrics(
    results: list[RecommendationResult],
    segment: str = "all",
) -> MetricReport:
    """Aggregate per-query metrics into a single report for a mode/segment."""
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

    return MetricReport(mode=mode, segment=segment, n=n, metrics=metrics)


# --- 5. Segmented analysis + reporting --------------------------------------


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


def report(
    per_mode: list[MetricReport],
    per_segment: dict[str, dict[str, list[MetricReport]]] | None = None,
    out_dir: Path | None = None,
) -> None:
    """Render the comparison tables (per mode + per segment) and optionally save them."""
    blocks: list[str] = []
    blocks.append("## Comparativa por modo de búsqueda\n")
    blocks.append(
        _markdown_table(
            sorted(per_mode, key=lambda r: -r.metrics.get("mrr", 0.0)),
            "mode",
            _HEADLINE,
        )
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


# --- I/O helpers -------------------------------------------------------------


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically (temp file + os.replace) so partial writes never corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _recs_cache_path(mode: str) -> Path:
    return CACHE_DIR / f"recommendations_{mode}.json"


def _load_recs_cache(mode: str) -> dict[str, RecommendationResult]:
    path = _recs_cache_path(mode)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {r["gold_id"]: RecommendationResult.model_validate(r) for r in raw}


def _save_recs_cache(mode: str, cache: dict[str, RecommendationResult]) -> None:
    _atomic_write_json(_recs_cache_path(mode), [r.model_dump() for r in cache.values()])


# --- Command implementations -------------------------------------------------


def cmd_build_silver(args: argparse.Namespace) -> int:
    gold = load_gold_standard()
    build_silver_standard(gold, limit=args.limit)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from dbs.vector import get_vector_db

    gold = load_gold_standard()
    gold_by_id = {g.gold_id: g for g in gold}
    silver = load_silver_standard()
    if not silver:
        logger.error(
            "No silver standard found at {}. Run 'build-silver' first.", SILVER_PATH
        )
        return 1

    if args.limit is not None:
        silver = silver[: args.limit]

    db = get_vector_db()
    warn_unresolved_directors(gold, db)

    per_mode: list[MetricReport] = []
    per_segment: dict[str, dict[str, list[MetricReport]]] = {}

    for mode in args.modes:
        cache = {} if args.no_cache else _load_recs_cache(mode)
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
            if i % 25 == 0:
                _save_recs_cache(mode, cache)
                logger.info("[{}] {}/{} done", mode, i, len(silver))
        _save_recs_cache(mode, cache)

        per_mode.append(compute_metrics(results, segment="all"))
        per_segment[mode] = {
            key: analyze_by_segment(results, gold_by_id, key)
            for key in ("study_type", "knowledge_area", "year")
        }

    out_dir = REPORTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    report(per_mode, per_segment, out_dir=out_dir)
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
    print("selftest OK — all metric assertions passed.")
    return 0


# --- Entry point -------------------------------------------------------------


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
        help="Only evaluate this many silver records.",
    )
    p_eval.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached recommendation runs and recompute.",
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
