"""
UPM Match — Recommendation quality evaluation harness (SKELETON).

This is a scaffold only: functions raise NotImplementedError and carry TODOs.
It lays out the methodology so the implementation can be filled in incrementally.

==============================================================================
METHODOLOGY
==============================================================================

1. GOLD STANDARD  (hand-made, evaluation/gold_standard.json)
   100 real TFG/TFM records curated by hand. Each record holds:
     - description : the abstract / description of the thesis work
     - year        : academic year of the work
     - director    : the actual supervisor (ideally a researcher_id known to
                     the system, so it can be matched against recommendations)
     - study_type  : "TFG" | "TFM"            (needed for the per-type analysis)
     - knowledge_area : UNESCO / department label (needed for per-area analysis)
   See gold_standard.example.json for the exact schema.

2. SILVER STANDARD  (synthetic, evaluation/silver_standard.json)
   For each gold record, 3 LLMs each rewrite the thesis `description` into a *student-style proposal* (the kind of free text a student would type into) UPM Match when looking for a supervisor.
   The 3 candidate proposals are then scored by the same 3 LLMs (LLM-as-judge); the highest-rated proposal is kept as the silver record.
   Each silver record keeps a back-reference to its gold record so the true director / year / area / type are available at scoring.

3. RECOMMENDATION RUN
   Each silver proposal is fed to UPM Match in every search mode (keywords / summary_direct / summary_enriched) to obtain a ranked list of recommended supervisors.
   The ranked researcher_ids are compared against the gold record's true director.

4. METRICS + ANALYSIS
   Compute classification and ranking metrics (see compute_metrics) and break them down by year, knowledge area and study type, to determine which search approach performs best and where each one fails.

==============================================================================
TODO CHECKLIST
==============================================================================
  [ ] Hand-curate evaluation/gold_standard.json (100 records).
  [ ] Resolve each gold `director` to a researcher_id present in Qdrant/Memgraph.
  [ ] Implement the 3-LLM proposal generation (generate_proposal_candidates).
  [ ] Implement the 3-LLM judging + selection (judge_and_select).
  [ ] Persist the silver standard and make generation reproducible (seed/cache).
  [ ] Implement run_recommendations against app.search for all 3 modes.
  [ ] Implement the metric functions (precision@k, recall@k, MRR, MAP, NDCG, hit@k).
  [ ] Implement segmented analysis (by year / area / study_type).
  [ ] Implement reporting (per-mode comparison table + per-segment breakdowns).
  [ ] Decide how to handle directors with no indexed profile (exclude vs penalise).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, Field

# --- Reference to the system under test -------------------------------------
# from app.search import search_pdis, search_pdis_by_summary, PDIResult
# from app.agent import extract_keywords, enrich_query
# from dbs.vector import QdrantBackend

# --- Configuration ----------------------------------------------------------

GOLD_PATH = Path("evaluation/gold_standard.json")
SILVER_PATH = Path("evaluation/silver_standard.json")

# The three LLMs used both as proposal generators and as judges.
# TODO: confirm model ids / providers (Ollama local vs API).
GENERATOR_MODELS: list[str] = [
    # "qwen2.5:7b-instruct",
    # "llama3.1:8b-instruct",
    # "mistral-nemo:12b",
]

# Search modes exposed by the web app (see app/routes.py).
SEARCH_MODES: list[str] = ["keywords", "summary_direct", "summary_enriched"]

# Cut-offs at which ranking metrics are reported.
K_VALUES: list[int] = [1, 3, 5, 10, 25]


# --- Data models -------------------------------------------------------------


class GoldRecord(BaseModel):
    """One hand-curated TFG/TFM ground-truth record."""

    id: str
    description: str
    year: int
    director: str  # TODO: normalise to a researcher_id known to the system
    study_type: str  # "TFG" | "TFM"
    knowledge_area: str
    # TODO: add any extra fields used for slicing (department, language, ...).


class ProposalCandidate(BaseModel):
    """A student-style proposal generated from a gold description by one LLM."""

    gold_id: str
    model: str
    proposal: str
    judge_score: float | None = None  # filled in during judging


class SilverRecord(BaseModel):
    """The best-rated synthetic proposal for a gold record."""

    gold_id: str
    proposal: str
    source_model: str
    judge_score: float
    # TODO: keep the losing candidates too for error analysis?


class RecommendationResult(BaseModel):
    """Ranked recommendations produced for one silver proposal in one mode."""

    gold_id: str
    mode: str
    ranked_researcher_ids: list[str] = Field(default_factory=list)
    true_director: str = ""
    # TODO: keep scores so threshold-based classification metrics are possible.


class MetricReport(BaseModel):
    """Aggregated metrics for one search mode (optionally one segment)."""

    mode: str
    segment: str = "all"  # e.g. "all", "year=2023", "study_type=TFM"
    n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)


# --- 1. Gold standard --------------------------------------------------------


def load_gold_standard(path: Path = GOLD_PATH) -> list[GoldRecord]:
    """Load and validate the hand-made gold standard JSON.

    TODO:
      - Read JSON, validate the schema (consider a Pydantic model).
      - Assert exactly the expected fields and non-empty descriptions.
      - Warn on directors that cannot be resolved to an indexed researcher_id.
    """
    raise NotImplementedError


# --- 2. Silver standard ------------------------------------------------------


def generate_proposal_candidates(
    gold: GoldRecord,
    models: list[str] = GENERATOR_MODELS,
) -> list[ProposalCandidate]:
    """Rewrite a thesis description into student-style proposals (one per LLM).

    TODO:
      - Prompt each model to produce a realistic first-person student proposal (interests, goals) WITHOUT leaking the director name or paper titles.
      - Keep the prompt identical across models for fairness.
      - Cache raw outputs to disk for reproducibility.
    """
    raise NotImplementedError


def judge_and_select(
    candidates: list[ProposalCandidate],
    judges: list[str] = GENERATOR_MODELS,
) -> SilverRecord:
    """Score candidate proposals with the LLM panel and keep the best.

    TODO:
      - Each judge rates each candidate (e.g. 1-5) on faithfulness to the gold
        description + realism as a student query. Blind to which model wrote it.
      - Aggregate scores (mean / majority) and select the top candidate.
      - Handle ties deterministically.
    """
    raise NotImplementedError


def build_silver_standard(
    gold_records: list[GoldRecord],
    output_path: Path = SILVER_PATH,
) -> list[SilverRecord]:
    """Generate the full silver standard from the gold standard and persist it.

    TODO:
      - For each gold record: generate_proposal_candidates -> judge_and_select.
      - Skip records already present in output_path (resumable).
      - Save SilverRecords as JSON.
    """
    raise NotImplementedError


def load_silver_standard(path: Path = SILVER_PATH) -> list[SilverRecord]:
    """Load a previously generated silver standard. TODO."""
    raise NotImplementedError


# --- 3. Recommendation run ---------------------------------------------------


def run_recommendations(
    silver: SilverRecord,
    gold: GoldRecord,
    mode: str,
    limit: int = max(K_VALUES),
) -> RecommendationResult:
    """Query UPM Match with a silver proposal and capture the ranked directors.

    TODO:
      - Dispatch by mode against app.search:
          keywords         -> extract_keywords + search_pdis
          summary_direct   -> search_pdis_by_summary(proposal)
          summary_enriched -> enrich_query + search_pdis_by_summary
      - Return ranked researcher_ids + the gold true_director.
    """
    raise NotImplementedError


# --- 4. Metrics --------------------------------------------------------------


def hit_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """1.0 if `relevant` appears in the top-k, else 0.0. TODO."""
    raise NotImplementedError


def precision_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Precision@k for a single relevant item. TODO."""
    raise NotImplementedError


def recall_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Recall@k (with one relevant item, equals hit@k). TODO."""
    raise NotImplementedError


def reciprocal_rank(ranked: list[str], relevant: str) -> float:
    """1/rank of the relevant item (0 if absent) — for MRR. TODO."""
    raise NotImplementedError


def average_precision(ranked: list[str], relevant: str) -> float:
    """Average precision for a single relevant item — for MAP. TODO."""
    raise NotImplementedError


def ndcg_at_k(ranked: list[str], relevant: str, k: int) -> float:
    """Normalised DCG@k. TODO."""
    raise NotImplementedError


def compute_metrics(
    results: list[RecommendationResult],
    segment: str = "all",
) -> MetricReport:
    """Aggregate per-query metrics into a single report for a mode/segment.

    TODO:
      - For each result compute the per-query metrics above at every K in K_VALUES.
      - Aggregate: MRR, MAP, mean Hit@k / Precision@k / Recall@k / NDCG@k.
      - Add classification view: treat "director in top-k" as the positive class
        and report accuracy / precision / recall / F1.
    """
    raise NotImplementedError


# --- 5. Segmented analysis + reporting --------------------------------------


def analyze_by_segment(
    results: list[RecommendationResult],
    gold_by_id: dict[str, GoldRecord],
    segment_key: str,
) -> list[MetricReport]:
    """Slice results by a gold attribute and compute metrics per slice.

    segment_key in {"year", "knowledge_area", "study_type"}.

    TODO:
      - Group results by gold_by_id[result.gold_id].<segment_key>.
      - compute_metrics per group; return one MetricReport per slice.
    """
    raise NotImplementedError


def report(reports: list[MetricReport]) -> None:
    """Render the comparison tables (per mode + per segment).

    TODO:
      - Print/save a mode-comparison table (which approach wins overall).
      - Print/save per-segment breakdowns (year / area / study_type).
      - Optionally export CSV / markdown for the TFM write-up.
    """
    raise NotImplementedError


# --- Entry point -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "build-silver",
        help="Generate the silver standard from the gold standard (3 LLMs).",
    )
    p_eval = sub.add_parser(
        "evaluate",
        help="Run recommendations for every mode and compute metrics.",
    )
    p_eval.add_argument(
        "--modes",
        nargs="+",
        default=SEARCH_MODES,
        help="Search modes to evaluate.",
    )

    args = parser.parse_args()

    # TODO: wire the subcommands:
    #   build-silver -> load_gold_standard -> build_silver_standard
    #   evaluate     -> load_gold/silver -> run_recommendations (all modes)
    #                   -> compute_metrics + analyze_by_segment -> report
    raise NotImplementedError(f"Command {args.command!r} not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
