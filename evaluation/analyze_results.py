"""
UPM Match — statistical analysis of the evaluation runs.

Reads every evaluation run published by `evaluate_recommendations.py --publish-minio` and mirrored back into the repo (`evaluation/reports/<source>/<ts>/...`), and turns the raw per-query rankings into the tables we need, with proper uncertainty quantification.

WHAT IT READS
   For each source ("gold", "silver") it discovers every timestamped run under `<reports>/<source>/<ts>/` (the overwriting `latest/` pointer is ignored so no run is counted twice) and loads the per-query rankings from each run's `cache/recommendations_<mode>.json`. Those files hold one RecommendationResult per query (ranked researcher_ids + the true director + timings), which is all the metric math needs.

THE TWO SOURCES OF UNCERTAINTY (kept separate, on purpose)
   1. Sampling over QUERIES — the real statistical power.
   Every metric is a mean over the ~N queries of the common universe, and every mode is evaluated on the SAME queries, so modes are compared PAIRED, query by query.
   This is where significance comes from and it needs a single run; with several hundred queries the tests have ample power.
   2. Run-to-run variation from the LLM — the 3+ repetitions only re-roll the unseeded query-time LLM (keyword extraction / enrichment).
   It is reported as a robustness band (between-run mean +/- std), NOT as the basis of the significance tests, and the deterministic modes (lexical_direct, summary_direct) collapse to a single value.
   The repetitions are used to DENOISE the per-query estimates (averaging a query's metric across runs), which keeps n = #queries while reducing measurement noise.

TABLES PRODUCED (per source unless noted), as Markdown + standalone LaTeX (booktabs):
   T1  Per-mode headline metrics with 95% bootstrap CIs over queries.
   T2  Pairwise significance between modes on MRR (Wilcoxon signed-rank) and on Hit@5 (exact McNemar), with paired-bootstrap effect-size CIs and Holm + Benjamini-Hochberg corrected p-values across the 10 mode pairs.
   T3  Between-run robustness (mean +/- std of MRR / Hit@5 / NDCG@10 across runs).
   T4  Per-query timings (total/LLM/retrieval seconds), pooled over runs.
   T5  Segment breakdown of MRR by study_type / knowledge_area / year (mode x segment).
   T6  Cross-source comparison (gold vs silver) of MRR / Hit@5 / NDCG@10 per mode — the "flip" discussion.
   T7  Recall ceiling recall@{25,50,100} of the first-stage candidate list (go/no-go for the re-ranking extension).
   T8  Equity of the recommendation across productivity terciles of the true supervisor (popularity-bias guard); needs the productivity map.
   T9  Off-line late-fusion sensitivity sweep over the graph-grounded re-ranker's stored components: quality + equity per weight config, without re-invoking the LLM.

USAGE
   uv run --module evaluation.analyze_results            # all sources, defaults
   uv run --module evaluation.analyze_results --reports-dir evaluation/reports
   uv run --module evaluation.analyze_results --sources silver --bootstrap 10000
   uv run --module evaluation.analyze_results --selftest # offline math checks, no data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# Make the project root importable so `from evaluation.* import` works when this file is
# launched directly (uv run python evaluation/analyze_results.py) as well as via -m.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402

# Reuse the harness so the metric math AND the K cut-offs are a single source of truth.
from evaluation.evaluate_recommendations import (  # noqa: E402
    K_VALUES,
    MODE_LABELS,
    SEARCH_MODES,
    GoldRecord,
    RecommendationResult,
    hit_at_k,
    load_gold_standard,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Metrics reported per mode (mrr == map and recall@k == hit@k for a single relevant item, both kept for fidelity).
# Hit@k cut-offs are the "shallow" (presentation-range) values of K_VALUES.
HEADLINE_METRICS = (
    ["mrr", "map"] + [f"hit@{k}" for k in K_VALUES if k <= 25] + ["ndcg@5", "ndcg@10"]
)
SOURCES_DEFAULT = ["gold", "silver"]
# A published run must cover at least this fraction of the largest run's query
# set to take part in the paired analysis. Guards against a smoke test (an
# `evaluate --limit N`) or a crashed run silently shrinking every table, since
# the paired query set is an intersection across runs.
MIN_RUN_COVERAGE = 0.5
SEGMENT_KEYS = ["study_type", "knowledge_area", "year"]

# Metric used for the pairwise significance tests and most breakdowns.
PRIMARY_METRIC = "mrr"
MCNEMAR_K = 5  # Hit@k cut-off for the binary (McNemar) significance test.

# Cut-offs for the recall ceiling of the first-stage candidate list: the "deep" values of the single K_VALUES list.
RECALL_CEILING_K = [k for k in K_VALUES if k >= 25]

# Default location of the per-PDI productivity map used by the equity metric.
# JSON: {"pdi_<id>": <activity count>}.
# Built and published to MinIO by the upm_match_productivity DAG (the server runs
# only through Airflow) and mirrored here by fetch-minio.
PRODUCTIVITY_DEFAULT = "evaluation/reports/productividad_pdis.json"
EQUITY_METRICS = ["mrr", f"hit@{MCNEMAR_K}"]
TERCILE_LABELS = ["T1 (baja)", "T2 (media)", "T3 (alta)"]

# Co-supervision re-scoring (T11). The ground truth records ONE director per
# work, but the portal lists every PDI who tutored it and a sizeable minority of
# the reference works have more than one, so a genuine co-supervisor of that very
# work is scored as a miss (memoria sec:eval-validez). T11 re-scores the same
# cached rankings counting any real tutor as a hit. Complementary reading only:
# relaxing relevance makes the task easier by construction, so absolute figures
# rise and only the RELATIVE comparison between strategies stays comparable.
COSUP_METRICS = ["mrr", f"hit@{MCNEMAR_K}", "hit@25"]
# Cached {work_id: [pdi ids]} so the table survives without Memgraph (the
# analysis often runs where only the reports are mirrored).
TUTORS_DEFAULT = "evaluation/reports/tutores_por_trabajo.json"

# Off-line late-fusion sensitivity sweep (T9). The graph-grounded re-ranker
# stores, per candidate, its listwise textual rank plus the raw graph signals in
# `rerank_components`; we re-compose the ranking under many fusion weights WITHOUT
# re-invoking the LLM (the dominant cost), and read off which signals/weights help
# quality and at what cost in equity (memoria sec:eval-estrategias-rerank).
SWEEP_MODE = "rerank_listwise_kg"  # the only mode carrying graph components
# Leave-one-out twin of the sweep mode: same ranking, graph signals recomputed
# with each query's own supervised work masked (harness subcommand `signals-loo`).
# Not a search condition, so it never enters T1/T2/T3; it only feeds a second
# sweep that shows what the signals are worth once the echo of the query's own
# work is removed (memoria sec:eval-validez).
SWEEP_MODE_LOO = "rerank_listwise_kg_loo"
# Same protocol applied to the OTHER channel: the evidence the re-ranker reads in
# its prompt, which can only be masked by re-ranking again. It is a full run, but
# NOT a competing strategy: it is `rerank_listwise_kg` measured under a stricter
# protocol. Keeping it out of the headline family matters, because the
# multiple-comparison correction must span the hypotheses of one question, and
# adding a robustness variant would penalise every other pair for a question that
# is not theirs (it would also leave gold and silver corrected over families of
# different size). Reported instead in its own table, with its own paired test.
EXCL_MODE = "rerank_listwise_kg_excl"
EXTRA_SWEEP_MODES = (SWEEP_MODE_LOO, EXCL_MODE)
# Graph signals available per candidate (higher = better), fused with the text rank.
GRAPH_SIGNALS = ["supervised_total", "supervised_fit", "teaching_fit", "coauthors_in_set"]
GRAPH_SIGNAL_LABELS = {
    "supervised_total": "supervisión (volumen)",
    "supervised_fit": "ajuste de supervisión",
    "teaching_fit": "ajuste docente",
    "coauthors_in_set": "coautoría",
}
# Late fusion is rank-based RRF (scale-free, so the raw-count signals and the
# text rank combine without per-signal normalisation); text rank is anchored at
# weight 1.0 and each graph signal weight is swept over this grid.
RRF_K = 60
SWEEP_GRID = [0.0, 0.5, 1.0]
SWEEP_METRICS = ["mrr", f"hit@{MCNEMAR_K}", "ndcg@10"]


# Per-query metric extraction
def metric_on_ranking(ranked: list[str], rel: str, metric: str) -> float:
    """Value of one ranking metric for an arbitrary ranked id list (single relevant item).

    Split from `per_query_value` so the off-line sweep (T9) can score a ranking
    it re-composes from stored components, not just the one stored in a result.
    """
    if metric in ("mrr", "map"):  # average precision == reciprocal rank here
        return reciprocal_rank(ranked, rel)
    name, _, k_str = metric.partition("@")
    k = int(k_str)
    if name == "hit":
        return hit_at_k(ranked, rel, k)
    if name == "recall":
        return recall_at_k(ranked, rel, k)
    if name == "precision":
        return precision_at_k(ranked, rel, k)
    if name == "ndcg":
        return ndcg_at_k(ranked, rel, k)
    raise ValueError(f"Unknown metric: {metric!r}")


def per_query_value(res: RecommendationResult, metric: str) -> float:
    """Value of one ranking metric for one query (single relevant item)."""
    return metric_on_ranking(res.ranked_researcher_ids, res.true_director, metric)


# Run discovery + loading
def discover_runs(reports_dir: Path, source: str, tag: str | None = None) -> list[Path]:
    """Return the timestamped run directories for a source (skipping `latest/`).
    A run directory is one that holds a `cache/` with at least one `recommendations_<mode>.json`.
    Sorted by name (timestamp), so order is chronological.

    With ``tag`` set, only the campaign under ``<source>/<tag>/`` is scanned, so a
    re-run published with ``evaluate --tag <tag>`` is analysed in isolation from
    earlier runs stored under the flat ``<source>/`` layout.
    """
    base = reports_dir / source / tag if tag else reports_dir / source
    if not base.is_dir():
        return []
    runs = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name == "latest":
            continue
        if list((child / "cache").glob("recommendations_*.json")):
            runs.append(child)
    return runs


def load_run_mode(run_dir: Path, mode: str) -> dict[str, RecommendationResult]:
    """Load one mode's per-query results from a run, keyed by gold_id (errors dropped)."""
    path = run_dir / "cache" / f"recommendations_{mode}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, RecommendationResult] = {}
    for item in raw:
        res = RecommendationResult.model_validate(item)
        if res.error is None:
            out[res.gold_id] = res
    return out


def load_productivity(path: Path) -> dict[str, float] | None:
    """Load the per-PDI productivity map used by the equity metric, or None.

    Expected JSON: ``{"pdi_<id>": <activity count>}``, produced by the evaluation DAG on the server and fetched with the reports.
    Returns None when the file is missing so the equity table is skipped gracefully, mirroring how the segment tables degrade when the gold standard is absent.
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in raw.items()}
    except (ValueError, TypeError, OSError) as exc:
        logger.warning("Could not read productivity map {}: {}", path, exc)
        return None


def load_tutor_sets(
    work_ids: list[str], cache_path: Path | None = None
) -> dict[str, set[str]] | None:
    """Map every reference work to the FULL set of PDIs the portal lists as its tutors.

    The ``Tutoring`` edge carries no role, so the portal cannot tell a director
    from a co-tutor or a ponente; every listed supervisor is therefore treated
    alike, which is the same assumption the rest of the corpus makes about the
    source. Used by T11 to re-score with a relaxed relevance criterion.

    Reads ``cache_path`` when present so the analysis runs where only the reports
    are mirrored; otherwise queries Memgraph and writes the cache. Returns None
    when neither is available, so T11 is skipped gracefully like T8.
    """
    if cache_path is not None and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.info("Tutor sets read from {} ({} works).", cache_path, len(raw))
            return {str(k): set(v) for k, v in raw.items()}
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("Could not read tutor map {}: {}", cache_path, exc)

    try:
        from dbs.graph.memgraph import MemgraphBackend  # lazy: needs MEMGRAPH_* env

        db = MemgraphBackend()
        out: dict[str, set[str]] = {}
        for row in db.query(
            """
            UNWIND $ids AS pid
            MATCH (t:LastYearProject {id: pid})<-[:Tutoring]-(p:Pdi)
            RETURN pid, collect(DISTINCT p.id) AS tutors
            """,
            {"ids": list(work_ids)},
        ):
            out[row["pid"]] = set(row["tutors"] or [])
        multi = sum(1 for v in out.values() if len(v) > 1)
        logger.info(
            "Tutor sets from Memgraph: {} works, {} with more than one tutor.",
            len(out),
            multi,
        )
        if cache_path is not None:
            # Best effort: the map is already in memory, so a read-only mount
            # must cost the cache, never the table.
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {k: sorted(v) for k, v in out.items()},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not cache tutor map to {}: {}", cache_path, exc)
        return out or None
    except Exception:  # noqa: BLE001 — degrade: T11 is skipped upstream
        logger.warning("Memgraph unavailable and no tutor cache; T11 skipped.")
        return None


def _assign_terciles(prod_by_pdi: dict[str, float]) -> dict[str, int]:
    """Assign each PDI to a productivity tercile 0 (low) / 1 / 2 (high).

    Boundaries are the 33rd and 66th percentiles over the PDI population given, so the split describes the candidates, not the queries (a prolific supervisor with many queries still counts once towards the boundaries).
    """
    values = np.array(sorted(prod_by_pdi.values()), dtype=float)
    q33, q66 = np.percentile(values, [100 / 3, 200 / 3])
    out: dict[str, int] = {}
    for pdi, v in prod_by_pdi.items():
        out[pdi] = 0 if v <= q33 else (1 if v <= q66 else 2)
    return out


class SourceData:
    """All runs of one source, organised for paired per-query analysis.

    Attributes
    ----------
    :runs: chronological list of run directories.
    :by_mode: mode -> list of {gold_id: RecommendationResult}, one dict per run that has the mode (so deterministic modes simply repeat the same numbers).
    :modes: modes present, in SEARCH_MODES order.
    :common: gold_ids present (non-error) in EVERY (mode, run), so every comparison is fully paired over the same queries.
    """

    def __init__(self, source: str, runs: list[Path]):
        self.source = source
        self.runs = runs
        self.by_mode: dict[str, list[dict[str, RecommendationResult]]] = {}
        for mode in (m for m in SEARCH_MODES if m not in EXTRA_SWEEP_MODES):
            per_run = [load_run_mode(r, mode) for r in runs]
            per_run = [d for d in per_run if d]  # keep only runs that have this mode
            if per_run:
                self.by_mode[mode] = per_run
        self.modes = [m for m in SEARCH_MODES if m in self.by_mode]

        # `common` is an INTERSECTION, so a partial run (a smoke test with
        # --limit, or a crashed one) would silently shrink every table to its
        # size while every number still looked plausible. Such runs are dropped
        # with a warning instead: a published run is only comparable when it
        # covers the same query set as the campaign it belongs to.
        sizes = [len(d) for runs_ in self.by_mode.values() for d in runs_]
        floor = max(sizes) * MIN_RUN_COVERAGE if sizes else 0
        for mode, runs_ in list(self.by_mode.items()):
            kept = [d for d in runs_ if len(d) >= floor]
            if len(kept) != len(runs_):
                logger.warning(
                    "Mode {}: {} partial run(s) ignored ({} < {:.0f} queries); "
                    "they would have shrunk the paired query set.",
                    mode,
                    len(runs_) - len(kept),
                    ", ".join(str(len(d)) for d in runs_ if len(d) < floor),
                    floor,
                )
            if kept:
                self.by_mode[mode] = kept
            else:
                del self.by_mode[mode]
        self.modes = [m for m in SEARCH_MODES if m in self.by_mode]

        id_sets = [set(d) for runs_ in self.by_mode.values() for d in runs_]
        self.common: list[str] = sorted(set.intersection(*id_sets)) if id_sets else []

        # Modes that are NOT search conditions and must stay out of the headline
        # tables: they carry the same ranking as an existing mode with some
        # component recomputed, and exist only for the off-line sweep. Loaded
        # here so the sweep can address them, never added to `self.modes`.
        self.extra_modes: dict[str, list[dict[str, RecommendationResult]]] = {}
        for mode in EXTRA_SWEEP_MODES:
            per_run = [load_run_mode(r, mode) for r in runs]
            per_run = [d for d in per_run if d]
            if per_run:
                self.extra_modes[mode] = per_run

    def runs_for_mode(
        self, mode: str
    ) -> list[dict[str, RecommendationResult]] | None:
        """Runs of a mode, whether it is a search condition or a sweep-only extra."""
        return self.by_mode.get(mode) or self.extra_modes.get(mode)

    def n_runs(self, mode: str) -> int:
        return len(self.by_mode.get(mode, []))

    def denoised(self, mode: str, metric: str) -> np.ndarray:
        """Per-query metric for a mode, averaged across its runs (denoised).

        Returns one value per common query, in `self.common` order.
        """
        runs_ = self.by_mode[mode]
        out = np.empty(len(self.common))
        for i, gid in enumerate(self.common):
            out[i] = float(np.mean([per_query_value(d[gid], metric) for d in runs_]))
        return out

    def majority_hit(self, mode: str, k: int) -> np.ndarray:
        """Binary Hit@k per common query by majority vote across the mode's runs."""
        runs_ = self.by_mode[mode]
        out = np.empty(len(self.common))
        for i, gid in enumerate(self.common):
            hits = [per_query_value(d[gid], f"hit@{k}") for d in runs_]
            out[i] = 1.0 if np.mean(hits) >= 0.5 else 0.0
        return out

    def per_run_metric(self, mode: str, metric: str) -> np.ndarray:
        """Aggregate (mean over common queries) of a metric, one value per run."""
        runs_ = self.by_mode[mode]
        return np.array(
            [
                float(np.mean([per_query_value(d[gid], metric) for gid in self.common]))
                for d in runs_
            ]
        )

    def pooled_timings(self, mode: str) -> dict[str, float]:
        """Pool every (run, common-query) timing for a mode into summary seconds."""
        totals, llms, searches = [], [], []
        for d in self.by_mode[mode]:
            for gid in self.common:
                r = d[gid]
                if r.total_seconds is None:
                    continue
                totals.append(r.total_seconds)
                llms.append(r.llm_seconds or 0.0)
                searches.append(r.search_seconds or 0.0)
        if not totals:
            return {}
        t = np.array(totals)
        return {
            "n": float(len(totals)),
            "total_mean_s": float(t.mean()),
            "total_median_s": float(np.median(t)),
            "total_p90_s": float(np.percentile(t, 90)),
            "llm_mean_s": float(np.mean(llms)),
            "search_mean_s": float(np.mean(searches)),
        }


# Statistics
def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, b: int, alpha: float
) -> tuple[float, float, float]:
    """Mean and a (1-alpha) percentile bootstrap CI over the sample `values`."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(b, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float
) -> tuple[float, float, float]:
    """Mean paired difference a-b with a paired percentile bootstrap CI (resample queries)."""
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired a vs b (1.0 if all ties)."""
    d = a - b
    if not np.any(d != 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def mcnemar_p(hit_a: np.ndarray, hit_b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar p-value on binary hits; returns (p, b, c) discordants.

    b = #(A hit, B miss), c = #(A miss, B hit). p=1.0 when there are no discordant pairs.
    """
    b = int(np.sum((hit_a == 1) & (hit_b == 0)))
    c = int(np.sum((hit_a == 0) & (hit_b == 1)))
    if b + c == 0:
        return 1.0, b, c
    p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return float(p), b, c


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, aligned to the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg (FDR) adjusted p-values, aligned to the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m / (rank + 1) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# Table assembly (each builder returns (markdown, latex))
def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _latex_table(
    header: list[str], rows: list[list[str]], caption: str, label: str
) -> str:
    cols = "l" + "r" * (len(header) - 1)
    head = " & ".join(header) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "|" + "---|" * len(header)
    lines = ["| " + " | ".join(header) + " |", sep]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_t1_headline(
    sd: SourceData, rng: np.random.Generator, b: int, alpha: float
) -> tuple[str, str]:
    """T1 — per-mode headline metrics with bootstrap CIs over queries."""
    header = ["Modo", "n"] + HEADLINE_METRICS
    rows = []
    for mode in sd.modes:
        cells = [MODE_LABELS.get(mode, mode), str(len(sd.common))]
        for metric in HEADLINE_METRICS:
            mean, lo, hi = bootstrap_ci(sd.denoised(mode, metric), rng, b, alpha)
            cells.append(f"{_fmt(mean)} [{_fmt(lo)}, {_fmt(hi)}]")
        rows.append(cells)
    cap = (
        f"Métricas por modo de búsqueda (fuente {sd.source}); media e IC bootstrap al "
        f"{int((1 - alpha) * 100)}\\% sobre {len(sd.common)} consultas."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:metrics-{sd.source}"
    )


def build_t2_significance(
    sd: SourceData, rng: np.random.Generator, b: int, alpha: float
) -> tuple[str, str]:
    """T2 — pairwise significance on MRR (Wilcoxon) and Hit@k (McNemar), corrected."""
    pairs = list(combinations(sd.modes, 2))
    diffs, ci_los, ci_his, p_wil, p_mcn, disc = [], [], [], [], [], []
    for a_mode, b_mode in pairs:
        a = sd.denoised(a_mode, PRIMARY_METRIC)
        bb = sd.denoised(b_mode, PRIMARY_METRIC)
        diff, lo, hi = paired_diff_ci(a, bb, rng, b, alpha)
        diffs.append(diff)
        ci_los.append(lo)
        ci_his.append(hi)
        p_wil.append(wilcoxon_p(a, bb))
        p, nb, nc = mcnemar_p(
            sd.majority_hit(a_mode, MCNEMAR_K), sd.majority_hit(b_mode, MCNEMAR_K)
        )
        p_mcn.append(p)
        disc.append((nb, nc))

    wil_holm, wil_bh = holm(p_wil), benjamini_hochberg(p_wil)
    mcn_holm = holm(p_mcn)

    header = [
        "Modo A",
        "Modo B",
        "ΔMRR",
        f"IC{int((1 - alpha) * 100)}\\%",
        "p Wilcoxon",
        "p Holm",
        "p BH",
        f"p McNemar(Hit@{MCNEMAR_K})",
        "p Holm",
        "sig.",
    ]
    rows = []
    for i, (a_mode, b_mode) in enumerate(pairs):
        sig = "sí" if wil_holm[i] < alpha else "no"
        rows.append(
            [
                MODE_LABELS.get(a_mode, a_mode),
                MODE_LABELS.get(b_mode, b_mode),
                _fmt(diffs[i]),
                f"[{_fmt(ci_los[i])}, {_fmt(ci_his[i])}]",
                _fmt(p_wil[i], 4),
                _fmt(wil_holm[i], 4),
                _fmt(wil_bh[i], 4),
                _fmt(p_mcn[i], 4),
                _fmt(mcn_holm[i], 4),
                sig,
            ]
        )
    cap = (
        f"Significancia pareada entre modos (fuente {sd.source}). ΔMRR>0 indica que el "
        "modo A supera al B; IC por bootstrap pareado; p de Wilcoxon (MRR) y McNemar "
        f"(Hit@{MCNEMAR_K}) corregidas por comparaciones múltiples (Holm/BH); "
        f'"sig."=Holm<{alpha} sobre MRR.'
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:significance-{sd.source}"
    )


def build_t3_robustness(sd: SourceData) -> tuple[str, str]:
    """T3 — between-run robustness (mean +/- std across runs)."""
    metrics = ["mrr", f"hit@{MCNEMAR_K}", "ndcg@10"]
    header = ["Modo", "nº runs"] + [m + " (media±σ)" for m in metrics]
    rows = []
    for mode in sd.modes:
        cells = [MODE_LABELS.get(mode, mode), str(sd.n_runs(mode))]
        for metric in metrics:
            vals = sd.per_run_metric(mode, metric)
            cells.append(f"{_fmt(vals.mean())} ± {_fmt(vals.std(ddof=0))}")
        rows.append(cells)
    cap = (
        f"Robustez entre ejecuciones (fuente {sd.source}): media y desviación típica de "
        "cada métrica a través de las repeticiones. Los modos deterministas "
        "(léxica/semántica directa) tienen σ=0."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:robustness-{sd.source}"
    )


def build_t4_timings(sd: SourceData) -> tuple[str, str]:
    """T4 — per-query timings pooled over runs."""
    keys = [
        "n",
        "total_mean_s",
        "total_median_s",
        "total_p90_s",
        "llm_mean_s",
        "search_mean_s",
    ]
    header = ["Modo"] + keys
    rows = []
    for mode in sd.modes:
        t = sd.pooled_timings(mode)
        cells = [MODE_LABELS.get(mode, mode)]
        for k in keys:
            v = t.get(k, 0.0)
            cells.append(_fmt(v, 0) if k == "n" else _fmt(v))
        rows.append(cells)
    cap = (
        f"Tiempo por consulta en segundos (fuente {sd.source}), agregado sobre todas las "
        "repeticiones; desglose en porción LLM (generación) y recuperación."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:timings-{sd.source}"
    )


def build_t5_segments(
    sd: SourceData, gold_by_id: dict[str, GoldRecord]
) -> list[tuple[str, str, str]]:
    """T5 — MRR by segment (one table per segment dimension): mode x segment value."""
    out: list[tuple[str, str, str]] = []
    for seg_key in SEGMENT_KEYS:
        groups: dict[str, list[int]] = {}
        for i, gid in enumerate(sd.common):
            g = gold_by_id.get(gid)
            if g is None:
                continue
            groups.setdefault(str(getattr(g, seg_key)), []).append(i)
        if not groups:
            continue
        seg_values = sorted(groups)
        header = ["Modo"] + [f"{v} (n={len(groups[v])})" for v in seg_values]
        rows = []
        for mode in sd.modes:
            vals = sd.denoised(mode, PRIMARY_METRIC)
            cells = [MODE_LABELS.get(mode, mode)]
            for v in seg_values:
                idx = groups[v]
                cells.append(_fmt(float(vals[idx].mean())))
            rows.append(cells)
        cap = (
            f"MRR por modo y {seg_key} (fuente {sd.source}); media sobre las consultas de "
            "cada segmento."
        )
        out.append(
            (
                seg_key,
                _md_table(header, rows),
                _latex_table(header, rows, cap, f"tab:segment-{sd.source}-{seg_key}"),
            )
        )
    return out


def build_t6_cross_source(sources: dict[str, SourceData]) -> tuple[str, str]:
    """T6 — gold vs silver headline metrics per mode (the flip)."""
    metrics = ["mrr", f"hit@{MCNEMAR_K}", "ndcg@10"]
    names = list(sources)
    header = ["Modo"] + [f"{m} ({s})" for m in metrics for s in names]
    rows = []
    modes = [m for m in SEARCH_MODES if all(m in sd.by_mode for sd in sources.values())]
    for mode in modes:
        cells = [MODE_LABELS.get(mode, mode)]
        for metric in metrics:
            for s in names:
                cells.append(_fmt(float(sources[s].denoised(mode, metric).mean())))
        rows.append(cells)
    cap = (
        "Comparativa entre fuentes (gold vs silver) de las métricas principales por modo; "
        "evidencia el cambio de ranking entre condiciones."
    )
    return _md_table(header, rows), _latex_table(header, rows, cap, "tab:cross-source")


def build_t7_recall_ceiling(sd: SourceData) -> tuple[str, str]:
    """T7 — recall ceiling recall@{25,50,100} of the first-stage candidate list.

    The re-ranking extension can only reorder candidates the first stage retrieved, so this is the upper bound of Hit@k after re-ranking and the go/no-go for the extension.
    Every strategy retrieves max(K_VALUES) candidates, so these cut-offs are always populated from a normal evaluation run.
    """
    header = ["Modo", "n"] + [f"recall@{k}" for k in RECALL_CEILING_K]
    rows = []
    for mode in sd.modes:
        cells = [MODE_LABELS.get(mode, mode), str(len(sd.common))]
        for k in RECALL_CEILING_K:
            cells.append(_fmt(float(sd.denoised(mode, f"recall@{k}").mean())))
        rows.append(cells)
    cap = (
        f"Techo de recall de la primera etapa (fuente {sd.source}): fracción de consultas "
        f"cuyo supervisor real aparece entre los primeros $K$ candidatos, para "
        f"$K\\in\\{{{', '.join(map(str, RECALL_CEILING_K))}\\}}$. Cota superior del "
        "$\\mathrm{Hit@}k$ tras la reordenación; requiere una profundidad de candidatos "
        "$\\ge$ 100 para ser informativo más allá de recall@25."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:recall-ceiling-{sd.source}"
    )


def build_t8_equity(
    sd: SourceData, productivity: dict[str, float]
) -> tuple[str, str] | None:
    """T8 — equity of the recommendation across productivity terciles of the true supervisor.

    Each query is placed in the tercile of its true supervisor's productivity; per mode we report a quality metric per tercile plus the gap T3-T1.
    A large gap in favour of the most productive tercile signals reliance on popularity — the risk the graph-grounded re-ranking must not aggravate.
    Returns None when too few supervisors have a productivity value to form terciles.
    """
    # True supervisor per common query (identical across modes/runs: it is the gold truth), taken from any loaded result.
    ref_run = sd.by_mode[sd.modes[0]][0]
    director_by_gid = {gid: ref_run[gid].true_director for gid in sd.common}

    directors = {d for d in director_by_gid.values() if d in productivity}
    if len(directors) < 3:
        logger.warning(
            "Equity table skipped (fuente {}): only {} supervisors have a productivity "
            "value; need at least 3 for terciles.",
            sd.source,
            len(directors),
        )
        return None

    tercile_of_pdi = _assign_terciles({d: productivity[d] for d in directors})
    # Query indices (into sd.common) per tercile.
    idx_by_tercile: dict[int, list[int]] = {0: [], 1: [], 2: []}
    dropped = 0
    for i, gid in enumerate(sd.common):
        d = director_by_gid[gid]
        if d in tercile_of_pdi:
            idx_by_tercile[tercile_of_pdi[d]].append(i)
        else:
            dropped += 1

    header = (
        ["Modo", "Métrica"]
        + [f"{TERCILE_LABELS[t]} (n={len(idx_by_tercile[t])})" for t in (0, 1, 2)]
        + ["brecha T3−T1"]
    )
    rows = []
    for mode in sd.modes:
        for metric in EQUITY_METRICS:
            vals = sd.denoised(mode, metric)
            per_t = [
                float(vals[idx_by_tercile[t]].mean())
                if idx_by_tercile[t]
                else float("nan")
                for t in (0, 1, 2)
            ]
            gap = per_t[2] - per_t[0]
            rows.append(
                [MODE_LABELS.get(mode, mode), metric]
                + [_fmt(v) for v in per_t]
                + [_fmt(gap)]
            )
    cap = (
        f"Equidad por tercil de productividad del supervisor real (fuente {sd.source}): "
        "calidad de la recomendación en el tercil menos, medianamente y más productivo. "
        "Una brecha T3$-$T1 grande delata dependencia de la popularidad. "
        f"{dropped} consultas sin dato de productividad quedaron fuera."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:equity-{sd.source}"
    )


# Late-fusion sensitivity sweep (T9)
def _ranks_by_signal(
    comp: dict[str, dict[str, float]], ids: list[str], signal: str
) -> dict[str, int]:
    """Rank candidates 1..N by DESCENDING signal value (ties -> better stage-1 rank).

    Turns a raw graph signal (a count, higher = better) into a per-query rank so
    it can enter the rank-based RRF fusion on the same footing as the text rank.
    """
    order = sorted(
        ids,
        key=lambda c: (-comp[c].get(signal, 0.0), comp[c].get("stage1_rank", 0.0)),
    )
    return {c: i + 1 for i, c in enumerate(order)}


def _fused_order(comp: dict[str, dict[str, float]], weights: dict[str, float]) -> list[str]:
    """Re-compose the candidate ranking under `weights` via rank-based RRF.

    `weights["text"]` scales the stored listwise rank; each graph signal with a
    positive weight contributes its own descending-value rank. Higher fused score
    first; the text rank breaks ties so a text-only config reproduces exactly the
    re-ranker's stored ordering.
    """
    ids = list(comp)
    text_rank = {
        c: int(comp[c].get("listwise_rank", comp[c].get("stage1_rank", i + 1)))
        for i, c in enumerate(ids)
    }
    active = {s: w for s, w in weights.items() if s != "text" and w > 0}
    sig_ranks = {s: _ranks_by_signal(comp, ids, s) for s in active}

    def score(c: str) -> float:
        s = weights.get("text", 0.0) / (RRF_K + text_rank[c])
        for sig, w in active.items():
            s += w / (RRF_K + sig_ranks[sig][c])
        return s

    return sorted(ids, key=lambda c: (-score(c), text_rank[c]))


def _sweep_per_query(
    runs: list[dict[str, RecommendationResult]],
    common: list[str],
    weights: dict[str, float],
    metric: str,
) -> np.ndarray:
    """Per-query metric of the re-composed ranking, denoised across the mode's runs.

    Mirrors `SourceData.denoised`: one value per common query, averaged over the
    runs that carry graph components for that query.
    """
    out = np.zeros(len(common))
    for i, gid in enumerate(common):
        vals = []
        for run in runs:
            res = run.get(gid)
            if res is None or not res.rerank_components:
                continue
            order = _fused_order(res.rerank_components, weights)
            vals.append(metric_on_ranking(order, res.true_director, metric))
        if vals:
            out[i] = float(np.mean(vals))
    return out


def _sweep_configs() -> list[tuple[str, dict[str, float]]]:
    """Named ablation configs: text-only, each graph signal alone, then all together."""
    configs: list[tuple[str, dict[str, float]]] = [("solo texto (listwise)", {"text": 1.0})]
    for sig in GRAPH_SIGNALS:
        configs.append((f"+ {GRAPH_SIGNAL_LABELS[sig]}", {"text": 1.0, sig: 0.5}))
    configs.append(("todas las señales (0.5)", {"text": 1.0, **{s: 0.5 for s in GRAPH_SIGNALS}}))
    configs.append(("todas las señales (1.0)", {"text": 1.0, **{s: 1.0 for s in GRAPH_SIGNALS}}))
    return configs


def _grid_configs() -> list[tuple[str, dict[str, float]]]:
    """Full |SWEEP_GRID|^|GRAPH_SIGNALS| grid (text anchored at 1.0) for the argmax search."""
    from itertools import product

    out: list[tuple[str, dict[str, float]]] = []
    for combo in product(SWEEP_GRID, repeat=len(GRAPH_SIGNALS)):
        weights = {"text": 1.0}
        weights.update({s: w for s, w in zip(GRAPH_SIGNALS, combo)})
        label = ", ".join(
            f"{GRAPH_SIGNAL_LABELS[s]}={w:g}" for s, w in zip(GRAPH_SIGNALS, combo) if w > 0
        )
        out.append((label or "solo texto", weights))
    return out


def build_t9_sweep(
    sd: SourceData,
    productivity: dict[str, float] | None,
    sweep_mode: str = SWEEP_MODE,
) -> tuple[str, str] | None:
    """T9 — late-fusion sensitivity sweep over the graph-grounded re-ranker's signals.

    Re-composes `rerank_listwise_kg`'s ranking off-line under a grid of fusion
    weights and reports quality (MRR / Hit@k / NDCG) plus, when a productivity map
    is available, the equity gap T3-T1 on MRR. Rows: a text-only reference, each
    graph signal added alone, all signals together, and the grid-search optimum by
    MRR — so the reader sees each signal's marginal value AND whether chasing MRR
    widens the popularity gap. Returns None when the graph-grounded mode is absent.
    """
    runs = sd.runs_for_mode(sweep_mode)
    if not runs:
        return None
    loo = sweep_mode != SWEEP_MODE

    # Optional equity: query indices per productivity tercile of the true supervisor.
    idx_by_tercile: dict[int, list[int]] | None = None
    if productivity is not None:
        ref = runs[0]
        director_by_gid = {gid: ref[gid].true_director for gid in sd.common if gid in ref}
        directors = {d for d in director_by_gid.values() if d in productivity}
        if len(directors) >= 3:
            tercile_of_pdi = _assign_terciles({d: productivity[d] for d in directors})
            idx_by_tercile = {0: [], 1: [], 2: []}
            for i, gid in enumerate(sd.common):
                d = director_by_gid.get(gid)
                if d in tercile_of_pdi:
                    idx_by_tercile[tercile_of_pdi[d]].append(i)

    def equity_gap(mrr_per_query: np.ndarray) -> float:
        if not idx_by_tercile or not idx_by_tercile[0] or not idx_by_tercile[2]:
            return float("nan")
        return float(
            mrr_per_query[idx_by_tercile[2]].mean() - mrr_per_query[idx_by_tercile[0]].mean()
        )

    def row_for(label: str, weights: dict[str, float]) -> tuple[list[str], float]:
        cells = [label]
        mrr_pq = _sweep_per_query(runs, sd.common, weights, "mrr")
        for metric in SWEEP_METRICS:
            pq = mrr_pq if metric == "mrr" else _sweep_per_query(runs, sd.common, weights, metric)
            cells.append(_fmt(float(pq.mean())))
        if idx_by_tercile is not None:
            cells.append(_fmt(equity_gap(mrr_pq)))
        return cells, float(mrr_pq.mean())

    header = ["Configuración"] + [m.upper() if m == "mrr" else m for m in SWEEP_METRICS]
    if idx_by_tercile is not None:
        header.append("brecha T3−T1 (MRR)")

    rows = [row_for(label, w)[0] for label, w in _sweep_configs()]

    # Grid-search optimum by MRR (many configs, so only the winner is reported here;
    # the full grid goes to a machine-readable CSV for the appendix / reproducibility).
    grid = [(label, w, *row_for(label, w)) for label, w in _grid_configs()]
    best_label, _best_w, best_cells, _best_mrr = max(grid, key=lambda g: g[3])
    best_cells[0] = f"óptimo por MRR ({best_label})"
    rows.append(best_cells)

    cap = (
        f"Barrido de sensibilidad de la fusión tardía (fuente {sd.source}) sobre las "
        "señales del reordenador con grafo, recompuesto sin volver a invocar el LLM. "
        + (
            "Señales calculadas con protocolo leave-one-out: el trabajo que origina "
            "cada consulta queda oculto al calcular las señales de supervisión de su "
            "propio director, de modo que la aportación medida ya no contiene el eco "
            "del enunciado. El rango textual de partida sigue procediendo del "
            "reordenador que sí vio esa evidencia en su contexto. "
            if loo
            else ""
        )
        + "Cada fila fija el peso del texto en 1.0 y varía las señales de grafo; la última "
        f"es el óptimo de una rejilla de ${len(_grid_configs())}$ combinaciones por MRR. "
        + (
            "La brecha T3$-$T1 vigila que ganar MRR no ensanche el sesgo de popularidad."
            if idx_by_tercile is not None
            else "(Sin mapa de productividad: columna de equidad omitida.)"
        )
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:rerank-sweep-{sd.source}{'-loo' if loo else ''}"
    )


def write_sweep_grid(
    sd: SourceData, out_dir: Path, sweep_mode: str = SWEEP_MODE
) -> Path | None:
    """Dump the full weight grid × metrics to CSV for the appendix (one row per config)."""
    runs = sd.runs_for_mode(sweep_mode)
    if not runs:
        return None
    suffix = "" if sweep_mode == SWEEP_MODE else "_loo"
    lines = ["config," + ",".join(GRAPH_SIGNALS) + "," + ",".join(SWEEP_METRICS)]
    for label, weights in _grid_configs():
        cols = [f'"{label}"'] + [f"{weights.get(s, 0.0):g}" for s in GRAPH_SIGNALS]
        for metric in SWEEP_METRICS:
            cols.append(
                _fmt(float(_sweep_per_query(runs, sd.common, weights, metric).mean()))
            )
        lines.append(",".join(cols))
    path = out_dir / f"t9_sweep_grid_{sd.source}{suffix}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Clarification effect (T10)
CLARIFY_MODE = "clarify_hybrid"
CLARIFY_BASELINE = "hybrid"


def build_t10_clarify(sd: SourceData) -> tuple[str, str] | None:
    """T10 — effect of the ASK-style clarification loop (memoria sec:eval-clarify).

    Reports how often the detector decided to ask (activation rate, overall and
    by ambiguity class) and what clarification bought, comparing the clarified
    ranking against the SAME queries' hybrid ranking. The gain is reported on
    the activated subset — where clarification actually happened — and on the
    whole set, since a loop that fires rarely can help a lot where it fires and
    still barely move the aggregate.
    Returns None when the clarification mode is absent from the campaign.
    """
    if CLARIFY_MODE not in sd.by_mode or CLARIFY_BASELINE not in sd.by_mode:
        return None
    runs = sd.by_mode[CLARIFY_MODE]

    # A query counts as activated when the detector asked at least once in the
    # majority of runs (the detector is deterministic per state, but the query
    # interpretation that precedes it is not).
    activated: list[int] = []
    rounds_total = 0
    levels: Counter[str] = Counter()
    for i, gid in enumerate(sd.common):
        fired = 0
        for run in runs:
            info = (run.get(gid).clarify_info or {}) if run.get(gid) else {}
            rounds = int(info.get("rounds") or 0)
            rounds_total += rounds
            levels.update(info.get("levels") or [])
            fired += 1 if rounds else 0
        if fired * 2 >= len(runs):
            activated.append(i)

    n_total = len(sd.common)
    n_act = len(activated)
    rows: list[list[str]] = []
    header = ["Métrica", "Valor"]
    rows.append(["Consultas evaluadas", str(n_total)])
    rows.append(
        [
            "Tasa de activación",
            f"{n_act / n_total:.1%} ({n_act})" if n_total else "—",
        ]
    )
    rows.append(
        [
            "Rondas por consulta (media)",
            _fmt(rounds_total / (len(runs) * n_total), 2) if n_total and runs else "—",
        ]
    )
    for level, count in sorted(levels.items()):
        rows.append([f"Preguntas de tipo «{level}»", str(count)])

    for metric in (PRIMARY_METRIC, f"hit@{MCNEMAR_K}"):
        base = sd.denoised(CLARIFY_BASELINE, metric)
        clar = sd.denoised(CLARIFY_MODE, metric)
        rows.append(
            [f"{metric} — todas: híbrida $\\rightarrow$ clarificada",
             f"{_fmt(float(base.mean()))} $\\rightarrow$ {_fmt(float(clar.mean()))}"]
        )
        if activated:
            idx = np.array(activated)
            rows.append(
                [
                    f"{metric} — activadas: híbrida $\\rightarrow$ clarificada",
                    f"{_fmt(float(base[idx].mean()))} $\\rightarrow$ "
                    f"{_fmt(float(clar[idx].mean()))}",
                ]
            )
    if activated:
        idx = np.array(activated)
        diff = sd.denoised(CLARIFY_MODE, PRIMARY_METRIC)[idx] - sd.denoised(
            CLARIFY_BASELINE, PRIMARY_METRIC
        )[idx]
        better = int(np.sum(diff > 0))
        worse = int(np.sum(diff < 0))
        rows.append(["Consultas activadas que mejoran / empeoran", f"{better} / {worse}"])
        rows.append(["p Wilcoxon (activadas, MRR)", _fmt(wilcoxon_p(
            sd.denoised(CLARIFY_MODE, PRIMARY_METRIC)[idx],
            sd.denoised(CLARIFY_BASELINE, PRIMARY_METRIC)[idx]), 4)])

    cap = (
        f"Efecto de la clarificación conversacional (fuente {sd.source}): con qué "
        "frecuencia el detector decide preguntar y qué aporta hacerlo, frente al "
        "mismo ranking híbrido sin clarificar. La ganancia se reporta sobre el "
        "subconjunto activado y sobre el total."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:clarify-{sd.source}"
    )


def build_t11_cosupervision(
    sd: SourceData, tutors: dict[str, set[str]]
) -> tuple[str, str] | None:
    """T11 — re-scoring when ANY real supervisor of the work counts as a hit.

    The default criterion credits only the single `director` recorded in the
    ground truth, yet the portal lists every PDI who tutored the work. When a
    work was co-supervised, the other supervisor is scored as an error even
    though they demonstrably supervised that exact work: a sharper, factual
    instance of the single-relevant limitation (memoria sec:eval-validez).

    Re-scores the SAME cached rankings with the relaxed criterion, taking the
    best-ranked real tutor as the relevant item, so nothing is re-run. Reported
    as a complementary reading: the relaxation makes the task easier by
    construction, so absolute values rise for every strategy and it is the
    relative ordering, unchanged, that carries the conclusions.
    """
    covered = [gid for gid in sd.common if tutors.get(gid)]
    if not covered:
        return None
    multi = sum(1 for gid in covered if len(tutors[gid]) > 1)
    if not multi:
        return None

    def relaxed_value(res: RecommendationResult, metric: str) -> float:
        """Metric for one query crediting whichever real tutor ranks best."""
        rels = tutors.get(res.gold_id) or {res.true_director}
        best = next((r for r in res.ranked_researcher_ids if r in rels), None)
        return metric_on_ranking(
            res.ranked_researcher_ids, best or res.true_director, metric
        )

    def mode_mean(mode: str, metric: str, relaxed: bool) -> float:
        runs_ = sd.by_mode[mode]
        per_query = [
            float(
                np.mean(
                    [
                        relaxed_value(d[gid], metric)
                        if relaxed
                        else per_query_value(d[gid], metric)
                        for d in runs_
                    ]
                )
            )
            for gid in covered
        ]
        return float(np.mean(per_query))

    header = ["Modo"]
    for metric in COSUP_METRICS:
        header += [f"{metric} (director)", f"{metric} (cualquiera)", "Δ"]

    rows: list[list[str]] = []
    for mode in sd.modes:
        row = [MODE_LABELS.get(mode, mode)]
        for metric in COSUP_METRICS:
            base = mode_mean(mode, metric, False)
            rel = mode_mean(mode, metric, True)
            pct = f" ({(rel - base) / base * 100:+.1f}\\%)" if base else ""
            row += [_fmt(base), _fmt(rel), f"{rel - base:+.3f}{pct}"]
        rows.append(row)

    cap = (
        f"Relectura con criterio de relevancia relajado (fuente {sd.source}, "
        f"n={len(covered)}): se cuenta como acierto a cualquiera de los tutores "
        f"reales del trabajo, no solo al director registrado en la verdad de "
        f"referencia. {multi} de los {len(covered)} trabajos tienen más de un "
        "tutor en el portal. Se reutilizan los mismos rankings sin recalcular "
        "nada: lo único que cambia es a quién se considera respuesta correcta. "
        "Lectura complementaria, no sustitutiva, pues relajar la relevancia "
        "facilita la tarea por construcción y eleva a todas las estrategias por "
        "igual."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:cosupervision-{sd.source}"
    )


def build_t13_excl(
    sd: SourceData, rng: np.random.Generator, b: int, alpha: float
) -> tuple[str, str] | None:
    """T13 — the graph-grounded re-ranking with the query's own work masked in the prompt.

    Companion to T12: that one cleans the numeric signals off-line, this one
    cleans the evidence the model actually read, which requires re-ranking. The
    comparison is paired query by query (both consume the same first stage) and
    is reported on its own, deliberately outside the headline family of T2.
    """
    runs_a = sd.by_mode.get(SWEEP_MODE)
    runs_b = sd.extra_modes.get(EXCL_MODE)
    if not runs_a or not runs_b:
        return None

    def vals(runs_, metric: str) -> np.ndarray:
        out = np.empty(len(sd.common))
        for i, gid in enumerate(sd.common):
            per_run = [per_query_value(d[gid], metric) for d in runs_ if gid in d]
            out[i] = float(np.mean(per_run)) if per_run else 0.0
        return out

    header = ["Métrica", "Sin exclusión", "Con exclusión", "Δ", "IC 95\\%", "$p$ Wilcoxon"]
    rows: list[list[str]] = []
    for metric in ["mrr", f"hit@{MCNEMAR_K}", "hit@25", "ndcg@10"]:
        a, bb = vals(runs_a, metric), vals(runs_b, metric)
        _, lo, hi = paired_diff_ci(bb, a, rng, b, alpha)
        rows.append(
            [
                metric,
                _fmt(float(a.mean())),
                _fmt(float(bb.mean())),
                f"{bb.mean() - a.mean():+.3f}",
                f"[{_fmt(lo)}, {_fmt(hi)}]",
                f"{wilcoxon_p(a, bb):.4f}",
            ]
        )

    cap = (
        f"Efecto de ocultar al reordenador el trabajo que origina cada consulta "
        f"(fuente {sd.source}, $n={len(sd.common)}$). Ambas condiciones parten del "
        "mismo ranking de primera etapa, de modo que la comparación es pareada y "
        "toda diferencia procede de la evidencia enmascarada. Se reporta aparte de "
        "la Tabla~\\ref{tab:benchmark-significancia} porque no es una estrategia "
        "rival sino la misma condición bajo un protocolo más estricto, y añadirla "
        "a aquella familia corregiría el resto de contrastes por una pregunta "
        "distinta de la suya."
    )
    return _md_table(header, rows), _latex_table(
        header, rows, cap, f"tab:excl-{sd.source}"
    )


def publish_analysis_to_minio(out_dir: Path, tag: str | None) -> None:
    """Upload the analysis outputs (report.md, LaTeX tables, sweep CSVs) to MinIO.

    Written for the Airflow-only server: the campaign DAG runs the analysis right
    after the evaluation and this publishes it next to the raw runs, so the whole
    campaign (data + analysis) can be pulled from anywhere with MinIO credentials.
    Layout mirrors the harness: ``analysis/<tag>/<timestamp>/`` (immutable) plus an
    overwriting ``analysis/<tag>/latest/``. Failures are logged, never raised — the
    analysis is already safe on disk.
    """
    try:
        from dbs import minio
    except Exception as exc:  # noqa: BLE001 — MinIO is optional for local runs
        logger.warning("MinIO not available, skipping analysis publish: {}", exc)
        return

    import os
    from datetime import datetime

    bucket = os.environ.get("MINIO_EVAL_BUCKET", "tfm-evaluation")
    ns = f"analysis/{tag}" if tag else "analysis"
    base = f"{ns}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    latest = f"{ns}/latest"

    uploads = [p for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != ".gitignore"]
    ok = 0
    for fp in uploads:
        rel = fp.relative_to(out_dir).as_posix()
        if minio.upload_file(bucket, fp, f"{base}/{rel}") and minio.upload_file(
            bucket, fp, f"{latest}/{rel}"
        ):
            ok += 1
    logger.info(
        "Published {}/{} analysis artefacts to MinIO bucket {!r} under {}/ (and {}/)",
        ok,
        len(uploads),
        bucket,
        base,
        latest,
    )


# Orchestration
def analyze(args: argparse.Namespace) -> int:
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    tex_dir = out_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)

    try:
        gold = load_gold_standard()
        gold_by_id = {g.gold_id: g for g in gold}
    except FileNotFoundError as exc:
        logger.warning("Gold standard not found ({}); segment tables skipped.", exc)
        gold_by_id = {}

    tutors = load_tutor_sets(
        sorted(gold_by_id), Path(args.tutors) if args.tutors else None
    )

    productivity = load_productivity(Path(args.productivity))
    if productivity is None:
        logger.warning(
            "Productivity map not found ({}); equity table (T8) skipped. "
            "Run the upm_match_productivity DAG and fetch the reports to get it.",
            args.productivity,
        )

    tag = getattr(args, "tag", None) or None
    sources: dict[str, SourceData] = {}
    for source in args.sources:
        runs = discover_runs(reports_dir, source, tag)
        if not runs:
            logger.warning(
                "No runs found for source {!r}{} under {}.",
                source,
                f" (tag {tag!r})" if tag else "",
                reports_dir,
            )
            continue
        sd = SourceData(source, runs)
        if not sd.modes or not sd.common:
            logger.warning("Source {!r}: no paired queries/modes — skipping.", source)
            continue
        logger.info(
            "Source {}: {} run(s), {} mode(s), {} common queries.",
            source,
            len(runs),
            len(sd.modes),
            len(sd.common),
        )
        sources[source] = sd

    if not sources:
        logger.error(
            "Nothing to analyze. Did you commit the results under {}?", reports_dir
        )
        return 1

    md_blocks: list[str] = ["# UPM Match — análisis estadístico de la evaluación\n"]

    def emit(name: str, md: str, tex: str, title: str) -> None:
        md_blocks.append(f"\n## {title}\n\n{md}\n")
        (tex_dir / f"{name}.tex").write_text(tex, encoding="utf-8")

    for source, sd in sources.items():
        rng = np.random.default_rng(args.seed)
        md_blocks.append(f"\n# Fuente: {source}\n")

        md, tex = build_t1_headline(sd, rng, args.bootstrap, args.alpha)
        emit(f"t1_headline_{source}", md, tex, f"T1 · Métricas por modo — {source}")

        md, tex = build_t2_significance(sd, rng, args.bootstrap, args.alpha)
        emit(
            f"t2_significance_{source}",
            md,
            tex,
            f"T2 · Significancia pareada — {source}",
        )

        md, tex = build_t3_robustness(sd)
        emit(f"t3_robustness_{source}", md, tex, f"T3 · Robustez entre runs — {source}")

        md, tex = build_t4_timings(sd)
        emit(f"t4_timings_{source}", md, tex, f"T4 · Tiempos por consulta — {source}")

        for seg_key, md, tex in build_t5_segments(sd, gold_by_id):
            emit(
                f"t5_segment_{source}_{seg_key}",
                md,
                tex,
                f"T5 · MRR por {seg_key} — {source}",
            )

        md, tex = build_t7_recall_ceiling(sd)
        emit(
            f"t7_recall_ceiling_{source}",
            md,
            tex,
            f"T7 · Techo de recall (etapa 1) — {source}",
        )

        if productivity is not None:
            equity = build_t8_equity(sd, productivity)
            if equity is not None:
                emit(
                    f"t8_equity_{source}",
                    equity[0],
                    equity[1],
                    f"T8 · Equidad por tercil de productividad — {source}",
                )

        sweep = build_t9_sweep(sd, productivity)
        if sweep is not None:
            emit(
                f"t9_rerank_sweep_{source}",
                sweep[0],
                sweep[1],
                f"T9 · Barrido de sensibilidad del reordenador — {source}",
            )
            grid_path = write_sweep_grid(sd, out_dir)
            if grid_path is not None:
                logger.info("Full re-ranking weight grid → {}", grid_path)

        sweep_loo = build_t9_sweep(sd, productivity, sweep_mode=SWEEP_MODE_LOO)
        if sweep_loo is not None:
            emit(
                f"t12_rerank_sweep_loo_{source}",
                sweep_loo[0],
                sweep_loo[1],
                f"T12 · Barrido con señales leave-one-out — {source}",
            )
            grid_loo = write_sweep_grid(sd, out_dir, sweep_mode=SWEEP_MODE_LOO)
            if grid_loo is not None:
                logger.info("Leave-one-out weight grid → {}", grid_loo)

        clarify = build_t10_clarify(sd)
        if clarify is not None:
            emit(
                f"t10_clarify_{source}",
                clarify[0],
                clarify[1],
                f"T10 · Efecto de la clarificación — {source}",
            )

        excl = build_t13_excl(sd, rng, args.bootstrap, args.alpha)
        if excl is not None:
            emit(
                f"t13_excl_{source}",
                excl[0],
                excl[1],
                f"T13 · Reordenación con exclusión del trabajo de la consulta — {source}",
            )

        if tutors:
            cosup = build_t11_cosupervision(sd, tutors)
            if cosup is not None:
                emit(
                    f"t11_cosupervision_{source}",
                    cosup[0],
                    cosup[1],
                    f"T11 · Relevancia relajada por cosupervisión — {source}",
                )

    if len(sources) > 1:
        md, tex = build_t6_cross_source(sources)
        emit("t6_cross_source", md, tex, "T6 · Comparativa gold vs silver")

    report_md = "\n".join(md_blocks)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / ".gitignore").write_text("*", encoding="utf-8")
    logger.info(
        "Wrote {} and {} LaTeX tables under {}.",
        out_dir / "report.md",
        len(list(tex_dir.glob("*.tex"))),
        tex_dir,
    )
    if getattr(args, "publish_minio", False):
        publish_analysis_to_minio(out_dir, tag)
    print("\n" + report_md + "\n")
    return 0


# Offline self-test (no data / services)
def selftest(_args: argparse.Namespace) -> int:
    """Validate the statistical helpers against known references."""
    rng = np.random.default_rng(0)

    # Bootstrap CI of a constant sample is the constant.
    mean, lo, hi = bootstrap_ci(np.ones(50), rng, 1000, 0.05)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0, (mean, lo, hi)

    # Wilcoxon: identical vectors -> p = 1.0; a strictly dominating b -> small p.
    a = np.array([0.0, 0.0, 0.0, 0.0])
    assert wilcoxon_p(a, a.copy()) == 1.0
    big = np.arange(1, 21, dtype=float)
    assert wilcoxon_p(big + 1.0, big) < 0.01

    # McNemar: no discordants -> p=1; all discordant one way -> tiny p.
    p, nb, nc = mcnemar_p(np.array([1, 1, 1]), np.array([1, 1, 1]))
    assert p == 1.0 and nb == 0 and nc == 0
    p, nb, nc = mcnemar_p(np.ones(12), np.zeros(12))
    assert nb == 12 and nc == 0 and p < 0.01, (p, nb, nc)
    # Exact McNemar with b=8,c=2 matches the two-sided binomial.
    p, nb, nc = mcnemar_p(
        np.array([1] * 8 + [0] * 2 + [1] * 5), np.array([0] * 8 + [1] * 2 + [1] * 5)
    )
    assert (nb, nc) == (8, 2)
    assert abs(p - binomtest(8, 10, 0.5).pvalue) < 1e-12

    # Holm / BH: monotone, bounded, and a textbook check.
    raw = [0.01, 0.02, 0.03, 0.04]
    h = holm(raw)
    assert abs(h[0] - 0.04) < 1e-9 and all(0 <= x <= 1 for x in h)
    bh = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert all(0 <= x <= 1 for x in bh)

    # Paired diff CI brackets the true mean difference for a clear shift.
    d, lo, hi = paired_diff_ci(big + 2.0, big, rng, 2000, 0.05)
    assert lo <= 2.0 <= hi and abs(d - 2.0) < 1e-9

    # Terciles: 9 evenly-spread values split 3/3/3 into low/mid/high.
    ter = _assign_terciles({f"pdi_{i}": float(i) for i in range(9)})
    assert [ter[f"pdi_{i}"] for i in range(9)] == [0, 0, 0, 1, 1, 1, 2, 2, 2], ter

    # Late fusion: text-only reproduces the listwise order; a strong graph signal
    # on the tail candidate lifts it above the others.
    comp = {
        "pdi_a": {"stage1_rank": 1, "listwise_rank": 1, "supervised_fit": 0.0},
        "pdi_b": {"stage1_rank": 2, "listwise_rank": 2, "supervised_fit": 0.0},
        "pdi_c": {"stage1_rank": 3, "listwise_rank": 3, "supervised_fit": 9.0},
    }
    assert _fused_order(comp, {"text": 1.0}) == ["pdi_a", "pdi_b", "pdi_c"]
    lifted = _fused_order(comp, {"text": 1.0, "supervised_fit": 5.0})
    assert lifted[0] == "pdi_c", lifted
    # metric_on_ranking scores an arbitrary re-composed list, not a stored one.
    assert metric_on_ranking(["x", "y", "z"], "y", "mrr") == 0.5
    assert metric_on_ranking(["x", "y", "z"], "y", "hit@2") == 1.0
    assert metric_on_ranking(["x", "y", "z"], "w", "hit@3") == 0.0

    print(
        "selftest OK — bootstrap, Wilcoxon, McNemar, Holm/BH, terciles, "
        "late-fusion all pass."
    )
    return 0


def main() -> int:
    import os

    # The tables use Unicode (Δ, σ, ±, accents); force UTF-8 so a Windows cp1252 console doesn't crash on print/log.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        description="Statistical analysis of UPM Match evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir",
        default="evaluation/reports",
        help="Directory mirroring the MinIO eval bucket (<source>/<ts>/...).",
    )
    parser.add_argument(
        "--out-dir",
        default="evaluation/reports/analysis",
        help="Where to write report.md and tables/*.tex.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=SOURCES_DEFAULT,
        choices=SOURCES_DEFAULT,
        help="Reference sets to analyze (default: both).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Only analyze the campaign published with 'evaluate --tag <tag>' "
        "(runs under <source>/<tag>/). Omit to analyze the flat <source>/ layout. "
        "Use it to exclude earlier runs made on outdated data.",
    )
    parser.add_argument(
        "--publish-minio",
        action="store_true",
        help="Upload the analysis outputs (report.md, tables/, sweep CSVs) to the "
        "evaluation MinIO bucket under analysis/<tag>/<timestamp>/ (plus an "
        "overwriting analysis/<tag>/latest/). Used by the campaign DAG so the "
        "whole campaign is retrievable off-server.",
    )
    parser.add_argument(
        "--productivity",
        default=PRODUCTIVITY_DEFAULT,
        help="JSON map {pdi_<id>: activity count} for the equity table (T8), "
        "produced by the evaluation DAG and fetched with the reports; T8 is "
        "skipped if absent.",
    )
    parser.add_argument(
        "--tutors",
        default=TUTORS_DEFAULT,
        help="JSON map {work_id: [pdi ids]} with EVERY tutor the portal lists per "
        "reference work, for the relaxed-relevance table (T11). Read if present; "
        "otherwise built from Memgraph and written there. T11 is skipped when "
        "neither is available.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=10000, help="Bootstrap resamples."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level / CI width."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (reproducible CIs)."
    )
    parser.add_argument(
        "--selftest", action="store_true", help="Run offline math checks and exit."
    )
    args = parser.parse_args()
    return selftest(args) if args.selftest else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
