"""
UPM Match — statistical analysis of the evaluation runs (memoria tables).

Reads every evaluation run published by `evaluate_recommendations.py --publish-minio` and mirrored back into the repo (`evaluation/reports/<source>/<ts>/...`), and turns the raw per-query rankings into the tables the memoria needs, with proper uncertainty quantification.

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

# Reuse the harness so the metric math is the single source of truth.
from evaluation.evaluate_recommendations import (  # noqa: E402
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

# Metrics reported per mode (mrr == map and recall@k == hit@k for a single relevant item,
# both kept for fidelity with the existing memoria tables).
HEADLINE_METRICS = [
    "mrr",
    "map",
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "ndcg@5",
    "ndcg@10",
]
SOURCES_DEFAULT = ["gold", "silver"]
SEGMENT_KEYS = ["study_type", "knowledge_area", "year"]

# Metric used for the pairwise significance tests and most breakdowns.
PRIMARY_METRIC = "mrr"
MCNEMAR_K = 5  # Hit@k cut-off for the binary (McNemar) significance test.


# Per-query metric extraction
def per_query_value(res: RecommendationResult, metric: str) -> float:
    """Value of one ranking metric for one query (single relevant item)."""
    ranked, rel = res.ranked_researcher_ids, res.true_director
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


# Run discovery + loading
def discover_runs(reports_dir: Path, source: str) -> list[Path]:
    """Return the timestamped run directories for a source (skipping `latest/`).
    A run directory is one that holds a `cache/` with at least one `recommendations_<mode>.json`.
    Sorted by name (timestamp), so order is chronological.
    """
    base = reports_dir / source
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
        for mode in SEARCH_MODES:
            per_run = [load_run_mode(r, mode) for r in runs]
            per_run = [d for d in per_run if d]  # keep only runs that have this mode
            if per_run:
                self.by_mode[mode] = per_run
        self.modes = [m for m in SEARCH_MODES if m in self.by_mode]

        id_sets = [set(d) for runs_ in self.by_mode.values() for d in runs_]
        self.common: list[str] = sorted(set.intersection(*id_sets)) if id_sets else []

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

    sources: dict[str, SourceData] = {}
    for source in args.sources:
        runs = discover_runs(reports_dir, source)
        if not runs:
            logger.warning(
                "No runs found for source {!r} under {}.", source, reports_dir
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

    print("selftest OK — bootstrap, Wilcoxon, McNemar, Holm/BH all pass.")
    return 0


def main() -> int:
    import os

    # The tables use Unicode (Δ, σ, ±, accents); force UTF-8 so a Windows cp1252
    # console doesn't crash on print/log.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(
        description="Statistical analysis of UPM Match evaluation runs (memoria tables).",
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
