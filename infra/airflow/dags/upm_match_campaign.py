"""UPM Match — DEFINITIVE evaluation campaign, end to end, in one trigger.

Runs the complete measurement campaign of the TFM (phase B4) unattended, so a
single manual trigger leaves everything computed, analysed and published:

  1. selftests        — offline math checks of the harness and the analysis.
  2. productivity     — per-PDI productivity map (Memgraph) for the equity
                        table (T8), published to the eval bucket root.
  3. gold  x3 reps    — repetition 1 evaluates ALL modes (incl. the two
                        deterministic baselines); repetitions 2-3 only the
                        LLM-in-the-loop modes (keywords, summary_enriched,
                        hybrid, rerank_listwise, rerank_listwise_kg), per the
                        3-run variance protocol of the memoria.
  4. silver x3 reps   — same over the silver standard (student-style proposals;
                        requires evaluation/silver/*.json, already committed).
  5. analyze          — statistical analysis (tables T1-T9, incl. the recall
                        ceiling, the equity terciles and the late-fusion
                        sensitivity sweep) over the whole campaign, published to
                        MinIO under analysis/<tag>/ (report.md + LaTeX tables +
                        sweep CSVs).
  6. fetch_to_repo    — mirrors the entire eval bucket (runs + productivity +
                        analysis) into the mounted repo checkout, so the results
                        are on the host when someone comes back to the machine.

Freshness vs resumability: each repetition is preceded by a `wipe_cache` task
that clears the campaign cache namespace for that source, and the evaluation
itself runs WITHOUT --no-cache. First attempt therefore computes everything
fresh (the wipe emptied the cache), while a crashed/retried `evaluate` attempt
RESUMES from the partial cache instead of starting the repetition over — the
wipe is a separate task with no retries, so Airflow will not re-run it once it
has succeeded.

The rerank modes consume the hybrid rankings cached by the SAME repetition
(mode order guarantees hybrid runs first), which is the registered protocol:
stage 1 fixed within a repetition, so rerank vs hybrid is perfectly paired.

Everything is sequential (a single chain): the LLM host is shared with
production and parallel repetitions would skew the timing measurements (T4).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.evaluate_recommendations"
_ANALYZE = "evaluation.analyze_results"
_UV = "uv run"

# Container-writable working dirs (same convention as upm_match_evaluation).
_CACHE_DIR = "/home/airflow/upm_eval/cache"
_REPORTS_DIR = "/home/airflow/upm_eval/reports"
# Local mirror of the eval bucket used by the analysis step (container-writable).
_MIRROR_DIR = "/home/airflow/upm_eval/mirror"

_ALL_MODES = [
    "lexical_direct",
    "keywords",
    "summary_direct",
    "summary_enriched",
    "hybrid",
    "rerank_listwise",
    "rerank_listwise_kg",
    "clarify_hybrid",
]
# Modes with an LLM in the query loop: the only ones whose run-to-run variance
# the repetitions measure. Order matters: hybrid before the modes that consume
# its cached ranking (the rerank pair and the clarification loop).
_LLM_MODES = [
    "keywords",
    "summary_enriched",
    "hybrid",
    "rerank_listwise",
    "rerank_listwise_kg",
    "clarify_hybrid",
]

# 3-run protocol of the memoria. Parse-time constant (Airflow cannot grow the
# task graph from a runtime param); edit here if the protocol ever changes.
_REPETITIONS = 3

_SOURCES = ["gold", "silver"]

_default_args = {
    "owner": "upm-match",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

_eval_env = {
    "EVAL_CACHE_DIR": _CACHE_DIR,
    "EVAL_REPORTS_DIR": _REPORTS_DIR,
}


def _evaluate_cmd(source: str, modes: list[str]) -> str:
    return (
        f"cd {_PROJECT} && {_UV} --module {_MODULE} "
        f"evaluate --source {source} "
        f"--modes {' '.join(modes)} "
        "--publish-minio --tag {{ params.tag }}"
    )


with DAG(
    dag_id="upm_match_campaign",
    description="Definitive end-to-end evaluation campaign: gold+silver x3, productivity, analysis, publish.",
    start_date=datetime(2026, 1, 30),
    # schedule=None -> manual trigger only (this is a one-shot campaign).
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    params={
        "tag": Param(
            "clean_abstracts",
            type="string",
            minLength=1,
            description="Campaign tag (REQUIRED): isolates cache, reports and MinIO "
            "under <source>/<tag>/ and analysis/<tag>/. Re-triggering with the same "
            "tag restarts the campaign fresh (the wipe tasks clear its cache).",
        ),
    },
    tags=["upm", "evaluation", "campaign"],
    doc_md=__doc__,
) as dag:
    selftest_harness = BashOperator(
        task_id="selftest_harness",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} selftest",
        retries=0,
        doc_md="Offline metric-math checks of the evaluation harness. Fail-fast guard.",
    )

    selftest_analysis = BashOperator(
        task_id="selftest_analysis",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_ANALYZE} --selftest",
        retries=0,
        doc_md="Offline checks of the statistics (bootstrap, tests, late fusion). Fail-fast guard.",
    )

    productivity = BashOperator(
        task_id="productivity",
        env=_eval_env,
        append_env=True,
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} productivity --publish-minio"
        ),
        doc_md=(
            "Per-PDI productivity map from Memgraph (publications + supervised "
            "works), published to the eval bucket root. Feeds the equity table (T8) "
            "and the equity column of the sensitivity sweep (T9)."
        ),
    )

    chain = [selftest_harness, selftest_analysis, productivity]

    for source in _SOURCES:
        for rep in range(1, _REPETITIONS + 1):
            modes = _ALL_MODES if rep == 1 else _LLM_MODES
            wipe = BashOperator(
                task_id=f"wipe_cache_{source}_rep{rep}",
                env=_eval_env,
                append_env=True,
                # Guard on a non-empty tag so an empty value can never wipe the
                # flat <source>/ cache of other runs.
                bash_command=(
                    'test -n "{{ params.tag }}" && '
                    f'rm -rf "{_CACHE_DIR}/{source}/{{{{ params.tag }}}}" && '
                    "echo wiped"
                ),
                retries=0,
                doc_md=(
                    "Clears this campaign's cache namespace so the repetition "
                    "computes fresh LLM outputs. Separate task with no retries: a "
                    "retried evaluate resumes from partial cache instead of "
                    "restarting the repetition."
                ),
            )
            evaluate = BashOperator(
                task_id=f"evaluate_{source}_rep{rep}",
                env=_eval_env,
                append_env=True,
                bash_command=_evaluate_cmd(source, modes),
                doc_md=(
                    f"Repetition {rep}/{_REPETITIONS} over the {source} standard "
                    f"({'all modes' if rep == 1 else 'LLM modes only'}). Publishes "
                    "report + per-mode rankings to MinIO under its own timestamp. "
                    "Resumable on retry via the on-disk cache."
                ),
            )
            chain += [wipe, evaluate]

    fetch_for_analysis = BashOperator(
        task_id="fetch_for_analysis",
        env={"EVAL_DOWNLOAD_DIR": _MIRROR_DIR},
        append_env=True,
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} fetch-minio",
        doc_md=(
            "Mirrors the whole eval bucket into a container-writable dir so the "
            "analysis reads every repetition of the campaign (plus the "
            "productivity map) regardless of project-mount permissions."
        ),
    )

    analyze = BashOperator(
        task_id="analyze",
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_ANALYZE} "
            f"--reports-dir {_MIRROR_DIR} "
            f"--out-dir {_MIRROR_DIR}/analysis "
            f"--productivity {_MIRROR_DIR}/productividad_pdis.json "
            f"--tutors {_MIRROR_DIR}/tutores_por_trabajo.json "
            "--tag {{ params.tag }} "
            "--publish-minio"
        ),
        doc_md=(
            "Statistical analysis of the whole campaign (both sources, all "
            "repetitions): tables T1-T11 as Markdown + LaTeX, including the recall "
            "ceiling (T7), equity terciles (T8), the late-fusion sensitivity "
            "sweep with its full grid CSVs (T9), the clarification loop (T10) and "
            "the relaxed-relevance re-scoring that credits any real co-supervisor "
            "(T11, needs Memgraph the first time; the tutor map is then cached in "
            "the mirror dir). Published to MinIO under analysis/<tag>/<timestamp>/ "
            "and analysis/<tag>/latest/."
        ),
    )

    fetch_to_repo = BashOperator(
        task_id="fetch_to_repo",
        env={"EVAL_DOWNLOAD_DIR": f"{_PROJECT}/evaluation/reports"},
        append_env=True,
        # Best effort, and pulls whatever exists even if a step above failed.
        trigger_rule="all_done",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} fetch-minio",
        doc_md=(
            "Mirrors the eval bucket (runs + productivity + analysis) into the "
            "mounted repo checkout so everything is on the host when you're back. "
            "evaluation/reports must be writable by the container user "
            "(mkdir -p evaluation/reports && chmod -R a+rwx evaluation/reports)."
        ),
    )

    chain += [fetch_for_analysis, analyze, fetch_to_repo]
    for upstream, downstream in zip(chain, chain[1:]):
        upstream >> downstream
