"""UPM Match — recommendation-quality evaluation DAG.

Runs the offline evaluation harness (evaluation/evaluate_recommendations.py) inside the Airflow container, which already has uv, the project mounted at /opt/airflow/project and network access to Qdrant + Ollama.

It evaluates every search mode over the chosen reference set, computes the ranking metrics + timings, and publishes the report (report.md + metrics.json) plus the raw per-mode rankings to MinIO,
so the results can be pulled from anywhere with MinIO credentials (bucket MINIO_EVAL_BUCKET, default 'tfm-evaluation', under <source>/<timestamp>/ and an overwriting <source>/latest/).

Reference set (DAG param `source`):
  - "gold" (default): query = the real thesis descriptions. Needs no silver standard, so the benchmark can run before the silver set exists.
  - "silver": query = the generated student-style proposals. Build the silver standard first with the `upm_match_silver` DAG, which writes evaluation/silver/silver_standard_{tfg,tfm}.json (gold schema), before triggering with source=silver.

Resumability: the harness caches every (mode, project) result to disk under evaluation/cache/<source>/ and skips the ones already done, so a crashed or retried task resumes from the last completed project rather than starting over.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.evaluate_recommendations"
_UV = "uv run"

# Search modes the harness can evaluate (mirrors SEARCH_MODES in the eval script).
# Kept here so the DAG stays self-contained; the script validates the choice too.
_SEARCH_MODES = [
    "lexical_direct",
    "keywords",
    "summary_direct",
    "summary_enriched",
    "hybrid",
]

_default_args = {
    "owner": "upm-match",
    # The evaluation is resumable from the on-disk cache, so a retry picks up where
    # a crashed run left off instead of repeating completed queries.
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_evaluation",
    description="Offline recommendation-quality evaluation over the gold/silver standard.",
    start_date=datetime(2026, 1, 30),
    # schedule=None -> manual trigger only.
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    params={
        "source": Param(
            "gold",
            enum=["gold", "silver"],
            description="Query source: 'gold' = thesis descriptions (no silver needed); "
            "'silver' = generated student proposals (run the upm_match_silver DAG first).",
        ),
        "tag": Param(
            "",
            type="string",
            description="Campaign tag to isolate this run under <source>/<tag>/ "
            "(cache, reports and MinIO). Set it after regenerating the data (e.g. "
            "'clean_abstracts') so a fresh cache is built on the new data and the "
            "analysis never mixes it with earlier runs. Leave empty for the flat layout.",
        ),
        "modes": Param(
            _SEARCH_MODES,
            type="array",
            items={"type": "string", "enum": _SEARCH_MODES},
            description="Search modes to evaluate (default = all five). Narrow it to "
            "re-run a subset, e.g. only the LLM modes (keywords, summary_enriched, "
            "hybrid) for the run-to-run variance repetitions.",
        ),
        "no_cache": Param(
            False,
            type="boolean",
            description="Recompute every (mode, query) from scratch, ignoring the "
            "on-disk cache. Needed to get fresh LLM rankings on each run (e.g. "
            "repeating the LLM modes to measure variance); otherwise cached results "
            "are reused and the run reproduces the previous one.",
        ),
        "no_universe_filter": Param(
            False,
            type="boolean",
            description="Do NOT restrict queries to supervisors present in BOTH "
            "indices (the common-universe protocol). Off (default) keeps the fair "
            "comparison; turn on only for an ablation of how the filter affects the "
            "metrics.",
        ),
    },
    tags=["upm", "evaluation"],
    doc_md=__doc__,
) as dag:
    selftest = BashOperator(
        task_id="selftest",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} selftest",
        retries=0,
        doc_md="Validates the metric math offline (no Qdrant/Ollama). Fast fail-fast guard.",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        env={
            "EVAL_CACHE_DIR": "/home/airflow/upm_eval/cache",
            "EVAL_REPORTS_DIR": "/home/airflow/upm_eval/reports",
        },
        append_env=True,
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} "
            "evaluate --source {{ params.source }} "
            "--modes {{ params.modes | join(' ') }} "
            "--publish-minio"
            "{{ ' --tag ' ~ params.tag if params.tag else '' }}"
            "{{ ' --no-cache' if params.no_cache else '' }}"
            "{{ ' --no-universe-filter' if params.no_universe_filter else '' }}"
        ),
        doc_md=(
            "Runs every search mode over the chosen reference set, computes metrics + "
            "timings + segment breakdowns, writes the report under "
            "evaluation/reports/<source>/<ts>/ and uploads it to MinIO. Resumable via "
            "the on-disk per-mode cache, so the configured retries resume from the "
            "last completed project."
        ),
    )

    fetch_results = BashOperator(
        task_id="fetch_results",
        # Mirror the whole eval bucket into the mounted project, so the results land in
        # the repo checkout on the host and can be pulled/copied from there. The target
        # dir must be writable by the container task user; make it so once on the host:
        #   mkdir -p evaluation/reports && chmod -R a+rwx evaluation/reports
        env={"EVAL_DOWNLOAD_DIR": f"{_PROJECT}/evaluation/reports"},
        append_env=True,
        # Run regardless of the evaluate outcome, to pull whatever is already in MinIO.
        trigger_rule="all_done",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} fetch-minio",
        doc_md=(
            "Downloads ALL evaluation artefacts stored in MinIO into "
            "evaluation/reports/ of the mounted project (the host repo checkout), "
            "mirroring the bucket layout (<source>/<ts>/... and <source>/latest/...). "
            "Runs even if 'evaluate' failed/retried, so it always pulls whatever "
            "results exist. The target dir must be writable by the container user."
        ),
    )

    selftest >> evaluate >> fetch_results
