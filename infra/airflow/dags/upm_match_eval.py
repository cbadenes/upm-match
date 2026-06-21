"""UPM Match — recommendation-quality evaluation DAG.

Runs the offline evaluation harness (evaluation/evaluate_recommendations.py) inside the Airflow container, which already has uv, the project mounted at /opt/airflow/project and network access to Qdrant + Ollama.

It evaluates every search mode over the chosen reference set, computes the ranking metrics + timings, and publishes the report (report.md + metrics.json) plus the raw per-mode rankings to MinIO,
so the results can be pulled from anywhere with MinIO credentials (bucket MINIO_EVAL_BUCKET, default 'tfm-evaluation', under <source>/<timestamp>/ and an overwriting <source>/latest/).

Reference set (DAG param `source`):
  - "gold" (default): query = the real thesis descriptions. Needs no silver standard, so the benchmark can run before the silver set exists.
  - "silver": query = the generated student-style proposals. Build the silver standard first (locally with `build-silver`, or as a separate run) and place evaluation/silver_standard.json in the repo before triggering with source=silver.

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
            "'silver' = generated student proposals (requires silver_standard.json).",
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
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} "
            "evaluate --source {{ params.source }} --publish-minio"
        ),
        doc_md=(
            "Runs every search mode over the chosen reference set, computes metrics + "
            "timings + segment breakdowns, writes the report under "
            "evaluation/reports/<source>/<ts>/ and uploads it to MinIO. Resumable via "
            "the on-disk per-mode cache, so the configured retries resume from the "
            "last completed project."
        ),
    )

    selftest >> evaluate
