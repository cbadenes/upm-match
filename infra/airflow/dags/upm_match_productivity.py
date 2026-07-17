"""UPM Match — per-PDI productivity map for the equity metric (T8).

Computes, from the knowledge graph (Memgraph), the productivity of every
researcher (number of publications + supervised works) and publishes the
resulting map to the evaluation bucket in MinIO, so it travels with the rest of
the evaluation artefacts: the `fetch_results` step of the evaluation DAG (or a
manual `fetch-minio`) mirrors it to `evaluation/reports/productividad_pdis.json`,
which is exactly where `evaluation/analyze_results.py` expects it for the
equity-by-tercile table (T8). While the file is absent, T8 is skipped gracefully.

Run it once after each corpus regeneration (phases 03+ re-run), BEFORE analysing
an evaluation campaign. It is idempotent: re-running simply overwrites the map
with fresh counts.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.evaluate_recommendations"
_UV = "uv run"

_default_args = {
    "owner": "upm-match",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_productivity",
    description="Per-PDI productivity map (Memgraph) for the equity metric T8.",
    start_date=datetime(2026, 1, 30),
    # schedule=None -> manual trigger only (run after each corpus regeneration).
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    tags=["upm", "evaluation"],
    doc_md=__doc__,
) as dag:
    productivity = BashOperator(
        task_id="build_and_publish",
        env={"EVAL_REPORTS_DIR": "/home/airflow/upm_eval/reports"},
        append_env=True,
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} productivity --publish-minio"
        ),
        doc_md=(
            "Counts publications (Authorship) + supervised works (Tutoring) per PDI "
            "in Memgraph, writes productividad_pdis.json and uploads it to the eval "
            "bucket root, next to the campaign reports."
        ),
    )
