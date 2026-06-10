"""UPM Match — main pipeline DAG.

Task dependency graph:
  01_find_ids -> 02_get_pdis_data -> 03_create_pdis -> 04_enrich_profile -> [05_create_keywords, 06_create_summaries]

Phases 05 and 06 are independent and run in parallel.
Estimated full pipeline duration: 3-6 hours.

To run only part of the pipeline, select tasks from the UI using
"Clear" + "Only Failed", or use the trigger with parameters.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_UV = "uv run"

_default_args = {
    "owner": "upm-match",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_pipeline",
    description="Full pipeline: scraping -> Memgraph graph -> LLM enrichment -> Qdrant indexing",
    start_date=datetime(2026, 1, 30),
    # schedule=None  -> manual trigger only
    # For weekly automatic runs (Sundays at 3am): schedule="0 3 * * 0"
    schedule=None,
    catchup=False,
    default_args=_default_args,
    tags=["upm", "pipeline"],
    doc_md=__doc__,
) as dag:
    fase1 = BashOperator(
        task_id="01_find_ids",
        bash_command=f"cd {_PROJECT} && {_UV} 01_find_ids.py",
        doc_md="Extracts researcher IDs from the UPM portal and uploads `researchers.csv` to MinIO.",
    )

    fase2 = BashOperator(
        task_id="02_get_pdis_data",
        bash_command=f"cd {_PROJECT} && {_UV} 02_get_pdis_data.py",
        retries=2,
        retry_delay=timedelta(minutes=5),
        doc_md="Downloads profile and activity HTML for each researcher (async, concurrency=10, exponential backoff).",
    )

    fase3 = BashOperator(
        task_id="03_create_pdis",
        bash_command=f"cd {_PROJECT} && {_UV} 03_create_pdis.py",
        doc_md="Parses HTML files and builds the graph in Memgraph (nodes: Pdi, Publication, Position, Subject, LastYearProject).",
    )

    fase4 = BashOperator(
        task_id="04_enrich_profile",
        bash_command=f"cd {_PROJECT} && {_UV} 04_enrich_profile.py",
        retries=2,
        retry_delay=timedelta(minutes=15),
        doc_md="Enriches each researcher with Claude (keywords, summaries, UNESCO categories). Output: enrichment.json in MinIO.",
    )

    fase5 = BashOperator(
        task_id="05_create_keywords",
        bash_command=f"cd {_PROJECT} && {_UV} 05_create_keywords.py --recreate",
        doc_md="Indexes keywords in Qdrant (sparse BM42 collection `researcher_keywords`). --recreate rebuilds the collection so schema/metadata refresh on every scheduled run.",
    )

    fase6 = BashOperator(
        task_id="06_create_summaries",
        bash_command=f"cd {_PROJECT} && {_UV} 06_create_summaries.py --recreate",
        doc_md="Generates rolling biographies with Ollama and indexes them in Qdrant (dense collection `researcher_summaries`). --recreate rebuilds the collection so the named vector, payload indexes and metadata refresh on every scheduled run.",
    )

    fase1 >> fase2 >> fase3 >> fase4 >> [fase5, fase6]
