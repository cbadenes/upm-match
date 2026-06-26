"""UPM Match — main pipeline DAG.

Task dependency graph:
  01_find_ids -> 02_get_pdis_data -> 03_create_pdis -> 04_enrich_profile -> [05_create_keywords, 06_create_summaries -> 07_index_summaries]

Phase 05 (keywords) runs in parallel with the 06 -> 07 chain (per-year summaries -> weighted indexing).
Estimated full pipeline duration: 3-6 hours.

To run only part of the pipeline, select tasks from the UI using
"Clear" + "Only Failed", or use the trigger with parameters.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.models.param import Param
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
    description="Full pipeline: scraping -> Memgraph graph -> profile consolidation -> LLM summaries -> Qdrant indexing",
    start_date=datetime(2026, 1, 30),
    # schedule=None  -> manual trigger only
    # For weekly automatic runs (Sundays at 3am): schedule="0 3 * * 0"
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    params={
        "recreate": Param(
            True,
            type="boolean",
            description="Rebuild the Qdrant collections from scratch (phases 05 and "
            "07): drop+recreate the keywords collection, and build phase 07 into a "
            "fresh versioned collection swapped via alias on success. Leave on for a "
            "full run; turn off to index into the existing collections.",
        ),
        "skip_existing": Param(
            False,
            type="boolean",
            description="Skip researchers already processed in each phase (03 graph, "
            "04 enrichment, 05 keywords, 06 summaries), so a re-run only does the new "
            "ones and avoids repeating the expensive phase-06 LLM work. Off = "
            "reprocess all. NOTE: phase 02 (scraping) has no skip flag and always "
            "re-downloads.",
        ),
    },
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
        bash_command=(
            f"cd {_PROJECT} && {_UV} 03_create_pdis.py"
            "{{ ' --skip-existing' if params.skip_existing else '' }}"
        ),
        doc_md="Parses HTML files and builds the graph in Memgraph (nodes: Pdi, Publication, Position, Subject, LastYearProject).",
    )

    fase4 = BashOperator(
        task_id="04_enrich_profile",
        bash_command=(
            f"cd {_PROJECT} && {_UV} 04_enrich_profile.py"
            "{{ ' --skip-existing' if params.skip_existing else '' }}"
        ),
        retries=2,
        retry_delay=timedelta(minutes=15),
        doc_md="Builds the structured, Pydantic-validated enrichment document for each PDI from the graph (positions, normalised topics, cleaned publications, supervised works), keeping only researchers in the four target schools, and persists enrichment.json to MinIO. No LLM.",
    )

    fase5 = BashOperator(
        task_id="05_create_keywords",
        bash_command=(
            f"cd {_PROJECT} && {_UV} 05_create_keywords.py"
            "{{ ' --recreate' if params.recreate else '' }}"
            "{{ ' --skip-existing' if params.skip_existing else '' }}"
        ),
        doc_md="Indexes keywords in Qdrant (sparse BM42 collection `researcher_keywords`). --recreate (params.recreate, default on) rebuilds the collection so schema/metadata refresh on every scheduled run.",
    )

    fase6 = BashOperator(
        task_id="06_create_summaries",
        bash_command=(
            f"cd {_PROJECT} && {_UV} 06_create_summaries.py"
            "{{ ' --skip-existing' if params.skip_existing else '' }}"
        ),
        doc_md="Generates standalone per-year research summaries + domains with Ollama and stores them as `history` in each enrichment.json in MinIO. No Qdrant.",
    )

    fase7 = BashOperator(
        task_id="07_index_summaries",
        bash_command=(
            f"cd {_PROJECT} && {_UV} 07_index_summaries.py"
            "{{ ' --recreate' if params.recreate else '' }}"
        ),
        retries=0,
        doc_md="Fuses the per-year summaries into a weighted current summary (recent-weighted, Zheng et al.) and indexes one point per PDI in Qdrant (`researcher_summaries`). --recreate builds into a fresh versioned collection and atomically swaps the `researcher_summaries` alias only on a fully successful run, so a failed/restarted run never destroys the live data.",
    )

    fase1 >> fase2 >> fase3 >> fase4 >> [fase5, fase6]
    fase6 >> fase7
