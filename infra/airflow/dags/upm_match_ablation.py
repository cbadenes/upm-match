"""UPM Match — ablation: raw publication text vs the LLM-written biography.

The dense index represents each researcher by a biography that a local model
writes from their production. That step (phases 6 and 7) is the most expensive
of the whole corpus pipeline, and the benchmark never questions it: it compares
ways of *querying* the index, never the representation being queried.

This DAG runs the missing comparison. It builds a parallel Qdrant collection
whose vectors come from the raw publication text (title plus abstract, newest
first until the embedding context is full) and evaluates it with the
`summary_direct` strategy, which invokes no model at query time and is therefore
deterministic: one run suffices and the only difference against the campaign's
`summary_direct` is what was indexed.

Cost: embeddings only, no generative inference. Roughly a quarter of an hour.

The ablation collection is disposable and never aliased, so production search is
untouched. Two asymmetries belong in the reading of the result and are recorded
in the script's own documentation: the biography condenses the whole trajectory
while the raw text only holds what fits, and only two thirds of publications
carry a usable abstract.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.ablation_rawtext"
_UV = "uv run"
_REPORTS_DIR = "/home/airflow/upm_eval/reports"

with DAG(
    dag_id="upm_match_ablation",
    description="Raw publication text vs generated biography as the dense representation.",
    start_date=datetime(2026, 1, 30),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "upm-match",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
        "email_on_retry": False,
    },
    params={
        "source": Param(
            "silver",
            type="string",
            enum=["silver", "gold"],
            description="Reference set. Silver is the primary scenario of the memoria.",
        ),
        "skip_build": Param(
            False,
            type="boolean",
            description="Reuse the existing ablation collection instead of rebuilding "
            "it (only useful when re-evaluating the other source).",
        ),
    },
    tags=["upm", "evaluation", "ablation"],
    doc_md=__doc__,
) as dag:
    ablate = BashOperator(
        task_id="ablate",
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} "
            "--source {{ params.source }} "
            f"--out {_REPORTS_DIR}/ablation_rawtext_{{{{ params.source }}}}.json "
            "{% if params.skip_build %}--skip-build{% endif %}"
        ),
        doc_md=(
            "Builds the raw-text collection and evaluates it. Prints the metrics and "
            "writes them as JSON, including the build statistics (how many "
            "researchers were truncated and how many publications were dropped), "
            "which is the truncation the ablation has to declare."
        ),
    )
