"""UPM Match — leave-one-out re-ranking: the graph evidence, with the query's own work hidden.

The graph-grounded re-ranker reads, under each candidate profile, the topics of
the works that candidate supervised. For the query's true supervisor that list
contains the very work the query was derived from, so the evidence carries an
echo of the question being asked (memoria sec:eval-validez).

The `signals-loo` post-process already removes that echo from the *numeric*
signals, off-line and for free. It cannot remove it from the evidence the model
read in its prompt, because that would change the ranking: the only way is to
re-rank with the work masked. That is what this DAG does, through the
`rerank_listwise_kg_excl` mode.

  1. fetch_stage1 — mirror the eval bucket, so stage 1 can be seeded from the
                    campaign's published hybrid rankings even if the working
                    cache is gone.
  2. evaluate     — re-rank with the masked evidence, consuming that same stage 1
                    (fixed stage 1: the comparison against `rerank_listwise_kg`
                    stays paired query by query) and publishing to MinIO.

Cost: one LLM call per window per query, so roughly the same as one repetition of
the graph-grounded condition (a few hours per source). Set `limit` to a small
number for a smoke test before committing to the full run.

Afterwards, trigger `upm_match_analysis` with the same tag: the new mode is a
search condition, so it enters the headline tables and the paired significance
test against the unmasked condition with no further work.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.evaluate_recommendations"
_UV = "uv run"

_CACHE_DIR = "/home/airflow/upm_eval/cache"
_REPORTS_DIR = "/home/airflow/upm_eval/reports"
_MIRROR_DIR = "/home/airflow/upm_eval/mirror"

_MODE = "rerank_listwise_kg_excl"

_default_args = {
    "owner": "upm-match",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_loo",
    description="Re-rank with the query's own supervised work masked (leave-one-out).",
    start_date=datetime(2026, 1, 30),
    schedule=None,  # manual trigger only
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    params={
        "tag": Param(
            "final",
            type="string",
            minLength=1,
            description="Campaign tag whose hybrid rankings feed stage 1, and under "
            "which the new condition is published.",
        ),
        "source": Param(
            "silver",
            type="string",
            enum=["silver", "gold"],
            description="Reference set. Silver is the primary scenario of the memoria.",
        ),
        "limit": Param(
            0,
            type="integer",
            minimum=0,
            description="Evaluate only this many queries (0 = all). Use a small "
            "value first to check the wiring before the full run.",
        ),
    },
    tags=["upm", "evaluation", "leave-one-out"],
    doc_md=__doc__,
) as dag:
    fetch_stage1 = BashOperator(
        task_id="fetch_stage1",
        env={"EVAL_DOWNLOAD_DIR": _MIRROR_DIR},
        append_env=True,
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} fetch-minio",
        doc_md=(
            "Mirrors the eval bucket so the campaign's hybrid rankings are available "
            "as stage 1 even when the container's working cache has been recreated."
        ),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        env={"EVAL_CACHE_DIR": _CACHE_DIR, "EVAL_REPORTS_DIR": _REPORTS_DIR},
        append_env=True,
        bash_command=(
            f"cd {_PROJECT} && {_UV} --module {_MODULE} evaluate "
            "--source {{ params.source }} "
            f"--modes {_MODE} "
            "--tag {{ params.tag }} "
            f"--stage1-from-reports {_MIRROR_DIR} "
            "{% if params.limit %}--limit {{ params.limit }}{% endif %} "
            "--publish-minio"
        ),
        doc_md=(
            "Re-ranks the campaign's hybrid candidates with each query's own "
            "supervised work hidden from its supervisor's evidence. Resumable: the "
            "harness caches per query, so a retry continues instead of restarting."
        ),
    )

    fetch_stage1 >> evaluate
