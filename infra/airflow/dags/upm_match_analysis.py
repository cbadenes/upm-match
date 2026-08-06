"""UPM Match — re-run ONLY the statistical analysis of an existing campaign.

The measurement campaign (`upm_match_campaign`) takes days: six evaluation
repetitions dominated by LLM inference. The analysis that reads its results,
however, takes minutes and is pure post-processing over the cached per-query
rankings. Whenever the analysis changes (a new table, a corrected metric, a
different cut-off) there is no reason to pay for the campaign again.

This DAG is that shortcut: it mirrors the evaluation bucket and re-runs the
statistics over a campaign that already exists, publishing a fresh analysis
under the same tag. Nothing is evaluated, no cache is wiped and no model is
invoked, so it is safe to trigger as often as needed.

  1. selftest_analysis  — offline math checks of the statistics. Fail-fast.
  2. fetch_for_analysis — mirror the eval bucket (every repetition of every
                          source, plus the productivity map) into a
                          container-writable dir.
  3. analyze            — tables T1-T11 as Markdown + LaTeX, published to MinIO
                          under analysis/<tag>/<timestamp>/ and .../latest/.
  4. fetch_to_repo      — best-effort mirror of the bucket into the mounted repo
                          checkout, so the results land on the host.

Use `upm_match_campaign` when the *evaluation* has to run again; use this one
when only the *reading* of it changes.

Note on T11 (relaxed relevance crediting any real co-supervisor): it needs
Memgraph on the first run, to resolve the full tutor set of every reference
work. The map is then cached in the mirror dir, so later runs need no graph.
When neither the cache nor Memgraph is available the table is simply skipped.
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

# Container-writable local mirror of the eval bucket (same convention as the
# campaign DAG, and deliberately the same path so the tutor-map cache and the
# downloaded runs are reused across both).
_MIRROR_DIR = "/home/airflow/upm_eval/mirror"

_default_args = {
    "owner": "upm-match",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_analysis",
    description="Re-run the statistical analysis of an existing campaign (no evaluation, no LLM).",
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
            description="Tag of the campaign to analyse (REQUIRED): reads the runs "
            "published under <source>/<tag>/ and writes analysis/<tag>/. Must match "
            "the tag the campaign was launched with.",
        ),
    },
    tags=["upm", "evaluation", "analysis"],
    doc_md=__doc__,
) as dag:
    selftest_analysis = BashOperator(
        task_id="selftest_analysis",
        bash_command=f"cd {_PROJECT} && {_UV} --module {_ANALYZE} --selftest",
        retries=0,
        doc_md="Offline checks of the statistics (bootstrap, tests, late fusion). Fail-fast guard.",
    )

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
            "Statistical analysis over the whole campaign (both sources, all "
            "repetitions): tables T1-T11 as Markdown + LaTeX, including the recall "
            "ceiling (T7), equity terciles (T8), the late-fusion sensitivity sweep "
            "with its grid CSVs (T9), the clarification loop (T10) and the "
            "relaxed-relevance re-scoring crediting any real co-supervisor (T11). "
            "Published to MinIO under analysis/<tag>/<timestamp>/ and "
            "analysis/<tag>/latest/."
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
            "mounted repo checkout so everything is on the host afterwards. "
            "evaluation/reports must be writable by the container user "
            "(mkdir -p evaluation/reports && chmod -R a+rwx evaluation/reports)."
        ),
    )

    selftest_analysis >> fetch_for_analysis >> analyze >> fetch_to_repo
