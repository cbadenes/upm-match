"""UPM Match — Silver Standard generation DAG.

Builds the silver standard from the gold standard in two artefacts per study type, written to the mounted project (the host repo checkout) so the gold files stay untouched:
  - silver_candidates_{tfg,tfm}.json: a copy of the gold records plus a `silver` field (one candidate per generator model, each ``{model, proposal, grades:{judge: score}}``), written by the generate/grade steps.
  - silver_standard_{tfg,tfm}.json: the finalized set in the gold schema (description = best-graded proposal), written by the finalize step.

Runs inside the Airflow container (uv + project mounted + Ollama on the network).
Ollama loads a single model at a time, so generation and grading run one model per task, sequentially:

    generate(model1) -> generate(model2) -> generate(model3) -> score(judge1) -> score(judge2) -> score(judge3)

Generation must finish before scoring (each judge grades all three texts).
Each task is resumable: it skips works/candidates already done and flushes to disk often, so a crash/retry resumes from the last completed item.

Models (generators == judges): mistral:7b, llama3.1:8b, gemma3:12b.
They must be pulled in the server's Ollama.

Setup (once on the host, so the container can write into the repo checkout):
    mkdir -p evaluation/silver && chmod -R a+rwx evaluation/silver
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

_PROJECT = "/opt/airflow/project"
_MODULE = "evaluation.evaluate_recommendations"
_UV = "uv run"
_SILVER_DIR = f"{_PROJECT}/evaluation/silver"

# Generator models == judge models: each generated text is graded by all three.
MODELS = ["mistral:7b", "llama3.1:8b", "gemma3:12b"]

_default_args = {
    "owner": "upm-match",
    # Generation/grading are resumable from disk, so a retry resumes where it stopped.
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_silver",
    description="Generate the silver standard (per-model proposals + cross-model grades) from the gold standard.",
    start_date=datetime(2026, 1, 30),
    # schedule=None -> manual trigger only.
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    tags=["upm", "evaluation", "silver"],
    doc_md=__doc__,
) as dag:
    _env = {"SILVER_DIR": _SILVER_DIR}

    # 1-3. Generation: one task per model, each writing its proposal into every work.
    gen_tasks = [
        BashOperator(
            task_id=f"generate_model_{i}",
            env=_env,
            append_env=True,
            bash_command=(
                f"cd {_PROJECT} && {_UV} --module {_MODULE} silver-generate --model {m}"
            ),
            doc_md=f"Generate student-style proposals with `{m}` for every gold work (TFG + TFM).",
        )
        for i, m in enumerate(MODELS, 1)
    ]

    # 4. Scoring: one task per judge, each loading its model once and grading all texts.
    grade_tasks = [
        BashOperator(
            task_id=f"score_model_{i}",
            env=_env,
            append_env=True,
            bash_command=(
                f"cd {_PROJECT} && {_UV} --module {_MODULE} silver-grade --judge {m}"
            ),
            doc_md=f"Score every generated proposal with judge `{m}` (model loaded once).",
        )
        for i, m in enumerate(MODELS, 1)
    ]

    # 5. Finalize: pick the best-graded proposal per work (random tie-break) and write
    #    the final silver standard in the gold schema, ready for `evaluate --source silver`.
    finalize = BashOperator(
        task_id="finalize",
        env=_env,
        append_env=True,
        bash_command=f"cd {_PROJECT} && {_UV} --module {_MODULE} silver-finalize",
        doc_md=(
            "Select the best-graded proposal per work (ties broken at random, "
            "reproducibly) and write evaluation/silver/silver_standard_{tfg,tfm}.json "
            "in the SAME schema as the gold standard (description = chosen proposal). "
            "The rich candidates + grades stay in silver_candidates_{tfg,tfm}.json."
        ),
    )

    # All sequential: generate (per model) -> grade (per judge) -> finalize.
    chain(*gen_tasks, *grade_tasks, finalize)
