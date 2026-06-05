"""UPM Match — pipeline DAG principal.

Grafo de dependencias:
  01_find_ids => 02_get_pdis_data => 03_create_pdis => 04_enrich_profile => [05_create_keywords, 06_create_summaries]

Las fases 05 y 06 son independientes entre sí y se ejecutan en paralelo.
Duración estimada del pipeline completo: 3-6 horas.

Para ejecutar solo parte del pipeline selecciona las tareas desde la UI
con "Clear" + "Only Failed" o usa el trigger con parámetros.
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
    description="Pipeline completo: scraping => grafo Memgraph => enriquecimiento LLM => indexación Qdrant",
    start_date=datetime(2026, 1, 30),
    # schedule=None  => sólo ejecución manual
    # Para ejecución automática semanal (domingos 3am): schedule="0 3 * * 0"
    schedule=None,
    catchup=False,
    default_args=_default_args,
    tags=["upm", "pipeline"],
    doc_md=__doc__,
) as dag:
    fase1 = BashOperator(
        task_id="01_find_ids",
        bash_command=f"cd {_PROJECT} && {_UV} 01_find_ids.py",
        doc_md="Extrae IDs de investigadores del portal UPM y sube `researchers.csv` a MinIO.",
    )

    fase2 = BashOperator(
        task_id="02_get_pdis_data",
        bash_command=f"cd {_PROJECT} && {_UV} 02_get_pdis_data.py",
        retries=2,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=3),
        doc_md="Descarga HTML de perfil y actividad de cada investigador (async, concurrencia=10, backoff exponencial).",
    )

    fase3 = BashOperator(
        task_id="03_create_pdis",
        bash_command=f"cd {_PROJECT} && {_UV} 03_create_pdis.py",
        execution_timeout=timedelta(hours=2),
        doc_md="Parsea HTMLs y construye el grafo en Memgraph (nodos Pdi, Publication, Position, Subject, LastYearProject).",
    )

    fase4 = BashOperator(
        task_id="04_enrich_profile",
        bash_command=f"cd {_PROJECT} && {_UV} 04_enrich_profile.py",
        retries=2,
        retry_delay=timedelta(minutes=15),
        execution_timeout=timedelta(hours=4),
        doc_md="Enriquece cada investigador con Claude (keywords, resúmenes, UNESCO). Output: enrichment.json en MinIO.",
    )

    fase5 = BashOperator(
        task_id="05_create_keywords",
        bash_command=f"cd {_PROJECT} && {_UV} 05_create_keywords.py --skip-existing",
        execution_timeout=timedelta(hours=1),
        doc_md="Indexa keywords en Qdrant (colección sparse BM42 `researcher_keywords`). --skip-existing evita re-indexar.",
    )

    fase6 = BashOperator(
        task_id="06_create_summaries",
        bash_command=f"cd {_PROJECT} && {_UV} 06_create_summaries.py",
        execution_timeout=timedelta(hours=3),
        doc_md="Genera biografías rolling con Ollama e indexa en Qdrant (colección densa `researcher_summaries`).",
    )

    fase1 >> fase2 >> fase3 >> fase4 >> [fase5, fase6]
