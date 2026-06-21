# Evaluation — UPM Match recommendation quality

Módulo para medir la calidad de las recomendaciones de director/a de UPM Match.
Compara **cinco modos de búsqueda** (un control léxico + los cuatro modos del
sistema) contra la verdad conocida: el director real de cada TFG/TFM.

## Idea general

Hay **dos conjuntos de referencia** y la consulta puede salir de cualquiera de ellos
(`--source`):

- **`--source gold`** — usa directamente la `description` del trabajo como consulta.
  No necesita el silver standard, así que **se puede lanzar primero** para tener
  resultados cuanto antes.
- **`--source silver`** — usa la propuesta "estilo alumno" generada por LLMs (más
  realista respecto a lo que escribiría un estudiante). Requiere `silver_standard.json`.

```
gold_standard_tfg.json ┐
gold_standard_tfm.json ┘─┬─────────────────────────────────────────────► (--source gold)
        │                │ director real, año, área, tipo
        │                └─(3 LLMs: generan + juzgan)─► silver_standard.json ─► (--source silver)
        ▼                                                       │ propuesta estilo alumno
        └──────────►  UPM Match (5 modos de búsqueda)  ◄────────┘
                                   │
                                   ▼
                     métricas de ranking (single-relevant)
                     + tiempos (LLM vs retrieval)
                     desglosadas por año / área / tipo
```

### 1. Gold standard (`gold_standard_tfg.json`, `gold_standard_tfm.json`)

500 registros de TFG/TFM curados **a mano** (250 + 250), proporcionales al tamaño
de cada escuela (ETSII 79, ETSIT 77, ETSISI 57, ETSIST 37 por tipo) y con un máximo
de 3 trabajos por director. Esquema de cada registro:

| campo            | descripción                                                |
|------------------|------------------------------------------------------------|
| `id`             | índice (string) por fichero, p. ej. `gold_tfg_0001` (la clave estable es `_project_id`)|
| `description`    | título + abstract + keywords del trabajo                   |
| `year`           | año académico                                              |
| `director`       | director real, como `researcher_id` (`pdi_<n>`)            |
| `study_type`     | `"TFG"` o `"TFM"`                                          |
| `knowledge_area` | escuela: `ETSII` / `ETSIT` / `ETSISI` / `ETSIST`          |
| `_project_id`    | id único global del trabajo (`tft_<n>`), usado como `gold_id`|
| `_title`/`_keywords` | campos de depuración                                   |

### 2. Silver standard (`silver_standard.json`)

Por cada registro gold, **3 LLMs locales** (Ollama) reescriben la `description` en
una **propuesta estilo alumno**. Las 3 propuestas se puntúan con los **mismos 3
LLMs** (*LLM-as-judge*, 1–5) y se queda la mejor. Generación **reproducible**
(semilla fija `SEED=42`) y **reanudable** (se cachea a disco tras cada registro).

Modelos por defecto: `qwen2.5:7b-instruct`, `llama3.1:latest`, `qwen3:4b`.
Se pueden cambiar con `EVAL_LLMS="modelA,modelB,modelC"`.

### 3. Run de recomendaciones — los 5 modos

| modo               | etiqueta (memoria)        | qué hace                                                              |
|--------------------|---------------------------|----------------------------------------------------------------------|
| `lexical_direct`   | Léxica directa (control)  | texto original → índice BM42 disperso, **sin LLM** (baseline léxico). |
| `keywords`         | Léxica con LLM            | extracción de keywords con LLM → BM42 + RRF.                          |
| `summary_direct`   | Semántica directa         | embebe la consulta → búsqueda densa contra los resúmenes.             |
| `summary_enriched` | Semántica enriquecida     | consulta enriquecida con LLM → búsqueda densa.                        |
| `hybrid`           | Híbrida                   | keywords (sparse) + texto enriquecido (dense), fusionados con RRF.    |

Los cuatro últimos reutilizan el mismo código que la ruta `/api/search`
(`app.agent` + `app.search`). El control `lexical_direct` salta la app y el LLM a
propósito, para aislar la contribución del LLM en la interpretación de la consulta.
De cada modo se captura la lista ordenada de `researcher_id` y se compara con el
director real.

**Universo común.** Por defecto, las consultas se restringen a los directores
presentes en **ambos** índices (keywords BM42 *y* resúmenes densos), para que todos
los modos puedan rankear al director real y la comparación sea justa. Con
`--no-universe-filter` se evalúan todas las consultas (los directores ausentes de un
índice cuentan como fallo en ese modo).

### 4. Métricas, tiempos y análisis

Cada consulta tiene **exactamente un ítem relevante** (el director real), así que se
usan las formas *single-relevant* de las métricas IR estándar:

- **Ranking:** Hit@k, Precision@k, Recall@k, NDCG@k, F1@k para `K = [1,3,5,10,25]`.
- **Globales:** MRR y MAP.
- **Tiempos por consulta:** total (media / mediana / p90) y su desglose en parte
  **LLM** (extracción de keywords / enriquecimiento) y parte **retrieval**
  (embedding + consulta a Qdrant), para reportar el coste de meter el LLM en el bucle.
- **Desglose:** por `study_type`, `knowledge_area` y `year`.

El informe se imprime y se guarda en `<reports>/<source>/<timestamp>/`
(`report.md` + `metrics.json`). Con `--publish-minio` se sube además a MinIO
(ver más abajo).

## Requisitos

- **Qdrant** con las colecciones pobladas: `researcher_keywords` (fase 5) y
  `researcher_summaries` (fase 7). Si la fase 5 falla al cargar enrichments, el
  índice de keywords queda casi vacío y el universo común se reduce a 0; reconstrúyelo
  con `uv run 05_create_keywords.py --recreate`.
- **Ollama** con los modelos de la app (`OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`) para
  evaluar. Los 3 `EVAL_LLMS` solo hacen falta para `build-silver`.

## Ficheros

| fichero                          | contenido                                      |
|----------------------------------|------------------------------------------------|
| `evaluate_recommendations.py`    | harness completo (build-silver / evaluate / selftest) |
| `gold_standard_tfg.json`         | 250 registros ground-truth de TFG              |
| `gold_standard_tfm.json`         | 250 registros ground-truth de TFM              |
| `gold_standard.example.json`     | ejemplo del esquema de un registro             |
| `silver_standard.json`           | (generado) propuestas sintéticas elegidas      |
| `cache/<source>/`, `reports/<source>/` | (generados) caché de runs e informes, separados por fuente |

> La caché y los informes se **namespacean por fuente** (`cache/gold/` vs
> `cache/silver/`), de modo que un run de gold nunca pisa al de silver. Sus rutas son
> configurables con `EVAL_CACHE_DIR` y `EVAL_REPORTS_DIR` (útil cuando el directorio
> del proyecto no es escribible, p. ej. dentro del contenedor de Airflow).

## Uso (desde el host)

> Las dependencias (Qdrant + Ollama) deben estar accesibles con las mismas
> variables que la app. Al ejecutar **desde el host** (no dentro de docker),
> apunta Ollama al puerto publicado: `OLLAMA_HOST=http://localhost:11434`.
> Qdrant ya resuelve a `localhost:6333`.

```bash
# 0. Comprobar la aritmética de las métricas (sin servicios)
uv run --module evaluation.evaluate_recommendations selftest

# 1. (Opcional) Generar el silver standard a partir del gold (3 LLMs). Reanudable.
OLLAMA_HOST=http://localhost:11434 \
  uv run --module evaluation.evaluate_recommendations build-silver
#   --limit N   para generar solo N registros nuevos (pruebas)

# 2. Evaluar. --source gold no necesita silver y se puede lanzar primero.
OLLAMA_HOST=http://localhost:11434 \
  uv run --module evaluation.evaluate_recommendations evaluate --source gold
#   --source silver           usa las propuestas estilo alumno (requiere build-silver)
#   --modes keywords hybrid   para un subconjunto de modos
#   --limit N                 para evaluar solo N registros
#   --no-cache                ignora la caché de runs y recalcula
#   --no-universe-filter      evalúa todas las consultas (sin restringir al universo común)
#   --publish-minio           sube el informe + rankings a MinIO
```

## Ejecución en el servidor (Airflow)

Como Qdrant/Ollama/datos viven en el servidor y el host de Airflow no tiene `uv`,
la evaluación se lanza con el DAG [`upm_match_evaluation`](../infra/airflow/dags/upm_match_eval.py),
que corre **dentro del contenedor de Airflow** (ya trae `uv`, el proyecto montado y
red a Qdrant/Ollama). No hace falta instalar nada ni reconstruir contenedores: el
código se lee del proyecto montado y los DAGs se reescanean solos.

- Tareas: `selftest` → `evaluate`.
- Parámetro `source` (`gold` por defecto / `silver`), elegible al disparar el DAG.
- **Reanudable:** la caché por modo se escribe en `/home/airflow/upm_eval/` (escribible
  por el contenedor) vía `EVAL_CACHE_DIR`/`EVAL_REPORTS_DIR`; un reintento retoma desde
  la última consulta completada.
- Los resultados se publican a MinIO con `--publish-minio`.

### Recuperar los resultados desde MinIO

El informe se sube al bucket `MINIO_EVAL_BUCKET` (por defecto `tfm-evaluation`, se
crea solo) bajo `<source>/<timestamp>/` y un puntero sobrescrito `<source>/latest/`:

```bash
# report.md + metrics.json + rankings crudos por modo (cache/)
mc cp upm/tfm-evaluation/gold/latest/report.md   ./report_gold.md
mc cp upm/tfm-evaluation/gold/latest/metrics.json ./metrics_gold.json
```

## Convenciones

- Modelos de datos con **Pydantic** (`GoldRecord`, `ProposalCandidate`,
  `SilverRecord`, `RecommendationResult`, `MetricReport`).
- Generación de propuestas **reproducible** (semilla fija) y **cacheada** en disco,
  para que el silver standard y los runs no cambien entre ejecuciones.
- Caché y reportes **separados por fuente** (`gold` / `silver`) y por modo.
