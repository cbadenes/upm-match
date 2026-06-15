# Evaluation — UPM Match recommendation quality

Módulo para medir la calidad de las recomendaciones de director/a de UPM Match.
Compara los cuatro modos de búsqueda del sistema contra la verdad conocida (el
director real de cada TFG/TFM).

## Idea general

```
gold_standard_tfg.json ┐
gold_standard_tfm.json ┘──(3 LLMs: generan + juzgan)──►  silver_standard.json
        │                                                       │
        │ director real, año, área, tipo                        │ propuesta estilo alumno
        ▼                                                       ▼
        └─────►  UPM Match (distintos modos de búsqueda)  ◄─────┘
                                   │
                                   ▼
                     métricas de ranking (single-relevant)
                     desglosadas por año / área / tipo
```

### 1. Gold standard (`gold_standard_tfg.json`, `gold_standard_tfm.json`)

500 registros de TFG/TFM curados **a mano** (250 + 250), proporcionales al tamaño
de cada escuela (ETSII 79, ETSIT 77, ETSISI 57, ETSIST 37 por tipo) y con un máximo
de 3 trabajos por director. Esquema de cada registro:

| campo            | descripción                                                |
|------------------|------------------------------------------------------------|
| `id`             | índice entero por fichero (la clave estable es `_project_id`)|
| `description`    | título + abstract + keywords del trabajo                   |
| `year`           | año académico                                              |
| `director`       | director real, como `researcher_id` (`pdi_<n>`)            |
| `study_type`     | `"TFG"` o `"TFM"`                                          |
| `knowledge_area` | escuela: `ETSII` / `ETSIT` / `ETSISI` / `ETSIST`          |
| `_project_id`    | id único global del trabajo (`tft_<n>`), usado como `gold_id`|
| `_title`/`_keywords` | campos de depuración                                   |

> **Nota:** todos los directores están indexados (la muestra de
> [`04_enrich_profile.py`](../04_enrich_profile.py) se restringe a un conjunto de
> `Position`). `evaluate` avisa de cualquier director no indexado.

### 2. Silver standard (`silver_standard.json`)

Por cada registro gold, **3 LLMs locales** (Ollama) reescriben la `description` en
una **propuesta estilo alumno**. Las 3 propuestas se puntúan con los **mismos 3
LLMs** (*LLM-as-judge*, 1–5) y se queda la mejor. Generación **reproducible**
(semilla fija `SEED=42`) y **reanudable** (se cachea a disco tras cada registro).

Modelos por defecto: `qwen2.5:7b-instruct`, `llama3.1:latest`, `qwen3:4b`.
Se pueden cambiar con `EVAL_LLMS="modelA,modelB,modelC"`.

### 3. Run de recomendaciones

Cada propuesta silver se lanza contra UPM Match en los **cuatro modos**
(`keywords`, `summary_direct`, `summary_enriched`, `hybrid`) reutilizando el mismo
código que la ruta `/api/search` (`app.agent` + `app.search`). Se captura la lista
ordenada de `researcher_id` y se compara con el director real.

### 4. Métricas y análisis

Cada consulta tiene **exactamente un ítem relevante** (el director real), así que se
usan las formas *single-relevant* de las métricas IR estándar:

- **Ranking:** Hit@k, Precision@k, Recall@k, NDCG@k, F1@k para `K = [1,3,5,10,25]`.
- **Globales:** MRR y MAP (que coinciden con un único relevante).
- **Desglose:** por `study_type`, `knowledge_area` y `year`, para ver **qué modo
  rinde mejor** y dónde falla cada uno.

El informe se imprime y se guarda en `evaluation/reports/<timestamp>/`
(`report.md` + `metrics.json`).

## Ficheros

| fichero                          | contenido                                      |
|----------------------------------|------------------------------------------------|
| `evaluate_recommendations.py`    | harness completo (build-silver / evaluate / selftest) |
| `gold_standard_tfg.json`         | 250 registros ground-truth de TFG              |
| `gold_standard_tfm.json`         | 250 registros ground-truth de TFM              |
| `gold_standard.example.json`     | ejemplo del esquema de un registro             |
| `silver_standard.json`           | (generado) propuestas sintéticas elegidas      |
| `cache/`, `reports/`             | (generados) caché de runs e informes           |

## Uso

> Las dependencias (Qdrant + Ollama) deben estar accesibles con las mismas
> variables que la app. Al ejecutar **desde el host** (no dentro de docker),
> apunta Ollama al puerto publicado:
> `OLLAMA_HOST=http://localhost:11434`. Qdrant ya resuelve a `localhost:6333`.

```bash
# 0. Comprobar la aritmética de las métricas (sin servicios)
uv run --module evaluation.evaluate_recommendations selftest

# 1. Generar el silver standard a partir del gold (3 LLMs). Reanudable.
OLLAMA_HOST=http://localhost:11434 \
  uv run --module evaluation.evaluate_recommendations build-silver
#   --limit N   para generar solo N registros nuevos (pruebas)

# 2. Evaluar todos los modos y sacar métricas + desgloses + informe
OLLAMA_HOST=http://localhost:11434 \
  uv run --module evaluation.evaluate_recommendations evaluate
#   --modes keywords hybrid   para un subconjunto de modos
#   --limit N                 para evaluar solo N registros silver
#   --no-cache                para ignorar la caché de runs y recalcular
```

## Convenciones

- Modelos de datos con **Pydantic** (`GoldRecord`, `ProposalCandidate`,
  `SilverRecord`, `RecommendationResult`, `MetricReport`).
- Generación de propuestas **reproducible** (semilla fija) y **cacheada** en disco,
  para que el silver standard y los runs no cambien entre ejecuciones.
