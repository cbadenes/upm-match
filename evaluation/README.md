# Evaluation — UPM Match recommendation quality

Mini-módulo para medir la calidad de las recomendaciones de director/a de
UPM Match. Está pensado para **ir creciendo**: hoy es un esqueleto
(`evaluate_recommendations.py`) con la metodología y los *stubs* a implementar.

## Idea general

Construimos dos datasets y comparamos las recomendaciones del sistema contra la
verdad conocida (el director real de cada TFG/TFM).

```
gold_standard.json  ──(3 LLMs: generan + juzgan)──►  silver_standard.json
        │                                                     │
        │ director real, año, área, tipo                      │ propuesta estilo alumno
        ▼                                                     ▼
        └────────────►   UPM Match (3 modos de búsqueda)  ◄────┘
                                   │
                                   ▼
                    métricas de clasificación + ranking
                    desglosadas por año / área / tipo
```

### 1. Gold standard (`gold_standard.json`)
100 registros de TFG/TFM curados **a mano**. Esquema (ver
`gold_standard.example.json`):

| campo            | descripción                                            |
|------------------|--------------------------------------------------------|
| `id`             | identificador único del registro                       |
| `description`    | descripción/abstract del trabajo                       |
| `year`           | año académico                                          |
| `director`       | director real, como `researcher_id` (`pdi_<n>`)        |
| `study_type`     | `"TFG"` o `"TFM"` (para el análisis por tipo)          |
| `knowledge_area` | área de conocimiento / departamento (análisis por área)|

> **Nota:** todos los directores estarán indexados. Es la "trampa" del proyecto:
> en [`04_enrich_profile.py`](../04_enrich_profile.py) restringimos la muestra a
> un conjunto de `Position` (departamentos), así que el gold standard debe
> curarse con directores que pertenezcan a ese subconjunto.

### 2. Silver standard (`silver_standard.json`)
Por cada registro gold, **3 LLMs** reescriben la `description` del trabajo en una
**propuesta estilo alumno** (el texto libre que un estudiante escribiría en UPM
Match buscando director/a). Las 3 propuestas se puntúan con los **mismos 3 LLMs**
(*LLM-as-judge*) y se queda la mejor valorada como registro silver, con
referencia a su registro gold.

### 3. Run de recomendaciones
Cada propuesta silver se lanza contra UPM Match en los **tres modos** de búsqueda
(`keywords`, `summary_direct`, `summary_enriched`) y se captura la lista
ordenada de directores recomendados.

### 4. Métricas y análisis
- **Clasificación:** accuracy / precision / recall / F1 (tratando "director en
  top-k" como clase positiva).
- **Ranking / recomendación:** Hit@k, Precision@k, Recall@k, MRR, MAP, NDCG@k
  para `K_VALUES = [1, 3, 5, 10, 25]`.
- **Desglose:** por año, por área de conocimiento y por tipo de estudios, para
  determinar **qué aproximación de búsqueda rinde mejor** y dónde falla cada una.

## Ficheros

| fichero                          | contenido                                   |
|----------------------------------|---------------------------------------------|
| `evaluate_recommendations.py`    | esqueleto + metodología + stubs (TODOs)     |
| `gold_standard.example.json`     | ejemplo del esquema del gold standard       |
| `gold_standard.json`             | (a crear a mano) 100 registros ground-truth |
| `silver_standard.json`           | (generado) propuestas sintéticas elegidas   |

## Uso (cuando esté implementado)

```bash
# 1. Generar el silver standard a partir del gold (3 LLMs)
uv run evaluation/evaluate_recommendations.py build-silver

# 2. Evaluar todos los modos y sacar métricas + desgloses
uv run evaluation/evaluate_recommendations.py evaluate
```

## Estado / TODO

Ver el *checklist* en la cabecera de `evaluate_recommendations.py`. En resumen:
crear el gold a mano, implementar generación+juicio de propuestas (3 LLMs),
el run de recomendaciones contra `app.search`, las funciones de métricas y el
análisis segmentado + reporte.

## Convenciones

- Modelos de datos con **Pydantic** (`GoldRecord`, `ProposalCandidate`,
  `SilverRecord`, `RecommendationResult`, `MetricReport`).
- Generación de propuestas **reproducible** (semilla / caché en disco) para que
  el silver standard no cambie entre ejecuciones.
