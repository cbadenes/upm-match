# UPM Match — Contexto del Proyecto

## Objetivo

Sistema de recomendación inteligente para la Universidad Politécnica de Madrid (UPM) que ayuda a:
1. **Estudiantes** a encontrar directores de TFG/TFM/Tesis adecuados
2. **Investigadores** a detectar posibles colaboraciones interdisciplinares

Fuente de datos: Portal Científico UPM (datos públicos de investigadores/PDIs).

---

## Pipeline de Datos (6 fases)

### Fase 1 — `01_find_ids.py`: Extracción de IDs
- Scrapa endpoint PHP del Portal Científico UPM
- Output: `researchers.csv` con `idPersonal`, `rfSignature`, `rfAdscription`, `rfRoll`
- Sube CSV a MinIO (`tfm/researchers.csv`)

### Fase 2 — `02_get_pdis_data.py`: Descarga HTML
- Por cada investigador descarga 2 ficheros HTML:
  - `profile.html` — página principal del investigador
  - `activity.html` — respuesta AJAX con toda la actividad investigadora
- Async HTTP (concurrencia=10), reintentos con backoff exponencial (3 intentos, 2-10 seg)
- Sube a MinIO: `researchers/{idPersonal}/profile.html` y `activity.html`
- 404s se manejan sin error (algunos perfiles no existen)

### Fase 3 — `03_create_pdis.py`: Extracción y Almacenamiento en Grafo (~1920 líneas)
- **Extrae** datos estructurados del HTML con selectores Scrapy:
  - Datos personales: nombre, email
  - IDs externos: ORCID, Scopus, Google Scholar, ResearchGate, Mendeley, LinkedIn, Dialnet
  - Posiciones actuales: título, entidad, rol, fecha de inicio
  - Topics de investigación: rankeados por frecuencia desde datos JavaScript
  - Publicaciones: título, autores, año, keywords, descripción (limpiada), URLs, agencia validadora
  - Asignaturas impartidas: nombre, tipo (oficial/internacional), fechas
  - Proyectos de último año (TFG/TFM): título, autores, keywords, año, descripción
- **Limpieza y normalización**:
  - Descripciones de publicaciones: 8 regex + 14 marcadores de metadatos eliminados
  - Topics: normalización a minúsculas, pluralización simple
  - Keywords: split y deduplicación preservando orden
  - Fechas: parsing YYYY-MM-DD con strict=False como fallback
- **Almacenamiento en grafo**:
  - Nodos: `Pdi`, `Publication`, `Position`, `Subject`, `LastYearProject`
  - Aristas: `Authorship` (Pdi→Publication), `Works` (Pdi→Position), `Teaching` (Pdi→Subject), `Tutoring` (Pdi→LastYearProject)
  - Backends: Memgraph (por defecto), ArangoDB, JSON local, MinIO

### Fase 4 — `04_enrich_profile.py`: Enriquecimiento con LLM
- Consulta Memgraph para reconstruir el subgrafo del investigador
- **Una sola llamada LLM por investigador** (Claude 3.5 Sonnet):
  - 15-30 keywords normalizadas (en idioma predominante)
  - 3 resúmenes: investigación, docencia, global (2-4 frases cada uno)
  - Clasificación UNESCO nivel 2 (scores 0.0-1.0, sparse)
- **Evidencia estructurada** (sin LLM): docencia activa/histórica, TFG/TFM tutorizados
- **Embeddings**: Voyage AI (voyage-3) o OpenAI (text-embedding-3-small)
- Output: `researchers/{id}/enrichment.json` en MinIO

### Fase 5 — `05_create_keywords.py`: Indexación Sparse en Qdrant
- Lee `enrichment.json` de MinIO por cada investigador
- Embebe los `topics` del investigador con BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`, fastembed)
- Indexa en Qdrant colección `researcher_keywords`, campo sparse `keywords_bm42`
- Payload: `researcher_id`, `name`, `topics`, `since`
- `--skip-existing`: precarga IDs en un scroll paginado

### Fase 6 — `06_create_summaries.py`: Biografía Rolling + Indexación Densa
- Por cada investigador agrupa publicaciones por año y construye una biografía acumulativa:
  - Agno + Ollama (`OLLAMA_SUMMARY_MODEL`, por defecto `qwen2.5:7b-instruct`)
  - Output estructurado Pydantic: `domains` (lista) + `profile` (3-5 frases en inglés)
  - Cada año actualiza los dominios previos sin borrarlos (rolling)
- Embebe el perfil final con Ollama (`OLLAMA_EMBED_MODEL`, por defecto `nomic-embed-text-v2-moe`)
- Indexa en Qdrant colección `researcher_summaries` (vector denso coseno, campo por defecto)
- Payload: `researcher_id`, `name`, `summary`, `topics`, `first_year`, `last_year`, `positions`, `publication_count`, etc.

---

## Aplicación Web — `app/`

Interfaz de búsqueda para estudiantes.
- **Desarrollo local**: `uv run main.py` (Flask dev server, puerto 5000).
- **Docker**: Gunicorn en `upm_match_app`, accesible en `http://localhost/app`. Traefik hace strip de `/app` y añade `X-Forwarded-Prefix: /app`; Flask lee ese header con `ProxyFix(x_prefix=1)` para que `url_for()` genere rutas correctas.

### Tres modos de búsqueda (POST `/api/search`)

| Modo | `mode` | Cómo funciona |
|------|--------|---------------|
| Por keywords | `keywords` | Agno+Ollama extrae keywords → RRF fusion de dos búsquedas BM42 sparse en `researcher_keywords` |
| Por resumen directo | `summary_direct` | Embebe el texto del usuario con `nomic-embed-text-v2-moe` → cosine similarity en `researcher_summaries` |
| Por resumen enriquecido | `summary_enriched` | Agno+Ollama traduce/enriquece el texto → embebe → cosine similarity en `researcher_summaries` |

### Módulos
- `app/__init__.py` — factory `create_app()`
- `app/routes.py` — Blueprint `main`; dispatch por `mode`; helper `_serialise_result`
- `app/agent.py` — `extract_keywords` (KeywordSet) + `enrich_query` (EnrichedQuery), ambos Agno+Ollama
- `app/search.py` — `search_pdis` (sparse) + `search_pdis_by_summary` (densa); `PDIResult` dataclass
- `app/templates/index.html` — UI: selector de modo, sub-modo, tarjetas de resultado adaptativas

### `PDIResult`
`rank`, `researcher_id`, `name`, `score`, `score_pct` (0-100 relativo al top), `matched_topics`, `all_topics`, `since`, `summary`.

---

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Base de datos grafo | Memgraph (principal), ArangoDB (alternativo) |
| Almacenamiento objetos | MinIO |
| Base de datos vectorial | Qdrant |
| LLM local | Ollama (Agno framework) |
| Embeddings nube | Voyage AI (voyage-3) / OpenAI (text-embedding-3-small) |
| Embeddings local | Ollama (`nomic-embed-text-v2-moe`) + fastembed BM42 |
| Web framework | Flask + Gunicorn (producción) |
| HTTP Async | httpx + tenacity |
| Parsing HTML | Scrapy selectors |
| Validación datos | Pydantic 2.0 |
| Logging | loguru |
| Package manager | uv (Python 3.13+) |
| Reverse proxy | Traefik v3 (file provider) |
| Orquestador pipeline | Airflow (LocalExecutor + PostgreSQL) |
| Infraestructura | Docker Compose |

---

## Módulo Vector — `dbs/vector/`

Espeja el patrón de `dbs/graph/`:
- `base.py` — Protocolo `VectorDataBase` + dataclasses `VectorPoint`, `SearchResult`
- `qdrant.py` — `QdrantBackend`: BM42 sparse (lazy) + `search_dense` para colecciones densas
- `__init__.py` — factory `get_vector_db(backend="qdrant")`

Métodos clave de `QdrantBackend`:
- `embed(keywords)` → `(indices, values)` BM42
- `search_by_keyword_sets(collection, sparse_field, general, specific, limit)` — RRF fusion
- `search_dense(collection, vector, limit)` — búsqueda en campo denso por defecto

---

## Modelos de Datos (Pydantic)

### Nodos
- **Pdi** (`pdi_{id}`): id, full_name, first/last_name, email, external_ids, topics
- **Publication** (`pub_{id}`): title, description, year, type, authors, keywords, source, validation_agency, url
- **Position** (`loc_{hash}`): name, entity, role, since
- **Subject** (`subj_{hash}`): name, type, start_date, end_date
- **LastYearProject** (`tft_{id}`): title, description, year, type, authors, keywords

### Aristas
- **Authorship**: Pdi → Publication
- **Works**: Pdi → Position (con role, since)
- **Teaching**: Pdi → Subject (con start_date, end_date)
- **Tutoring**: Pdi → LastYearProject

### Documento de Enriquecimiento (`models/enrichment.py`)
- `ResearcherEnrichmentDocument`: topics, positions, publications, subjects, last_year_projects
- Resúmenes LLM, keywords normalizadas, categorías UNESCO

---

## Estructura de Directorios

```
UPM_Match/
├── 01_find_ids.py              # Fase 1: extracción IDs
├── 02_get_pdis_data.py         # Fase 2: scraping HTML
├── 03_create_pdis.py           # Fase 3: extracción + grafo
├── 04_enrich_profile.py        # Fase 4: enriquecimiento de datos
├── 05_create_keywords.py       # Fase 5: indexación BM42 sparse en Qdrant
├── 06_create_summaries.py      # Fase 6: biografía rolling + indexación densa en Qdrant
├── main.py                     # Entry point Flask (uv run main.py)
├── app/
│   ├── __init__.py            # create_app()
│   ├── routes.py              # /api/search con tres modos
│   ├── agent.py               # extract_keywords + enrich_query (Agno+Ollama)
│   ├── search.py              # PDIResult, search_pdis, search_pdis_by_summary
│   └── templates/
│       └── index.html         # UI: selector de modo + resultados
├── dbs/
│   ├── minio.py               # Wrapper cliente MinIO
│   ├── graph/
│   │   ├── __init__.py        # Factory get_graph_db()
│   │   ├── base.py            # Protocolo GraphDataBase
│   │   ├── memgraph.py        # Implementación Memgraph
│   │   └── arango.py          # Implementación ArangoDB
│   └── vector/
│       ├── __init__.py        # Factory get_vector_db()
│       ├── base.py            # Protocolo VectorDataBase
│       └── qdrant.py          # QdrantBackend (BM42 sparse + dense)
├── models/
│   ├── pdi.py
│   ├── publication.py
│   ├── position.py
│   ├── subject.py
│   ├── last_year_project.py
│   ├── relationships.py
│   └── enrichment.py
├── utils/
│   ├── logging.py             # setup_logging() con loguru
│   ├── researchers.py         # load_researcher_ids*, load_enrichment
│   └── text.py                # normalize_topic, split_publication_keywords, clean_description, choose_*
├── schema/                    # Templates JSON de esquema
├── data/
│   ├── raw/                   # HTML (almacenamiento local)
│   ├── processed/             # JSON output (almacenamiento local)
│   └── researchers.csv        # Lista maestra de investigadores
├── infra/
│   ├── docker-compose.yaml    # Todos los servicios Docker
│   ├── app/
│   │   └── Dockerfile         # Imagen Flask app (python:3.13-slim + uv + gunicorn)
│   ├── airflow/
│   │   ├── Dockerfile         # Imagen Airflow extendida con uv + Python 3.13
│   │   └── dags/
│   │       └── upm_match.py   # DAG del pipeline de 6 fases
│   └── traefik/
│       └── dynamic.yml        # Rutas y middlewares de Traefik (file provider)
├── pyproject.toml             # Dependencias (uv)
└── .env                       # Variables de entorno
```

---

## Estructura MinIO

Tres buckets separados (nombres configurados en `.env`):

```
[MINIO_RAW_BUCKET = tfm-raw]
├── researchers.csv
└── researchers/{id}/
    ├── profile.html
    └── activity.html

[MINIO_PROCESSED_BUCKET = tfm-processed]
└── pdi_{id}.json              # AJAX JSON bruto (fase 3)

[MINIO_CURATED_BUCKET = tfm-curated]
└── researchers/{id}/
    └── enrichment.json        # Enriquecimiento LLM (fase 4)
```

---

## Estado Actual del Desarrollo

### Completado
- Fases 1-6: pipeline completo funcionando end-to-end con datos reales
- Todos los modelos Pydantic
- Backends Memgraph y ArangoDB
- Integración MinIO (tres buckets: raw, processed, curated)
- Módulo `dbs/vector/` con QdrantBackend (sparse BM42 + dense cosine)
- Colecciones Qdrant pobladas: `researcher_keywords` (sparse) y `researcher_summaries` (densa)
- Docker Compose con todos los servicios: Memgraph, MinIO, Qdrant, Ollama, Lab, Traefik, Airflow, PostgreSQL, Flask app
- Airflow DAG `upm_match_pipeline` orquestando las 6 fases (05 y 06 en paralelo)
- Aplicación web Flask con tres modos de búsqueda (keywords, summary_direct, summary_enriched)
- App containerizada con Gunicorn, servida en `http://localhost/app` vía Traefik
- Módulo `utils/text.py` con normalización de topics/keywords y limpieza de abstracts compartida entre fases

### No Implementado Aún
- Detección de colaboraciones entre investigadores
- Combinación de resultados sparse + dense (hybrid search)
- Scoring con señales adicionales (actividad reciente, carga de tutorización, etc.)

---

## Variables de Entorno Relevantes

```env
MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, MINIO_PORT, MINIO_RAW_BUCKET, MINIO_PROCESSED_BUCKET, MINIO_CURATED_BUCKET
MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD
QDRANT_PORT, QDRANT_API_KEY, QDRANT_HOST
OLLAMA_PORT=11434, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_SUMMARY_MODEL, OLLAMA_EMBED_MODEL
```

---

## Notas de Implementación

- **Deduplicación de posiciones**: hash SHA1(title+entity)
- **Deduplicación de asignaturas**: hash SHA1(name+type)
- **Limpieza de descripciones**: 8 regex + 14 marcadores de metadatos eliminados
- **Contexto LLM**: máx. 30 publicaciones + 20 tutorizaciones para caber en la ventana de contexto
- **Tutorización**: ausencia de datos TFG/TFM se trata como neutral (política de subida voluntaria), nunca penaliza
- **Topics**: extraídos del objeto JavaScript `data1` del portal UPM, normalizados a minúsculas
- **Agno + Ollama**: los errores de conexión/modelo llegan como string en `response.content` en vez de excepción — todos los agentes comprueban `isinstance(response.content, str)` explícitamente
- **score_pct**: normalización lineal relativa al primer resultado (0-100), no es una escala absoluta ni comparable entre modos de búsqueda
