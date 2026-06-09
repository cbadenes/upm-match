"""
04_enrich_profiles.py
=====================
Fase 4 del pipeline UPM Match: enriquecimiento de perfiles de investigadores.

Para cada investigador con datos en Memgraph y HTML crudo en MinIO, este script:
  1. Reconstruye el corpus de texto a partir del subgrafo (publicaciones, asignaturas, posiciones, trabajos tutelados).
  2. Llama al LLM una sola vez por investigador para generar en un único pase:
       - Lista normalizada de keywords (Aproximación 1)
       - Tres resúmenes breves: investigación, docencia y global (Aproximación 2)
       - Clasificación sparse en categorías UNESCO nivel-2 (Aproximación 3)
  3. Genera los embeddings de los tres resúmenes con el modelo de embeddings configurado.
  4. Extrae evidencia estructural de docencia y tutorización directamente del grafo, sin pasar por el LLM (Aproximación 4).
  5. Persiste el documento de enriquecimiento en MinIO bajo researchers/{id}/enrichment.json siguiendo el esquema schema/enrichment_profile.schema.json.
  6. Opcionalmente actualiza propiedades del nodo Researcher en Memgraph.

Lógica de skip-if-already-enriched: si enrichment.json existe en MinIO y su generated_at es posterior al profile.html correspondiente, el investigador se omite salvo que se pase --force.

NOTA DE SESGO - tutorización:
  La subida de TFGs/TFMs al Portal Científico es voluntaria.
  La ausencia de trabajos tutelados en los datos NO implica falta de experiencia; puede reflejar simplemente que el investigador no ha subido sus trabajos. Por ello, la evidencia de tutorización se trata siempre como señal positiva aditiva y nunca penaliza en el scoring.

Dependencias externas:
  - anthropic        : generación LLM (keywords, resúmenes, UNESCO)
  - voyageai         : embeddings (voyage-3 o voyage-3-lite)  ← opción recomendada
                       Alternativa: openai (text-embedding-3-small)
  - gqlalchemy       : consultas a Memgraph
  - dbs/minio.py     : I/O con MinIO (helper interno del proyecto)
  - jsonschema       : validación del documento de salida contra el esquema

Uso:
  python 04_enrich_profiles.py
  python 04_enrich_profiles.py --force          # regenera aunque exista enriquecimiento
  python 04_enrich_profiles.py --ids 12345 67890 # solo esos investigadores
  python 04_enrich_profiles.py --dry-run        # construye corpus pero no llama al LLM ni persiste
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración - ajustar según entorno
# ---------------------------------------------------------------------------
load_dotenv()
MINIO_BUCKET = getenv("MINIO_BUCKET_NAME", "upm-match")
EMBEDDING_MODEL = "voyage-3"  # cambiar a "text-embedding-3-small" si se usa OpenAI
LLM_MODEL = "claude-sonnet-4-20250514"
ENRICHMENT_KEY_TEMPLATE = "researchers/{researcher_id}/enrichment.json"
SCHEMA_PATH = Path("schema/enrichment_profile.schema.json")

# Número máximo de publicaciones incluidas en el corpus del LLM.
# Las más recientes tienen prioridad para no exceder la ventana de contexto.
MAX_PUBLICATIONS_IN_CORPUS = 30
MAX_TUTORING_IN_CORPUS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paso 1 - Carga de datos desde el grafo
# ---------------------------------------------------------------------------


def load_researcher_subgraph(researcher_id: str) -> dict[str, Any]:
    """
    Consulta Memgraph y devuelve el subgrafo del investigador como diccionario
    con las siguientes claves:
      - researcher   : dict con los campos del nodo Researcher
      - publications : lista de dicts (campos Publication)
      - subjects     : lista de dicts (campos Subject + campos Teaching: start_date, end_date)
      - positions    : lista de dicts (campos Position + campos Works: role, since)
      - tutoring     : lista de dicts (campos Publication de TFGs/TFMs/tesis)

    TODO: implementar con el driver de Memgraph del proyecto (gqlalchemy o neo4j-driver).
    Las queries Cypher de referencia se incluyen como comentarios inline.
    """

    # -- Researcher base -------------------------------------------------------
    # MATCH (r:Researcher {id: $researcher_id}) RETURN r
    researcher: dict[str, Any] = {}

    # -- Publicaciones (Authorship) --------------------------------------------
    # MATCH (r:Researcher {id: $researcher_id})-[:Authorship]->(p:Publication)
    # WHERE p.type NOT IN ['TFG', 'TFM', 'Tesis']
    # RETURN p ORDER BY p.year DESC LIMIT $max_pubs
    publications: list[dict] = []

    # -- Asignaturas (Teaching) ------------------------------------------------
    # MATCH (r:Researcher {id: $researcher_id})-[t:Teaching]->(s:Subject)
    # RETURN s, t.start_date, t.end_date
    subjects: list[dict] = []

    # -- Posiciones (Works) ---------------------------------------------------
    # MATCH (r:Researcher {id: $researcher_id})-[w:Works]->(pos:Position)
    # RETURN pos, w.role, w.since
    positions: list[dict] = []

    # -- Trabajos tutelados ---------------------------------------------------
    # MATCH (r:Researcher {id: $researcher_id})-[:Authorship]->(p:Publication)
    # WHERE p.type IN ['TFG', 'TFM', 'Tesis']
    # RETURN p ORDER BY p.year DESC LIMIT $max_tutoring
    #
    # NOTA: cuando se implemente la carga de TFGs/TFMs en el pipeline principal,
    # estas queries serán funcionales. Mientras tanto devuelve lista vacía.
    tutoring: list[dict] = []

    return {
        "researcher": researcher,
        "publications": publications,
        "subjects": subjects,
        "positions": positions,
        "tutoring": tutoring,
    }


# ---------------------------------------------------------------------------
# Paso 2 - Construcción del corpus de texto
# ---------------------------------------------------------------------------


def build_corpus(subgraph: dict[str, Any]) -> dict[str, list[str]]:
    """
    Agrega y limpia el texto del subgrafo. Devuelve el objeto `corpus` del esquema.
    No llama a ningún servicio externo; es transformación pura sobre los datos del grafo.
    """
    pubs = subgraph["publications"]
    subjects = subgraph["subjects"]
    positions = subgraph["positions"]
    tutoring = subgraph["tutoring"]
    researcher = subgraph["researcher"]

    def clean(value: str | None) -> str | None:
        """Elimina espacios redundantes y devuelve None si el resultado es vacío."""
        if not value:
            return None
        cleaned = " ".join(value.split())
        return cleaned if cleaned else None

    def flatten_keywords(items: list[dict], field: str = "keywords") -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            for kw in item.get(field) or []:
                normalized = kw.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    result.append(normalized)
        return result

    return {
        "publication_titles": [t for p in pubs if (t := clean(p.get("title")))],
        "publication_descriptions": [
            d for p in pubs if (d := clean(p.get("description")))
        ],
        "publication_keywords": flatten_keywords(pubs),
        "researcher_topics": [
            t.strip() for t in (researcher.get("topics") or []) if t.strip()
        ],
        "subject_titles": [t for s in subjects if (t := clean(s.get("title")))],
        "position_labels": [
            label
            for pos in positions
            if (
                label := clean(
                    " ".join(filter(None, [pos.get("name"), pos.get("entity")]))
                )
            )
        ],
        "tutoring_titles": [t for w in tutoring if (t := clean(w.get("title")))],
        "tutoring_descriptions": [
            d for w in tutoring if (d := clean(w.get("description")))
        ],
        "tutoring_keywords": flatten_keywords(tutoring),
    }


# ---------------------------------------------------------------------------
# Paso 3 - Llamada LLM: keywords + resúmenes + UNESCO en un único pase
# ---------------------------------------------------------------------------


def build_llm_prompt(researcher: dict, corpus: dict[str, list[str]]) -> str:
    """
    Construye el prompt para el LLM a partir del corpus.
    El prompt solicita una única respuesta JSON estructurada con las tres
    representaciones (keywords, summaries, unesco_categories) para minimizar
    el número de llamadas a la API.
    """
    name = researcher.get("full_name", "Unknown researcher")

    # Truncar para no exceder contexto en investigadores muy prolíficos
    pub_titles_text = "\n".join(
        f"- {t}" for t in corpus["publication_titles"][:MAX_PUBLICATIONS_IN_CORPUS]
    )
    pub_desc_text = "\n".join(f"- {d}" for d in corpus["publication_descriptions"][:15])
    pub_kw_text = ", ".join(corpus["publication_keywords"][:50])
    topics_text = ", ".join(corpus["researcher_topics"])
    subjects_text = "\n".join(f"- {s}" for s in corpus["subject_titles"])
    positions_text = "\n".join(f"- {p}" for p in corpus["position_labels"])
    tutoring_text = "\n".join(
        f"- {t}" for t in corpus["tutoring_titles"][:MAX_TUTORING_IN_CORPUS]
    )

    return f"""You are analyzing the academic profile of a researcher at Universidad Politécnica de Madrid (UPM).
Your task is to produce a structured JSON enrichment profile based on the data provided.

Researcher: {name}

=== RAW DATA ===

Publication titles:
{pub_titles_text or "(none)"}

Publication abstracts/descriptions:
{pub_desc_text or "(none)"}

Publication keywords (from portal):
{pub_kw_text or "(none)"}

Researcher declared topics:
{topics_text or "(none)"}

Teaching subjects:
{subjects_text or "(none)"}

Positions held:
{positions_text or "(none)"}

Supervised works (TFG/TFM/Thesis) registered in portal:
{tutoring_text or "(none - may reflect voluntary upload policy, not absence of supervision activity)"}

=== TASK ===

Return ONLY a valid JSON object with exactly these three keys. No markdown, no preamble.

{{
  "keywords": {{
    "terms": [<list of 15-30 normalized lowercase keywords or short phrases in the predominant language of the data>],
    "language": "<es|en|mixed>"
  }},
  "summaries": {{
    "research": "<2-3 sentence summary of research activity, main topics, and publication profile>",
    "teaching": "<2-3 sentence summary of teaching activity and supervised works if any>",
    "global": "<3-4 sentence integrated summary suitable for matching with student queries. Write in the same language as most of the data.>"
  }},
  "unesco_categories": [
    {{
      "code": "<UNESCO level-2 code, 4 digits>",
      "label": "<official English label>",
      "score": <relevance 0.0-1.0>
    }}
  ]
}}

For unesco_categories: include 3-8 categories, level-2 (4-digit codes preferred). Use the UNESCO Nomenclature for Fields of Science and Technology. Only include categories with score >= 0.15. Scores should reflect relative importance; they need not sum to 1.
"""


def call_llm(prompt: str, client: anthropic.Anthropic) -> dict[str, Any]:
    """
    Llama al LLM y parsea la respuesta como JSON.
    Lanza ValueError si la respuesta no es JSON válido.
    """
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Eliminar posibles delimitadores de bloque de código si el modelo los añade
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}\n\nRaw response:\n{raw}"
        ) from exc


# ---------------------------------------------------------------------------
# Paso 4 - Embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(summaries: dict[str, str]) -> dict[str, list[float]]:
    """
    Genera embeddings para los tres resúmenes.

    La implementación concreta depende del proveedor configurado en EMBEDDING_MODEL.
    Esta función es un stub que debe completarse con el cliente correspondiente:

      Voyage AI (recomendado):
        import voyageai
        vo = voyageai.Client()
        result = vo.embed([text], model="voyage-3")
        return result.embeddings[0]

      OpenAI (alternativa):
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding

    TODO: implementar según el proveedor elegido.
    """
    log.warning("Embedding generation not yet implemented. Returning empty vectors.")
    return {
        "research": [],
        "teaching": [],
        "global": [],
    }


# ---------------------------------------------------------------------------
# Paso 5 - Evidencia estructural (sin LLM)
# ---------------------------------------------------------------------------


def build_teaching_evidence(subgraph: dict[str, Any]) -> dict[str, Any]:
    """
    Extrae evidencia docente directamente del grafo.
    No llama a ningún servicio externo.
    """
    subjects = subgraph["subjects"]
    now = datetime.now(timezone.utc)

    active, historical, subject_types = [], [], set()
    for s in subjects:
        title = (s.get("title") or "").strip()
        if not title:
            continue
        end_raw = s.get("end_date")
        is_active = end_raw is None or _parse_date(end_raw) >= now
        (active if is_active else historical).append(title)
        if s_type := s.get("type"):
            subject_types.add(s_type)

    return {
        "subject_count": len(subjects),
        "active_subjects": active,
        "historical_subjects": historical,
        "subject_types": list(subject_types),
    }


def build_tutoring_evidence(subgraph: dict[str, Any]) -> dict[str, Any]:
    """
    Agrega evidencia de trabajos tutelados.
    Trata la ausencia de datos como señal neutral, no negativa.
    """
    tutoring = subgraph["tutoring"]

    tfg_count = sum(1 for w in tutoring if _classify_tutoring_type(w) == "TFG")
    tfm_count = sum(1 for w in tutoring if _classify_tutoring_type(w) == "TFM")
    thesis_count = sum(1 for w in tutoring if _classify_tutoring_type(w) == "Tesis")

    # Títulos representativos: los 5 más recientes
    sorted_works = sorted(tutoring, key=lambda w: w.get("year") or 0, reverse=True)
    representative_titles = [
        t for w in sorted_works[:5] if (t := (w.get("title") or "").strip())
    ]

    # Consolidar keywords de todos los trabajos
    seen_kw: set[str] = set()
    topic_keywords: list[str] = []
    for w in tutoring:
        for kw in w.get("keywords") or []:
            normalized = kw.strip().lower()
            if normalized and normalized not in seen_kw:
                seen_kw.add(normalized)
                topic_keywords.append(normalized)

    return {
        "tfg_count": tfg_count,
        "tfm_count": tfm_count,
        "thesis_count": thesis_count,
        "has_tutoring_data": len(tutoring) > 0,
        "representative_titles": representative_titles,
        "topic_keywords": topic_keywords,
    }


def _classify_tutoring_type(work: dict) -> str:
    """Clasifica un trabajo tutelado a partir de su campo type."""
    t = (work.get("type") or "").upper()
    if "TFG" in t or "GRADO" in t:
        return "TFG"
    if "TFM" in t or "MASTER" in t or "MÁSTER" in t:
        return "TFM"
    if "TESIS" in t or "DOCTORAL" in t or "PHD" in t:
        return "Tesis"
    return "Other"


def _parse_date(value: str | None):
    """Intento de parseo de fecha con fallback. Devuelve datetime o datetime.min."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y"):
        try:
            return datetime.strptime(value[: len(fmt)], fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Paso 6 - Ensamblado y persistencia
# ---------------------------------------------------------------------------


def assemble_enrichment_document(
    researcher_id: str,
    corpus: dict,
    llm_output: dict,
    embeddings: dict,
    teaching_evidence: dict,
    tutoring_evidence: dict,
) -> dict[str, Any]:
    """
    Ensambla el documento final siguiendo el esquema enrichment_profile.schema.json.
    """
    return {
        "researcher_id": researcher_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "corpus": corpus,
        "keywords": {
            "terms": llm_output.get("keywords", {}).get("terms", []),
            "source": "llm",
            "language": llm_output.get("keywords", {}).get("language", "unknown"),
        },
        "summaries": llm_output.get(
            "summaries",
            {
                "research": "",
                "teaching": "",
                "global": "",
            },
        ),
        "embeddings": embeddings,
        "unesco_categories": llm_output.get("unesco_categories", []),
        "teaching_evidence": teaching_evidence,
        "tutoring_evidence": tutoring_evidence,
    }


def persist_to_minio(researcher_id: str, document: dict) -> None:
    """
    Persiste el documento de enriquecimiento en MinIO.

    TODO: integrar con dbs/minio.py del proyecto.
    Ruta: researchers/{researcher_id}/enrichment.json
    """
    key = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    log.info("Persisting enrichment document to MinIO: %s/%s", MINIO_BUCKET, key)

    # Ejemplo de llamada con el helper interno del proyecto:
    #   from dbs.minio import get_minio_client
    #   client = get_minio_client()
    #   data = json.dumps(document, ensure_ascii=False, indent=2).encode()
    #   client.put_object(MINIO_BUCKET, key, io.BytesIO(data), len(data),
    #                     content_type="application/json")
    raise NotImplementedError("Integrate with dbs/minio.py")


def enrichment_exists_and_is_fresh(researcher_id: str) -> bool:
    """
    Comprueba si ya existe un enrichment.json en MinIO y si es más reciente que
    el profile.html del investigador. Si es así, el investigador puede omitirse.

    TODO: implementar con dbs/minio.py.
    """
    return False  # Por defecto, no omitir ninguno hasta que esté implementado


def enrich_researcher(
    researcher_id: str, client: anthropic.Anthropic, dry_run: bool = False
) -> None:
    """
    Ejecuta el pipeline completo de enriquecimiento para un investigador.
    """
    log.info("Enriching researcher %s", researcher_id)

    # 1. Carga del subgrafo
    subgraph = load_researcher_subgraph(researcher_id)
    if not subgraph["researcher"]:
        log.warning("Researcher %s not found in graph. Skipping.", researcher_id)
        return

    # 2. Corpus
    corpus = build_corpus(subgraph)

    # Comprobar si hay suficiente texto para enriquecer
    total_text_items = sum(len(v) for v in corpus.values())
    if total_text_items == 0:
        log.warning("Researcher %s has empty corpus. Skipping LLM call.", researcher_id)
        return

    if dry_run:
        log.info(
            "[DRY RUN] Corpus built for %s. %d text items. Skipping LLM.",
            researcher_id,
            total_text_items,
        )
        return

    # 3. LLM: keywords + resúmenes + UNESCO en un pase
    prompt = build_llm_prompt(subgraph["researcher"], corpus)
    llm_output = call_llm(prompt, client)

    # 4. Embeddings
    embeddings = generate_embeddings(llm_output.get("summaries", {}))

    # 5. Evidencia estructural
    teaching_evidence = build_teaching_evidence(subgraph)
    tutoring_evidence = build_tutoring_evidence(subgraph)

    # 6. Ensamblado
    document = assemble_enrichment_document(
        researcher_id=researcher_id,
        corpus=corpus,
        llm_output=llm_output,
        embeddings=embeddings,
        teaching_evidence=teaching_evidence,
        tutoring_evidence=tutoring_evidence,
    )

    # 7. Persistencia
    persist_to_minio(researcher_id, document)
    log.info("Researcher %s enriched successfully.", researcher_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich researcher profiles for UPM Match."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich even if enrichment.json already exists.",
    )
    parser.add_argument(
        "--ids", nargs="+", metavar="ID", help="Enrich only these researcher IDs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build corpus but skip LLM calls and persistence.",
    )
    args = parser.parse_args()

    anthropic_client = anthropic.Anthropic()  # Lee ANTHROPIC_API_KEY del entorno

    # Cargar IDs: o los pasados por argumento, o todos desde MinIO/CSV
    if args.ids:
        researcher_ids = args.ids
    else:
        # TODO: leer desde MinIO bucket tfm / researchers.csv
        # from dbs.minio import read_researchers_csv
        # researcher_ids = read_researchers_csv()["idPersonal"].to_list()
        researcher_ids = []
        log.warning("No researcher IDs loaded. Implement CSV loading from MinIO.")

    log.info("Starting enrichment for %d researchers.", len(researcher_ids))

    skipped, succeeded, failed = 0, 0, 0

    for researcher_id in researcher_ids:
        if not args.force and enrichment_exists_and_is_fresh(researcher_id):
            log.info("Skipping %s (enrichment is fresh).", researcher_id)
            skipped += 1
            continue
        try:
            enrich_researcher(researcher_id, anthropic_client, dry_run=args.dry_run)
            succeeded += 1
        except Exception as exc:
            log.error("Failed to enrich researcher %s: %s", researcher_id, exc)
            failed += 1

    log.info(
        "Enrichment complete. succeeded=%d skipped=%d failed=%d",
        succeeded,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
