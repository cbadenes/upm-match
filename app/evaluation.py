"""Evaluation questionnaire for UPM Match (PDIs and, later, students).

The questionnaire is reachable by URL only (never linked from the main UI). It
collects qualitative feedback about the generated researcher profiles and the
recommendation system, and stores one JSON document per submission in MinIO.

GDPR / proportionality:
  - No identifying data (name, e-mail, researcher id) is ever stored with the
    responses. Only the answers, the generated *stimulus* the respondent saw
    (summary, keywords, per-year summaries) and non-identifying metadata are
    persisted, so the data stays useful for aggregate academic analysis.
  - The selected researcher id lives only in the signed session cookie while the
    form is being filled, to render the right profile, and is dropped on submit.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import datetime, timezone

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from loguru import logger

from dbs.vector import QdrantBackend, get_vector_db

eval_bp = Blueprint("eval", __name__)

# Data sources

SUMMARY_COLLECTION = "researcher_summaries"
SUMMARY_DENSE_VECTOR = "dense"
# Number of similar PDIs (including the seed) that form one recommendation case.
CASE_GROUP_SIZE = 10
# Enrichment document layout in MinIO (see utils.researchers.ENRICHMENT_KEY_TEMPLATE).
_ENRICHMENT_KEY_TEMPLATE = "researchers/{researcher_id}/enrichment.json"
PORTAL_URL_TEMPLATE = "https://portalcientifico.upm.es/es/ipublic/researcher/{id}"
PORTAL_ITEM_URL_TEMPLATE = "https://portalcientifico.upm.es/es/ipublic/item/{id}"

# Years shown in the temporal-evolution section (most recent first at render time).
EVOLUTION_YEARS = (2023, 2024, 2025, 2026)

# MinIO bucket that receives the survey responses. Auto-created on first upload.
EVAL_BUCKET = os.environ.get("MINIO_EVAL_BUCKET", "upm-match-evaluations")

FORM_VERSION = "pdi-v3"
# Bumped when the shape of the stored record (metadata/answers layout) changes,
# so later analysis can branch on it independently of the questionnaire version.
RECORD_SCHEMA = "1"
SCALE_MIN, SCALE_MAX = 1, 5
SCALE = f"{SCALE_MIN}-{SCALE_MAX}"

_db: QdrantBackend = get_vector_db()

# Cache the PDI directory (id + name) for the searchable selector.
# The set of indexed researchers only changes when the pipeline reindexes, so a lazy cache is plenty and avoids a full Qdrant scan on every consent-page load.
_pdi_cache: list[dict] | None = None


# Questionnaire schema (single source of truth for render + storage).
# Each Likert item carries a polarity so the analysis can reverse-code the negatively worded statements.
# Open questions are stored verbatim.

# Block 1 — profile validation (all positive, 1=disagree ... 5=agree).
SECTION2_LIKERT = [
    (
        "s2_repr_actual",
        "El perfil generado representa adecuadamente mi actividad investigadora actual.",
        "positive",
    ),
    (
        "s2_cubre",
        "El perfil cubre correctamente mis principales líneas de investigación.",
        "positive",
    ),
    (
        "s2_temas",
        "Los temas atribuidos en el perfil reflejan realmente mi actividad investigadora.",
        "positive",
    ),
    (
        "s2_distingue",
        "El perfil distingue adecuadamente entre líneas principales y líneas secundarias o puntuales.",
        "positive",
    ),
    (
        "s2_util_estudiante",
        "El perfil sería útil para que un estudiante valore si puedo tutorizar un TFG/TFM sobre sus intereses.",
        "positive",
    ),
    (
        "s2_claridad",
        "El perfil está redactado de forma clara y comprensible.",
        "positive",
    ),
]
SECTION2_OPEN = [
    ("s2o_correcciones", "¿Qué corregiría o añadiría en el perfil generado?"),
]

# Block 2 — knowledge-domain relevance: importance level per domain (0-3), preselected at 2.
KEYWORD_LEVELS = [
    (0, "No relacionada"),
    (1, "Relación puntual o secundaria"),
    (2, "Línea relevante"),
    (3, "Línea principal para recomendación de TFG/TFM"),
]
KEYWORD_DEFAULT_LEVEL = 2

# Block 4 — system utility (kept short).
SECTION6_LIKERT = [
    (
        "s6_ayuda_identificar",
        "El sistema podría ayudar a estudiantes a identificar posibles tutores de TFG/TFM.",
        "positive",
    ),
    (
        "s6_descubrir_menos_visibles",
        "El sistema podría ayudar a descubrir PDI adecuados pero menos visibles para los estudiantes.",
        "positive",
    ),
    (
        "s6_solo_apoyo",
        "El sistema debería usarse solo como apoyo a la decisión, no como mecanismo automático de asignación.",
        "neutral",
    ),
]
SECTION6_OPEN = [
    (
        "s6o_info_adicional",
        "¿Qué información adicional debería mostrar el sistema para que la recomendación fuera más fiable?",
    ),
    ("s6o_riesgos", "¿Qué riesgos ve en el uso de este tipo de sistema?"),
    ("s6o_mejoras", "¿Qué mejoras considera prioritarias?"),
]

# Bundle passed to the template so render and storage never drift.
QUESTIONS = {
    "section2_likert": SECTION2_LIKERT,
    "section2_open": SECTION2_OPEN,
    "keyword_levels": KEYWORD_LEVELS,
    "keyword_default_level": KEYWORD_DEFAULT_LEVEL,
    "section6_likert": SECTION6_LIKERT,
    "section6_open": SECTION6_OPEN,
}

# Block 3 — "Evaluación de un caso de recomendación". One case auto-generated on load.
# Number of recommended PDIs shown (top-N).
CASE_TOP_N = 5
SECTION5_LIKERT = [
    (
        "plausible",
        "La consulta generada representa un caso plausible de búsqueda de tutor/a de TFG/TFM.",
        "positive",
    ),
    (
        "conjunto",
        "El conjunto de PDI recomendados es razonable para esta consulta.",
        "positive",
    ),
    (
        "orden",
        "El orden relativo de los PDI recomendados es razonable.",
        "positive",
    ),
    (
        "no_omite",
        "No echo en falta ningún PDI claramente relevante entre las primeras recomendaciones.",
        "positive",
    ),
]
# Fifth item: text depends on whether the evaluated PDI appears in the shown ranking.
SECTION5_POSITION = {
    "id": "posicion",
    "in_top": "Mi posición en el ranking es razonable para esta consulta.",
    "not_in_top": "Considero que yo debería aparecer entre las primeras recomendaciones para esta consulta.",
}
SECTION5_OPINION = (
    "opinion",
    "¿Algún comentario sobre esta consulta o sus recomendaciones?",
)


# --- Student questionnaire schema (see docs/cuestionario_alumnos.md) ---------
# Feedback about the app itself, not a student profile. Single page, incremental
# intake shown/hidden client-side. No identifying data stored.

STUDENT_FORM_VERSION = "student-v3"

SCHOOLS = ["ETSIST", "ETSISI", "ETSIT", "ETSII"]
LEVELS = [
    ("grado", "Grado"),
    ("master", "Máster"),
    ("tesis", "Doctorado / Tesis"),
]
USED_OPTIONS = [
    ("yes_useful", "Sí, y encontré resultados útiles"),
    ("yes_unconvincing", "Sí, pero no me convencieron"),
    ("no", "No, aún no la he usado"),
]

STUDENT_S1_LIKERT = [
    ("st1_facil", "La aplicación fue fácil de usar.", "positive"),
    (
        "st1_relevantes",
        "Las recomendaciones fueron relevantes para mi tema de interés.",
        "positive",
    ),
    (
        "st1_clara",
        "La información mostrada de cada profesor/a fue clara.",
        "positive",
    ),
    ("st1_entendi", "Entendí por qué se me recomendaba a cada profesor/a.", "positive"),
    (
        "st1_encontre",
        "Encontré al menos un posible tutor/a que encajaba con lo que buscaba.",
        "positive",
    ),
]
STUDENT_S1_OPEN = [
    ("st1o_experiencia", "¿Qué mejorarías de la experiencia de uso?"),
]

STUDENT_S2_MULTI = (
    "st2_info",
    "¿Qué información te resultaría más útil al ver una recomendación? (elige hasta 3)",
    [
        ("resumen", "Resumen del profesor"),
        ("temas", "Temas coincidentes"),
        ("afinidad", "Puntuación de afinidad"),
        ("publicaciones", "Publicaciones recientes"),
        ("proyectos", "Proyectos"),
        ("tfg_dirigidos", "Trabajos TFG/TFM dirigidos anteriormente"),
        ("contacto", "Forma de contacto o disponibilidad para tutorizar"),
        ("portal", "Enlace al Portal Científico"),
    ],
)
STUDENT_S2_MULTI_MAX = 3
STUDENT_S2_LIKERT = [
    (
        "st2_otra_escuela",
        "Me resultaría útil que me recomendara profesores de otra escuela, antes que de la mía, si encajan con mi tema.",
        "positive",
    ),
    (
        "st2_jovenes",
        "Me resultaría útil que el sistema incluyera investigadores jóvenes o perfiles emergentes si encajan bien con mi tema.",
        "positive",
    ),
    (
        "st2_apoyo",
        "Confiaría en el sistema como apoyo para decidir tutor (no como decisión final).",
        "positive",
    ),
    (
        "st2_explicacion",
        "Necesitaría ver por qué se recomienda cada profesor/a para confiar en el sistema.",
        "neutral",
    ),
]
STUDENT_S2_SINGLE = (
    "st2_prioridad",
    "¿Qué preferirías que priorice el sistema?",
    [
        ("afinidad", "Afinidad temática pura"),
        ("cercania", "Cercanía académica: misma escuela, titulación o profesores conocidos"),
        ("equilibrio", "Un equilibrio de ambas"),
        ("no_claro", "No lo tengo claro"),
    ],
)

STUDENT_S3_LIKERT = [
    ("st3_tiempo", "Un sistema así me ahorraría tiempo al buscar tutor/a.", "positive"),
    ("st3_descubrir", "Me ayudaría a descubrir tutores/as que no conocía.", "positive"),
    (
        "st3_usaria",
        "Lo usaría si estuviera disponible para mi TFG/TFM/Tesis.",
        "positive",
    ),
    (
        "st3_confianza",
        "Me daría más confianza para contactar con un profesor/a.",
        "positive",
    ),
]
STUDENT_S4_OPEN = [
    (
        "st4_mejoras",
        "¿Qué mejorarías o qué información adicional te gustaría ver en UPM Match?",
    ),
]

STUDENT_QUESTIONS = {
    "s1_likert": STUDENT_S1_LIKERT,
    "s1_open": STUDENT_S1_OPEN,
    "s2_multi": STUDENT_S2_MULTI,
    "s2_multi_max": STUDENT_S2_MULTI_MAX,
    "s2_likert": STUDENT_S2_LIKERT,
    "s2_single": STUDENT_S2_SINGLE,
    "s3_likert": STUDENT_S3_LIKERT,
    "s4_open": STUDENT_S4_OPEN,
}
STUDENT_LEVEL_VALUES = {value for value, _ in LEVELS}
STUDENT_USED_VALUES = {value for value, _ in USED_OPTIONS}


# Helpers


def _summaries_collection() -> str:
    return _db.latest_collection(SUMMARY_COLLECTION) or SUMMARY_COLLECTION


def _numeric_id(value) -> int | None:
    """Extract the numeric portal id from an int or a string like 'pdi_308607'."""
    if value is None:
        return None
    match = re.search(r"(\d{3,})", str(value))
    return int(match.group(1)) if match else None


def list_pdis() -> list[dict]:
    """Return [{'id': <numeric portal id>, 'name': <full name>}] for every indexed researcher, sorted by name. Cached after the first scan."""
    global _pdi_cache
    if _pdi_cache is not None:
        return _pdi_cache

    try:
        rows = _db.scroll_payloads(
            _summaries_collection(), payload_fields=["researcher_id", "name"]
        )
    except Exception:  # noqa: BLE001 — el formulario no debe romperse si Qdrant falla
        logger.exception("Could not scan researchers for the PDI selector")
        return []

    pdis: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        rid = _numeric_id(payload.get("researcher_id")) or _numeric_id(row.get("id"))
        name = (payload.get("name") or "").strip()
        if rid and name:
            pdis.append({"id": rid, "name": name})

    pdis.sort(key=lambda p: p["name"].lower())
    _pdi_cache = pdis
    logger.info("Loaded {} PDIs for the evaluation selector", len(pdis))
    return pdis


# In-memory cache of Spanish translations, keyed by the (English) summary text.
# Per-process / per-worker; cleared on restart (acceptable for the survey window).
_summary_es_cache: dict[str, str] = {}


def _translate_summary(summary: str) -> str | None:
    """Translate the English summary to Spanish with the local LLM (cached).

    Returns None if translation fails, so the UI can fall back to the original.
    """
    summary = (summary or "").strip()
    if not summary:
        return None
    if summary in _summary_es_cache:
        return _summary_es_cache[summary]
    try:
        from app.agent import translate_to_spanish  # lazy: pulls in agno/ollama

        translated = translate_to_spanish(summary)
    except Exception:  # noqa: BLE001 — degrade gracefully to the English original
        logger.warning("Summary translation to Spanish failed; using original only")
        translated = None
    if translated:
        _summary_es_cache[summary] = translated
    return translated


def load_profile(researcher_id: int, translate: bool = False) -> dict | None:
    """Load the consolidated generated profile shown to the respondent.

    :param translate: when True, also add `summary_es` (Spanish translation of the
        summary). Left False for cheap existence checks.
    :return: dict with name, portal_url, summary, summary_es, keywords, topics, year;
        or None when the researcher is not present in the current index.
    """
    payload = _db.get_payload(_summaries_collection(), researcher_id)
    if not payload:
        return None
    summary = (payload.get("summary") or "").strip()
    return {
        "researcher_id": researcher_id,
        "name": (payload.get("name") or "").strip() or f"PDI {researcher_id}",
        "portal_url": PORTAL_URL_TEMPLATE.format(id=researcher_id),
        "summary": summary,
        "summary_es": _translate_summary(summary) if translate else None,
        "keywords": payload.get("domains") or [],
        "topics": payload.get("topics") or [],
        "year": payload.get("year"),
    }


def _publications_for_year(publications: list | None, year: int) -> list[dict]:
    """Publications used to build a given year's summary: title, abstract (if any)
    and a Portal Científico item link when the id is stored in the bucket.

    Near-duplicate titles are collapsed keeping the longest abstract, mirroring the
    de-duplication phase 6 applies before summarising.
    """
    seen: dict[str, dict] = {}
    for pub in publications or []:
        if pub.get("year") != year:
            continue
        title = (pub.get("title") or "").strip()
        abstract = (pub.get("abstract") or "").strip()
        if not title and not abstract:
            continue
        item_id = pub.get("id")
        key = title.lower() or f"__id__{item_id}"
        existing = seen.get(key)
        if existing and len(abstract) <= len(existing["abstract"] or ""):
            continue
        seen[key] = {
            "title": title or "(sin título)",
            "abstract": abstract or None,
            "item_id": item_id,
            "url": PORTAL_ITEM_URL_TEMPLATE.format(id=item_id.removeprefix("pub_"))
            if item_id
            else None,
        }
    return list(seen.values())


def load_yearly(researcher_id: int) -> list[dict]:
    """Load the per-year generated summaries (MinIO `history`) for the evolution
    section, restricted to EVOLUTION_YEARS and ordered most-recent-first. Each year
    also carries the publications used to generate its summary."""
    try:
        from dbs import minio  # lazy: keeps the main app import light

        key = _ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
        data = minio.read_json(minio.CURATED_BUCKET, key)
    except Exception:  # noqa: BLE001 — degradar con elegancia si MinIO no responde
        logger.exception("Could not read yearly summaries for {}", researcher_id)
        return []

    history = (data or {}).get("history") or {}
    publications = (data or {}).get("publications") or []
    years: list[dict] = []
    for year in EVOLUTION_YEARS:
        snap = history.get(str(year))
        profile = (snap or {}).get("profile")
        if profile:
            years.append(
                {
                    "year": year,
                    "profile": profile.strip(),
                    "domains": (snap or {}).get("domains") or [],
                    "publications": _publications_for_year(publications, year),
                }
            )
    years.sort(key=lambda y: y["year"], reverse=True)
    return years


# --- Section 5: recommendation-case generation -----------------------------


def _dense(raw_vector) -> list[float] | None:
    """Extract the dense vector from Qdrant's (possibly named) vector container."""
    if isinstance(raw_vector, dict):
        return raw_vector.get(SUMMARY_DENSE_VECTOR) or raw_vector.get("dense")
    return raw_vector


def _centroid(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _matched_domains(keywords: list[str], domains: list[str]) -> list[str]:
    """Domains that overlap (containment, case-insensitive) with any keyword."""
    matched: list[str] = []
    for domain in domains:
        dl = domain.lower()
        for kw in keywords:
            kl = kw.lower()
            if kl == dl or kl in dl or dl in kl:
                matched.append(domain)
                break
    return matched


def build_recommendation_case(seed_id: int) -> dict | None:
    """Build one recommendation-evaluation case for a seed PDI.

    Retrieves the seed's point plus its nearest neighbours, reverse-engineers a
    plausible student query from the group with an LLM, and returns both a rich
    display payload (with names) and a sanitised `store` payload (no identities).
    """
    from app.agent import generate_student_case  # lazy: pulls in agno/ollama

    collection = _summaries_collection()
    seed = _db.get_point(collection, seed_id)
    if not seed:
        return None
    seed_vec = _dense(seed["vector"])
    if not seed_vec:
        return None

    neighbours = _db.nearest_by_vector(
        collection, seed_vec, using=SUMMARY_DENSE_VECTOR, limit=CASE_GROUP_SIZE
    )

    members: list[dict] = []
    seen_ids: set = set()
    for n in neighbours:
        vec = _dense(n["vector"])
        if not vec or n["id"] in seen_ids:
            continue
        seen_ids.add(n["id"])
        payload = n["payload"]
        members.append(
            {
                "id": n["id"],
                "name": (payload.get("name") or "").strip() or f"PDI {n['id']}",
                "summary": (payload.get("summary") or "").strip(),
                "domains": payload.get("domains") or [],
                "vector": vec,
            }
        )
    if not members:
        return None

    centroid = _centroid([m["vector"] for m in members])
    case = generate_student_case(
        [{"summary": m["summary"], "domains": m["domains"]} for m in members]
    )
    keywords = case.keywords or []

    for m in members:
        m["central"] = _cosine(m["vector"], centroid)
        m["matched"] = _matched_domains(keywords, m["domains"])
    # Rank by query relevance (matched domains), tie-broken by centrality.
    members.sort(key=lambda m: (len(m["matched"]), m["central"]), reverse=True)

    case_id = uuid.uuid4().hex[:12]
    # Where the evaluated PDI lands, and whether it is within the shown top-N.
    seed_rank = next((r for r, m in enumerate(members, 1) if m["id"] == seed_id), None)
    seed_in_top = seed_rank is not None and seed_rank <= CASE_TOP_N

    display: list[dict] = []
    store_recs: list[dict] = []
    for rank, m in enumerate(members, 1):
        is_seed = m["id"] == seed_id
        # Display: only the top-N, with names but NO domains/keywords (avoid bias).
        if rank <= CASE_TOP_N:
            numeric = _numeric_id(m["id"]) or m["id"]
            display.append(
                {
                    "rank": rank,
                    "name": m["name"],
                    "portal_url": PORTAL_URL_TEMPLATE.format(id=numeric),
                    "is_seed": is_seed,
                }
            )
        # Store: full ranking (anonymous); matched domains kept for later analysis.
        store_recs.append({"rank": rank, "is_seed": is_seed, "matched": m["matched"]})

    variant = "in_top" if seed_in_top else "not_in_top"
    position_text = SECTION5_POSITION[variant]

    return {
        "case_id": case_id,
        "query": case.query,
        "recommendations": display,
        "seed_rank": seed_rank,
        "seed_in_top": seed_in_top,
        "position_question": {
            "id": SECTION5_POSITION["id"],
            "text": position_text,
            "variant": variant,
        },
        "store": {
            "query": case.query,
            "keywords": keywords,
            "seed_rank": seed_rank,
            "seed_in_top": seed_in_top,
            "position_variant": variant,
            "position_text": position_text,
            "recommendations": store_recs,
        },
    }


def _int_answer(raw: str | None) -> int | None:
    if raw and raw.isdigit():
        value = int(raw)
        if SCALE_MIN <= value <= SCALE_MAX:
            return value
    return None


def _level_answer(raw: str | None) -> int | None:
    """Parse a keyword importance level (0-3)."""
    if raw is not None and str(raw).isdigit():
        value = int(raw)
        if 0 <= value <= 3:
            return value
    return None


def _collect_likert(items, form) -> list[dict]:
    return [
        {
            "id": qid,
            "text": text,
            "polarity": polarity,
            "scale": SCALE,
            "value": _int_answer(form.get(qid)),
        }
        for (qid, text, polarity) in items
    ]


def _collect_open(items, form) -> list[dict]:
    return [
        {"id": qid, "text": text, "value": (form.get(qid) or "").strip()}
        for (qid, text) in items
    ]


def _collect_multi(spec, form) -> dict:
    qid, text, options = spec
    valid = {value for value, _ in options}
    selected = [v for v in form.getlist(qid) if v in valid]
    return {
        "id": qid,
        "text": text,
        "options": [{"value": v, "label": lbl} for v, lbl in options],
        "selected": selected,
    }


def _collect_single(spec, form) -> dict:
    qid, text, options = spec
    valid = {value for value, _ in options}
    value = form.get(qid)
    return {
        "id": qid,
        "text": text,
        "options": [{"value": v, "label": lbl} for v, lbl in options],
        "value": value if value in valid else None,
    }


def _stats_from_lists(likert: list[dict], open_items: list[dict]) -> dict:
    return {
        "likert_total": len(likert),
        "likert_answered": sum(1 for x in likert if x.get("value") is not None),
        "open_total": len(open_items),
        "open_answered": sum(1 for x in open_items if (x.get("value") or "").strip()),
    }


def _device_from_ua() -> str:
    """Coarse device class from the User-Agent (no raw UA is stored).

    Kept intentionally low-resolution ('mobile'/'tablet'/'desktop') so it helps
    segment analysis without contributing to fingerprinting.
    """
    ua = (request.user_agent.string or "").lower()
    if any(t in ua for t in ("ipad", "tablet", "kindle", "silk", "playbook")):
        return "tablet"
    if any(t in ua for t in ("mobi", "android", "iphone", "ipod")):
        return "mobile"
    return "desktop"


def _duration_seconds(started_at_iso: str | None, now: datetime) -> int | None:
    """Whole seconds between the session start and submission, or None."""
    if not started_at_iso:
        return None
    try:
        start = datetime.fromisoformat(started_at_iso)
    except (ValueError, TypeError):
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    delta = (now - start).total_seconds()
    return int(delta) if delta >= 0 else None


def _build_metadata(role: str, form_version: str, started_at_iso: str | None) -> dict:
    """Non-identifying metadata attached to every submission, useful for later
    retrieval and analysis (time, completion time, device, language, versions)."""
    now = datetime.now(timezone.utc)
    return {
        "submission_id": str(uuid.uuid4()),
        "record_schema": RECORD_SCHEMA,
        "form_version": form_version,
        "role": role,
        "submitted_at": now.isoformat(),
        "submitted_date": now.date().isoformat(),
        "started_at": started_at_iso,
        "duration_seconds": _duration_seconds(started_at_iso, now),
        "device": _device_from_ua(),
        "language": (request.accept_languages.best or "").split("-")[0] or None,
    }


def _response_stats(answers: dict) -> dict:
    """Completion counts across the whole instrument, to flag partial/careless
    responses at analysis time."""
    likert_total = likert_answered = open_total = open_answered = 0

    def tally_likert(items):
        nonlocal likert_total, likert_answered
        for it in items or []:
            likert_total += 1
            if it.get("value") is not None:
                likert_answered += 1

    def tally_open(items):
        nonlocal open_total, open_answered
        for it in items or []:
            open_total += 1
            if (it.get("value") or "").strip():
                open_answered += 1

    section2 = answers.get("section2") or {}
    tally_likert(section2.get("likert"))
    tally_open(section2.get("open"))
    section6 = answers.get("section6") or {}
    tally_likert(section6.get("likert"))
    tally_open(section6.get("open"))

    return {
        "likert_total": likert_total,
        "likert_answered": likert_answered,
        "open_total": open_total,
        "open_answered": open_answered,
    }


def _collect_keywords(profile: dict, form) -> dict:
    """PDI validation of the generated keywords: importance level (0-3) per generated
    domain, plus any domains the PDI added themselves.

    Levels: 0 no relacionada, 1 puntual/secundaria, 2 línea relevante,
    3 línea principal para recomendación. Missing values fall back to the default.
    """
    generated = []
    for i, keyword in enumerate(profile.get("keywords") or []):
        level = _level_answer(form.get(f"kw_{i}"))
        generated.append(
            {"keyword": keyword, "level": KEYWORD_DEFAULT_LEVEL if level is None else level}
        )
    added: list[str] = []
    seen: set[str] = set()
    for raw in form.getlist("kw_added"):
        value = (raw or "").strip()
        low = value.lower()
        if value and low not in seen:
            seen.add(low)
            added.append(value)
    return {"generated": generated, "added": added}


def _collect_section5(form) -> list[dict]:
    """Collect the (optional) recommendation-evaluation cases and their ratings.

    Each generated case posts a hidden `s5_case_<id>` field carrying the sanitised
    stimulus (query + keywords + anonymous recommendations) plus per-case ratings.
    """
    cases: list[dict] = []
    for key in form.keys():
        if not key.startswith("s5_case_"):
            continue
        case_id = key[len("s5_case_") :]
        try:
            stored = json.loads(form.get(key) or "{}")
        except (ValueError, TypeError):
            continue
        likert = [
            {
                "id": f"s5_{case_id}_{qkey}",
                "text": text,
                "polarity": polarity,
                "scale": SCALE,
                "value": _int_answer(form.get(f"s5_{case_id}_{qkey}")),
            }
            for (qkey, text, polarity) in SECTION5_LIKERT
        ]
        # Fifth item: text/variant depend on whether the PDI was shown in the top-N.
        pos_id = SECTION5_POSITION["id"]
        likert.append(
            {
                "id": f"s5_{case_id}_{pos_id}",
                "text": stored.get("position_text"),
                "polarity": "positive",
                "variant": stored.get("position_variant"),
                "scale": SCALE,
                "value": _int_answer(form.get(f"s5_{case_id}_{pos_id}")),
            }
        )
        cases.append(
            {
                "case_id": case_id,
                "query": stored.get("query"),
                "keywords": stored.get("keywords") or [],
                "seed_rank": stored.get("seed_rank"),
                "seed_in_top": stored.get("seed_in_top"),
                "recommendations": stored.get("recommendations") or [],
                "likert": likert,
                "opinion": (
                    form.get(f"s5_{case_id}_{SECTION5_OPINION[0]}") or ""
                ).strip(),
            }
        )
    return cases


def _build_record(profile: dict, form, started_at: str | None = None) -> dict:
    """Assemble the PDI submission document (linked to the evaluated profile)."""
    answers = {
        "section2": {
            "likert": _collect_likert(SECTION2_LIKERT, form),
            "open": _collect_open(SECTION2_OPEN, form),
        },
        "keywords": _collect_keywords(profile, form),
        "section5": _collect_section5(form),
        "section6": {
            "likert": _collect_likert(SECTION6_LIKERT, form),
            "open": _collect_open(SECTION6_OPEN, form),
        },
    }

    metadata = _build_metadata("pdi", FORM_VERSION, started_at)
    metadata["response_stats"] = _response_stats(answers)

    return {
        "metadata": metadata,
        # Responses ARE linked to the evaluated profile (no longer anonymous):
        # the portal id is stored so results can be validated against the profile.
        "researcher_id": profile.get("researcher_id"),
        "consent": {
            "academic_aggregated_use": True,
            "privacy_terms": True,
        },
        # Generated profile the respondent evaluated (the stimulus).
        "stimulus": {
            "profile_summary": profile.get("summary"),
            "profile_summary_es": profile.get("summary_es"),
            "keywords": profile.get("keywords"),
            "topics": profile.get("topics"),
            "last_active_year": profile.get("year"),
        },
        "answers": answers,
    }


def _build_student_record(form, started_at: str | None = None) -> dict:
    """Assemble the anonymised student submission (no identifying fields).

    Section 1 (usage experience) is only included when the student reports having
    used the app. Academic context (school, level, degrees, interest) is stored as
    coarse categories/interests — no name, e-mail or enrolment id.
    """
    used = form.get("used_app")
    used_yes = (used or "").startswith("yes")

    context = {
        "school": form.get("school"),
        "level": form.get("level"),
        "degree": (form.get("degree") or "").strip() or None,
        "master": (form.get("master") or "").strip() or None,
        "thesis_topic": (form.get("thesis_topic") or "").strip() or None,
        "favorite_topic": (form.get("favorite_topic") or "").strip(),
        "used_app": used,
    }

    section1 = None
    if used_yes:
        section1 = {
            "likert": _collect_likert(STUDENT_S1_LIKERT, form),
            "open": _collect_open(STUDENT_S1_OPEN, form),
        }
    section2 = {
        "multi": _collect_multi(STUDENT_S2_MULTI, form),
        "likert": _collect_likert(STUDENT_S2_LIKERT, form),
        "single": _collect_single(STUDENT_S2_SINGLE, form),
    }
    section3 = {"likert": _collect_likert(STUDENT_S3_LIKERT, form)}
    section4 = {"open": _collect_open(STUDENT_S4_OPEN, form)}

    answers = {
        "section1": section1,
        "section2": section2,
        "section3": section3,
        "section4": section4,
    }

    likert_all = list(section2["likert"]) + list(section3["likert"])
    open_all = list(section4["open"])
    if section1:
        likert_all += section1["likert"]
        open_all += section1["open"]

    metadata = _build_metadata("student", STUDENT_FORM_VERSION, started_at)
    metadata["response_stats"] = _stats_from_lists(likert_all, open_all)

    return {
        "metadata": metadata,
        "consent": {"academic_aggregated_use": True, "privacy_terms": True},
        "context": context,
        "answers": answers,
    }


def _persist(record: dict) -> bool:
    from dbs import minio  # lazy

    meta = record["metadata"]
    role = meta.get("role") or "unknown"
    date = meta.get("submitted_date") or datetime.now(timezone.utc).date().isoformat()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Partitioned by role and date for easy listing/retrieval and analysis.
    object_name = f"responses/{role}/{date}/{ts}_{meta['submission_id']}.json"
    return minio.upload_json(EVAL_BUCKET, record, object_name)


# --- Routes ----------------------------------------------------------------


@eval_bp.get("/evaluation")
def eval_home():
    """Role choice: PDI (implemented) or student (placeholder)."""
    return render_template("evaluation/home.html")


def _student_form_context(**extra) -> dict:
    ctx = {
        "schools": SCHOOLS,
        "levels": LEVELS,
        "used_options": USED_OPTIONS,
        "main_app_url": url_for("main.match"),
        "q": STUDENT_QUESTIONS,
    }
    ctx.update(extra)
    return ctx


@eval_bp.get("/evaluation/student")
def eval_student():
    # Single page; store the start time to measure completion duration.
    session["eval"] = {
        "role": "student",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return render_template("evaluation/student_form.html", **_student_form_context())


@eval_bp.post("/evaluation/student/submit")
def eval_student_submit():
    form = request.form
    if not form.get("consent"):
        return render_template(
            "evaluation/student_form.html",
            **_student_form_context(consent_error=True),
        ), 400

    school = form.get("school")
    level = form.get("level")
    favorite = (form.get("favorite_topic") or "").strip()
    used = form.get("used_app")
    if (
        school not in SCHOOLS
        or level not in STUDENT_LEVEL_VALUES
        or used not in STUDENT_USED_VALUES
        or not favorite
    ):
        return render_template(
            "evaluation/student_form.html",
            **_student_form_context(validation_error=True),
        ), 400

    started_at = (session.get("eval") or {}).get("started_at")
    record = _build_student_record(form, started_at=started_at)
    submission_id = record["metadata"]["submission_id"]
    if not _persist(record):
        logger.error("Failed to persist student submission {}", submission_id)
        return render_template(
            "evaluation/student_form.html",
            **_student_form_context(save_error=True),
        ), 500

    session.pop("eval", None)
    logger.info("Stored student submission {}", submission_id)
    return redirect(url_for("eval.eval_thanks"))


@eval_bp.get("/evaluation/pdi")
def eval_pdi_consent():
    not_found = request.args.get("notfound")
    return render_template(
        "evaluation/consent.html",
        pdis=list_pdis(),
        portal_url_prefix="https://portalcientifico.upm.es/es/ipublic/researcher/",
        contact_email="carlos.badenes@upm.es",
        not_found=not_found,
    )


@eval_bp.post("/evaluation/pdi/start")
def eval_pdi_start():
    """Validate consent + PDI selection, open a session, go to the form."""
    if not (
        request.form.get("consent_aggregated") and request.form.get("privacy_terms")
    ):
        return redirect(url_for("eval.eval_pdi_consent"))

    researcher_id = _numeric_id(request.form.get("pdi_id")) or _numeric_id(
        request.form.get("portal_id")
    )
    if researcher_id is None or load_profile(researcher_id) is None:
        # Unknown / not indexed: send them back with the contact-Carlos notice.
        return redirect(url_for("eval.eval_pdi_consent", notfound=1))

    session["eval"] = {
        "role": "pdi",
        "pdi_id": researcher_id,
        "consent": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return redirect(url_for("eval.eval_pdi_form"))


def _session_pdi() -> int | None:
    ev = session.get("eval")
    if not ev or ev.get("role") != "pdi" or not ev.get("consent"):
        return None
    return ev.get("pdi_id")


@eval_bp.get("/evaluation/pdi/questionnaire")
def eval_pdi_form():
    researcher_id = _session_pdi()
    if researcher_id is None:
        return redirect(url_for("eval.eval_pdi_consent"))

    profile = load_profile(researcher_id, translate=True)
    if profile is None:
        session.pop("eval", None)
        return redirect(url_for("eval.eval_pdi_consent", notfound=1))

    return render_template(
        "evaluation/questionnaire.html",
        profile=profile,
        q=QUESTIONS,
        s5_likert=[{"key": k, "text": t} for k, t, _ in SECTION5_LIKERT],
        s5_opinion=SECTION5_OPINION[1],
    )


@eval_bp.post("/evaluation/pdi/case")
def eval_pdi_case():
    """Generate one recommendation-evaluation case for the session's PDI (JSON)."""
    seed_id = _session_pdi()
    if seed_id is None:
        return jsonify({"error": "Sesión no válida."}), 403
    try:
        case = build_recommendation_case(seed_id)
    except Exception as exc:  # noqa: BLE001 — surface a friendly error to the UI
        logger.exception("Recommendation case generation failed")
        return jsonify({"error": f"No se pudo generar el caso: {exc}"}), 500
    if not case:
        return jsonify({"error": "No hay datos suficientes para generar un caso."}), 500
    return jsonify(case)


@eval_bp.post("/evaluation/pdi/submit")
def eval_pdi_submit():
    researcher_id = _session_pdi()
    if researcher_id is None:
        return redirect(url_for("eval.eval_pdi_consent"))

    profile = load_profile(researcher_id, translate=True)
    if profile is None:
        session.pop("eval", None)
        return redirect(url_for("eval.eval_pdi_consent", notfound=1))

    started_at = (session.get("eval") or {}).get("started_at")
    record = _build_record(profile, request.form, started_at=started_at)
    submission_id = record["metadata"]["submission_id"]
    ok = _persist(record)
    if not ok:
        logger.error("Failed to persist evaluation submission {}", submission_id)
        return render_template(
            "evaluation/questionnaire.html",
            profile=profile,
            q=QUESTIONS,
            save_error=True,
        ), 500

    session.pop("eval", None)
    logger.info("Stored evaluation submission {}", submission_id)
    return redirect(url_for("eval.eval_thanks"))


@eval_bp.get("/evaluation/thanks")
def eval_thanks():
    return render_template("evaluation/thanks.html")
