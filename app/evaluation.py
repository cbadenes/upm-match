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

import os
import re
import uuid
from datetime import datetime, timezone

from flask import (
    Blueprint,
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
# Enrichment document layout in MinIO (see utils.researchers.ENRICHMENT_KEY_TEMPLATE).
_ENRICHMENT_KEY_TEMPLATE = "researchers/{researcher_id}/enrichment.json"
PORTAL_URL_TEMPLATE = "https://portalcientifico.upm.es/es/ipublic/researcher/{id}"

# Years shown in the temporal-evolution section (most recent first at render time).
EVOLUTION_YEARS = (2023, 2024, 2025, 2026)

# MinIO bucket that receives the survey responses. Auto-created on first upload.
EVAL_BUCKET = os.environ.get("MINIO_EVAL_BUCKET", "upm-match-evaluations")

FORM_VERSION = "pdi-v1"
SCALE_MIN, SCALE_MAX = 1, 5
SCALE = f"{SCALE_MIN}-{SCALE_MAX}"

_db: QdrantBackend = get_vector_db()

# Cache the PDI directory (id + name) for the searchable selector.
# The set of indexed researchers only changes when the pipeline reindexes, so a lazy cache is plenty and avoids a full Qdrant scan on every consent-page load.
_pdi_cache: list[dict] | None = None


# Questionnaire schema (single source of truth for render + storage).
# Each Likert item carries a polarity so the analysis can reverse-code the negatively worded statements.
# Open questions are stored verbatim.

SECTION2_LIKERT = [
    (
        "s2_repr_actual",
        "El resumen representa correctamente mi actividad investigadora actual.",
        "positive",
    ),
    (
        "s2_lineas_principales",
        "El resumen identifica adecuadamente mis principales líneas de investigación.",
        "positive",
    ),
    (
        "s2_omite_lineas",
        "El resumen omite líneas relevantes de mi actividad investigadora.",
        "negative",
    ),
    (
        "s2_sobrepeso_antiguas",
        "El resumen da demasiada importancia a líneas secundarias o antiguas.",
        "negative",
    ),
    (
        "s2_util_estudiante",
        "El resumen sería útil para que un estudiante valore si puedo tutorizar su TFG/TFM.",
        "positive",
    ),
    (
        "s2_claridad",
        "El resumen está redactado de forma clara y comprensible.",
        "positive",
    ),
]
SECTION2_OPEN = [
    (
        "s2o_correctos",
        "¿Qué aspectos del resumen considera correctos o bien representados?",
    ),
    (
        "s2o_faltan",
        "¿Qué líneas de investigación relevantes faltan o están poco representadas?",
    ),
    (
        "s2o_sobre",
        "¿Qué líneas aparecen sobrerrepresentadas o no reflejan bien su actividad actual?",
    ),
    ("s2o_reformular", "Si tuviera que corregir este resumen, ¿cómo lo reformularía?"),
]

SECTION3_LIKERT = [
    (
        "refleja",
        "El resumen refleja adecuadamente mi actividad investigadora en ese año.",
        "positive",
    ),
    (
        "evolucion",
        "El resumen capta la evolución de mis líneas respecto a años anteriores.",
        "positive",
    ),
    (
        "no_centrales",
        "El resumen incluye líneas que no fueron realmente centrales en ese año.",
        "negative",
    ),
]
SECTION3_OPEN = ("cambios", "¿Qué cambiaría en la descripción de este año concreto?")

SECTION4_OPEN = [
    (
        "s4_descripcion_libre",
        "Escriba una descripción de proyecto, interés o tema de TFG/TFM para la que considere que debería aparecer entre los primeros candidatos recomendados.",
    ),
    (
        "s4_palabras_clave",
        "Indique entre 5 y 10 palabras clave que deberían activar su recomendación como tutor.",
    ),
    (
        "s4_temas_no_adecuados",
        "Indique temas que podrían aparecer asociados a su perfil, pero para los que no considera que debería ser recomendado como tutor principal.",
    ),
]

SECTION6_LIKERT = [
    (
        "s6_ayuda_identificar",
        "El sistema podría ayudar a estudiantes a identificar posibles tutores.",
        "positive",
    ),
    (
        "s6_descubrir_menos_visibles",
        "El sistema podría ayudar a descubrir PDIs menos visibles pero adecuados.",
        "positive",
    ),
    (
        "s6_solo_apoyo",
        "El sistema debería usarse solo como apoyo y no como decisión automática.",
        "neutral",
    ),
    (
        "s6_explicaciones_necesarias",
        "Las explicaciones del sistema son necesarias para confiar en la recomendación.",
        "neutral",
    ),
    (
        "s6_integracion_portal",
        "Me parecería útil que el sistema estuviera integrado en un portal institucional.",
        "positive",
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
    "section3_likert": SECTION3_LIKERT,
    "section3_open": SECTION3_OPEN,
    "section4_open": SECTION4_OPEN,
    "section6_likert": SECTION6_LIKERT,
    "section6_open": SECTION6_OPEN,
}


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


def load_profile(researcher_id: int) -> dict | None:
    """Load the consolidated generated profile shown to the respondent.

    :return: dict with name, portal_url, summary, keywords, topics, year; or None
        when the researcher is not present in the current index.
    """
    payload = _db.get_payload(_summaries_collection(), researcher_id)
    if not payload:
        return None
    return {
        "researcher_id": researcher_id,
        "name": (payload.get("name") or "").strip() or f"PDI {researcher_id}",
        "portal_url": PORTAL_URL_TEMPLATE.format(id=researcher_id),
        "summary": (payload.get("summary") or "").strip(),
        "keywords": payload.get("domains") or [],
        "topics": payload.get("topics") or [],
        "year": payload.get("year"),
    }


def load_yearly(researcher_id: int) -> list[dict]:
    """Load the per-year generated summaries (MinIO `history`) for the evolution
    section, restricted to EVOLUTION_YEARS and ordered most-recent-first."""
    try:
        from dbs import minio  # lazy: keeps the main app import light

        key = _ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
        data = minio.read_json(minio.CURATED_BUCKET, key)
    except Exception:  # noqa: BLE001 — degradar con elegancia si MinIO no responde
        logger.exception("Could not read yearly summaries for {}", researcher_id)
        return []

    history = (data or {}).get("history") or {}
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
                }
            )
    years.sort(key=lambda y: y["year"], reverse=True)
    return years


def _int_answer(raw: str | None) -> int | None:
    if raw and raw.isdigit():
        value = int(raw)
        if SCALE_MIN <= value <= SCALE_MAX:
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


def _build_record(profile: dict, yearly: list[dict], form) -> dict:
    """Assemble the anonymised submission document (no identifying fields)."""
    section3: list[dict] = []
    for year in yearly:
        prefix = f"s3_{year['year']}"
        likert = [
            {
                "id": f"{prefix}_{key}",
                "text": text,
                "polarity": polarity,
                "scale": SCALE,
                "value": _int_answer(form.get(f"{prefix}_{key}")),
            }
            for (key, text, polarity) in SECTION3_LIKERT
        ]
        open_key = f"{prefix}_{SECTION3_OPEN[0]}"
        section3.append(
            {
                "year": year["year"],
                "likert": likert,
                "open": {
                    "id": open_key,
                    "text": SECTION3_OPEN[1],
                    "value": (form.get(open_key) or "").strip(),
                },
            }
        )

    return {
        "submission_id": str(uuid.uuid4()),
        "form_version": FORM_VERSION,
        "role": "pdi",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "consent": {
            "academic_aggregated_use": True,
            "privacy_terms": True,
        },
        # Stimulus the respondent evaluated. Derived from public portal data;
        # kept WITHOUT any identifier so responses remain interpretable.
        "stimulus": {
            "profile_summary": profile.get("summary"),
            "keywords": profile.get("keywords"),
            "topics": profile.get("topics"),
            "last_active_year": profile.get("year"),
            "yearly": [
                {"year": y["year"], "profile": y["profile"], "domains": y["domains"]}
                for y in yearly
            ],
        },
        "answers": {
            "section2": {
                "likert": _collect_likert(SECTION2_LIKERT, form),
                "open": _collect_open(SECTION2_OPEN, form),
            },
            "section3": section3,
            "section4": {"open": _collect_open(SECTION4_OPEN, form)},
            "section6": {
                "likert": _collect_likert(SECTION6_LIKERT, form),
                "open": _collect_open(SECTION6_OPEN, form),
            },
        },
    }


def _persist(record: dict) -> bool:
    from dbs import minio  # lazy

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    object_name = f"responses/pdi/{ts}_{record['submission_id']}.json"
    return minio.upload_json(EVAL_BUCKET, record, object_name)


# --- Routes ----------------------------------------------------------------


@eval_bp.get("/evaluacion")
def eval_home():
    """Role choice: PDI (implemented) or student (placeholder)."""
    return render_template("evaluacion/home.html")


@eval_bp.get("/evaluacion/estudiante")
def eval_student():
    return render_template("evaluacion/student_placeholder.html")


@eval_bp.get("/evaluacion/pdi")
def eval_pdi_consent():
    not_found = request.args.get("notfound")
    return render_template(
        "evaluacion/consent.html",
        pdis=list_pdis(),
        portal_url_prefix="https://portalcientifico.upm.es/es/ipublic/researcher/",
        contact_email="carlos.badenes@upm.es",
        not_found=not_found,
    )


@eval_bp.post("/evaluacion/pdi/iniciar")
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


@eval_bp.get("/evaluacion/pdi/cuestionario")
def eval_pdi_form():
    researcher_id = _session_pdi()
    if researcher_id is None:
        return redirect(url_for("eval.eval_pdi_consent"))

    profile = load_profile(researcher_id)
    if profile is None:
        session.pop("eval", None)
        return redirect(url_for("eval.eval_pdi_consent", notfound=1))

    return render_template(
        "evaluacion/questionnaire.html",
        profile=profile,
        yearly=load_yearly(researcher_id),
        q=QUESTIONS,
    )


@eval_bp.post("/evaluacion/pdi/enviar")
def eval_pdi_submit():
    researcher_id = _session_pdi()
    if researcher_id is None:
        return redirect(url_for("eval.eval_pdi_consent"))

    profile = load_profile(researcher_id)
    if profile is None:
        session.pop("eval", None)
        return redirect(url_for("eval.eval_pdi_consent", notfound=1))

    record = _build_record(profile, load_yearly(researcher_id), request.form)
    ok = _persist(record)
    if not ok:
        logger.error(
            "Failed to persist evaluation submission {}", record["submission_id"]
        )
        return render_template(
            "evaluacion/questionnaire.html",
            profile=profile,
            yearly=load_yearly(researcher_id),
            q=QUESTIONS,
            save_error=True,
        ), 500

    session.pop("eval", None)
    logger.info("Stored evaluation submission {}", record["submission_id"])
    return redirect(url_for("eval.eval_thanks"))


@eval_bp.get("/evaluacion/gracias")
def eval_thanks():
    return render_template("evaluacion/thanks.html")
