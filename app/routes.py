"""Flask routes for the UPM Match recommendation API and UI."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request
from loguru import logger

from dbs.vector import QdrantBackend, get_vector_db

from .agent import enrich_query, extract_keywords
from .evaluation import list_pdis, load_profile, load_yearly
from .search import (
    COLLECTION_NAME,
    SUMMARY_COLLECTION,
    PDIResult,
    search_pdis,
    search_pdis_by_summary,
    search_pdis_hybrid,
)

bp = Blueprint("main", __name__)

# Shared Qdrant client - created once and reused across requests.
# BM42 model is loaded lazily on first search call.
_db: QdrantBackend = get_vector_db()


def _serialise_result(r: PDIResult) -> dict:
    return {
        "rank": r.rank,
        "researcher_id": r.researcher_id,
        "name": r.name,
        "score": r.score,
        "score_pct": r.score_pct,
        "matched_topics": r.matched_topics,
        "all_topics": r.all_topics,
        "since": r.since,
        "summary": r.summary,
    }


@bp.get("/")
def landing():
    try:
        indexed_pdis = _db.count_points(SUMMARY_COLLECTION)
    except Exception:  # noqa: BLE001 — el landing no debe romperse si Qdrant no responde
        indexed_pdis = None
    return render_template(
        "landing.html", active_page="landing", indexed_pdis=indexed_pdis
    )


@bp.get("/match")
def match():
    return render_template("match.html", active_page="match")


@bp.get("/about")
def about():
    return render_template("about.html", active_page="about")


@bp.get("/search")
def pdi_search():
    """
    Public researcher explorer:
      pick a PDI and browse everything we store about them (consolidated profile + per-year summaries, keywords and publications).
    The per-PDI data is loaded lazily via `pdi_data`.
    """
    return render_template("search.html", active_page="search", pdis=list_pdis())


@bp.get("/api/pdi/<int:researcher_id>")
def pdi_data(researcher_id: int):
    """Return the full stored profile for one researcher (JSON).

    Combines the consolidated summary/keywords/topics (Qdrant payload) with the
    per-year generated summaries, domains and publications (MinIO enrichment).
    """
    profile = load_profile(researcher_id, translate=True)
    if profile is None:
        return jsonify({"error": "No encontramos información de este PDI."}), 404
    return jsonify({"profile": profile, "years": load_yearly(researcher_id)})


@bp.get("/api/stats")
def stats():
    """Return indexed PDI counts from the latest Qdrant collections."""
    keywords_collection = _db.latest_collection(COLLECTION_NAME) or COLLECTION_NAME
    summaries_collection = (
        _db.latest_collection(SUMMARY_COLLECTION) or SUMMARY_COLLECTION
    )
    return jsonify(
        {
            "keywords_collection": keywords_collection,
            "indexed_keywords": _db.count_points(keywords_collection),
            "summaries_collection": summaries_collection,
            "indexed_summaries": _db.count_points(summaries_collection),
        }
    )


@bp.post("/api/search")
def search():
    """Search for PDI recommendations given a free-text research idea.

    Request body (JSON):
      - idea (str): free-text research idea in any language.
      - mode (str, optional): 'keywords' (default), 'summary_direct' or 'summary_enriched'.
      - limit (int, optional): number of results to return (default 10, max 20).

    Response (JSON) common fields:
      - mode: the mode that was used.
      - results: list of ranked PDI objects.

    Additional fields by mode:
      - keywords: general_keywords, specific_keywords.
      - summary_enriched: enriched_query (the LLM-expanded English paragraph).
    """
    body = request.get_json(silent=True) or {}
    idea: str = (body.get("idea") or "").strip()
    if not idea:
        return jsonify({"error": "El campo 'idea' es obligatorio."}), 422

    limit: int = min(int(body.get("limit") or 10), 20)
    mode: str = (body.get("mode") or "keywords").strip()

    logger.info("Search request - mode={} idea_len={} limit={}", mode, len(idea), limit)

    if mode == "keywords":
        try:
            keywords = extract_keywords(idea)
            logger.debug(
                "Keywords extracted - general={} specific={}",
                keywords.general,
                keywords.specific,
            )
        except Exception as exc:
            logger.exception("Keyword extraction failed")
            return jsonify({"error": f"Error al extraer keywords: {exc}"}), 500

        try:
            results = search_pdis(_db, keywords.general, keywords.specific, limit=limit)
            logger.info("Keyword search returned {} results", len(results))
        except Exception as exc:
            logger.exception("Qdrant keyword search failed")
            return jsonify({"error": f"Error en la búsqueda: {exc}"}), 500

        return jsonify(
            {
                "mode": "keywords",
                "general_keywords": keywords.general,
                "specific_keywords": keywords.specific,
                "results": [_serialise_result(r) for r in results],
            }
        )

    if mode == "hybrid":
        # Sparse leg: keyword extraction. Dense leg: enriched query (falls back
        # to the raw idea if enrichment fails) so the summaries (written in
        # English) are matched on comparable text.
        try:
            keywords = extract_keywords(idea)
        except Exception as exc:
            logger.exception("Keyword extraction failed")
            return jsonify({"error": f"Error al extraer keywords: {exc}"}), 500

        enriched_query: str | None = None
        dense_text = idea
        try:
            dense_text = enrich_query(idea)
            enriched_query = dense_text
        except Exception:
            logger.warning("Query enrichment failed; using raw idea for dense leg")

        try:
            results = search_pdis_hybrid(
                _db, keywords.general, keywords.specific, dense_text, limit=limit
            )
            logger.info("Hybrid search returned {} results", len(results))
        except Exception as exc:
            logger.exception("Qdrant hybrid search failed")
            return jsonify({"error": f"Error en la búsqueda híbrida: {exc}"}), 500

        return jsonify(
            {
                "mode": "hybrid",
                "general_keywords": keywords.general,
                "specific_keywords": keywords.specific,
                "enriched_query": enriched_query,
                "results": [_serialise_result(r) for r in results],
            }
        )

    if mode in ("summary_direct", "summary_enriched"):
        query_text = idea
        enriched_query: str | None = None

        if mode == "summary_enriched":
            try:
                query_text = enrich_query(idea)
                enriched_query = query_text
                logger.debug("Query enriched - length={}", len(query_text))
            except Exception as exc:
                logger.exception("Query enrichment failed")
                return jsonify(
                    {"error": f"Error al enriquecer la consulta: {exc}"}
                ), 500

        try:
            results = search_pdis_by_summary(_db, query_text, limit=limit)
            logger.info("Summary search returned {} results", len(results))
        except Exception as exc:
            logger.exception("Qdrant dense summary search failed")
            return jsonify({"error": f"Error en la búsqueda por resumen: {exc}"}), 500

        return jsonify(
            {
                "mode": mode,
                "enriched_query": enriched_query,
                "results": [_serialise_result(r) for r in results],
            }
        )

    return jsonify({"error": f"Modo de búsqueda desconocido: '{mode}'."}), 422
