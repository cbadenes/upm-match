"""Flask routes for the UPM Match recommendation API and UI."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request
from loguru import logger

from dbs.vector import get_vector_db

from .agent import enrich_query, extract_keywords
from .search import PDIResult, search_pdis, search_pdis_by_summary

bp = Blueprint("main", __name__)

# Shared Qdrant client - created once and reused across requests.
# BM42 model is loaded lazily on first search call.
_db = get_vector_db()


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
def index():
    return render_template("index.html")


@bp.post("/api/search")
def search():
    """Search for PDI recommendations given a free-text research idea.

    Request body (JSON):
      - idea (str): free-text research idea in any language.
      - mode (str, optional): search mode — 'keywords' (default), 'summary_direct',
        or 'summary_enriched'.
      - limit (int, optional): number of results to return (default 10, max 20).

    Response (JSON) — common fields:
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
                return jsonify({"error": f"Error al enriquecer la consulta: {exc}"}), 500

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
