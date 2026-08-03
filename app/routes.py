"""Flask routes for the UPM Match recommendation API and UI."""

from __future__ import annotations

import os

from flask import Blueprint, Response, jsonify, render_template, request, url_for
from loguru import logger

from dbs.vector import QdrantBackend, get_vector_db

from .agent import enrich_query, extract_keywords
from .clarify import (
    CLARIFY_MAX_ROUNDS,
    ClarifyTurn,
    effective_idea,
    next_clarification,
)
from .evaluation import _numeric_id, list_pdis, load_profile, load_yearly
from .graph import load_graph_profile
from .search import (
    COLLECTION_NAME,
    RERANK_POOL,
    SUMMARY_COLLECTION,
    PDIResult,
    rerank_pdi_results,
    search_pdis,
    search_pdis_by_summary,
    search_pdis_hybrid,
)

# Retrieval depth when the clarification loop is active: domain-level questions draw their aspects from this whole pool (depth as taxonomy substitute).
CLARIFY_POOL = int(os.environ.get("CLARIFY_POOL", "100"))

# Rights channel published on /legal: the address a PDI writes to in order to correct or remove their profile. Kept in sync with the questionnaire contact.
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "carlos.badenes@upm.es")
# Shown on /legal so the notice carries a visible date, as a privacy notice should.
LEGAL_LAST_UPDATED = os.environ.get("LEGAL_LAST_UPDATED", "agosto de 2026")

bp = Blueprint("main", __name__)

# Shared Qdrant client - created once and reused across requests.
# BM42 model is loaded lazily on first search call.
_db: QdrantBackend = get_vector_db()


def _serialise_result(r: PDIResult) -> dict:
    return {
        "rank": r.rank,
        "researcher_id": r.researcher_id,
        # Numeric portal id, so the UI can deep-link to the profile explorer (/search?pdi=<id>) from each result card.
        "id_num": _numeric_id(r.researcher_id),
        "name": r.name,
        "score": r.score,
        "score_pct": r.score_pct,
        "matched_topics": r.matched_topics,
        "all_topics": r.all_topics,
        "since": r.since,
        "summary": r.summary,
        # Rank before the second-stage re-ranking (None for single-stage modes).
        "stage1_rank": r.stage1_rank,
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
    """User-facing search: describe an idea (enriched by default, with a quick
    direct variant), pick your own topics, or run the smart (hybrid) search."""
    return render_template("match.html", active_page="match")


@bp.get("/test")
def test():
    """Method-comparison bench (the former /match UI). URL-only: not linked from
    the navigation and disallowed in robots.txt, like the questionnaire."""
    return render_template("test.html", active_page=None, is_test=True)


@bp.get("/about")
def about():
    return render_template("about.html", active_page="about")


@bp.get("/legal")
def legal():
    """Legal notice + privacy information (RGPD art. 14, LSSI art. 10).

    Linked from the footer of every page: it is where a PDI finds out that this system exists, on what legal basis, and how to have their profile corrected or removed.
    """
    return render_template(
        "legal.html",
        active_page="legal",
        contact_email=CONTACT_EMAIL,
        last_updated=LEGAL_LAST_UPDATED,
    )


@bp.get("/healthz")
def healthz():
    """Liveness probe for monitoring/orchestration. Intentionally does not touch
    Qdrant/MinIO/Ollama: it reports whether the web process is up and serving."""
    return jsonify({"status": "ok"}), 200


@bp.get("/robots.txt")
def robots():
    """Public pages are indexable; the questionnaire and the JSON API are not."""
    api_root = url_for("main.stats").rsplit("/", 1)[0]  # .../api
    body = "\n".join(
        [
            "User-agent: *",
            f"Disallow: {url_for('eval.eval_home')}",
            f"Disallow: {url_for('main.test')}",
            f"Disallow: {api_root}",
            "",
        ]
    )
    return Response(body, mimetype="text/plain")


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
    per-year generated summaries, domains and publications (MinIO enrichment)
    and the academic trajectory (Memgraph: supervised works, subjects, positions).
    """
    profile = load_profile(researcher_id, translate=True)
    if profile is None:
        return jsonify({"error": "No encontramos información de este PDI."}), 404
    return jsonify(
        {
            "profile": profile,
            "years": load_yearly(researcher_id),
            "graph": load_graph_profile(researcher_id),
        }
    )


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
      - idea (str): free-text research idea in any language (all modes except 'keywords_user').
      - keywords (list[str]): user-provided topic keywords ('keywords_user' only).
      - mode (str, optional): 'keywords' (default), 'keywords_user', 'hybrid', 'summary_direct' or 'summary_enriched'.
      - rerank (str, optional, 'hybrid' only): '' (no re-ranking, default), 'text' (second-stage listwise LLM re-ranking) or 'kg' (listwise with knowledge-graph evidence per candidate; degrades to 'text' without Memgraph). Powers the smart search and the /test bench.
      - clarify (bool, optional, 'hybrid' only): enable the ASK-style clarification loop (smart search). When the retrieved space looks ambiguous the response carries a `clarify` question instead of results.
      - clarification (list, optional): completed rounds so far, as [{question, answer}, ...]; their answers refine the query (ASK).
      - clarify_skip (bool, optional): answer now, skipping the detector ("buscar sin aclarar").
      - limit (int, optional): number of results to return (default 10, max 20).

    Response (JSON) common fields:
      - mode: the mode that was used.
      - results: list of ranked PDI objects (with stage1_rank when re-ranked).

    Additional fields by mode:
      - keywords: general_keywords, specific_keywords.
      - summary_enriched: enriched_query (the LLM-expanded English paragraph).
      - hybrid with rerank: rerank {variant, model, pool, windows, seconds,
        kg_degraded?, error?} (null when re-ranking was not requested).
      - hybrid with clarify, when a question is needed: NO results; instead clarify {question, options, ambiguity, round, max_rounds}. The client re-posts with the answer appended to `clarification`.
    """
    body = request.get_json(silent=True) or {}
    idea: str = (body.get("idea") or "").strip()
    limit: int = min(int(body.get("limit") or 10), 20)
    mode: str = (body.get("mode") or "keywords").strip()

    if mode != "keywords_user" and not idea:
        return jsonify({"error": "El campo 'idea' es obligatorio."}), 422

    logger.info("Search request - mode={} idea_len={} limit={}", mode, len(idea), limit)

    if mode == "keywords_user":
        # Topics typed by the user (no LLM involved): dedupe, trim and cap them, then run the same sparse BM42 search used for LLM-extracted keywords (single keyword set; the backend falls back to a one-leg query).
        seen: set[str] = set()
        keywords: list[str] = []
        for raw in body.get("keywords") or []:
            kw = str(raw).strip()[:60]
            if kw and kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
            if len(keywords) >= 15:
                break
        if not keywords:
            return jsonify({"error": "Añade al menos una temática para buscar."}), 422

        try:
            results = search_pdis(_db, keywords, [], limit=limit)
            logger.info("User-keyword search returned {} results", len(results))
        except Exception as exc:
            logger.exception("Qdrant user-keyword search failed")
            return jsonify({"error": f"Error en la búsqueda: {exc}"}), 500

        return jsonify(
            {
                "mode": "keywords_user",
                "keywords": keywords,
                "results": [_serialise_result(r) for r in results],
            }
        )

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
        # Optional second stage: 'text' re-ranks the top candidates with the listwise LLM; 'kg' additionally grounds each candidate with its knowledge-graph evidence (fase B). Empty = classic single-stage RRF.
        rerank_variant: str = (body.get("rerank") or "").strip().lower()
        if rerank_variant not in ("", "text", "kg"):
            return jsonify(
                {"error": f"Variante de re-ranking desconocida: '{rerank_variant}'."}
            ), 422

        # Optional ASK-style clarification loop (fase C, smart search): the client sends the completed rounds in `clarification` and may set  `clarify_skip` to force results ("buscar sin aclarar").
        clarify_on: bool = bool(body.get("clarify"))
        clarify_skip: bool = bool(body.get("clarify_skip"))
        history: list[ClarifyTurn] = []
        for raw in (body.get("clarification") or [])[: CLARIFY_MAX_ROUNDS * 2]:
            question = str((raw or {}).get("question") or "").strip()[:300]
            answer = str((raw or {}).get("answer") or "").strip()[:300]
            if question and answer:
                history.append(ClarifyTurn(question=question, answer=answer))

        # Refined query: the idea plus the clarification exchanges (ASK appends the answers to the query and relaunches the search).
        query_text = effective_idea(idea, history) if history else idea

        # Sparse leg: keyword extraction. Dense leg: enriched query (falls back to the raw idea if enrichment fails) so the summaries (written in English) are matched on comparable text.
        try:
            keywords = extract_keywords(query_text)
        except Exception as exc:
            logger.exception("Keyword extraction failed")
            return jsonify({"error": f"Error al extraer keywords: {exc}"}), 500

        enriched_query: str | None = None
        dense_text = query_text
        try:
            dense_text = enrich_query(query_text)
            enriched_query = dense_text
        except Exception:
            logger.warning("Query enrichment failed; using raw idea for dense leg")

        # The re-ranker reorders a pool deeper than the displayed page; the clarification loop retrieves deeper still, because the domain-level question draws its aspects from the whole pool.
        pool = limit
        if rerank_variant:
            pool = max(pool, RERANK_POOL)
        if clarify_on:
            pool = max(pool, CLARIFY_POOL)
        try:
            results = search_pdis_hybrid(
                _db, keywords.general, keywords.specific, dense_text, limit=pool
            )
            logger.info("Hybrid search returned {} results", len(results))
        except Exception as exc:
            logger.exception("Qdrant hybrid search failed")
            return jsonify({"error": f"Error en la búsqueda híbrida: {exc}"}), 500

        # ASK loop step: decide whether to ask ONE more clarifying question before ranking.
        # Every failure path inside degrades to answering.
        if clarify_on and not clarify_skip and results:
            clarification = next_clarification(idea, history, results)
            if clarification is not None:
                logger.info(
                    "Clarification round {} ({}): {}",
                    len(history) + 1,
                    clarification.ambiguity,
                    clarification.question,
                )
                return jsonify(
                    {
                        "mode": "hybrid",
                        "clarify": {
                            "question": clarification.question,
                            "options": clarification.options,
                            "ambiguity": clarification.ambiguity,
                            "round": len(history) + 1,
                            "max_rounds": CLARIFY_MAX_ROUNDS,
                        },
                    }
                )

        rerank_info: dict | None = None
        if rerank_variant and results:
            try:
                head, tail = results[:RERANK_POOL], results[RERANK_POOL:]
                head, rerank_info = rerank_pdi_results(
                    head, dense_text, use_kg=rerank_variant == "kg"
                )
                results = head + tail
                logger.info(
                    "Re-ranked {} candidates ({}, {} windows, {:.1f}s)",
                    rerank_info["pool"],
                    rerank_info["variant"],
                    rerank_info["windows"],
                    rerank_info["seconds"],
                )
            except Exception as exc:
                # Stage 1 already answered the query: degrade, don't fail.
                logger.exception("Re-ranking failed; returning stage-1 order")
                rerank_info = {"error": str(exc)}
        results = results[:limit]

        return jsonify(
            {
                "mode": "hybrid",
                "general_keywords": keywords.general,
                "specific_keywords": keywords.specific,
                "enriched_query": enriched_query,
                "rerank": rerank_info,
                "clarify": None,
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
