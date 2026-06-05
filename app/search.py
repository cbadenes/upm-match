"""PDI search logic: sparse keyword retrieval and dense summary retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import ollama as _ollama
from dotenv import load_dotenv

from dbs.vector import QdrantBackend, SearchResult

load_dotenv()

COLLECTION_NAME = "researcher_keywords"
SPARSE_FIELD = "keywords_bm42"
SUMMARY_COLLECTION = "researcher_summaries"
DEFAULT_LIMIT = 10

_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")


@dataclass
class PDIResult:
    """A ranked PDI recommendation with interpretability data."""

    rank: int
    researcher_id: str
    name: str
    score: float
    score_pct: int  # normalised 0-100 relative to the top result
    matched_topics: list[str] = field(default_factory=list)
    all_topics: list[str] = field(default_factory=list)
    since: int | None = None
    summary: str | None = None


def _matched_topics(query_keywords: list[str], topics: list[str]) -> list[str]:
    """Return the researcher topics that overlap with any query keyword.
    Uses containment matching at the phrase level: a topic matches if it contains a query keyword as a substring, or vice versa.

    :param query_keywords: List of keywords from the agent (both general and specific).
    :param topics: List of researcher topics from the PDI profile.
    :return: List of topics that match any of the query keywords.
    """
    matched: list[str] = []
    for topic in topics:
        topic_lower = topic.lower()
        for kw in query_keywords:
            kw_lower = kw.lower()
            if (
                kw_lower == topic_lower
                or kw_lower in topic_lower
                or topic_lower in kw_lower
            ):
                matched.append(topic)
                break
    return matched


def _to_pdi_results(
    raw: list[SearchResult],
    all_keywords: list[str],
) -> list[PDIResult]:
    """Convert raw Qdrant results to annotated PDIResult objects.
    This includes calculating the normalised score percentage and determining which researcher topics match the query keywords for interpretability.

    :param raw: List of SearchResult objects returned by the Qdrant search.
    :param all_keywords: Combined list of general and specific keywords used in the search, for interpretability scoring.
    :return: List of PDIResult objects with rank, score, matched topics and other metadata.
    """
    if not raw:
        return []

    max_score = raw[0].score or 1.0  # results are already sorted by score desc

    results: list[PDIResult] = []
    for rank, sr in enumerate(raw, 1):
        payload = sr.payload
        topics: list[str] = payload.get("topics") or []
        matched = _matched_topics(all_keywords, topics)
        results.append(
            PDIResult(
                rank=rank,
                researcher_id=payload.get("researcher_id", ""),
                name=payload.get("name") or "Unknown",
                score=round(sr.score, 4),
                score_pct=round((sr.score / max_score) * 100),
                matched_topics=matched,
                all_topics=topics[:12],
                since=payload.get("since") or None,
            )
        )
    return results


def search_pdis(
    db: QdrantBackend,
    general_keywords: list[str],
    specific_keywords: list[str],
    limit: int = DEFAULT_LIMIT,
) -> list[PDIResult]:
    """Find the most relevant PDIs for a pair of keyword lists.

    Runs two independent BM42 sparse queries (one per list) and merges the ranked results using Qdrant's Reciprocal Rank Fusion
    This ensures that both the broad domain signal and the specific technique signal contribute to the final ranking.

    :param db: Initialised QdrantBackend instance.
    :param general_keywords: Broad domain keywords from the agent.
    :param specific_keywords: Specific method/technique keywords from the agent.
    :param limit: Number of top results to return.
    :return: Ranked list of PDIResult objects.
    """
    raw = db.search_by_keyword_sets(
        collection=COLLECTION_NAME,
        sparse_field=SPARSE_FIELD,
        general_keywords=general_keywords,
        specific_keywords=specific_keywords,
        limit=limit,
    )
    all_keywords = general_keywords + specific_keywords
    return _to_pdi_results(raw, all_keywords)


# ── Dense summary search ───────────────────────────────────────────────────────


def _embed_for_summary(text: str) -> list[float]:
    """Embed a query text with the same Ollama model used to index researcher summaries."""
    client = _ollama.Client(host=_OLLAMA_HOST)
    response = client.embed(model=_EMBED_MODEL, input=text)
    return response.embeddings[0]


def _to_pdi_results_summary(raw: list[SearchResult]) -> list[PDIResult]:
    """Convert dense search results from the researcher_summaries collection to PDIResult objects.

    :param raw: List of SearchResult objects from the dense search.
    :return: List of PDIResult objects with rank, score, summary text, and topics.
    """
    if not raw:
        return []

    max_score = raw[0].score or 1.0

    results: list[PDIResult] = []
    for rank, sr in enumerate(raw, 1):
        payload = sr.payload
        topics: list[str] = payload.get("topics") or []
        results.append(
            PDIResult(
                rank=rank,
                researcher_id=str(payload.get("researcher_id", "")),
                name=payload.get("name") or "Unknown",
                score=round(sr.score, 4),
                score_pct=round((sr.score / max_score) * 100),
                matched_topics=[],
                all_topics=topics[:12],
                since=payload.get("first_year") or None,
                summary=payload.get("summary"),
            )
        )
    return results


def search_pdis_by_summary(
    db: QdrantBackend,
    query_text: str,
    limit: int = DEFAULT_LIMIT,
) -> list[PDIResult]:
    """Find the most relevant PDIs by dense cosine similarity against researcher summaries.

    Embeds query_text with the same model used during indexing (OLLAMA_EMBED_MODEL)
    and queries the researcher_summaries Qdrant collection.

    :param db: Initialised QdrantBackend instance.
    :param query_text: Text to embed — either the raw student idea or an LLM-enriched version.
    :param limit: Number of top results to return.
    :return: Ranked list of PDIResult objects.
    """
    embedding = _embed_for_summary(query_text)
    raw = db.search_dense(
        collection=SUMMARY_COLLECTION,
        vector=embedding,
        limit=limit,
    )
    return _to_pdi_results_summary(raw)
