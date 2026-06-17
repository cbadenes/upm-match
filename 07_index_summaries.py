"""
Phase 7 — build the weighted "current" summary and index it in Qdrant.

Reads the per-year standalone summaries that phase 6 stored in MinIO (`history` in each researcher's enrichment document) and, for each researcher,
builds a single "current" summary by **recurrent fusion** over the last ``WINDOW_YEARS`` years (relative to the researcher's most recent active year),
giving more weight to recent work — following the recurrent-summarisation idea of *Harnessing LLMs for Text-Rich Sequential Recommendation* (Zheng et al., 2024).

For each researcher it then:
  - persists the fused `summary` and `domains` back into the enrichment JSON in
    MinIO (top level, outside `history`);
  - embeds the summary (dense, Ollama) + BM42 over the domains (sparse);
  - upserts a SINGLE point per researcher into the `researcher_summaries`
    Qdrant collection (used for recommendations).

The per-year summaries stay in MinIO for trend analysis; only the "current"
point is indexed.

USAGE
    uv run 07_index_summaries.py --recreate
    uv run 07_index_summaries.py --id 260726
    uv run 07_index_summaries.py --recreate --verbose

Requires MinIO, Qdrant and Ollama reachable (MINIO_*, QDRANT_*, OLLAMA_*).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import ollama as _ollama
from agno.agent import Agent
from agno.models.ollama import Ollama
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from dbs import minio
from dbs.vector import QdrantBackend
from utils.logging import setup_logging
from utils.researchers import ENRICHMENT_KEY_TEMPLATE as _ENRICHMENT_KEY_TEMPLATE

load_dotenv()

# Configuration

COLLECTION_NAME = "researcher_summaries"
DENSE_VECTOR_NAME = "dense"
DOMAINS_SPARSE_NAME = "domains_bm42"

WINDOW_YEARS = 10  # backwards from the most recent year in the researcher's history to consider for recurrent fusion.

_SUMMARY_MODEL = os.environ.get("OLLAMA_SUMMARY_MODEL", "qwen2.5:7b-instruct")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")

_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# BM42 sparse embedder, lazy (loads the model once on first use).
_sparse_backend: QdrantBackend | None = None


def _sparse() -> QdrantBackend:
    global _sparse_backend
    if _sparse_backend is None:
        _sparse_backend = QdrantBackend()
    return _sparse_backend


# Recurrent fusion (LLM)
class ExpertiseUpdate(BaseModel):
    domains: list[str] = Field(
        description="4–12 expertise areas, each 2–6 words, lowercase."
    )
    profile: str = Field(
        description="3–5 sentences, present tense, no tool/paper names, no time markers."
    )


_FUSION_DESCRIPTION = (
    "You maintain a researcher's CURRENT expertise profile as you fold in one "
    "more recent year of their work at a time. Your output matches students with "
    "thesis supervisors, so it must reflect what they work on NOW."
)
_FUSION_INSTRUCTIONS = [
    "You receive an ACCUMULATED profile (older years) and a MORE RECENT year's profile.",
    "Produce updated `domains` and `profile` that PRIORITISE the recent research direction,",
    "while keeping earlier expertise that is still clearly relevant.",
    "DOMAINS: 4–12 items, 2–6 words, lowercase. Merge both lists; favour recent ones;",
    "  drop clearly abandoned lines only if needed to stay within 12.",
    "PROFILE: 3 to 5 sentences (hard max 5), present tense. Lead with the current focus.",
    "  No semicolons to chain items. No tool/system/dataset names, no paper/journal titles.",
    "  NEVER use time markers ('recently', 'currently', 'this year', 'continued', etc.).",
]


def _run_fusion(prompt: str) -> ExpertiseUpdate | None:
    """Call the fusion agent and return an ExpertiseUpdate, or None on error."""
    agent = Agent(
        model=Ollama(id=_SUMMARY_MODEL, host=_OLLAMA_HOST),
        description=_FUSION_DESCRIPTION,
        instructions=_FUSION_INSTRUCTIONS,
        output_schema=ExpertiseUpdate,
        markdown=False,
    )
    try:
        response = agent.run(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fusion agent failed: {}", exc)
        return None
    content = response.content
    if isinstance(content, ExpertiseUpdate):
        return content
    if isinstance(content, dict):
        try:
            return ExpertiseUpdate.model_validate(content)
        except Exception:  # noqa: BLE001
            pass
    logger.error("Fusion agent returned unexpected content: {}", type(content))
    return None


def _window_years(history: dict) -> list[int]:
    """Sorted (ascending) years in `history` within the last WINDOW_YEARS of the most recent one."""
    years: list[int] = []
    for y in history:
        try:
            years.append(int(y))
        except (TypeError, ValueError):
            continue
    if not years:
        return []
    cutoff = max(years) - (WINDOW_YEARS - 1)
    return sorted(y for y in years if y >= cutoff)


def _build_current(history: dict) -> tuple[str, list[str], int] | None:
    """Recurrently fuse the windowed per-year summaries into the current summary.

    :param history: the `history` dict from a researcher's enrichment document.
    :return summary, domains, last_year: the fused summary, domains, and the most recent year used; or None if no valid years.
    """
    years = _window_years(history)
    if not years:
        return None

    def snap(y: int) -> tuple[str, list[str]]:
        s = history.get(str(y)) or {}
        return (s.get("profile") or "").strip(), (s.get("domains") or [])

    # Seed with the oldest year in the window.
    cur_profile, cur_domains = snap(years[0])
    # Fold each more-recent year in (recurrent), giving weight to recent work.
    for y in years[1:]:
        new_profile, new_domains = snap(y)
        if not new_profile:
            continue
        if not cur_profile:
            cur_profile, cur_domains = new_profile, new_domains
            continue
        prompt = (
            "ACCUMULATED profile (older years):\n"
            f"domains: {', '.join(cur_domains)}\n{cur_profile}\n\n"
            f"MORE RECENT year ({y}) profile:\n"
            f"domains: {', '.join(new_domains)}\n{new_profile}\n\n"
            "Merge into the updated current profile and domains."
        )
        result = _run_fusion(prompt)
        if result is not None and result.profile.strip():
            cur_profile, cur_domains = result.profile.strip(), result.domains

    if not cur_profile:
        return None
    return cur_profile, cur_domains, years[-1]


# MinIO
def _iter_enrichment_keys(only_id: int | None) -> list[str]:
    if only_id is not None:
        return [_ENRICHMENT_KEY_TEMPLATE.format(researcher_id=only_id)]
    keys = minio.list_objects(
        minio.CURATED_BUCKET, prefix="researchers/", recursive=True
    )
    return [k for k in keys if k.endswith("/enrichment.json")]


def _persist_current(researcher_id: str, summary: str, domains: list[str]) -> None:
    """Write the fused `summary`/`domains` back to the enrichment JSON (top level)."""
    key = _ENRICHMENT_KEY_TEMPLATE.format(researcher_id=_num(researcher_id))
    data = minio.read_json(minio.CURATED_BUCKET, key)
    if data is None:
        logger.warning(
            "Enrichment JSON not found for {} — cannot persist summary", researcher_id
        )
        return
    data["summary"] = summary
    data["domains"] = domains
    minio.upload_json(minio.CURATED_BUCKET, data, key)


# Qdrant
def _qdrant() -> QdrantClient:
    return QdrantClient(
        url=f"http://{_QDRANT_HOST}:{_QDRANT_PORT}", api_key=_QDRANT_API_KEY
    )


def _create_collection(client: QdrantClient, collection: str, dim: int) -> None:
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: qdrant_models.VectorParams(
                size=dim, distance=qdrant_models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            DOMAINS_SPARSE_NAME: qdrant_models.SparseVectorParams(
                index=qdrant_models.SparseIndexParams(on_disk=True),
                modifier=qdrant_models.Modifier.IDF,
            )
        },
    )
    for field, schema in (
        ("researcher_id", qdrant_models.PayloadSchemaType.KEYWORD),
        ("year", qdrant_models.PayloadSchemaType.INTEGER),
        ("domains", qdrant_models.PayloadSchemaType.KEYWORD),
        ("position_names", qdrant_models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=collection, field_name=field, field_schema=schema
        )
    logger.info(
        "Created collection '{}' (dense '{}' {}-dim + sparse '{}' BM42/IDF)",
        collection,
        DENSE_VECTOR_NAME,
        dim,
        DOMAINS_SPARSE_NAME,
    )


def _versioned_name() -> str:
    """A unique physical collection name for a fresh build."""
    return f"{COLLECTION_NAME}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"


def _alias_target(client: QdrantClient) -> str | None:
    """Physical collection the `researcher_summaries` alias points to, or None."""
    try:
        for a in client.get_aliases().aliases:
            if a.alias_name == COLLECTION_NAME:
                return a.collection_name
    except Exception:  # noqa: BLE001
        pass
    return None


def _set_alias(client: QdrantClient, target: str) -> None:
    """Point the `researcher_summaries` alias at `target` atomically.

    If a *real* collection (legacy, pre-alias layout) occupies the alias name,
    drop it first so the alias can take over the name.
    """
    if target == COLLECTION_NAME:
        return  # legacy in-place collection; nothing to alias
    if client.collection_exists(COLLECTION_NAME) and _alias_target(client) is None:
        client.delete_collection(COLLECTION_NAME)
    client.update_collection_aliases(
        change_aliases_operations=[
            qdrant_models.CreateAliasOperation(
                create_alias=qdrant_models.CreateAlias(
                    collection_name=target, alias_name=COLLECTION_NAME
                )
            )
        ]
    )
    logger.info("Alias '{}' now points to '{}'", COLLECTION_NAME, target)


def _cleanup_old_collections(client: QdrantClient, keep: str) -> None:
    """Delete stale versioned build collections, keeping only `keep`."""
    prefix = f"{COLLECTION_NAME}_"
    for c in client.get_collections().collections:
        if c.name != keep and c.name.startswith(prefix):
            client.delete_collection(c.name)
            logger.info("Deleted stale collection '{}'", c.name)


def _embed_dense(text: str) -> list[float]:
    return (
        _ollama.Client(host=_OLLAMA_HOST)
        .embed(model=_EMBED_MODEL, input=text)
        .embeddings[0]
    )


def _num(researcher_id: str) -> int:
    """Numeric part of a researcher id ('pdi_303799' -> 303799)."""
    return int(str(researcher_id).split("_")[-1])


def _upsert(
    client: QdrantClient,
    collection: str,
    doc: dict,
    summary: str,
    domains: list[str],
    last_year: int,
    embedding: list[float],
) -> None:
    indices, values = _sparse().embed(domains)
    vector: dict[str, object] = {DENSE_VECTOR_NAME: embedding}
    if indices:
        vector[DOMAINS_SPARSE_NAME] = qdrant_models.SparseVector(
            indices=indices, values=values
        )
    full_name = f"{doc.get('first_name') or ''} {doc.get('last_name') or ''}".strip()
    positions = [
        p.get("name") or p.get("entity")
        for p in (doc.get("positions") or [])
        if (p.get("name") or p.get("entity"))
    ]
    payload = {
        "researcher_id": doc["id"],
        "name": full_name,
        "year": last_year,
        "summary": summary,
        "domains": domains,
        "topics": (doc.get("topics") or [])[:20],
        "position_names": positions,
        "publication_count": len(doc.get("publications") or []),
        "subject_count": len(doc.get("subjects") or []),
        "last_year_project_count": len(doc.get("last_year_projects") or []),
        "summary_model": _SUMMARY_MODEL,
        "embedding_model": _EMBED_MODEL,
        "embedding_dim": len(embedding),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(
        collection_name=collection,
        points=[
            qdrant_models.PointStruct(
                id=_num(doc["id"]), vector=vector, payload=payload
            )
        ],
    )


# Orchestration
def run(only_id: int | None, recreate: bool, limit: int | None) -> int:
    keys = _iter_enrichment_keys(only_id)
    if limit is not None:
        keys = keys[:limit]
    logger.info("{} enrichment documents to index", len(keys))

    client = _qdrant()

    # Resolve the physical collection to write into.
    # We NEVER write into the live collection the app serves:
    # a full rebuild (--recreate) goes into a fresh versioned collection and only swaps the `researcher_summaries` alias once the build completes with zero failures.
    # So a crash mid-run can never destroy the data already being served, the previous collection stays untouched.
    current = _alias_target(client)
    if current is None and client.collection_exists(COLLECTION_NAME):
        current = COLLECTION_NAME  # legacy in-place collection (pre-alias)

    build_fresh = recreate or current is None
    if build_fresh:
        target = _versioned_name()
        collection_ready = False
        logger.info(
            "Indexing into fresh collection '{}' (alias swap on success)", target
        )
    else:
        target = current
        collection_ready = True
        logger.info("Upserting in place into '{}'", target)

    stats = {"indexed": 0, "no_history": 0, "failed": 0}
    for i, key in enumerate(keys, 1):
        if i % 50 == 0:
            logger.info("Progress: {}/{} ({} indexed)", i, len(keys), stats["indexed"])
        doc = minio.read_json(minio.CURATED_BUCKET, key)
        if not doc or not doc.get("id"):
            stats["failed"] += 1
            continue
        history = doc.get("history") or {}
        if not history:
            stats["no_history"] += 1
            continue
        try:
            built = _build_current(history)
            if built is None:
                stats["no_history"] += 1
                continue
            summary, domains, last_year = built
            _persist_current(doc["id"], summary, domains)
            embedding = _embed_dense(summary)
            if not collection_ready:
                _create_collection(client, target, len(embedding))
                collection_ready = True
            _upsert(client, target, doc, summary, domains, last_year, embedding)
            stats["indexed"] += 1
        except Exception as exc:  # noqa: BLE001 — robustez por investigador
            logger.error("Failed to index {}: {}", doc.get("id"), exc)
            stats["failed"] += 1

    logger.info(
        "Done — indexed: {}, no history: {}, failed: {}",
        stats["indexed"],
        stats["no_history"],
        stats["failed"],
    )

    # Publish a freshly built collection only if it is complete.
    # On any failure we keep the previous live collection and discard the partial build, so the app never points at an incomplete or empty collection.
    if build_fresh and collection_ready:
        if stats["failed"] == 0 and stats["indexed"] > 0:
            _set_alias(client, target)
            _cleanup_old_collections(client, keep=target)
        else:
            logger.warning(
                "Build incomplete (failed={}); keeping previous collection and "
                "discarding partial build '{}'",
                stats["failed"],
                target,
            )
            try:
                client.delete_collection(target)
            except Exception:  # noqa: BLE001
                pass

    return 0 if stats["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7: build the weighted current summary and index it in Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--id",
        dest="only_id",
        type=int,
        default=None,
        help="Index only this researcher (e.g. 260726).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents (for testing).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info("=== UPM Match — Phase 7: current-summary indexing ===")
    logger.info("Summary model: {}  |  Embed model: {}", _SUMMARY_MODEL, _EMBED_MODEL)
    return run(args.only_id, args.recreate, args.limit)


if __name__ == "__main__":
    sys.exit(main())
