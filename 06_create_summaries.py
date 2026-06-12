"""
Rolling biography summarization pipeline for UPM Match.

For each researcher, groups publications by year and builds an iterative rolling biography using an LLM (Agno + Ollama).
The final summary is embedded with a dedicated dense model and indexed in Qdrant for semantic search against student TFG/TFM queries.

Usage:
    uv run 06_create_summaries.py --id 308607
    uv run 06_create_summaries.py --id 308607 --debug
    uv run 06_create_summaries.py --id 308607 --debug output/308607.md
    uv run 06_create_summaries.py --skip-existing --verbose
    uv run 06_create_summaries.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
from models import ResearcherEnrichmentDocument
from utils.logging import setup_logging
from utils.researchers import ENRICHMENT_KEY_TEMPLATE as _ENRICHMENT_KEY_TEMPLATE
from utils.researchers import load_enrichment as _load_enrichment
from utils.researchers import load_researcher_ids as _load_researcher_ids

load_dotenv()

# Ollama
_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")
_SUMMARY_MODEL = os.environ.get("OLLAMA_SUMMARY_MODEL", "qwen2.5:7b-instruct")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
# Qdrant
_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "researcher_summaries"
DENSE_VECTOR_NAME = "dense"
DOMAINS_SPARSE_NAME = "domains_bm42"
# A single year with many publications can overflow the model context window and make Ollama drop the connection (losing the whole year).
# When a year exceeds this many publications we feed them to the agent in successive batches so nothing is lost.
PUBS_PER_BATCH = int(os.environ.get("SUMMARY_PUBS_PER_BATCH", "10"))

_AGENT_DESCRIPTION = (
    "You are an academic research profiler building a cumulative picture of a researcher's expertise."
    "Your output is used to match students with thesis supervisors."
)
_AGENT_INSTRUCTIONS = [
    "You produce two outputs: `domains` (list) and `profile` (prose).",
    "DOMAINS — cumulative list of expertise areas:",
    "  * 4–12 items, each 2–6 words, lowercase (e.g. 'knowledge graphs', 'named entity recognition').",
    "  * When given established domains: output ALL of them unchanged, then append genuinely new areas found in the abstracts.",
    "  * When starting fresh: extract the main research areas from the abstracts.",
    "PROFILE — 3 to 5 proper English sentences (hard maximum: 5):",
    "  * Sentences 1–4: describe what this researcher is deeply expert in, grouping related domains naturally.",
    "  * Final sentence (optional, only when new domains were just added): naturally introduce the new research direction.",
    "  * If no new domains were added, write only the core expertise — do not reference time, recency, or change.",
    "  * No semicolons to chain items. Each sentence ends with a period. Present tense throughout.",
    "  * NEVER mention: software, system, tool, dataset, or platform names (no proper nouns).",
    "  * NEVER mention: paper titles, conference names, or journal names.",
    "  * NEVER use: 'continued', 'expanded', 'recently', 'this year', 'currently', or any time marker.",
]


class ExpertiseUpdate(BaseModel):
    domains: list[str] = Field(
        description=(
            "Cumulative list of 4–12 expertise areas, each 2–6 words, lowercase."
            "Includes ALL items from the established list plus any new areas from the current abstracts."
        )
    )
    profile: str = Field(
        description=(
            "3 to 5 sentences (max 5) covering all domains."
            "Core expertise first; new direction last only if new domains were added."
            "No tool/system names, no paper titles, no time markers."
        )
    )


_DEDUP_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _dedup_key(text: str | None) -> str:
    """Normalise a string for duplicate detection: lowercase, alnum-only."""
    if not text:
        return ""
    return _DEDUP_NORMALIZE_RE.sub("", text.lower()).strip()


def _dedupe_publications(publications: list) -> list:
    """Drop near-duplicate publications, keeping the most informative copy.

    Duplicates are detected by normalised title (falling back to the normalised abstract prefix when the title is missing).
    The portal frequently exposes the same work several times (e.g. preprint + journal, or across sources), which wastes context and over-weights its domains in the rolling summary.
    When two copies collide the one with the longer abstract is kept.
    """
    best: dict[str, object] = {}
    order: list[str] = []
    for pub in publications:
        key = _dedup_key(pub.title) or _dedup_key(pub.abstract)[:120]
        if not key:
            key = f"__id__{pub.id}"  # no text at all: never merge distinct items
        if key not in best:
            best[key] = pub
            order.append(key)
        elif len(pub.abstract or "") > len(getattr(best[key], "abstract", "") or ""):
            best[key] = pub
    return [best[key] for key in order]


def _group_pubs_by_year(
    enrichment: ResearcherEnrichmentDocument,
) -> dict[int, list[dict]]:
    """Return publications grouped by year, sorted chronologically.

    Near-duplicate publications are removed first.
    Only keeps fields that are meaningful for summarization (title, abstract, keywords).
    Publications with no year or no textual content are silently dropped.
    """
    publications = _dedupe_publications(enrichment.publications)
    removed = len(enrichment.publications) - len(publications)
    if removed:
        logger.debug(
            "{}: dropped {} near-duplicate publication(s)", enrichment.id, removed
        )

    by_year: defaultdict[int, list[dict]] = defaultdict(list)
    for pub in publications:
        if not pub.year or pub.year <= 0:
            continue
        abstract = pub.abstract or ""
        title = pub.title or ""
        # Skip pubs with no useful textual content
        if not abstract and not title:
            continue
        entry: dict = {"abstract": abstract} if abstract else {}
        # Include title when abstract is absent or too short to be informative
        if len(abstract) < 50 and title:
            entry["title"] = title
        if entry:
            by_year[pub.year].append(entry)

    # Within each year keep only the most informative entries (longest abstract first)
    return {
        year: sorted(
            entries, key=lambda e: len(e.get("abstract", "")), reverse=True
        )  # [:MAX_PUBS_PER_YEAR]
        for year, entries in sorted(by_year.items())
    }


def _run_agent(prompt: str) -> ExpertiseUpdate | None:
    """Call the summarization agent and return an ExpertiseUpdate, or None on error."""
    agent = Agent(
        model=Ollama(id=_SUMMARY_MODEL, host=_OLLAMA_HOST),
        description=_AGENT_DESCRIPTION,
        instructions=_AGENT_INSTRUCTIONS,
        output_schema=ExpertiseUpdate,
        markdown=False,
    )
    try:
        response = agent.run(prompt)
    except Exception as exc:
        logger.error("Agent call failed: {}", exc)
        return None

    # Agno wraps connection/model errors as plain strings instead of raising
    if isinstance(response.content, str):
        logger.error("Ollama returned an error string: {}", response.content)
        return None
    if isinstance(response.content, ExpertiseUpdate):
        return response.content
    if isinstance(response.content, dict):
        try:
            return ExpertiseUpdate.model_validate(response.content)
        except Exception:
            pass
    logger.error("Agent returned unexpected content type: {}", type(response.content))
    return None


def _chunk(items: list[dict], size: int) -> list[list[dict]]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_rolling_summary(
    enrichment: ResearcherEnrichmentDocument,
    debug_output: Path | bool | None,
) -> tuple[str, list[str], dict[str, dict]] | None:
    """Iterates over publication years and builds an accumulating biography.

    Returns (final summary text, final domains list, history), or None if no usable publications exist.

    history maps each processed year (as string) to its final ExpertiseUpdate snapshot: {"domains": [...], "profile": "..."}.
    Multi-batch years produce one entry per year (after the last batch), not one per batch.

    Years with more than PUBS_PER_BATCH publications are fed to the agent in successive batches.
    The continuation prompt explicitly tells the model it is still within the same year so it does not treat each batch as a new research period.

    `debug_output` options:
      None  - production mode, no intermediate output
      True  - print each batch's summary to stdout (no Qdrant write)
      Path  - write all intermediate summaries to a markdown file (no Qdrant write)
    """
    by_year = _group_pubs_by_year(enrichment)
    if not by_year:
        return None

    full_name = (
        f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
        or str(enrichment.id)
    )

    total_pubs = sum(len(pubs) for pubs in by_year.values())
    logger.info(
        "Generating rolling summary for {} ({}) - {} years, {} publications",
        enrichment.id,
        full_name,
        len(by_year),
        total_pubs,
    )

    prev_domains: list[str] = []
    prev_profile: str | None = None
    history: dict[str, dict] = {}
    md_lines: list[str] = []

    if isinstance(debug_output, Path):
        md_lines.append(f"# Research Biography: {full_name} ({enrichment.id})\n\n")

    for year, pubs in by_year.items():
        batches = _chunk(pubs, PUBS_PER_BATCH)
        multi_batch = len(batches) > 1
        if multi_batch:
            logger.debug(
                "{} - year {} has {} pubs, processing in {} batches of <= {}",
                enrichment.id,
                year,
                len(pubs),
                len(batches),
                PUBS_PER_BATCH,
            )

        for batch_idx, batch_pubs in enumerate(batches, 1):
            pubs_json = json.dumps(batch_pubs, ensure_ascii=False, indent=2)
            batch_label = (
                f"{year} (batch {batch_idx}/{len(batches)})"
                if multi_batch
                else str(year)
            )
            is_continuation = batch_idx > 1

            if not prev_domains:
                # First batch ever = no established profile yet.
                prompt = (
                    f"Researcher: {full_name}\n\n"
                    f"Abstracts ({batch_label}, {len(batch_pubs)} publication(s)):\n{pubs_json}\n\n"
                    "Extract the researcher's expertise domains and write the profile."
                )
            elif is_continuation:
                # More publications from the same year = the model must not treat this as a new research period; it is still accumulating year {year}.
                domains_bullets = "\n".join(f"- {d}" for d in prev_domains)
                prompt = (
                    f"Researcher: {full_name}\n\n"
                    f"Expertise domains so far in {year} (all must appear in `domains` output):\n{domains_bullets}\n\n"
                    f"More abstracts from {year} (batch {batch_idx}/{len(batches)}, {len(batch_pubs)} publication(s)):\n{pubs_json}\n\n"
                    f"These publications belong to the same year ({year}) as the previous batch — "
                    "do NOT treat this as a new research period.\n"
                    "1. Add to `domains` any research areas in these abstracts NOT already listed.\n"
                    "2. Write the `profile` covering all domains.\n"
                    "   If new domains were added, end with one sentence on the new direction.\n"
                    "   If not, describe expertise without any reference to time or change."
                )
            else:
                # First batch of a new year = standard rolling update.
                domains_bullets = "\n".join(f"- {d}" for d in prev_domains)
                prompt = (
                    f"Researcher: {full_name}\n\n"
                    f"Established expertise domains (all must appear in `domains` output):\n{domains_bullets}\n\n"
                    f"New abstracts ({batch_label}, {len(batch_pubs)} publication(s)):\n{pubs_json}\n\n"
                    "1. Add to `domains` any research areas in these abstracts NOT already in the established list.\n"
                    "2. Write the `profile` covering all domains."
                    "If new domains were added, end with one sentence introducing this new direction."
                    "If not, describe the researcher's expertise without any reference to time or change."
                )

            logger.debug(
                "Summarizing {} - {} ({} pubs)",
                enrichment.id,
                batch_label,
                len(batch_pubs),
            )
            result = _run_agent(prompt)

            if result is None:
                logger.warning(
                    "Agent produced no output for {} / {} - keeping previous result",
                    enrichment.id,
                    batch_label,
                )
                continue

            prev_domains = result.domains
            prev_profile = result.profile

            # Record the year snapshot after the final batch of this year.
            if batch_idx == len(batches):
                history[str(year)] = {
                    "domains": result.domains,
                    "profile": result.profile,
                }

            if debug_output is True:
                domains_str = ", ".join(result.domains)
                print(
                    f"\n--- {full_name} | {batch_label} ({len(batch_pubs)} pub(s)) ---"
                )
                print(f"Domains: [{domains_str}]")
                print(result.profile)
            elif isinstance(debug_output, Path):
                domains_str = ", ".join(result.domains)
                md_lines.append(
                    f"## {batch_label} ({len(batch_pubs)} publication(s))\n\n"
                    f"**Domains:** {domains_str}\n\n{result.profile}\n\n---\n\n"
                )

    if isinstance(debug_output, Path) and md_lines:
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        debug_output.write_text("".join(md_lines), encoding="utf-8")
        logger.info("Debug markdown written to {}", debug_output)

    if prev_profile is None:
        return None
    return prev_profile, prev_domains, history


# Qdrant dense helpers
def _embed_text(text: str) -> list[float]:
    """Generate a dense embedding via Ollama's embed endpoint."""
    client = _ollama.Client(host=_OLLAMA_HOST)
    response = client.embed(model=_EMBED_MODEL, input=text)
    return response.embeddings[0]


# Sparse (BM42) embedder for the expertise domains.
# Reuses QdrantBackend so the sparse model and text normalisation match phase 05 and the app's query side.
_sparse_backend: QdrantBackend | None = None


def _embed_domains_sparse(domains: list[str]) -> tuple[list[int], list[float]]:
    """Build a BM42 sparse vector from the expertise domains list."""
    global _sparse_backend
    if _sparse_backend is None:
        _sparse_backend = QdrantBackend()
    return _sparse_backend.embed(domains)


def _ensure_dense_collection(qdrant: QdrantClient, dim: int) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        logger.debug("Collection '{}' already exists", COLLECTION_NAME)
        return
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qdrant_models.VectorParams(
                size=dim,
                distance=qdrant_models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            DOMAINS_SPARSE_NAME: qdrant_models.SparseVectorParams(
                index=qdrant_models.SparseIndexParams(on_disk=False),
                modifier=qdrant_models.Modifier.IDF,
            )
        },
    )
    logger.info(
        "Created Qdrant collection '{}' (dense '{}' {}-dim cosine + sparse '{}' BM42/IDF)",
        COLLECTION_NAME,
        DENSE_VECTOR_NAME,
        dim,
        DOMAINS_SPARSE_NAME,
    )

    # Payload indexes enable fast filtering in future hybrid/filtered searches.
    for field_name, schema in (
        ("year", qdrant_models.PayloadSchemaType.INTEGER),
        ("position_names", qdrant_models.PayloadSchemaType.KEYWORD),
        ("domains", qdrant_models.PayloadSchemaType.KEYWORD),
    ):
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=schema,
        )
    logger.info("Created payload indexes on year, position_names, domains")


def _build_payload(
    researcher_id: int,
    summary: str,
    domains: list[str],
    enrichment: ResearcherEnrichmentDocument,
    embedding_dim: int,
    indexed_at: str,
) -> dict:
    """Build the payload dict for a researcher summary point, including provenance metadata.

    :param researcher_id: The researcher's unique ID (PDI).
    :param summary: The final rolling summary text for this researcher.
    :param domains: The final list of expertise domains for this researcher.
    :param enrichment: The full enrichment document, used to extract metadata like name, year, positions, and counts.
    :param embedding_dim: The dimension of the dense embedding vector, recorded in the payload for future reference.
    :param indexed_at: ISO timestamp of when this summary was indexed, recorded in the payload for future reference.
    :return: A dict containing all payload fields for the Qdrant point, including the summary, domains, metadata, and provenance information.
    """
    full_name = f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
    year = max(
        (p.year for p in enrichment.publications if p.year and p.year > 0),
        default=None,
    )
    # positions: {name → role} dict for rich payload access.
    # position_names: flat list of keys for Qdrant KEYWORD index filtering.
    positions: dict[str, str] = {}
    for p in enrichment.positions:
        key = p.name or p.entity
        if key:
            positions[key] = p.role or ""
    return {
        "researcher_id": enrichment.id,
        "name": full_name,
        "summary": summary,
        "domains": domains,
        "topics": enrichment.topics[:20],
        "year": year,
        "positions": positions,
        "position_names": list(positions.keys()),
        "publication_count": len(enrichment.publications),
        "subject_count": len(enrichment.subjects),
        "last_year_project_count": len(enrichment.last_year_projects),
        # Provenance metadata: which models produced the representation and text.
        "summary_model": _SUMMARY_MODEL,
        "embedding_model": _EMBED_MODEL,
        "embedding_dim": embedding_dim,
        "indexed_at": indexed_at,
    }


def _upsert_point(
    qdrant: QdrantClient,
    researcher_id: int,
    embedding: list[float],
    domains_sparse: tuple[list[int], list[float]],
    payload: dict,
) -> None:
    """Upsert a single point for this researcher into Qdrant, including the dense embedding, sparse domains vector, and payload.

    :param qdrant: An instance of QdrantClient connected to the target Qdrant service.
    :param researcher_id: The researcher's unique ID (PDI), used as the point ID in Qdrant.
    :param embedding: The dense embedding vector for the summary text, generated by the embed model.
    :param domains_sparse: A tuple containing the sparse vector indices and values for the expertise domains, generated by the BM42 sparse embedder.
    :param payload: A dict containing all payload fields for the point, including the summary, domains, metadata, and provenance information.
    """
    vector: dict[str, object] = {DENSE_VECTOR_NAME: embedding}
    sparse_indices, sparse_values = domains_sparse
    if sparse_indices:
        vector[DOMAINS_SPARSE_NAME] = qdrant_models.SparseVector(
            indices=sparse_indices, values=sparse_values
        )
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qdrant_models.PointStruct(
                id=researcher_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    logger.info("PDI {} upserted into dense summaries collection", researcher_id)


def _persist_history(researcher_id: int, history: dict[str, dict]) -> None:
    """Patch the researcher's enrichment JSON in MinIO with the generated history."""
    key = _ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    data = minio.read_json(minio.CURATED_BUCKET, key)
    if data is None:
        logger.warning(
            "Enrichment JSON not found for {} — cannot persist history to MinIO",
            researcher_id,
        )
        return
    data["history"] = history
    if not minio.upload_json(minio.CURATED_BUCKET, data, key):
        logger.warning(
            "Failed to update enrichment JSON for {} in MinIO", researcher_id
        )
    else:
        logger.debug(
            "History for {} written to enrichment JSON in MinIO", researcher_id
        )


# Main processing loop
def run(
    researcher_ids: list[int],
    debug_output: Path | bool | None,
    skip_existing: bool,
    recreate: bool = False,
) -> dict[str, int]:
    stats: dict[str, int] = {
        "indexed": 0,
        "skipped": 0,
        "no_enrichment": 0,
        "no_publications": 0,
        "failed": 0,
    }

    qdrant: QdrantClient | None = None
    collection_ready = False
    existing_ids: set[int] = set()

    if not debug_output:
        qdrant = QdrantClient(
            url=f"http://{_QDRANT_HOST}:{_QDRANT_PORT}",
            api_key=_QDRANT_API_KEY,
        )
        if recreate and qdrant.collection_exists(COLLECTION_NAME):
            logger.info("Recreating collection '{}' from scratch", COLLECTION_NAME)
            qdrant.delete_collection(COLLECTION_NAME)
        if skip_existing and qdrant.collection_exists(COLLECTION_NAME):
            records, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=10_000,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {r.id for r in records}
            logger.info("{} researchers already indexed", len(existing_ids))

    total = len(researcher_ids)
    for i, researcher_id in enumerate(researcher_ids, 1):
        if i % 50 == 0:
            logger.info("Progress: {}/{}", i, total)

        if researcher_id in existing_ids:
            logger.debug("Skipping {} (already indexed)", researcher_id)
            stats["skipped"] += 1
            continue

        enrichment = _load_enrichment(researcher_id)
        if enrichment is None:
            logger.warning(
                "No enrichment JSON in bucket for {} — skipping", researcher_id
            )
            stats["no_enrichment"] += 1
            continue

        result = _build_rolling_summary(enrichment, debug_output)
        if result is None:
            logger.debug("No usable publications for {}", researcher_id)
            stats["no_publications"] += 1
            continue
        summary, domains, history = result

        if debug_output:
            # Debug mode: count as processed but don't write to Qdrant
            stats["indexed"] += 1
            continue

        # Production mode: embed and upsert into Qdrant, then persist to MinIO
        try:
            embedding = _embed_text(summary)
            domains_sparse = _embed_domains_sparse(domains)
            if not collection_ready:
                _ensure_dense_collection(qdrant, len(embedding))
                collection_ready = True
            indexed_at = datetime.now(timezone.utc).isoformat()
            payload = _build_payload(
                researcher_id,
                summary,
                domains,
                enrichment,
                len(embedding),
                indexed_at,
            )
            _upsert_point(qdrant, researcher_id, embedding, domains_sparse, payload)
            _persist_history(researcher_id, history)
            stats["indexed"] += 1
        except Exception as exc:
            logger.error("Failed to index {}: {}", researcher_id, exc)
            stats["failed"] += 1

    return stats


# Entry point
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rolling research biographies and index them in Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--id",
        type=int,
        metavar="ID",
        help="Process only this researcher ID.",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=None,
        metavar="OUTPUT_PATH",
        help=(
            "Debug mode: show intermediate summaries without writing to Qdrant."
            "Omit the path to print to stdout; provide a .md path to save to a file."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip researchers already indexed in Qdrant.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Drop and recreate the collection before indexing so the named "
            "vector, payload indexes and metadata are applied. Used by the "
            "scheduled full run."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info("=== UPM Match - Rolling Biography Pipeline ===")
    logger.info(
        "Summary model: {}  |  Embed model: {}",
        _SUMMARY_MODEL,
        _EMBED_MODEL if not args.debug else "(skipped in debug mode)",
    )

    # Resolve debug_output value
    debug_output: Path | bool | None = None
    if args.debug is True:
        debug_output = True
        logger.info("Debug mode: printing intermediate summaries to stdout")
    elif args.debug is not None:
        debug_output = Path(args.debug)
        logger.info("Debug mode: saving intermediate summaries to {}", debug_output)

    if args.id:
        researcher_ids = [args.id]
        logger.info("Processing single researcher: {}", args.id)
    else:
        try:
            researcher_ids = _load_researcher_ids()
        except Exception as exc:
            logger.exception("Failed to load researcher IDs: {}", exc)
            return 1
        logger.info("Loaded {} researcher IDs from MinIO", len(researcher_ids))

    t0 = time.monotonic()
    stats = run(researcher_ids, debug_output, args.skip_existing, args.recreate)
    elapsed = time.monotonic() - t0

    logger.info(
        "Done in {:.1f}s - indexed: {}, skipped: {}, no enrichment: {}, no publications: {}, failed: {}",
        elapsed,
        stats["indexed"],
        stats["skipped"],
        stats["no_enrichment"],
        stats["no_publications"],
        stats["failed"],
    )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
