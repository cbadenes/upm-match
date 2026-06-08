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
import sys
import time
from collections import defaultdict
from pathlib import Path

import ollama as _ollama
from agno.agent import Agent
from agno.models.ollama import Ollama
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from models import ResearcherEnrichmentDocument
from utils.logging import setup_logging
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
# MAX_PUBS_PER_YEAR = 8  # cap per year to avoid overwhelming context on prolific years

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


def _group_pubs_by_year(
    enrichment: ResearcherEnrichmentDocument,
) -> dict[int, list[dict]]:
    """Return publications grouped by year, sorted chronologically.

    Only keeps fields that are meaningful for summarization (title, abstract, keywords).
    Publications with no year or no textual content are silently dropped.
    """
    by_year: defaultdict[int, list[dict]] = defaultdict(list)
    for pub in enrichment.publications:
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


def _build_rolling_summary(
    enrichment: ResearcherEnrichmentDocument,
    debug_output: Path | bool | None,
) -> str | None:
    """Iterates over publication years and builds an accumulating biography.

    Returns the final summary text, or None if no usable publications exist.

    debug_output semantics:
      None  - production mode, no intermediate output
      True  - print each year's summary to stdout (no Qdrant write)
      Path  - write all intermediate summaries to a markdown file (no Qdrant write)
    """
    by_year = _group_pubs_by_year(enrichment)
    if not by_year:
        return None

    full_name = (
        f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
        or str(enrichment.id)
    )
    prev_domains: list[str] = []
    prev_profile: str | None = None
    md_lines: list[str] = []

    if isinstance(debug_output, Path):
        md_lines.append(f"# Research Biography: {full_name} ({enrichment.id})\n\n")

    for year, pubs in by_year.items():
        pubs_json = json.dumps(pubs, ensure_ascii=False, indent=2)

        if not prev_domains:
            prompt = (
                f"Researcher: {full_name}\n\n"
                f"Abstracts ({year}, {len(pubs)} publication(s)):\n{pubs_json}\n\n"
                "Extract the researcher's expertise domains and write the profile."
            )
        else:
            domains_bullets = "\n".join(f"- {d}" for d in prev_domains)
            prompt = (
                f"Researcher: {full_name}\n\n"
                f"Established expertise domains (all must appear in `domains` output):\n{domains_bullets}\n\n"
                f"New abstracts ({year}, {len(pubs)} publication(s)):\n{pubs_json}\n\n"
                "1. Add to `domains` any research areas in these abstracts NOT already in the established list.\n"
                "2. Write the `profile` covering all domains."
                "If new domains were added, end with one sentence introducing this new direction."
                "If not, describe the researcher's expertise without any reference to time or change."
            )

        logger.debug(
            "Summarizing {} — year {} ({} pubs)", enrichment.id, year, len(pubs)
        )
        result = _run_agent(prompt)

        if result is None:
            logger.warning(
                "Agent produced no output for {} / year {} — keeping previous result",
                enrichment.id,
                year,
            )
            continue

        prev_domains = result.domains
        prev_profile = result.profile

        if debug_output is True:
            domains_str = ", ".join(result.domains)
            print(f"\n--- {full_name} | Year {year} ({len(pubs)} pub(s)) ---")
            print(f"Domains: [{domains_str}]")
            print(result.profile)
        elif isinstance(debug_output, Path):
            domains_str = ", ".join(result.domains)
            md_lines.append(
                f"## Year {year} ({len(pubs)} publication(s))\n\n"
                f"**Domains:** {domains_str}\n\n{result.profile}\n\n---\n\n"
            )

    if isinstance(debug_output, Path) and md_lines:
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        debug_output.write_text("".join(md_lines), encoding="utf-8")
        logger.info("Debug markdown written to {}", debug_output)

    return prev_profile


# Qdrant dense helpers
def _embed_text(text: str) -> list[float]:
    """Generate a dense embedding via Ollama's embed endpoint."""
    client = _ollama.Client(host=_OLLAMA_HOST)
    response = client.embed(model=_EMBED_MODEL, input=text)
    return response.embeddings[0]


def _ensure_dense_collection(qdrant: QdrantClient, dim: int) -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        logger.debug("Collection '{}' already exists", COLLECTION_NAME)
        return
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.info(
        "Created Qdrant collection '{}' ({}-dim dense cosine)", COLLECTION_NAME, dim
    )


def _build_payload(
    researcher_id: int,
    summary: str,
    enrichment: ResearcherEnrichmentDocument,
) -> dict:
    full_name = f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
    last_year = max(
        (p.year for p in enrichment.publications if p.year and p.year > 0),
        default=None,
    )
    positions = [
        p.name or p.entity or p.role
        for p in enrichment.positions
        if p.name or p.entity or p.role
    ][:3]
    return {
        "researcher_id": enrichment.id,
        "name": full_name,
        "summary": summary,
        "topics": enrichment.topics[:20],
        "first_year": enrichment.first_year,
        "last_year": last_year,
        "positions": positions,
        "publication_count": len(enrichment.publications),
        "subject_count": len(enrichment.subjects),
        "last_year_project_count": len(enrichment.last_year_projects),
    }


def _upsert_point(
    qdrant: QdrantClient,
    researcher_id: int,
    embedding: list[float],
    payload: dict,
) -> None:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qdrant_models.PointStruct(
                id=researcher_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )
    logger.info("PDI {} upserted into dense summaries collection", researcher_id)


# Main processing loop
def run(
    researcher_ids: list[int],
    debug_output: Path | bool | None,
    skip_existing: bool,
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
            logger.debug("No enrichment.json for {}", researcher_id)
            stats["no_enrichment"] += 1
            continue

        summary = _build_rolling_summary(enrichment, debug_output)
        if summary is None:
            logger.debug("No usable publications for {}", researcher_id)
            stats["no_publications"] += 1
            continue

        if debug_output:
            # Debug mode: count as processed but don't write to Qdrant
            stats["indexed"] += 1
            continue

        # Production mode: embed and upsert into Qdrant
        try:
            embedding = _embed_text(summary)
            if not collection_ready:
                _ensure_dense_collection(qdrant, len(embedding))
                collection_ready = True
            _upsert_point(
                qdrant,
                researcher_id,
                embedding,
                _build_payload(researcher_id, summary, enrichment),
            )
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
    stats = run(researcher_ids, debug_output, args.skip_existing)
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
