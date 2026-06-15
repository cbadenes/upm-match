"""
Phase 6 — per-year research summaries for UPM Match.

For each researcher, groups publications by year and generates a STANDALONE
summary + expertise domains for each year using an LLM (Agno + Ollama). The
result is stored as `history` in the researcher's enrichment document in MinIO.
Indexing into Qdrant and the weighted "current" summary are done in phase 7.

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
from pathlib import Path

from agno.agent import Agent
from agno.models.ollama import Ollama
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from dbs import minio
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


def _build_yearly_summaries(
    enrichment: ResearcherEnrichmentDocument,
    debug_output: Path | bool | None,
) -> dict[str, dict] | None:
    """Generate a STANDALONE summary + domains for each publication year.

    Returns `history` mapping each year (as string) to {"domains": [...], "profile": "..."},
    or None if there are no usable publications.

    Each year is summarised on its own — there is NO carry-over from previous years
    (the weighted fusion across years now happens in phase 7). Batches within the SAME
    year still accumulate so a year is never split into separate research periods.

    `debug_output` options:
      None  - production mode (the caller persists history to MinIO)
      True  - print each batch's summary to stdout
      Path  - write all intermediate summaries to a markdown file
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

    history: dict[str, dict] = {}
    md_lines: list[str] = []

    if isinstance(debug_output, Path):
        md_lines.append(f"# Research Biography: {full_name} ({enrichment.id})\n\n")

    for year, pubs in by_year.items():
        # Standalone per year: reset so each year is summarised on its own (no
        # carry-over from previous years). Batches within this year still accumulate.
        prev_domains: list[str] = []
        prev_profile: str | None = None
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

    return history if history else None


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
) -> dict[str, int]:
    stats: dict[str, int] = {
        "processed": 0,
        "skipped": 0,
        "no_enrichment": 0,
        "no_publications": 0,
        "failed": 0,
    }

    total = len(researcher_ids)
    for i, researcher_id in enumerate(researcher_ids, 1):
        if i % 50 == 0:
            logger.info("Progress: {}/{}", i, total)

        enrichment = _load_enrichment(researcher_id)
        if enrichment is None:
            logger.warning(
                "No enrichment JSON in bucket for {} — skipping", researcher_id
            )
            stats["no_enrichment"] += 1
            continue

        if skip_existing and not debug_output and enrichment.history:
            logger.debug("Skipping {} (history already present)", researcher_id)
            stats["skipped"] += 1
            continue

        history = _build_yearly_summaries(enrichment, debug_output)
        if history is None:
            logger.debug("No usable publications for {}", researcher_id)
            stats["no_publications"] += 1
            continue

        if debug_output:
            # Debug mode: don't write to MinIO
            stats["processed"] += 1
            continue

        try:
            _persist_history(researcher_id, history)
            stats["processed"] += 1
        except Exception as exc:
            logger.error("Failed to persist history for {}: {}", researcher_id, exc)
            stats["failed"] += 1

    return stats


# Entry point
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-year research summaries and store them in MinIO (phase 6).",
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
            "Debug mode: show intermediate summaries without writing to MinIO."
            "Omit the path to print to stdout; provide a .md path to save to a file."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip researchers whose enrichment already has summaries (history).",
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
    logger.info("Summary model: {}", _SUMMARY_MODEL)

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
        "Done in {:.1f}s - processed: {}, skipped: {}, no enrichment: {}, no publications: {}, failed: {}",
        elapsed,
        stats["processed"],
        stats["skipped"],
        stats["no_enrichment"],
        stats["no_publications"],
        stats["failed"],
    )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
