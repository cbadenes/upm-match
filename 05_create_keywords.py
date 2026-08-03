"""
Keyword indexing script for UPM Match.

Reads enrichment documents from MinIO, generates BM42 sparse embeddings from each
researcher's topics, and indexes them in Qdrant for keyword-based similarity search.

This is the first step towards the recommendation pipeline: once all researchers are
indexed, a student's interest keywords can be matched against this collection to rank
the most relevant supervisors.

Usage:
    uv run 05_create_keywords.py
    uv run 05_create_keywords.py --skip-existing
    uv run 05_create_keywords.py --ids 308607 261274          # test with specific IDs
    uv run 05_create_keywords.py --batch-size 50 --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from loguru import logger

from dbs.vector import QdrantBackend, VectorPoint
from dbs.vector.qdrant import SPARSE_MODEL_NAME
from utils.logging import setup_logging
from utils.exclusions import drop_excluded
from utils.researchers import load_enrichment, load_researcher_ids
from utils.text import canonicalize_topics

load_dotenv()

COLLECTION_NAME = "researcher_keywords"
SPARSE_FIELD = "keywords_bm42"


def run(
    researcher_ids: list[int],
    db: QdrantBackend,
    batch_size: int,
    skip_existing: bool,
) -> dict[str, int]:
    stats = {
        "indexed": 0,
        "skipped": 0,
        "no_enrichment": 0,
        "no_topics": 0,
        "failed": 0,
    }

    # Preload existing IDs in a single scan - much faster than per-point lookups
    existing_ids: set[int] = set()
    if skip_existing:
        existing_ids = db.get_all_ids(COLLECTION_NAME)
        logger.info("{} researchers already indexed", len(existing_ids))

    indexed_at = datetime.now(timezone.utc).isoformat()
    batch: list[VectorPoint] = []

    def flush_batch() -> None:
        if not batch:
            return
        try:
            n = db.upsert_points(COLLECTION_NAME, SPARSE_FIELD, batch)
            stats["indexed"] += n
            logger.debug("Upserted batch of {} points", n)
        except Exception as exc:
            logger.error("Batch upsert failed: {}", exc)
            stats["failed"] += len(batch)
        batch.clear()

    total = len(researcher_ids)
    for i, researcher_id in enumerate(researcher_ids, 1):
        if i % 100 == 0:
            logger.info("Progress: {}/{}", i, total)

        if researcher_id in existing_ids:
            logger.debug("Skipping {} (already indexed)", researcher_id)
            stats["skipped"] += 1
            continue

        enrichment = load_enrichment(researcher_id)
        if enrichment is None:
            logger.debug("No enrichment.json for {}", researcher_id)
            stats["no_enrichment"] += 1
            continue

        topics = canonicalize_topics(enrichment.topics)
        if not topics:
            full_name = (
                f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
            )
            logger.debug("No topics for {} ({})", researcher_id, full_name)
            stats["no_topics"] += 1
            continue

        try:
            indices, values = db.embed(topics)
        except Exception as exc:
            logger.error("Embedding failed for {}: {}", researcher_id, exc)
            stats["failed"] += 1
            continue

        full_name = (
            f"{enrichment.first_name or ''} {enrichment.last_name or ''}".strip()
        )
        batch.append(
            VectorPoint(
                id=researcher_id,
                sparse_indices=indices,
                sparse_values=values,
                payload={
                    "researcher_id": enrichment.id,
                    "name": full_name,
                    "topics": topics,
                    "since": enrichment.first_year or 0,
                    "embedding_model": SPARSE_MODEL_NAME,
                    "indexed_at": indexed_at,
                },
            )
        )
        logger.info(
            "PDI {} queued for keyword indexing: {} topics", researcher_id, len(topics)
        )

        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index researcher keyword embeddings into Qdrant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
            "Drop and recreate the collection before indexing so schema and "
            "payload metadata changes are applied. Used by the scheduled full run."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of points to upsert per batch.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="Process only these researcher IDs (useful for testing).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    setup_logging("create_keywords", level="DEBUG" if args.verbose else "INFO")
    logger.info("=== UPM Match - Keyword Indexing Pipeline ===")

    if args.ids:
        # --ids bypasses load_researcher_ids(), so the opt-out list is applied here
        # too: an explicit id must never override an exercised right to object.
        researcher_ids = drop_excluded(args.ids)
        if len(researcher_ids) != len(args.ids):
            logger.warning(
                "Skipped {} requested id(s) on the exclusion list",
                len(args.ids) - len(researcher_ids),
            )
        logger.info("Processing {} specific researcher IDs", len(researcher_ids))
    else:
        try:
            researcher_ids = load_researcher_ids()
        except Exception as exc:
            logger.exception("Failed to load researcher IDs: {}", exc)
            return 1
        logger.info("Loaded {} researcher IDs from MinIO", len(researcher_ids))

    db = QdrantBackend()
    if args.recreate and db.collection_exists(COLLECTION_NAME):
        logger.info("Recreating collection '{}' from scratch", COLLECTION_NAME)
        db.delete_collection(COLLECTION_NAME)
    db.ensure_collection(COLLECTION_NAME, SPARSE_FIELD)
    before = db.count_points(COLLECTION_NAME)
    logger.info(
        "Collection '{}' ready ({} points already indexed)",
        COLLECTION_NAME,
        before,
    )

    t0 = time.monotonic()
    stats = run(researcher_ids, db, args.batch_size, args.skip_existing)
    elapsed = time.monotonic() - t0

    logger.info(
        "Done in {:.1f}s - indexed: {}, skipped: {}, no enrichment: {}, no topics: {}, failed: {}",
        elapsed,
        stats["indexed"],
        stats["skipped"],
        stats["no_enrichment"],
        stats["no_topics"],
        stats["failed"],
    )
    after = db.count_points(COLLECTION_NAME)
    if after != before:
        logger.info(
            "Collection '{}' grew from {} to {} points", COLLECTION_NAME, before, after
        )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
