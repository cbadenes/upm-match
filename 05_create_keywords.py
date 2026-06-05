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
from io import StringIO

import polars as pl
from dotenv import load_dotenv
from loguru import logger

from dbs import minio
from dbs.vector import QdrantBackend, VectorPoint
from models import ResearcherEnrichmentDocument
from utils.logging import setup_logging

load_dotenv()

MINIO_BUCKET = minio.BUCKET_NAME
CSV_OBJECT_NAME = "researchers.csv"
ENRICHMENT_KEY_TEMPLATE = "researchers/{researcher_id}/enrichment.json"

COLLECTION_NAME = "researcher_keywords"
SPARSE_FIELD = "keywords_bm42"


def load_researcher_ids(bucket_name: str, object_name: str) -> list[int]:
    """Load researcher IDs from the CSV stored in MinIO."""
    csv_content = minio.read_file(bucket_name, object_name)
    if csv_content is None:
        raise FileNotFoundError(f"CSV not found at {bucket_name}/{object_name}")
    df = pl.read_csv(StringIO(csv_content))
    if "idPersonal" not in df.columns:
        raise ValueError(f"No 'idPersonal' column in CSV. Found: {df.columns}")
    return (
        df.select("idPersonal")
        .drop_nulls()
        .filter(pl.col("idPersonal").cast(pl.Int64) > 0)
        .get_column("idPersonal")
        .cast(pl.Int64)
        .to_list()
    )


def load_enrichment(
    bucket_name: str, researcher_id: int
) -> ResearcherEnrichmentDocument | None:
    """Read and validate the enrichment document for a researcher from MinIO."""
    key = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    data = minio.read_json(bucket_name, key)
    if data is None:
        return None
    try:
        return ResearcherEnrichmentDocument.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse enrichment for {}: {}", researcher_id, exc)
        return None


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

        enrichment = load_enrichment(MINIO_BUCKET, researcher_id)
        if enrichment is None:
            logger.debug("No enrichment.json for {}", researcher_id)
            stats["no_enrichment"] += 1
            continue

        topics = [t.strip().lower() for t in enrichment.topics if t.strip()]
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
                },
            )
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
        researcher_ids = args.ids
        logger.info("Processing {} specific researcher IDs", len(researcher_ids))
    else:
        try:
            researcher_ids = load_researcher_ids(MINIO_BUCKET, CSV_OBJECT_NAME)
        except Exception as exc:
            logger.exception("Failed to load researcher IDs: {}", exc)
            return 1
        logger.info("Loaded {} researcher IDs from MinIO", len(researcher_ids))

    db = QdrantBackend()
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
