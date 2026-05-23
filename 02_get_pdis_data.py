"""
Script to download PDI profiles from the 'Portal Científico UPM' .
It loads researcher IDs from a CSV stored in MinIO, scrapes their profiles and uploads the results back to MinIO.
The script can be run for all researchers or a specific one by ID.
If a profile has already been downloaded and exists in MinIO, it is skipped to save time and resources.

For each researcher, the script downloads two HTML files:
- `profile.html`: The main profile page of the researcher.
- `activity.html`: The page containing the researcher's activity, which is loaded via an AJAX request.
The files are stored in MinIO under the path `researchers/{idPersonal}/`.

Use:
> uv run 02_get_pdis_data.py
> uv run 02_get_pdis_data.py --id 308607
"""

import argparse
from io import StringIO
from pathlib import Path

import polars as pl
from loguru import logger

from dbs.minio import read_file as minio_read_file
from scraping.scraper import run_scraper

# Configuration
MINIO_BUCKET = "tfm"
CSV_OBJECT_NAME = "researchers.csv"
LOG_PATH = Path("logs/pipeline.log")


# ID loading
def load_researcher_ids_from_minio(
    bucket_name: str = MINIO_BUCKET,
    object_name: str = CSV_OBJECT_NAME,
) -> list[int]:
    """Load researcher IDs from a CSV stored in MinIO after validating format.

    :param bucket_name: MinIO bucket name.
    :param object_name: Object name for researchers CSV.
    :return: List of researcher IDs.
    """
    csv_content = minio_read_file(bucket_name, object_name)
    if csv_content is None:
        raise FileNotFoundError(
            f"CSV file not found in MinIO at {bucket_name}/{object_name}"
        )

    df = pl.read_csv(StringIO(csv_content))

    if "idPersonal" not in df.columns:
        raise ValueError(
            f"The CSV does not contain the 'idPersonal' column. Columns found: {df.columns}"
        )

    df = (
        df.select("idPersonal")
        .drop_nulls()
        .filter(pl.col("idPersonal").cast(pl.Int64) > 0)
        .unique()
        .sort("idPersonal")
    )

    ids = df["idPersonal"].to_list()
    logger.info(f"{len(ids)} IDs loaded from MinIO: {bucket_name}/{object_name}")
    return ids


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PDI download pipeline."""
    parser = argparse.ArgumentParser(
        description="Download one PDI or all PDI profiles from MinIO."
    )
    parser.add_argument(
        "-i",
        "--id",
        type=int,
        help="Download only the PDI with this idPersonal. If omitted, all PDIs are downloaded.",
    )
    return parser.parse_args()


async def main(researcher_id: int | None = None):
    """Run the full pipeline: load IDs, scrape and upload profiles to MinIO."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOG_PATH,
        rotation="50 MB",
        retention="10 days",
        level="INFO",
        encoding="utf-8",
    )

    logger.info("═" * 60)
    logger.info("Starting pipeline - Portal Cientifico UPM")
    logger.info("═" * 60)

    # Phase 1: load researcher IDs from MinIO.
    ids = load_researcher_ids_from_minio()
    if researcher_id is not None:
        if researcher_id not in ids:
            logger.warning(f"Requested researcher id {researcher_id} was not found in MinIO")
        ids = [researcher_id]

    # Phase 2: scrape profiles and upload directly to MinIO.
    stats = await run_scraper(ids, bucket_name=MINIO_BUCKET)

    logger.info("Pipeline finished")
    logger.info(f"  OK:       {stats['ok']}")
    logger.info(f"  Failed:   {stats['failed']}")
    logger.info(f"  Total:    {stats['total']}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.id))
