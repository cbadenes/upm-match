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
import asyncio

from loguru import logger

from utils.logging import setup_logging
from utils.researchers import load_researcher_ids

from dbs import minio
from scraping.scraper import run_scraper


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

    setup_logging("get_pdis_data")

    # Phase 1: load researcher IDs from MinIO.
    ids = load_researcher_ids()
    if researcher_id is not None:
        if researcher_id not in ids:
            logger.warning(
                f"Requested researcher id {researcher_id} was not found in MinIO"
            )
        ids = [researcher_id]

    # Phase 2: scrape profiles and upload directly to MinIO.
    stats = await run_scraper(ids, bucket_name=minio.RAW_BUCKET)

    logger.info("Pipeline finished")
    logger.info(f"  OK:       {stats['ok']}")
    logger.info(f"  Failed:   {stats['failed']}")
    logger.info(f"  Total:    {stats['total']}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.id))
