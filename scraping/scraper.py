import asyncio
import random

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from dbs.minio import object_exists, upload_text

# Configuration

BASE_URL = "https://portalcientifico.upm.es"
AJAX_URL = (
    f"{BASE_URL}/imarina_mod/templates/imarina/"
    "ipublic-researcher/get-ipublic-researcher-datatables.php"
)
CONCURRENCY = 10  # Concurrent requests; increase by 5 if there are no 429/503 errors.
RETRY_TRIES = 3
RETRY_MIN = 2  # Minimum seconds between retries.
RETRY_MAX = 10  # Maximum seconds between retries.
PAUSE_MIN = 0.3  # Minimum courtesy pause after each download.
PAUSE_MAX = 0.8  # Maximum courtesy pause.

HEADERS = {
    "User-Agent": (
        "UPM-ResearchPortal-Scraper/1.0 "
        "(PIE: UPM-Match; contacto: yago.boleas@alumnos.upm.es)"
    ),
    "Accept-Language": "es-ES,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Retry helpers
RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def make_retry(**kwargs):
    """Reusable retry decorator for HTTP requests.

    :param kwargs: Additional arguments passed to the tenacity decorator.
    :return: Configured decorator.
    """
    return retry(
        retry=retry_if_exception_type(RETRYABLE),
        stop=stop_after_attempt(RETRY_TRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN, max=RETRY_MAX),
        reraise=True,
        **kwargs,
    )


# Download helpers
@make_retry()
async def _get_main(client: httpx.AsyncClient, researcher_id: int) -> str:
    """Fetch the main page for a researcher.

    :param client: Async HTTP client.
    :param researcher_id: Researcher ID.
    :return: Main page HTML.
    """
    url = f"{BASE_URL}/es/ipublic/researcher/{researcher_id}"
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.text


@make_retry()
async def _post_ajax(client: httpx.AsyncClient, researcher_id: int) -> str:
    """Perform the AJAX request that returns additional researcher data.

    :param client: Async HTTP client.
    :param researcher_id: Researcher ID.
    :return: AJAX response HTML.
    """
    payload = {
        "filters_excluded_wf_personal": "",
        "filters_excluded_it_personal": "0",
        "id_personal": str(researcher_id),
        "lang": "spa",
        "listas_predeterminadas": "0",
        "produccion_validada": "0",
    }
    r = await client.post(AJAX_URL, data=payload, timeout=30)
    r.raise_for_status()
    return r.text


# Per-researcher task
async def scrape_researcher(
    client: httpx.AsyncClient,
    researcher_id: int,
    semaphore: asyncio.Semaphore,
    bucket_name: str,
) -> bool:
    """Download a researcher's HTML files (main + AJAX) and upload to MinIO.

    :param client: Async HTTP client.
    :param researcher_id: Researcher ID.
    :param semaphore: Semaphore used to limit concurrency.
    :param bucket_name: MinIO bucket name.
    :return: True when the download and upload succeed, False on error.
    """
    # Checkpoint: if both files already exist in MinIO, skip the download.
    profile_path = f"researchers/{researcher_id}/profile.html"
    activity_path = f"researchers/{researcher_id}/activity.html"
    if object_exists(bucket_name, profile_path) and object_exists(
        bucket_name, activity_path
    ):
        logger.debug("[{}] Already downloaded in MinIO, skipping", researcher_id)
        return True

    async with semaphore:
        try:
            html = await _get_main(client, researcher_id)
            profile_ok = upload_text(
                bucket_name,
                html,
                profile_path,
                content_type="text/html",
            )

            ajax_html = await _post_ajax(client, researcher_id)
            activity_ok = upload_text(
                bucket_name,
                ajax_html,
                activity_path,
                content_type="text/html",
            )

            if not (profile_ok and activity_ok):
                logger.error(
                    "[{}] Failed to upload one or more files to MinIO",
                    researcher_id,
                )
                return False

            # Random courtesy pause.
            await asyncio.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))
            logger.info("[{}] Downloaded and uploaded to MinIO", researcher_id)
            return True

        except httpx.HTTPStatusError as e:
            # 404 means the profile does not exist; log it and continue.
            if e.response.status_code == 404:
                logger.warning("[{}] Profile not found (404)", researcher_id)
                return False
            logger.error("[{}] HTTP {}: {}", researcher_id, e.response.status_code, e)
            return False

        except Exception as e:
            logger.error("[{}] Unexpected error: {}", researcher_id, e)
            return False


# Download orchestrator
async def run_scraper(researcher_ids: list[int], bucket_name: str) -> dict:
    """Run the scraper for a list of researcher IDs and upload to MinIO.

    :param researcher_ids: Researcher IDs to download.
    :param bucket_name: MinIO bucket name to upload to.
    :return: Dictionary with execution statistics.
    """
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(
        headers=HEADERS, http2=True, follow_redirects=True
    ) as client:
        tasks = [
            scrape_researcher(client, rid, semaphore, bucket_name)
            for rid in researcher_ids
        ]
        results = await tqdm.gather(*tasks, desc="Downloading and uploading profiles")

    total = len(results)
    ok = sum(results)
    failed = total - ok

    logger.info(
        "[{}] Scraping completed - OK: {} | Failed: {} | Total: {}",
        researcher_ids[0] if researcher_ids else "Unknown",
        ok,
        failed,
        total,
    )
    return {"ok": ok, "failed": failed, "total": total}
