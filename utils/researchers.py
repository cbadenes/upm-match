from __future__ import annotations

from io import StringIO

import polars as pl
from loguru import logger

from dbs import minio
from models import ResearcherEnrichmentDocument
from utils.exclusions import load_excluded_ids

CSV_OBJECT_NAME = "researchers.csv"
ENRICHMENT_KEY_TEMPLATE = "researchers/{researcher_id}/enrichment.json"


def load_researcher_ids(
    bucket_name: str = minio.RAW_BUCKET,
    object_name: str = CSV_OBJECT_NAME,
) -> list[int]:
    """Load researcher IDs from a CSV stored in MinIO.

    Researchers on the exclusion list (`utils.exclusions`) are dropped here, at
    the single entry point every pipeline phase uses, so that a PDI who objected
    is never downloaded, enriched, indexed nor evaluated again.

    :param bucket_name: MinIO bucket containing the CSV.
    :param object_name: Object path within the bucket.
    :return: Sorted, deduplicated list of researcher IDs, exclusions removed.
    :raises FileNotFoundError: If the CSV does not exist in MinIO.
    :raises ValueError: If the CSV lacks the required column.
    """
    csv_content = minio.read_file(bucket_name, object_name)
    if csv_content is None:
        raise FileNotFoundError(
            f"CSV file not found in MinIO at {bucket_name}/{object_name}"
        )
    df = pl.read_csv(StringIO(csv_content))
    if "idPersonal" not in df.columns:
        raise ValueError(
            f"The CSV does not contain the 'idPersonal' column. "
            f"Columns found: {df.columns}"
        )
    ids: list[int] = (
        df.select("idPersonal")
        .drop_nulls()
        .filter(pl.col("idPersonal").cast(pl.Int64) > 0)
        .unique()
        .sort("idPersonal")
        .get_column("idPersonal")
        .cast(pl.Int64)
        .to_list()
    )
    excluded = load_excluded_ids()
    if excluded:
        kept = [rid for rid in ids if rid not in excluded]
        if len(kept) != len(ids):
            logger.info(
                "Excluded {} researcher(s) on the opt-out list", len(ids) - len(kept)
            )
        ids = kept
    logger.info(
        "Loaded {} researcher IDs from {}/{}", len(ids), bucket_name, object_name
    )
    return ids


def load_researcher_ids_with_signatures(
    bucket_name: str = minio.RAW_BUCKET,
    object_name: str = CSV_OBJECT_NAME,
) -> tuple[list[int], list[str]]:
    """Load researcher IDs and rfSignature values from a CSV stored in MinIO.

    :param bucket_name: MinIO bucket containing the CSV.
    :param object_name: Object path within the bucket.
    :return: Tuple of (ids, signatures), both sorted by idPersonal.
    :raises FileNotFoundError: If the CSV does not exist in MinIO.
    :raises ValueError: If the CSV lacks the required columns.
    """
    csv_content = minio.read_file(bucket_name, object_name)
    if csv_content is None:
        raise FileNotFoundError(
            f"CSV file not found in MinIO at {bucket_name}/{object_name}"
        )
    df = pl.read_csv(StringIO(csv_content))
    if "idPersonal" not in df.columns:
        raise ValueError(
            f"The CSV does not contain the 'idPersonal' column. "
            f"Columns found: {df.columns}"
        )
    df = (
        df.select(["idPersonal", "rfSignature"])
        .drop_nulls()
        .filter(pl.col("idPersonal").cast(pl.Int64) > 0)
        .unique()
        .sort("idPersonal")
    )
    ids: list[int] = df["idPersonal"].to_list()
    signatures: list[str] = df["rfSignature"].to_list()
    logger.info(
        "Loaded {} researcher IDs from {}/{}", len(ids), bucket_name, object_name
    )
    return ids, signatures


def load_enrichment(
    researcher_id: int,
    bucket_name: str = minio.CURATED_BUCKET,
) -> ResearcherEnrichmentDocument | None:
    """Read and validate the enrichment document for a researcher from MinIO.

    :param researcher_id: Numeric researcher ID.
    :param bucket_name: MinIO bucket containing enrichment documents.
    :return: Validated document, or None if missing or unparseable.
    """
    key = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    data = minio.read_json(bucket_name, key)
    if data is None:
        return None
    try:
        return ResearcherEnrichmentDocument.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse enrichment for {}: {}", researcher_id, exc)
        return None
