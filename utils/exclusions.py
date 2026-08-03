"""Exclusion list: researchers who must not be processed nor published.

This is the technical channel that makes the right to object (art. 21 RGPD) and
the right to erasure (art. 17) effective. A PDI who asks to be left out is added
here and disappears from every surface of the system:

  - the pipeline stops downloading, enriching and indexing them
    (`load_researcher_ids` filters the list at the source, so phases 02/04/05/07
    and the evaluation scripts all inherit the exclusion);
  - the web app stops serving them, immediately and without waiting for a
    reindex (search results, PDI directory and profile endpoint).

The list is the single source of truth and lives in MinIO so that the app and
the Airflow DAGs read the same object without redeploying anything. Use
`08_manage_exclusions.py` to edit it — do not hand-edit the object.

Failure mode (deliberate): if MinIO cannot be reached the last successfully
loaded list is reused, and `EXCLUDED_RESEARCHER_IDS` acts as a permanent floor
that never depends on MinIO being up. Put the ids of anyone who has formally
objected in that env var as well: it is the belt-and-braces guarantee that a
storage outage can never resurrect a profile that must stay hidden.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# MinIO object holding the list (curated bucket).
EXCLUSIONS_KEY = "governance/exclusions.json"
# Seconds a loaded list is trusted before re-reading MinIO. Short enough that an
# exclusion takes effect almost immediately, long enough not to hit MinIO on
# every request.
EXCLUSIONS_TTL = int(os.environ.get("EXCLUSIONS_TTL", "60"))
SCHEMA_VERSION = 1

# Process-level cache: (expires_at, ids). Also survives MinIO outages.
_cache: tuple[float, set[int]] | None = None


def normalise_id(value: Any) -> int | None:
    """Coerce a researcher reference to its numeric portal id.

    Accepts ``305196``, ``"305196"`` and the graph/payload form ``"pdi_305196"``.

    :param value: Researcher reference in any of the accepted forms.
    :return: Numeric id, or None when the value carries no usable id.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; never a researcher id
        return None
    if isinstance(value, int):
        return value or None
    text = str(value).strip()
    if not text:
        return None
    digits = text.rsplit("_", 1)[-1]  # "pdi_305196" -> "305196"
    try:
        return int(digits) or None
    except ValueError:
        return None


def _env_excluded() -> set[int]:
    """Ids pinned through EXCLUDED_RESEARCHER_IDS (comma-separated)."""
    raw = os.environ.get("EXCLUDED_RESEARCHER_IDS", "")
    ids: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        rid = normalise_id(chunk)
        if rid:
            ids.add(rid)
    return ids


def _read_document() -> dict:
    """Read the raw exclusions document from MinIO (empty skeleton if absent)."""
    from dbs import minio  # noqa: PLC0415 — lazy: needs MINIO_* env at import time

    data = minio.read_json(minio.CURATED_BUCKET, EXCLUSIONS_KEY)
    if not isinstance(data, dict):
        return {"version": SCHEMA_VERSION, "updated_at": None, "entries": []}
    data.setdefault("entries", [])
    return data


def load_document() -> dict:
    """Return the full exclusions document, entries included.

    Used by the management CLI; the hot paths want `load_excluded_ids`.
    """
    return _read_document()


def load_excluded_ids(force: bool = False) -> set[int]:
    """Return the ids that must never be processed nor served.

    :param force: Skip the TTL cache and re-read MinIO.
    :return: Union of the MinIO list and EXCLUDED_RESEARCHER_IDS.
    """
    global _cache

    env_ids = _env_excluded()
    now = time.monotonic()
    if not force and _cache is not None and now < _cache[0]:
        return _cache[1] | env_ids

    try:
        document = _read_document()
        ids = {
            rid
            for rid in (
                normalise_id((entry or {}).get("researcher_id"))
                for entry in document.get("entries") or []
            )
            if rid
        }
        _cache = (now + EXCLUSIONS_TTL, ids)
        return ids | env_ids
    except Exception:  # noqa: BLE001 — an outage must not resurrect a hidden profile
        if _cache is not None:
            logger.warning(
                "Could not refresh the exclusion list; reusing the {} cached id(s)",
                len(_cache[1]),
            )
            return _cache[1] | env_ids
        logger.error(
            "Could not load the exclusion list and nothing is cached; falling back "
            "to EXCLUDED_RESEARCHER_IDS only ({} id(s)). Profiles excluded solely "
            "via MinIO may be visible until MinIO recovers.",
            len(env_ids),
        )
        return env_ids


def is_excluded(researcher_id: Any) -> bool:
    """True when this researcher must not be processed nor served."""
    rid = normalise_id(researcher_id)
    return bool(rid) and rid in load_excluded_ids()


def drop_excluded(researcher_ids: Iterable[Any]) -> list[Any]:
    """Filter an iterable of researcher references, preserving order and form."""
    excluded = load_excluded_ids()
    kept = [r for r in researcher_ids if (normalise_id(r) or 0) not in excluded]
    return kept


def invalidate_cache() -> None:
    """Drop the TTL cache so the next read hits MinIO (used after a write)."""
    global _cache
    _cache = None


def save_document(document: dict) -> bool:
    """Persist the exclusions document to MinIO and invalidate the cache."""
    from dbs import minio  # noqa: PLC0415 — lazy, as above

    document["version"] = SCHEMA_VERSION
    document["updated_at"] = datetime.now(timezone.utc).isoformat()
    ok = minio.upload_json(minio.CURATED_BUCKET, document, EXCLUSIONS_KEY)
    if ok:
        invalidate_cache()
    return ok


def add_exclusion(
    researcher_id: int,
    reason: str = "",
    requested_on: str | None = None,
    purged: bool = False,
) -> bool:
    """Add (or update) one researcher in the exclusion list.

    :param researcher_id: Numeric portal id.
    :param reason: Short free-text note (e.g. "oposición art. 21 RGPD").
    :param requested_on: ISO date the interested party made the request.
    :param purged: Whether the derived data was deleted as well.
    :return: True when the list was written successfully.
    """
    rid = normalise_id(researcher_id)
    if not rid:
        raise ValueError(f"Not a valid researcher id: {researcher_id!r}")

    document = _read_document()
    entries = [
        entry
        for entry in document.get("entries") or []
        if normalise_id((entry or {}).get("researcher_id")) != rid
    ]
    entries.append(
        {
            "researcher_id": rid,
            "reason": reason,
            "requested_on": requested_on,
            "excluded_at": datetime.now(timezone.utc).isoformat(),
            "purged": purged,
        }
    )
    entries.sort(key=lambda e: e["researcher_id"])
    document["entries"] = entries
    return save_document(document)


def remove_exclusion(researcher_id: int) -> bool:
    """Remove one researcher from the exclusion list (re-inclusion).

    Only lifts the block: the profile reappears once the pipeline regenerates
    and reindexes that researcher.

    :return: True when the id was present and the list was written.
    """
    rid = normalise_id(researcher_id)
    if not rid:
        raise ValueError(f"Not a valid researcher id: {researcher_id!r}")

    document = _read_document()
    before = document.get("entries") or []
    after = [
        entry
        for entry in before
        if normalise_id((entry or {}).get("researcher_id")) != rid
    ]
    if len(after) == len(before):
        return False
    document["entries"] = after
    return save_document(document)
