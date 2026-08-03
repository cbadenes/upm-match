"""
Manage the researcher exclusion list (right to object / right to erasure).

This is the operational tool behind the rights channel published at /legal. A
PDI who asks not to appear is added here; from that moment the pipeline stops
processing them and the web app stops serving them (see `utils/exclusions.py`).

Commands
--------
List the current exclusions:
> uv run 08_manage_exclusions.py list

Exclude a researcher (hides them everywhere, keeps the derived data):
> uv run 08_manage_exclusions.py add 305196 --reason "oposición art. 21 RGPD" --requested-on 2026-07-30

Exclude AND delete every derived artefact we generated about them:
> uv run 08_manage_exclusions.py add 305196 --reason "supresión art. 17 RGPD" --purge

Re-include someone who changed their mind (needs a reindex to reappear):
> uv run 08_manage_exclusions.py remove 305196

What --purge deletes
--------------------
  - the Qdrant points in both collections (keywords + summaries), including the
    versioned collections behind the aliases;
  - the enrichment document in the curated bucket (LLM-generated summary,
    per-year summaries, domains, publications);
  - with --purge-raw, also the scraped HTML in the raw bucket;
  - with --purge-graph, also the Memgraph nodes and edges.

The id itself stays on the exclusion list: that minimal record is what keeps the
next pipeline run from downloading them again, and is the standard way of making
an opt-out durable.

Nothing is deleted without confirmation unless --yes is passed.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from dbs import minio  # noqa: E402 — needs the env loaded first
from dbs.vector import QdrantBackend  # noqa: E402
from utils.exclusions import (  # noqa: E402
    add_exclusion,
    load_document,
    normalise_id,
    remove_exclusion,
)
from utils.logging import setup_logging  # noqa: E402
from utils.researchers import ENRICHMENT_KEY_TEMPLATE  # noqa: E402

KEYWORDS_COLLECTION = "researcher_keywords"
SUMMARIES_COLLECTION = "researcher_summaries"


def _confirm(prompt: str, assume_yes: bool) -> bool:
    """Ask for confirmation on stdin unless --yes was passed."""
    if assume_yes:
        return True
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes", "s", "si", "sí")
    except EOFError:  # non-interactive without --yes: refuse rather than guess
        return False


def cmd_list(_args: argparse.Namespace) -> int:
    """Print the current exclusion list."""
    document = load_document()
    entries = document.get("entries") or []
    if not entries:
        print("The exclusion list is empty.")
        return 0

    print(f"{len(entries)} researcher(s) excluded "
          f"(last updated: {document.get('updated_at') or 'unknown'}):\n")
    for entry in entries:
        rid = entry.get("researcher_id")
        flag = "purged" if entry.get("purged") else "hidden"
        requested = entry.get("requested_on") or "—"
        print(f"  {rid}  [{flag}]  requested: {requested}")
        if entry.get("reason"):
            print(f"        reason: {entry['reason']}")
    return 0


def _purge_qdrant(researcher_id: int) -> None:
    """Remove the researcher's points from every relevant Qdrant collection."""
    db = QdrantBackend()
    targets: set[str] = set()
    for prefix in (KEYWORDS_COLLECTION, SUMMARIES_COLLECTION):
        targets.add(prefix)
        latest = db.latest_collection(prefix)
        if latest:
            targets.add(latest)
    for collection in sorted(targets):
        db.delete_points(collection, [researcher_id])


def _purge_curated(researcher_id: int) -> None:
    """Delete the enrichment document (summaries, domains, publications)."""
    key = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    if minio.object_exists(minio.CURATED_BUCKET, key):
        minio.delete_file(minio.CURATED_BUCKET, key)
        logger.info("Deleted {}/{}", minio.CURATED_BUCKET, key)
    else:
        logger.info("No enrichment document for {}", researcher_id)


def _purge_raw(researcher_id: int) -> None:
    """Delete the scraped HTML captured from the portal."""
    keys = [
        f"researchers/{researcher_id}/profile.html",
        f"researchers/{researcher_id}/activity.html",
    ]
    present = [k for k in keys if minio.object_exists(minio.RAW_BUCKET, k)]
    if not present:
        logger.info("No raw HTML for {}", researcher_id)
        return
    minio.delete_files(minio.RAW_BUCKET, present)
    logger.info("Deleted {} raw object(s) for {}", len(present), researcher_id)


def _purge_graph(researcher_id: int) -> None:
    """Detach-delete the researcher's node from Memgraph."""
    try:
        from dbs.graph.memgraph import MemgraphBackend  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.warning("Memgraph backend unavailable; skipping graph purge")
        return
    try:
        backend = MemgraphBackend()
        backend.query(
            "MATCH (p:Pdi {id: $id}) DETACH DELETE p", {"id": f"pdi_{researcher_id}"}
        )
        logger.info("Deleted Pdi node pdi_{} from Memgraph", researcher_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not purge the graph for {}: {}", researcher_id, exc)


def cmd_add(args: argparse.Namespace) -> int:
    """Add a researcher to the exclusion list, optionally purging their data."""
    rid = normalise_id(args.researcher_id)
    if not rid:
        logger.error("Not a valid researcher id: {}", args.researcher_id)
        return 2

    actions = ["hide them from the app and stop the pipeline processing them"]
    if args.purge:
        actions.append("DELETE the Qdrant points and the enrichment document")
    if args.purge_raw:
        actions.append("DELETE the scraped HTML in the raw bucket")
    if args.purge_graph:
        actions.append("DELETE the Memgraph node and its edges")

    print(f"\nResearcher {rid} — this will:")
    for action in actions:
        print(f"  · {action}")
    if not _confirm("\nProceed?", args.yes):
        print("Aborted; nothing was changed.")
        return 1

    if not add_exclusion(
        rid,
        reason=args.reason,
        requested_on=args.requested_on,
        purged=bool(args.purge or args.purge_raw or args.purge_graph),
    ):
        logger.error("Could not write the exclusion list; aborting before any deletion")
        return 1
    logger.info("Researcher {} added to the exclusion list", rid)

    # Deletions run only after the list is safely persisted, so a failure here
    # still leaves the researcher hidden.
    if args.purge:
        _purge_qdrant(rid)
        _purge_curated(rid)
    if args.purge_raw:
        _purge_raw(rid)
    if args.purge_graph:
        _purge_graph(rid)

    print(f"\nDone. Researcher {rid} is excluded.")
    print("The app applies it within ~60 s (no restart needed).")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    """Remove a researcher from the exclusion list."""
    rid = normalise_id(args.researcher_id)
    if not rid:
        logger.error("Not a valid researcher id: {}", args.researcher_id)
        return 2

    if not remove_exclusion(rid):
        print(f"Researcher {rid} was not on the exclusion list.")
        return 1

    print(f"Researcher {rid} removed from the exclusion list.")
    print(
        "Their profile will reappear once the pipeline regenerates and reindexes "
        "them (phases 02 → 07). If the data was purged, run the full pipeline for "
        f"that id: uv run 02_get_pdis_data.py --id {rid}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage the researcher exclusion list (RGPD arts. 17 and 21).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show the current exclusion list.").set_defaults(
        func=cmd_list
    )

    add = sub.add_parser("add", help="Exclude a researcher.")
    add.add_argument("researcher_id", help="Numeric portal id (e.g. 305196).")
    add.add_argument(
        "--reason", default="", help='Short note, e.g. "oposición art. 21 RGPD".'
    )
    add.add_argument(
        "--requested-on", default=None, help="ISO date of the request (YYYY-MM-DD)."
    )
    add.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the Qdrant points and the enrichment document.",
    )
    add.add_argument(
        "--purge-raw",
        action="store_true",
        help="Also delete the scraped HTML in the raw bucket.",
    )
    add.add_argument(
        "--purge-graph",
        action="store_true",
        help="Also delete the Memgraph node and its edges.",
    )
    add.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
    add.set_defaults(func=cmd_add)

    remove = sub.add_parser("remove", help="Re-include a researcher.")
    remove.add_argument("researcher_id", help="Numeric portal id.")
    remove.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
    remove.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    setup_logging("manage_exclusions")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
