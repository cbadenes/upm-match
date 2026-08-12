"""Ablation: does the generated biography retrieve better than the raw publication text?

The dense index represents each researcher by a biography that a local LLM writes
from their production (phases 6 and 7), which is the single most expensive step of
the corpus pipeline. The benchmark of the memoria compares ways of *querying* that
index, never the representation itself, so the obvious question stays unanswered:
would embedding the publications directly do just as well?

This script answers it with the smallest fair experiment. It builds a parallel
Qdrant collection whose vectors come from the raw publication text, and runs the
`summary_direct` strategy against it. That strategy is chosen deliberately: it
invokes no LLM at query time, so it is deterministic and one run suffices, and the
only thing that differs from the campaign's `summary_direct` is what was indexed.

Two asymmetries are inherent to the comparison and must be read with the result,
not swept under it:

  - The biography condenses the WHOLE trajectory; the raw text only holds the most
    recent publications that fit the embedding context. That is not a handicap
    imposed here, it is the constraint any raw-text approach faces, and it is the
    reason the biography exists at all. The per-year breakdown quantifies its
    effect: if the biography's edge comes from covering the past, it should show up
    as an advantage on older works.
  - Only two thirds of publications carry a usable abstract (memoria
    sec:disc-fuente), so the raw text is thinner than its volume suggests.

Usage:
    uv run --module evaluation.ablation_rawtext --source silver
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ollama as _ollama  # noqa: E402
from loguru import logger  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from dbs import minio  # noqa: E402
from evaluation.evaluate_recommendations import (  # noqa: E402
    GOLD_TFG_PATH,
    GOLD_TFM_PATH,
    K_VALUES,
    SILVER_TFG_PATH,
    SILVER_TFM_PATH,
    common_indexed_universe,
    hit_at_k,
    load_gold_standard,
    ndcg_at_k,
    reciprocal_rank,
)

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
_QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

COLLECTION = "researcher_rawtext_ablation"
DENSE_VECTOR = "dense"
# Character budget for the concatenated raw text of one researcher. Kept well
# inside the embedding model's context so no publication is cut mid-abstract:
# whole publications are added, newest first, until the next one would not fit.
RAW_TEXT_BUDGET = int(os.environ.get("ABLATION_TEXT_BUDGET", "24000"))


def _embed(text: str) -> list[float]:
    return (
        _ollama.Client(host=_OLLAMA_HOST).embed(model=_EMBED_MODEL, input=text).embeddings[0]
    )


def build_raw_text(doc: dict) -> tuple[str, int, int]:
    """Concatenate title + abstract of the most recent publications that fit.

    :return: the text, how many publications went in, and how many were dropped
        for lack of room (the truncation this ablation has to declare).
    """
    pubs = [p for p in (doc.get("publications") or []) if p.get("title")]
    pubs.sort(key=lambda p: (p.get("year") or 0), reverse=True)
    parts: list[str] = []
    used = size = 0
    for p in pubs:
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        block = f"{title}. {abstract}".strip()
        if size + len(block) > RAW_TEXT_BUDGET:
            break
        parts.append(block)
        size += len(block)
        used += 1
    return "\n\n".join(parts), used, len(pubs) - used


def build_collection(client: QdrantClient, only_indexed: set[str]) -> dict[str, int]:
    """(Re)build the ablation collection from the raw publication text."""
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = len(_embed("dimension probe"))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
    )
    logger.info("Collection '{}' created ({} dims).", COLLECTION, dim)

    keys = minio.list_objects(minio.CURATED_BUCKET, prefix="researchers/", recursive=True)
    keys = [k for k in keys if k.endswith("enrichment.json")]
    stats = {"indexed": 0, "skipped_no_text": 0, "truncated": 0, "pubs_used": 0, "pubs_dropped": 0}
    points = []
    for key in keys:
        doc = minio.read_json(minio.CURATED_BUCKET, key)
        # The enrichment document keys the researcher as `id` ("pdi_303799");
        # the Qdrant payload written by phase 7 renames it to `researcher_id`.
        rid = str((doc or {}).get("id") or "")
        if rid not in only_indexed:
            continue
        text, used, dropped = build_raw_text(doc)
        if not text.strip():
            stats["skipped_no_text"] += 1
            continue
        points.append(
            models.PointStruct(
                id=int(rid.split("_")[-1]),
                vector={DENSE_VECTOR: _embed(text)},
                payload={"researcher_id": rid, "publications_used": used},
            )
        )
        stats["indexed"] += 1
        stats["pubs_used"] += used
        stats["pubs_dropped"] += dropped
        stats["truncated"] += 1 if dropped else 0
        if len(points) >= 64:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []
            logger.info("Indexed {} researchers so far...", stats["indexed"])
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    return stats


def evaluate(client: QdrantClient, source: str, universe: set[str]) -> dict:
    """Run the deterministic dense strategy over the ablation collection."""
    paths = (GOLD_TFG_PATH, GOLD_TFM_PATH) if source == "gold" else (SILVER_TFG_PATH, SILVER_TFM_PATH)
    records = [r for r in load_gold_standard(paths=paths) if r.director in universe]
    logger.info("Evaluating {} queries from the {} standard.", len(records), source)

    depth = max(K_VALUES)
    rr = 0.0
    hits = {k: 0 for k in K_VALUES}
    ndcg10 = 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for i, rec in enumerate(records, 1):
        found = client.query_points(
            collection_name=COLLECTION,
            query=_embed(rec.description),
            using=DENSE_VECTOR,
            limit=depth,
            with_payload=True,
        ).points
        ranked = [p.payload["researcher_id"] for p in found]
        rr += reciprocal_rank(ranked, rec.director)
        ndcg10 += ndcg_at_k(ranked, rec.director, 10)
        for k in K_VALUES:
            hits[k] += hit_at_k(ranked, rec.director, k)
        by_year[rec.year].append(reciprocal_rank(ranked, rec.director))
        if i % 50 == 0:
            logger.info("{}/{} queries done", i, len(records))

    n = len(records)
    return {
        "n": n,
        "mrr": rr / n,
        "ndcg@10": ndcg10 / n,
        **{f"hit@{k}": hits[k] / n for k in K_VALUES},
        "mrr_by_year": {y: sum(v) / len(v) for y, v in sorted(by_year.items())},
        "n_by_year": {y: len(v) for y, v in sorted(by_year.items())},
    }


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("gold", "silver"), default="silver")
    ap.add_argument(
        "--out",
        default="evaluation/reports/ablation_rawtext.json",
        help="Where to write the metrics (also printed).",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing ablation collection instead of rebuilding it.",
    )
    args = ap.parse_args()

    # Same connection style as the production backend (dbs/vector/qdrant.py).
    client = QdrantClient(url=f"http://{_QDRANT_HOST}:{_QDRANT_PORT}", api_key=_QDRANT_API_KEY)

    # EXACTLY the campaign's universe: researchers present in BOTH production
    # indices, computed with the harness's own helper. Using only the dense index
    # would evaluate over a different candidate set and a different query subset,
    # and the comparison against `summary_direct` would not mean anything.
    from dbs.vector import get_vector_db

    _, _, universe = common_indexed_universe(get_vector_db())
    logger.info("Common indexed universe (campaign protocol): {} researchers.", len(universe))

    stats = {}
    if not args.skip_build:
        stats = build_collection(client, universe)
        logger.info("Build stats: {}", stats)

    metrics = evaluate(client, args.source, universe)
    out = {"source": args.source, "text_budget_chars": RAW_TEXT_BUDGET, "build": stats, "metrics": metrics}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Wrote {}", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
