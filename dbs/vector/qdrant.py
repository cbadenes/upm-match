import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient, models

from .base import SearchResult, VectorPoint

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. Check your .env file."
        )
    return value


_HOST = os.environ.get("QDRANT_HOST", "localhost")
_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_API_KEY = _require_env("QDRANT_API_KEY")

# Sparse (BM42) embedding model.
SPARSE_MODEL_NAME = os.environ.get(
    "SPARSE_EMBED_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)
# BM42 is a tiny model (~30 MB) so CPU inference is fast enough and keeps the GPU free for Ollama.
# Change to None to let ONNX Runtime pick the device.
_FASTEMBED_PROVIDERS = ["CPUExecutionProvider"]


class QdrantBackend:
    """Qdrant vector database backend using BM42 sparse embeddings.

    The BM42 model is loaded lazily on first use to avoid startup cost when the instance is created but embedding is not immediately needed.
    """

    def __init__(
        self,
        host: str = _HOST,
        port: int = _PORT,
        api_key: str = _API_KEY,
    ):
        self._client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
        self._embedder: SparseTextEmbedding | None = None
        logger.debug("Connected to Qdrant at {}:{}", host, port)

    @property
    def _model(self) -> SparseTextEmbedding:
        if self._embedder is None:
            logger.debug("Loading BM42 sparse embedding model (first use)...")
            self._embedder = SparseTextEmbedding(
                SPARSE_MODEL_NAME, providers=_FASTEMBED_PROVIDERS
            )
            logger.debug("BM42 model ready")
        return self._embedder

    def embed(self, keywords: list[str]) -> tuple[list[int], list[float]]:
        """Generate a BM42 sparse vector from a list of keywords.

        Keywords are joined as a single text before embedding so the model computes token-level weights across the full keyword set.

        :param keywords: List of keyword strings to embed.
        :return: (indices, values) tuple for a SparseVector.
        """
        text = " ".join(k.strip().lower() for k in keywords if k.strip())
        if not text:
            return [], []
        result = list(self._model.embed([text]))[0]
        return result.indices.tolist(), result.values.tolist()

    def ensure_collection(
        self,
        collection: str,
        sparse_field: str,
        on_disk: bool = False,
    ) -> None:
        if self._client.collection_exists(collection):
            logger.debug("Collection '{}' already exists", collection)
            return
        self._client.create_collection(
            collection_name=collection,
            sparse_vectors_config={
                sparse_field: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=on_disk),
                    modifier=models.Modifier.IDF,
                )
            },
        )
        logger.info(
            "Created Qdrant collection '{}' with sparse field '{}'",
            collection,
            sparse_field,
        )

    def collection_exists(self, collection: str) -> bool:
        return self._client.collection_exists(collection)

    def list_collections(self) -> list[str]:
        """Return the names of all physical collections (aliases excluded)."""
        return [c.name for c in self._client.get_collections().collections]

    def latest_collection(self, prefix: str) -> str | None:
        """Return the newest physical collection for a family.

        Phase 7 builds into timestamp-suffixed collections (`<prefix>_<ts>`) and flips an alias once the build succeeds.
        Resolving the lexicographically largest matching name gives the most recent build directly, without depending on the alias being set.
        Falls back to a flat `<prefix>` collection (legacy, pre-alias layout) when no versioned one exists.

        :param prefix: Collection family prefix, e.g. "researcher_summaries".
        :return: Newest matching collection name, or None if none exist.
        """
        matches = [
            name
            for name in self.list_collections()
            if name == prefix or name.startswith(f"{prefix}_")
        ]
        return max(matches) if matches else None

    def count_points(self, collection: str) -> int:
        # `count(exact=True)` reflects every stored point. We avoid
        # `get_collection().points_count`, which counts only *indexed* points and
        # is None until the optimizer runs — reporting 0 for freshly built
        # collections that actually hold data.
        if not self.collection_exists(collection):
            return 0
        return self._client.count(collection_name=collection, exact=True).count

    def point_exists(self, collection: str, point_id: int) -> bool:
        results = self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def get_all_ids(self, collection: str) -> set[int]:
        """Fetch all point IDs from the collection in a single paginated scan.

        More efficient than calling point_exists per researcher when checking which researchers are already indexed.
        """
        if not self.collection_exists(collection):
            return set()

        ids: set[int] = set()
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=collection,
                offset=offset,
                limit=1000,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(r.id for r in records)
            if next_offset is None:
                break
            offset = next_offset

        return ids

    def upsert_points(
        self,
        collection: str,
        sparse_field: str,
        points: list[VectorPoint],
    ) -> int:
        if not points:
            return 0
        qdrant_points = [
            models.PointStruct(
                id=p.id,
                vector={
                    sparse_field: models.SparseVector(
                        indices=p.sparse_indices,
                        values=p.sparse_values,
                    )
                },
                payload=p.payload,
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=qdrant_points)
        return len(qdrant_points)

    def search_by_keywords(
        self,
        collection: str,
        sparse_field: str,
        keywords: list[str],
        limit: int = 10,
    ) -> list[SearchResult]:
        indices, values = self.embed(keywords)
        if not indices:
            logger.warning("Empty embedding for keywords: {}", keywords)
            return []
        results = self._client.query_points(
            collection_name=collection,
            query=models.SparseVector(indices=indices, values=values),
            using=sparse_field,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(id=p.id, score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    def search_by_keyword_sets(
        self,
        collection: str,
        sparse_field: str,
        general_keywords: list[str],
        specific_keywords: list[str],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search using Qdrant's native RRF fusion over two sparse keyword queries.

        Embeds general and specific keyword lists independently and merges the
        ranked results using Reciprocal Rank Fusion so neither list dominates.

        :param collection: Name of the Qdrant collection.
        :param sparse_field: Name of the sparse vector field to query.
        :param general_keywords: Broad domain keywords for the first query.
        :param specific_keywords: Precise technique/method keywords for the second query.
        :param limit: Maximum number of fused results to return.
        :return: List of SearchResult objects sorted by fused relevance score.
        """
        general_idx, general_val = (
            self.embed(general_keywords) if general_keywords else ([], [])
        )
        specific_idx, specific_val = (
            self.embed(specific_keywords) if specific_keywords else ([], [])
        )

        if not general_idx and not specific_idx:
            logger.warning("Both keyword lists produced empty embeddings")
            return []

        # Fall back to a single search if only one list produced a valid embedding
        if not general_idx:
            return self.search_by_keywords(
                collection, sparse_field, specific_keywords, limit
            )
        if not specific_idx:
            return self.search_by_keywords(
                collection, sparse_field, general_keywords, limit
            )

        results = self._client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=general_idx, values=general_val),
                    using=sparse_field,
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=specific_idx, values=specific_val
                    ),
                    using=sparse_field,
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(id=p.id, score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    def search_dense(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        using: str | None = None,
    ) -> list[SearchResult]:
        """Search a dense vector field of the collection.

        :param collection: Name of the Qdrant collection.
        :param vector: Pre-computed dense query vector.
        :param limit: Maximum number of results to return.
        :param using: Named dense vector to query, or None for the default vector.
        :return: List of SearchResult objects sorted by similarity score descending.
        """
        results = self._client.query_points(
            collection_name=collection,
            query=vector,
            using=using,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(id=p.id, score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    def search_hybrid(
        self,
        collection: str,
        dense_vector: list[float],
        dense_using: str,
        sparse_field: str,
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Fuse a dense query and a sparse query over the same collection with native RRF.

        Both legs run as Qdrant prefetches and are merged with Reciprocal Rank
        Fusion, so the semantic signal (dense summary vector) and the lexical
        signal (sparse expertise-domain vector) contribute jointly to the ranking.
        If the sparse query is empty (no usable keywords) the search degrades
        gracefully to a pure dense query.

        :param collection: Name of the collection to search.
        :param dense_vector: Pre-computed dense query vector.
        :param dense_using: Named dense vector field to query.
        :param sparse_field: Named sparse vector field to query.
        :param sparse_indices: Sparse vector indices for the keyword leg.
        :param sparse_values: Sparse vector values for the keyword leg.
        :param limit: Maximum number of fused results to return.
        :return: List of SearchResult objects sorted by fused relevance score.
        """
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=dense_using,
                limit=limit * 2,
            )
        ]
        if sparse_indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                    using=sparse_field,
                    limit=limit * 2,
                )
            )
        else:
            logger.debug("Hybrid search: empty sparse leg, falling back to dense only")

        results = self._client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(id=p.id, score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    def delete_collection(self, collection: str) -> bool:
        try:
            self._client.delete_collection(collection)
            logger.info("Deleted Qdrant collection '{}'", collection)
            return True
        except Exception as exc:
            logger.exception("Error deleting collection '{}': {}", collection, exc)
            return False
