"""
Vector database backends and utilities.

Usage:
>   from dbs.vector import get_vector_db, VectorDataBase
>
>   db = get_vector_db()  # defaults to Qdrant
>   db.ensure_collection("researcher_keywords", sparse_field="keywords_bm42")
>   db.upsert_points("researcher_keywords", "keywords_bm42", points)
>   results = db.search_by_keywords("researcher_keywords", "keywords_bm42", ["machine learning"])
"""

from .base import SearchResult, VectorDataBase, VectorPoint
from .qdrant import QdrantBackend

__all__ = [
    "VectorDataBase",
    "VectorPoint",
    "SearchResult",
    "QdrantBackend",
    "get_vector_db",
]

_BACKENDS: dict[str, type] = {
    "qdrant": QdrantBackend,
}


def get_vector_db(backend: str = "qdrant", **kwargs) -> VectorDataBase:
    """Factory function to get a vector database backend instance.

    :param backend: The name of the backend to use (currently only "qdrant").
    :param kwargs: Additional keyword arguments passed to the backend constructor.
    :return: An instance of the specified vector database backend.
    """
    if backend not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unsupported vector database backend '{backend}'. Available: {available}"
        )
    return _BACKENDS[backend](**kwargs)
