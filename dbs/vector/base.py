from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorPoint:
    """A single point to be stored in the vector database."""

    id: int
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result returned from a vector similarity search."""

    id: int
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorDataBase(Protocol):
    """Contract for vector database implementations."""

    def ensure_collection(
        self,
        collection: str,
        sparse_field: str,
        on_disk: bool = False,
    ) -> None:
        """Create the collection with a sparse vector field if it does not exist.
        If the collection already exists, this is a no-op.

        :param collection: Name of the collection to create or verify.
        :param sparse_field: Name of the sparse vector field to configure.
        :param on_disk: Whether to use on-disk storage for the sparse index (if supported by the backend).
        """
        ...

    def collection_exists(self, collection: str) -> bool:
        """Return True if the collection exists.

        :param collection: Name of the collection to check.
        :return: True if the collection exists, False otherwise.
        """
        ...

    def count_points(self, collection: str) -> int:
        """Return the number of points in a collection.

        :param collection: Name of the collection to count points in.
        :return: Number of points in the collection.
        """
        ...

    def point_exists(self, collection: str, point_id: int) -> bool:
        """Return True if a point with the given id exists in the collection.

        :param collection: Name of the collection to check.
        :param point_id: ID of the point to check for.
        :return: True if the point exists, False otherwise.
        """
        ...

    def get_all_ids(self, collection: str) -> set[int]:
        """Return the set of all point ids currently stored in the collection.

        :param collection: Name of the collection to retrieve ids from.
        :return: Set of all point ids in the collection.
        """
        ...

    def upsert_points(
        self,
        collection: str,
        sparse_field: str,
        points: list[VectorPoint],
    ) -> int:
        """Upsert points into the collection. Returns the number of points processed.
        If a point with the same id already exists, it will be updated with the new vector and payload.

        :param collection: Name of the collection to upsert into.
        :param sparse_field: Name of the sparse vector field to use for the point vectors.
        :param points: List of VectorPoint objects to upsert.
        :return: Number of points successfully upserted.
        """
        ...

    def search_by_keywords(
        self,
        collection: str,
        sparse_field: str,
        keywords: list[str],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for the most similar points using keyword-based sparse retrieval.

        :param collection: Name of the collection to search.
        :param sparse_field: Name of the sparse vector field to use for similarity.
        :param keywords: List of keyword strings to use as the search query.
        :param limit: Maximum number of results to return.
        :return: List of SearchResult objects containing id, score, and payload.
        """
        ...

    def delete_collection(self, collection: str) -> bool:
        """Delete a collection entirely. Returns True on success.
        Use with caution, as this will remove all data in the collection.

        :param collection: Name of the collection to delete.
        :return: True if the collection was successfully deleted, False otherwise.
        """
        ...
