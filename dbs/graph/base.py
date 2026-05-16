from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphDataBase(Protocol):
    """Contract for graph database implementations.
    This protocol defines the methods that any graph database class should implement to be compatible with the rest of the system.

    :param Protocol: The base class for defining protocols in Python, which are a way to specify the expected behavior of classes without requiring inheritance.
    """

    def ensure_collections(self) -> None:
        """Create the required collections/labels if they do not exist."""
        ...

    def query(
        self,
        statement: str,
        bind_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a query and return all results as a list of dicts."""
        ...

    def fetch_document(self, collection: str, id: str) -> dict[str, Any] | None:
        """Fetch a single document by id, or None if it does not exist."""
        ...

    def document_exists(self, collection: str, id: str) -> bool:
        """Return True if a document with the given id exists."""
        ...

    def count_documents(self, collection: str) -> int:
        """Return the number of documents in a collection."""
        ...

    def upsert_documents(self, collection: str, docs: list[dict[str, Any]]) -> int:
        """Create or update documents. Returns the number of documents processed."""
        ...

    def upsert_edges(self, collection: str, edges: list[dict[str, Any]]) -> int:
        """Create or update edge documents. Returns the number of edges processed."""
        ...

    def delete_document(self, collection: str, id: str) -> bool:
        """Delete a document by id. Returns True on success."""
        ...

    def delete_documents(self, collection: str, ids: list[str]) -> int:
        """Delete multiple documents by id. Returns the count of successful deletions."""
        ...

    def clear_collection(self, collection: str) -> bool:
        """Remove all documents from a collection. Returns True on success."""
        ...

    def clear_database(self) -> bool:
        """Clear all data from the database.
        For ArangoDB, clears all collections in the selected database.
        For Memgraph, clears all nodes and relationships in the entire database.

        :return: True if the operation was successful, False otherwise."""
        ...

    def adapt_document(self, collection: str, document: dict[str, Any]) -> dict[str, Any]:
        """Adapt a document dict to backend-specific format.
        
        This method handles field mappings specific to each backend:
        - ArangoDB: id → _key, normalizes node/edge references
        - Memgraph: title → name (for certain node types), other adaptations
        
        :param collection: The collection/label name (e.g., 'Pdi', 'Publication')
        :param document: The document dict from Pydantic model
        :return: Backend-specific document dict
        """
        ...
