import os
from typing import Any, Iterable

from arango import ArangoClient
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_USER = os.environ.get("ARANGO_ROOT_USERNAME")
_PASSWORD = os.environ.get("ARANGO_ROOT_PASSWORD")
_DB_NAME = os.environ.get("ARANGO_DB")
_HOST = os.environ.get("ARANGO_HOST", "localhost")
_PORT = os.environ.get("ARANGO_PORT")

API_HOST = f"http://{_HOST}:{_PORT}"
ARANGO_CLIENT = ArangoClient(hosts=API_HOST)
sys_db = ARANGO_CLIENT.db("_system", username=_USER, password=_PASSWORD)


class ArangoBackend:
    def __init__(self, db_name: str = _DB_NAME):
        db_name = db_name or _DB_NAME
        if not db_name:
            raise ValueError("No database name provided and ARANGO_DB is not set")

        client = ArangoClient(hosts=API_HOST)
        sys_db = client.db("_system", username=_USER, password=_PASSWORD)

        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            logger.info("Created ArangoDB database {}", db_name)

        self._db = client.db(db_name, username=_USER, password=_PASSWORD)
        logger.debug("Connected to ArangoDB database {}", db_name)

    @staticmethod
    def _normalize_identity_fields(document: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of a document with `id` and `_key` kept in sync."""
        normalized = dict(document)
        if "id" in normalized:
            normalized["_key"] = str(normalized["id"])
        elif "_key" in normalized:
            normalized["id"] = str(normalized["_key"])
        return normalized

    def _ensure_collection(self, name: str, edge: bool = False):
        if not self._db.has_collection(name):
            self._db.create_collection(name, edge=edge)
            logger.debug("Created ArangoDB collection {} (edge={})", name, edge)
        return self._db.collection(name)

    def ensure_collections(self) -> None:
        """Create the necessary collections in the ArangoDB database if they do not exist."""
        self._ensure_collection("pdi")
        self._ensure_collection("publication")
        self._ensure_collection("position")
        self._ensure_collection("subject")
        self._ensure_collection("last_year_project")
        # self._ensure_collection("organizations")

        self._ensure_collection("authorship", edge=True)
        self._ensure_collection("works", edge=True)
        self._ensure_collection("teaching", edge=True)
        self._ensure_collection("tutoring", edge=True)
        # self._ensure_collection("affiliation", edge=True)

    def query(
        self,
        statement: str,
        bind_vars: dict[str, Any] | None = None,
        *,
        count: bool = False,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """Run an AQL query and return all results as dictionaries.

        :param statement: The AQL query string to execute.
        :param bind_vars: Optional dictionary of bind variables to use in the query.
        :param count: Whether to include the total count of results in the cursor, default is False.
        :param batch_size: The number of results to fetch in each batch from the database, default is 1000.
        :return: A list of dictionaries representing the query results.
        """
        cursor = self._db.aql.execute(
            statement,
            bind_vars=bind_vars or {},
            count=count,
            batch_size=batch_size,
        )
        return list(cursor)

    def fetch_document(self, collection: str, id: str) -> dict[str, Any] | None:
        """Fetch a single document by id.

        :param collection: The name of the collection to fetch from.
        :param id: The document id to fetch.
        :return: The document as a dictionary, or None if it does not exist.
        """
        return self._db.collection(collection).get(id)

    def document_exists(self, collection: str, id: str) -> bool:
        """Check whether a document exists in a collection.

        :param collection: The name of the collection to check.
        :param id: The document id to check for existence.
        :return: True if the document exists, False otherwise.
        """
        return self.fetch_document(collection, id) is not None

    def count_documents(self, collection: str) -> int:
        """Return the number of documents in a collection.

        :param collection: The name of the collection to count documents in.
        :return: The number of documents in the specified collection.
        """
        return self._db.collection(collection).count()

    def upsert_documents(self, collection: str, docs: list[dict[str, Any]]) -> int:
        """Create or update documents in the specified collection of the ArangoDB database.

        :param collection: The name of the collection where the documents should be upserted.
        :param docs: A list of dictionaries representing the documents to be upserted.
            Each dictionary should contain an `id` field to identify the document for upsert operations.
        """
        if not docs:
            return 0
        col = self._ensure_collection(collection)
        normalized_docs = [self._normalize_identity_fields(doc) for doc in docs]
        col.insert_many(normalized_docs, overwrite_mode="update")
        return len(docs)

    def upsert_edges(self, collection: str, edges: list[dict[str, Any]]) -> int:
        """Create or update edge documents in the specified collection of the ArangoDB database.

        :param collection: The name of the collection where the edge documents should be upserted.
        :param edges: A list of dictionaries representing the edge documents to be upserted.
        """
        if not edges:
            return 0
        col = self._ensure_collection(collection, edge=True)
        normalized_edges = [self._normalize_identity_fields(edge) for edge in edges]
        col.insert_many(normalized_edges, overwrite_mode="update")
        return len(edges)

    def delete_document(self, collection_name: str, id: str) -> bool:
        """Delete a single document by id.

        :param collection_name: The name of the collection from which to delete the document.
        :param id: The document id to delete.
        :return: True if the document was successfully deleted, False otherwise."""
        try:
            self._db.collection(collection_name).delete(id, ignore_missing=True)
            return True
        except Exception as exc:
            logger.exception(
                "Error deleting document {} from {}: {}", id, collection_name, exc
            )
            return False

    def delete_documents(self, collection_name: str, ids: Iterable[str]) -> int:
        """Delete multiple documents by id.

        :param collection_name: The name of the collection from which to delete the documents.
        :param ids: An iterable of ids of the documents to delete.
        :return: The number of documents that were successfully deleted.
        """
        deleted = 0
        for id in ids:
            if self.delete_document(collection_name, id):
                deleted += 1
        return deleted

    def clear_collection(self, collection_name: str) -> bool:
        """Remove all documents from a collection.

        :param collection_name: The name of the collection to clear.
        :return: True if the collection was successfully cleared, False otherwise.
        """
        try:
            self._db.collection(collection_name).truncate()
            return True
        except Exception as exc:
            logger.exception("Error truncating collection {}: {}", collection_name, exc)
            return False

    def clear_database(self) -> bool:
        """Clear all data from the selected ArangoDB database by dropping all collections.

        :return: True if the database was successfully cleared, False otherwise.
        """
        try:
            # Get all collections in the database
            collections = self._db.collections()

            for collection in collections:
                # Skip system collections
                if collection["name"].startswith("_"):
                    continue
                try:
                    self._db.delete_collection(collection["name"])
                    logger.debug("Deleted collection {}", collection["name"])
                except Exception as exc:
                    logger.exception(
                        "Error deleting collection {}: {}", collection["name"], exc
                    )

            logger.info("Cleared all collections from ArangoDB database")
            return True
        except Exception as exc:
            logger.exception("Error clearing ArangoDB database: {}", exc)
            return False

    def adapt_document(
        self, collection: str, document: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapt document for ArangoDB format.

        ArangoDB specific mappings:
        - `id` field is mapped to `_key` (using existing _normalize_identity_fields)

        :param collection: Collection name (not used for ArangoDB, but kept for interface consistency)
        :param document: Document dict from Pydantic model
        :return: Backend-specific document dict with `_key` set from `id`
        """
        return self._normalize_identity_fields(document)
