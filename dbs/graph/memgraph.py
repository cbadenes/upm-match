import os
from typing import Any, Iterable

from dotenv import load_dotenv
from gqlalchemy import Memgraph
from loguru import logger

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. Check your .env file."
        )
    return value


_USERNAME = _require_env("MEMGRAPH_USERNAME")
_PASSWORD = _require_env("MEMGRAPH_PASSWORD")
_HOST = os.environ.get("MEMGRAPH_HOST", "localhost")
_PORT = int(os.environ.get("MEMGRAPH_PORT", "7687"))
_ENCRYPTED = os.environ.get("MEMGRAPH_ENCRYPTED", "false").lower() == "true"


class MemgraphBackend:
    def __init__(
        self,
        host: str = _HOST,
        port: int = _PORT,
        username: str = _USERNAME,
        password: str = _PASSWORD,
        encrypted: bool = _ENCRYPTED,
    ):
        """Initialize Memgraph backend connection.

        :param host: The hostname of the Memgraph server, default is localhost.
        :param port: The port number for the Memgraph server, default is 7687.
        :param username: The username for authentication, default is empty string.
        :param password: The password for authentication, default is empty string.
        :param encrypted: Whether to use encrypted connection, default is False.
        """
        self._db = Memgraph(
            host=host,
            port=port,
            username=username,
            password=password,
            encrypted=encrypted,
        )
        logger.debug("Connected to Memgraph at {}:{}", host, port)

    def ensure_collections(self) -> None:
        """Create the required labels/relationships in Memgraph if they do not exist.
        Memgraph doesn't require pre-creation of labels like some other graph DBs, but we ensure the connection is active.
        """
        try:
            list(
                self._db.execute_and_fetch(
                    "MATCH (n) RETURN COUNT(n) AS node_count LIMIT 1"
                )
            )
            logger.debug("Memgraph connection verified and ready for collections")
        except Exception as exc:
            logger.exception("Error verifying Memgraph connection: {}", exc)
            raise

    def query(
        self,
        statement: str,
        bind_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return all results as a list of dicts.

        :param statement: The Cypher query string to execute.
        :param bind_vars: Optional dictionary of bind variables to use in the query.
        :return: A list of dictionaries representing the query results.
        """
        try:
            results = list(
                self._db.execute_and_fetch(statement, parameters=bind_vars or {})
            )
            return results
        except Exception as exc:
            logger.exception("Error executing Cypher query: {}", exc)
            return []

    def fetch_document(self, collection: str, id: str) -> dict[str, Any] | None:
        """Fetch a single node by label and id property.

        :param collection: The label name (e.g., "pdi", "publication").
        :param id: The value of the id property to fetch.
        :return: The node as a dictionary, or None if it does not exist.
        """
        query = f"""
        MATCH (n:{collection} {{id: $id}})
        RETURN n
        """
        try:
            results = list(
                self._db.execute_and_fetch(
                    query,
                    parameters={"id": id},
                )
            )
            if results:
                return dict(results[0])
            return None
        except Exception as exc:
            logger.exception(
                "Error fetching document from {} with id {}: {}",
                collection,
                id,
                exc,
            )
            return None

    def document_exists(self, collection: str, id: str) -> bool:
        """Check whether a node exists with the given label and id.

        :param collection: The label name to check.
        :param id: The value of the id property to check.
        :return: True if the node exists, False otherwise.
        """
        return (
            self.query(
                f"MATCH (n:{collection.capitalize()} {{id: $id}}) RETURN n", {"id": id}
            )
            is not None
        )

    def count_documents(self, collection: str) -> int:
        """Return the number of nodes with the given label.

        :param collection: The label name to count.
        :return: The number of nodes with the specified label.
        """
        query = f"MATCH (n:{collection}) RETURN COUNT(n) AS count"
        try:
            results = list(self._db.execute_and_fetch(query))
            if results:
                return results[0].get("count", 0)
            return 0
        except Exception as exc:
            logger.exception(
                "Error counting documents in {}: {}",
                collection,
                exc,
            )
            return 0

    def upsert_documents(self, collection: str, docs: list[dict[str, Any]]) -> int:
        """Create or update nodes in the specified label of Memgraph.

        :param collection: The label name where nodes should be upserted.
        :param docs: A list of dictionaries representing the nodes to be upserted.
            Each dictionary should contain an `id` field to identify the node.
            Additional properties in each dict will be set as node properties.
        :return: The number of nodes processed.
        """
        if not docs:
            return 0

        processed = 0
        for doc in docs:
            if "id" not in doc:
                logger.warning(
                    "Document missing id field, skipping: {}",
                    doc,
                )
                continue

            id_value = doc["id"]
            merge_query = f"""
            MERGE (n:{collection} {{id: $id}})
            SET n += $properties
            """
            try:
                self._db.execute(
                    merge_query,
                    parameters={"id": id_value, "properties": dict(doc)},
                )
                processed += 1
            except Exception as exc:
                logger.exception(
                    "Error upserting document {} in {}: {}",
                    id_value,
                    collection,
                    exc,
                )

        logger.debug("Upserted {} documents into {}", processed, collection)
        return processed

    def upsert_edges(self, collection: str, edges: list[dict[str, Any]]) -> int:
        """Create or update relationship documents in Memgraph.

        :param collection: The relationship type name (e.g., "authorship", "teaching").
        :param edges: A list of dictionaries representing the relationships.
            Each edge must have "id", "_from", and "_to" fields.
            Additional properties will be set on the relationship.
        :return: The number of edges processed.
        """
        if not edges:
            return 0

        processed = 0

        for edge in edges:
            edge_id = edge.get("id")

            from_ref = edge.get("_from") or edge.get("from_node")
            to_ref = edge.get("_to") or edge.get("to_node")

            if not edge_id or not from_ref or not to_ref:
                logger.warning(
                    "Edge missing required fields (id, _from/_to), skipping: {}",
                    edge,
                )
                continue

            from_collection, fromid = from_ref.split("/", 1)
            to_collection, toid = to_ref.split("/", 1)

            merge_query = f"""
            MATCH (from:{from_collection} {{id: $from_id}})
            MATCH (to:{to_collection} {{id: $to_id}})
            MERGE (from)-[r:{collection}]->(to)
            SET r += $properties
            """

            try:
                self._db.execute(
                    merge_query,
                    parameters={
                        "from_id": fromid,
                        "to_id": toid,
                        "properties": {
                            k: v
                            for k, v in edge.items()
                            if k not in ["_from", "_to", "from_node", "to_node"]
                        },
                    },
                )
                processed += 1

            except Exception as exc:
                logger.exception(
                    "Error upserting edge {} of type {}: {}",
                    edge_id,
                    collection,
                    exc,
                )

        logger.debug("Upserted {} edges of type {}", processed, collection)
        return processed

    def delete_document(self, collection_name: str, id: str) -> bool:
        """Delete a single node by id.

        :param collection_name: The label name of the node to delete.
        :param id: The id value of the node to delete.
        :return: True if the node was successfully deleted, False otherwise.
        """
        delete_query = f"""
        MATCH (n:{collection_name} {{id: $id}})
        DETACH DELETE n
        """
        try:
            self._db.execute(delete_query, parameters={"id": id})
            return True
        except Exception as exc:
            logger.exception(
                "Error deleting document {} from {}: {}",
                id,
                collection_name,
                exc,
            )
            return False

    def delete_documents(self, collection_name: str, ids: Iterable[str]) -> int:
        """Delete multiple nodes by id.

        :param collection_name: The label name of the nodes to delete.
        :param ids: An iterable of ids of the nodes to delete.
        :return: The number of nodes that were successfully deleted.
        """
        deleted = 0
        for id in ids:
            if self.delete_document(collection_name, id):
                deleted += 1
        return deleted

    def clear_collection(self, collection_name: str) -> bool:
        """Remove all nodes with the given label.

        :param collection_name: The label name to clear.
        :return: True if the collection was successfully cleared, False otherwise.
        """
        clear_query = f"MATCH (n:{collection_name}) DETACH DELETE n"
        try:
            self._db.execute(clear_query)
            logger.debug("Cleared collection {}", collection_name)
            return True
        except Exception as exc:
            logger.exception(
                "Error clearing collection {}: {}",
                collection_name,
                exc,
            )
            return False

    def clear_database(self) -> bool:
        """Clear all data from the Memgraph database by deleting all nodes and relationships.

        :return: True if the database was successfully cleared, False otherwise.
        """
        clear_query = "MATCH (n) DETACH DELETE n"
        try:
            self._db.execute(clear_query)
            logger.info("Cleared all nodes and relationships from Memgraph database")
            return True
        except Exception as exc:
            logger.exception("Error clearing Memgraph database: {}", exc)
            return False

    def adapt_document(
        self, collection: str, document: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapt document for Memgraph format.

        Memgraph specific mappings:
        - Node labels should start with capital letter
        - `title` field may be mapped to `name` for certain node types
        - Remove None values to keep Cypher clean

        :param collection: Collection/label name (used to determine mappings)
        :param document: Document dict from Pydantic model
        :return: Backend-specific document dict adapted for Memgraph
        """
        adapted = dict(document)

        if "full_name" in adapted and collection == "Pdi":
            adapted["name"] = adapted.pop("full_name")

        # For certain node types, map 'title' to 'name' (if title exists)
        if "title" in adapted and collection in (
            "Publication",
            "Subject",
            "LastYearProject",
        ):
            adapted["name"] = adapted.pop("title")

        # Remove None values for cleaner Cypher properties
        # Keep id field (it's required for Memgraph lookups)
        adapted = {k: v for k, v in adapted.items() if v is not None or k == "id"}

        return adapted
