"""
Graph database backends and utilities.
"""

from .arango import ArangoBackend
from .base import GraphDataBase
from .memgraph import MemgraphBackend

__all__ = ["GraphDataBase", "ArangoBackend", "MemgraphBackend"]

_BACKENDS = {
    "arango": ArangoBackend,
    "memgraph": MemgraphBackend,
    "neo4j": None,  # Neo4jBackend,
}


def get_graph_db(backend: str = "arango", **kwargs) -> GraphDataBase:
    """Factory function to get a graph database backend instance.

    :param backend: The name of the backend to use (e.g., "arango", "memgraph", "neo4j").
    :param kwargs: Additional keyword arguments to pass to the backend constructor.
    :return: An instance of the specified graph database backend.
    """
    if backend not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unsupported graph database backend '{backend}'. Available options are: {available}"
        )

    cls = _BACKENDS[backend]
    if cls is None:
        raise NotImplementedError(f"The backend '{backend}' is not yet implemented.")

    return cls(**kwargs)
