"""
dbs - A package for database interactions, including graph databases, vector databases, and MinIO object storage.

This package provides a unified interface for working with different types of databases, such as graph databases (e.g., ArangoDB, Memgraph), vector databases (e.g., Qdrant), and object storage services (e.g., MinIO).

Use case:
>   from dbs import get_graph_db, GraphDataBase
>
>   db: GraphDataBase = get_graph_db()  # default backend from env
>   db: GraphDataBase = get_graph_db("arango")  # explicitly specify ArangoDB backend

>   from dbs import get_vector_db, VectorDataBase
>
>   vdb: VectorDataBase = get_vector_db()  # defaults to Qdrant

>   from dbs import minio
>   minio.upload_json(bucket, payload, "path/to/object.json")
>   # or import individual functions directly:
>   from dbs import upload_json, read_file
"""

from . import graph, minio, vector
from .graph import GraphDataBase, get_graph_db
from .minio import (
    bucket_exists,
    delete_file,
    delete_files,
    download_file,
    ensure_bucket,
    list_objects,
    object_exists,
    read_file,
    read_json,
    upload_bytes,
    upload_file,
    upload_json,
    upload_text,
)
from .vector import SearchResult, VectorDataBase, VectorPoint, get_vector_db

__all__ = [
    # Namespaces
    "graph",
    "minio",
    "vector",
    # Graph
    "GraphDataBase",
    "get_graph_db",
    # Vector
    "VectorDataBase",
    "VectorPoint",
    "SearchResult",
    "get_vector_db",
    # MinIO
    "bucket_exists",
    "delete_file",
    "delete_files",
    "download_file",
    "ensure_bucket",
    "list_objects",
    "object_exists",
    "read_file",
    "read_json",
    "upload_bytes",
    "upload_file",
    "upload_json",
    "upload_text",
]
