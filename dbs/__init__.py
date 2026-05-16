"""
dbs - A package for database interactions, including graph databases and MinIO object storage.

This package provides a unified interface for working with different types of databases, such as graph databases (e.g., ArangoDB) and object storage services (e.g., MinIO).
It includes modules for handling graph database operations, as well as functions for managing MinIO buckets and objects.

Use case:
>   from dbs import get_graph_db, GraphDataBase
>
>   db: GraphDataBase = get_graph_db()  # default backend from env
>   db: GraphDataBase = get_graph_db("arango")  # explicitly specify ArangoDB backend

>   from dbs import minio
>   minio.upload_json(bucket, payload, "path/to/object.json")
>   # or import individual functions directly:
>   from dbs import upload_json, read_file
"""

from . import graph, minio
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

__all__ = [
    # Namespaces
    "graph",
    "minio",
    # Graph
    "GraphDataBase",
    "get_graph_db",
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
