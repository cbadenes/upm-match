import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from minio import Minio
from minio.error import S3Error

load_dotenv()

__ACCESS_KEY = os.environ.get("MINIO_ROOT_USER")
__SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD")
__HOST = os.environ.get("MINIO_HOST", "localhost")
__PORT = os.environ.get("MINIO_PORT")

MINIO_CLIENT = Minio(
    f"{__HOST}:{__PORT}",
    access_key=__ACCESS_KEY,
    secret_key=__SECRET_KEY,
    secure=False,
)

BUCKET_NAME: str | None = os.environ.get("MINIO_BUCKET_NAME")


def bucket_exists(bucket_name: str) -> bool:
    """Check whether a bucket exists.

    :param bucket_name: The name of the bucket to check.
    :return: True if the bucket exists, False otherwise.
    """
    try:
        return MINIO_CLIENT.bucket_exists(bucket_name)
    except Exception as exc:
        logger.exception("Error checking bucket {}: {}", bucket_name, exc)
        return False


def ensure_bucket(bucket_name: str, must_exist: bool = False) -> bool:
    """Check if the specified bucket exists in MinIO and create it if it does not exist (unless must_exist is True).

    :param bucket_name: The name of the bucket to check or create.
    :param must_exist: Whether the bucket must already exist, defaults to False
    :return: True if the bucket exists or was created, False otherwise
    """
    try:
        if bucket_exists(bucket_name):
            return True
        if must_exist:
            logger.warning("Bucket {} not found", bucket_name)
            return False

        MINIO_CLIENT.make_bucket(bucket_name)
        logger.info("Created MinIO bucket {}", bucket_name)
        return True

    except Exception as exc:
        logger.exception("Error ensuring bucket {}: {}", bucket_name, exc)
        return False


def list_objects(
    bucket_name: str, prefix: str = "", recursive: bool = False, **kwargs
) -> list[str]:
    """List object names in a bucket for a given prefix.

    :param bucket_name: The name of the bucket to list objects from.
    :param prefix: The prefix to filter objects by (default is an empty string, which matches all objects).
    :param recursive: Whether to list objects recursively (default is False).
    :param kwargs: Additional keyword arguments to pass to the list_objects method of the MinIO client.
    :return: A list of object names that match the specified prefix in the bucket, or an empty list if the bucket does not exist.
    """
    if not ensure_bucket(bucket_name, must_exist=True):
        return []
    return [
        obj.object_name
        for obj in MINIO_CLIENT.list_objects(
            bucket_name, prefix=prefix, recursive=recursive, **kwargs
        )
    ]


def object_exists(bucket_name: str, object_name: str) -> bool:
    """Check whether an object exists in a bucket.

    :param bucket_name: The name of the bucket to check.
    :param object_name: The name of the object to check for.
    :return: True if the object exists in the bucket, False otherwise.
    """
    if not ensure_bucket(bucket_name, must_exist=True):
        return False
    try:
        MINIO_CLIENT.stat_object(bucket_name, object_name)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
            return False
        logger.exception(
            "Error checking object {} in bucket {}: {}", object_name, bucket_name, exc
        )
        return False
    except Exception as exc:
        logger.exception(
            "Error checking object {} in bucket {}: {}", object_name, bucket_name, exc
        )
        return False


def upload_bytes(
    bucket_name: str,
    data: bytes,
    object_name: str,
    content_type: str = "application/octet-stream",
) -> bool:
    """Upload raw bytes to a bucket.

    :param bucket_name: The name of the bucket to upload to.
    :param data: The raw bytes to upload.
    :param object_name: The name that the object should have in the bucket (including any desired path within the bucket).
    :param content_type: The content type of the data being uploaded, default is "application/octet-stream".
    :return: True if the upload was successful, False otherwise.
    """
    if not ensure_bucket(bucket_name):
        return False
    try:
        MINIO_CLIENT.put_object(
            bucket_name,
            object_name,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        logger.debug("Uploaded bytes to {}/{}", bucket_name, object_name)
        return True
    except Exception as exc:
        logger.exception("Error uploading to {}/{}: {}", bucket_name, object_name, exc)
        return False


def upload_text(
    bucket_name: str, text: str, object_name: str, content_type: str = "text/plain"
) -> bool:
    """Upload text content (UTF-8 encoded) to a bucket.

    :param bucket_name: The name of the bucket to upload to.
    :param text: The text content to upload.
    :param object_name: The name that the object should have in the bucket (including any desired path within the bucket).
    :param content_type: The content type of the text being uploaded, default is "text/plain".
    :return: True if the upload was successful, False otherwise.
    """
    return upload_bytes(
        bucket_name, text.encode("utf-8"), object_name, content_type=content_type
    )


def upload_json(bucket_name: str, payload: Any, object_name: str) -> bool:
    """Upload a JSON-serializable object to a bucket.

    :param bucket_name: The name of the bucket to upload to.
    :param payload: The JSON-serializable object to upload.
    :param object_name: The name that the object should have in the bucket (including any desired path within the bucket).
    :return: True if the upload was successful, False otherwise.
    """
    return upload_text(
        bucket_name,
        json.dumps(payload, ensure_ascii=False, indent=4),
        object_name,
        content_type="application/json",
    )


def upload_file(bucket_name: str, file_path: Path, object_name: str) -> bool:
    """Save a file to the specified bucket in MinIO.

    :param bucket_name: The name of the bucket where the file should be uploaded.
    :param file_path: The local path to the file that should be uploaded.
    :param object_name: The name that the file should have in the bucket (including any desired path within the bucket).
    """
    if not ensure_bucket(bucket_name):
        logger.error("Bucket {} could not be created or accessed", bucket_name)
        return False
    try:
        MINIO_CLIENT.fput_object(bucket_name, object_name, str(file_path))
        logger.debug("Uploaded {} to {}/{}", file_path, bucket_name, object_name)
        return True
    except Exception as exc:
        logger.exception(
            "Error uploading {} to {}/{}: {}", file_path, bucket_name, object_name, exc
        )
        return False


def download_file(bucket_name: str, object_name: str, destination: Path) -> bool:
    """Download an object into a local file.

    :param bucket_name: The name of the bucket where the object is stored.
    :param object_name: The name of the object in the bucket (including any path within the bucket).
    :param destination: The local path where the downloaded file should be saved.
    :return: True if the download was successful, False otherwise.
    """
    if not ensure_bucket(bucket_name, must_exist=True):
        return False
    try:
        MINIO_CLIENT.fget_object(bucket_name, object_name, str(destination))
        logger.debug("Downloaded {}/{} to {}", bucket_name, object_name, destination)
        return True
    except Exception as exc:
        logger.exception(
            "Error downloading {}/{} to {}: {}",
            bucket_name,
            object_name,
            destination,
            exc,
        )
        return False


def read_file(bucket_name: str, object_name: str) -> str | None:
    """Read the content of a file stored in the specified bucket in MinIO.

    :param bucket_name: The name of the bucket where the file is stored.
    :param object_name: The name of the file in the bucket (including any path within the bucket).
    :return: The content of the file as a string, or None if the bucket or file does not exist.
    """
    if not ensure_bucket(bucket_name, must_exist=True):
        logger.warning("Bucket {} does not exist", bucket_name)
        return None
    try:
        response = MINIO_CLIENT.get_object(bucket_name, object_name)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        logger.exception(
            "Error reading object {}/{}: {}", bucket_name, object_name, exc
        )
        return None
    except Exception as exc:
        logger.exception(
            "Error reading object {}/{}: {}", bucket_name, object_name, exc
        )
        return None


def read_json(bucket_name: str, object_name: str) -> Any | None:
    """Read and decode a JSON object from a bucket.

    :param bucket_name: The name of the bucket where the JSON object is stored.
    :param object_name: The name of the JSON object in the bucket (including any path within the bucket).
    :return: The decoded JSON object, or None if the bucket or object does not exist or if there was an error decoding the JSON.
    """
    content = read_file(bucket_name, object_name)
    if content is None:
        return None
    return json.loads(content)


def delete_file(bucket_name: str, object_name: str) -> bool:
    """Delete an object from a bucket.

    :param bucket_name: The name of the bucket where the object is stored.
    :param object_name: The name of the object in the bucket (including any path within the bucket).
    :return: True if the deletion was successful, False otherwise.
    """
    if not ensure_bucket(bucket_name, must_exist=True):
        return False
    try:
        MINIO_CLIENT.remove_object(bucket_name, object_name)
        logger.debug("Deleted {}/{}", bucket_name, object_name)
        return True
    except Exception as exc:
        logger.exception("Error deleting {}/{}: {}", bucket_name, object_name, exc)
        return False


def delete_files(bucket_name: str, object_names: list[str]) -> int:
    """Delete several objects from a bucket.

    :param bucket_name: The name of the bucket where the objects are stored.
    :param object_names: A list of names of the objects to delete from the bucket (including any paths within the bucket).
    :return: The number of objects that were successfully deleted.
    """
    return sum(delete_file(bucket_name, name) for name in object_names)
