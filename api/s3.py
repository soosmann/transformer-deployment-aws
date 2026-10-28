"""S3 model downloads and temporary image links using Boto3 credentials."""

import logging
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4

import boto3

logger = logging.getLogger(__name__)


def get_bucket_name():
    bucket = os.getenv("AWS_S3_BUCKET", "").strip()
    if not bucket:
        raise ValueError("Set AWS_S3_BUCKET to download models or upload images.")
    return bucket


def download_dir_from_s3(local_path, s3_prefix):
    """Download a directory from an S3 bucket.

    Publish the directory only after all objects have downloaded successfully.
    An interrupted download must not look like a usable cached model.
    """
    bucket_name = get_bucket_name()
    s3 = boto3.client("s3")
    destination = Path(local_path)
    if destination.exists():
        raise FileExistsError(f"Model directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = s3_prefix.strip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")

    with TemporaryDirectory(dir=destination.parent) as temporary_dir:
        staging = Path(temporary_dir) / "model"
        staging.mkdir()
        downloaded = 0
        for result in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in result.get("Contents", []):
                key = obj["Key"]
                if not key.startswith(prefix):
                    raise ValueError("S3 returned a key outside the model prefix.")
                if key.endswith("/"):
                    continue  # S3 can contain directory markers rather than files.
                relative = PurePosixPath(key[len(prefix) :])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("S3 object key would escape the model directory.")
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket_name, key, str(target))
                downloaded += 1
                logger.info("Downloaded %s", key)

        if not downloaded:
            raise FileNotFoundError(f"No model files found under s3://{bucket_name}/{prefix}")
        staging.rename(destination)


def upload_image_to_s3(file_name, s3_prefix="ml-images", object_name=None):
    """Upload an image to an S3 bucket and return a presigned URL."""
    bucket_name = get_bucket_name()
    s3 = boto3.client("s3")
    if object_name is None:
        # Separate concurrent uploads; the original demo used one shared temp.jpg.
        object_name = f"{uuid4().hex}{Path(file_name).suffix}"
    key = f"{s3_prefix.strip('/')}/{object_name}"
    s3.upload_file(str(file_name), bucket_name, key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=3600,  # URL expires in 1 hour
    )
