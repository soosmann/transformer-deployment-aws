"""Verify filesystem effects of S3 downloads with a fake cloud client."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from api import s3


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    client = Mock()
    monkeypatch.setattr(s3.boto3, "client", Mock(return_value=client))
    return client


def set_objects(storage, *pages):
    storage.get_paginator.return_value.paginate.return_value = pages


def write_download(bucket, key, target):
    Path(target).write_text(key)


def test_download_includes_pages_and_subdirectories(storage, tmp_path):
    set_objects(
        storage,
        {"Contents": [{"Key": "models/test/"}, {"Key": "models/test/config.json"}]},
        {"Contents": [{"Key": "models/test/nested/tokenizer.json"}]},
    )
    storage.download_file.side_effect = write_download
    destination = tmp_path / "test"
    s3.download_dir_from_s3(destination, "models/test/")
    assert (destination / "config.json").read_text() == "models/test/config.json"
    assert (destination / "nested/tokenizer.json").is_file()
    assert list(tmp_path.iterdir()) == [destination]
    storage.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="test-bucket",
        Prefix="models/test/",
    )


@pytest.mark.parametrize("page", [{}, {"Contents": [{"Key": "models/test/"}]}])
def test_empty_prefix_does_not_create_cached_model(storage, tmp_path, page):
    set_objects(storage, page)
    with pytest.raises(FileNotFoundError):
        s3.download_dir_from_s3(tmp_path / "test", "models/test")
    assert list(tmp_path.iterdir()) == []


def test_failed_download_cleans_partial_files(storage, tmp_path):
    set_objects(
        storage,
        {
            "Contents": [
                {"Key": "models/test/config.json"},
                {"Key": "models/test/weights.bin"},
            ]
        },
    )

    def download(bucket, key, target):
        if key.endswith("weights.bin"):
            raise OSError("transfer interrupted")
        write_download(bucket, key, target)

    storage.download_file.side_effect = download
    with pytest.raises(OSError):
        s3.download_dir_from_s3(tmp_path / "test", "models/test")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "key", ["models/test/../../escape", "models/test//absolute", "outside/file"]
)
def test_unsafe_object_keys_are_rejected(storage, tmp_path, key):
    set_objects(storage, {"Contents": [{"Key": key}]})
    with pytest.raises(ValueError):
        s3.download_dir_from_s3(tmp_path / "test", "models/test")
    storage.download_file.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_existing_cache_is_not_overwritten(storage, tmp_path):
    destination = tmp_path / "test"
    destination.mkdir()
    (destination / "config.json").write_text("original")
    with pytest.raises(FileExistsError):
        s3.download_dir_from_s3(destination, "models/test")
    assert (destination / "config.json").read_text() == "original"
    storage.download_file.assert_not_called()


def test_missing_bucket_does_not_contact_aws(monkeypatch, tmp_path):
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    client_factory = Mock(side_effect=AssertionError("AWS should not be contacted"))
    monkeypatch.setattr(s3.boto3, "client", client_factory)
    with pytest.raises(ValueError, match="AWS_S3_BUCKET"):
        s3.download_dir_from_s3(tmp_path / "test", "models/test")
    client_factory.assert_not_called()


def test_image_uploads_use_unique_keys_and_one_hour_links(storage, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    for _ in range(2):
        s3.upload_image_to_s3(image)
    keys = [call.args[2] for call in storage.upload_file.call_args_list]
    assert keys[0] != keys[1]
    assert all(key.startswith("ml-images/") and key.endswith(".png") for key in keys)
    storage.generate_presigned_url.assert_called_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": keys[-1]},
        ExpiresIn=3600,
    )
