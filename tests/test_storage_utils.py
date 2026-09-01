"""Tests for cloud storage helpers (GCS, R2, S3 URI handling and S3 upload)."""
import os

import pytest

from scdiag.storage_utils import (
    _upload_s3,
    parse_storage_uri,
    save_checkpoint,
    storage_upload,
)


class TestParseStorageUri:

  def test_s3_uri(self):
    assert parse_storage_uri("s3://my-bucket/runs/exp1") == ("s3", "my-bucket",
                                                             "runs/exp1")

  def test_s3_uri_bucket_only(self):
    assert parse_storage_uri("s3://my-bucket") == ("s3", "my-bucket", "")

  def test_gcs_and_r2_unchanged(self):
    assert parse_storage_uri("gs://b/p") == ("gs", "b", "p")
    assert parse_storage_uri("r2://b/p") == ("r2", "b", "p")

  def test_rejects_other_schemes(self):
    for uri in ("http://b/p", "azure://b/p", "my-bucket/p"):
      with pytest.raises(ValueError, match="URI must start with"):
        parse_storage_uri(uri)


class _FakeS3Client:

  def __init__(self):
    self.calls = []

  def upload_file(self, local_path, bucket, key):
    self.calls.append((local_path, bucket, key))


class TestUploadS3:

  @pytest.fixture
  def fake_client(self, monkeypatch):
    client = _FakeS3Client()
    captured = {}

    def fake_boto3_client(service, **kwargs):
      captured["service"] = service
      captured["kwargs"] = kwargs
      return client

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    client.captured = captured
    return client

  @pytest.fixture
  def local_file(self, tmp_path):
    path = tmp_path / "model_latest.pt"
    path.write_bytes(b"checkpoint-bytes")
    return str(path)

  def test_explicit_credentials_and_session_token(self, fake_client, local_file,
                                                  monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "TOKEN_TEST")

    result = _upload_s3("my-bucket", local_file, "runs/exp1")

    assert fake_client.captured["service"] == "s3"
    assert fake_client.captured["kwargs"] == {
        "aws_access_key_id": "AKIA_TEST",
        "aws_secret_access_key": "SECRET_TEST",
        "aws_session_token": "TOKEN_TEST",
    }
    assert fake_client.calls == [(local_file, "my-bucket", "runs/exp1/model_latest.pt")]
    assert result == "s3://my-bucket/runs/exp1/model_latest.pt"

  def test_session_token_defaults_to_none(self, fake_client, local_file, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    _upload_s3("my-bucket", local_file, "")

    assert fake_client.captured["kwargs"]["aws_session_token"] is None
    assert fake_client.calls == [(local_file, "my-bucket", "model_latest.pt")]

  def test_falls_back_to_default_chain_without_env_creds(self, fake_client, local_file,
                                                         monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
      monkeypatch.delenv(var, raising=False)

    _upload_s3("my-bucket", local_file, "prefix")

    assert fake_client.captured["kwargs"] == {}
    assert fake_client.calls == [(local_file, "my-bucket", "prefix/model_latest.pt")]


class TestStorageUploadDispatch:

  def test_s3_dispatch(self, monkeypatch, tmp_path):
    local = tmp_path / "ckpt.pt"
    local.write_bytes(b"x")
    seen = {}

    def fake_upload_s3(bucket, path, prefix):
      seen["args"] = (bucket, path, prefix)
      return f"s3://{bucket}/{prefix}/ckpt.pt"

    monkeypatch.setattr("scdiag.storage_utils._upload_s3", fake_upload_s3)
    result = storage_upload("b", str(local), "p", scheme="s3")
    assert seen["args"] == ("b", str(local), "p")
    assert result == "s3://b/p/ckpt.pt"

  def test_unknown_scheme_fatal(self, tmp_path):
    local = tmp_path / "ckpt.pt"
    local.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported storage scheme"):
      storage_upload("b", str(local), "p", scheme="wasabi")


class TestSaveCheckpointS3:

  def test_save_with_s3_remote(self, tmp_path, monkeypatch):
    local = tmp_path / "nested" / "ckpt.pt"
    seen = {}

    def fake_upload_s3(bucket, path, prefix):
      seen["args"] = (bucket, path, prefix)
      return f"s3://{bucket}/{prefix}/ckpt.pt"

    monkeypatch.setattr("scdiag.storage_utils._upload_s3", fake_upload_s3)
    result = save_checkpoint({"epoch": 1}, str(local), remote_uri="s3://my-bucket/runs")
    assert os.path.isfile(result)
    assert seen["args"] == ("my-bucket", str(local), "runs")

  def test_no_remote_no_upload(self, tmp_path):
    local = tmp_path / "ckpt.pt"
    result = save_checkpoint({"epoch": 1}, str(local))
    assert os.path.isfile(result)
    assert result == str(local)
