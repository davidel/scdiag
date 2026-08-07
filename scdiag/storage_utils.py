"""Unified cloud storage helpers for GCS and Cloudflare R2."""

import logging
import os

import torch

from scdiag.logging_utils import fatal


def parse_storage_uri(uri):
  """Parse a ``gs://`` or ``r2://`` URI into ``(scheme, bucket, prefix)``.

    Args:
        uri: A storage URI.  Must start with ``gs://`` (Google Cloud
            Storage) or ``r2://`` (Cloudflare R2).

    Returns:
        A tuple ``(scheme, bucket, prefix)`` where *scheme* is ``"gs"``
        or ``"r2"``.
  """
  if uri.startswith("gs://"):
    scheme = "gs"
    without_scheme = uri[len("gs://"):]
  elif uri.startswith("r2://"):
    scheme = "r2"
    without_scheme = uri[len("r2://"):]
  else:
    fatal(f"URI must start with gs:// or r2://, got: {uri}", ValueError)

  parts = without_scheme.split("/", 1)
  bucket = parts[0]
  prefix = parts[1] if len(parts) > 1 else ""
  return scheme, bucket, prefix


def _upload_gcs(bucket_name, local_path, prefix):
  """Upload *local_path* to a GCS bucket under *prefix*."""
  from google.cloud import storage

  client = storage.Client()
  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(local_path)
  return f"gs://{bucket_name}/{blob_name}"


def _upload_r2(bucket_name, local_path, prefix):
  """Upload *local_path* to a Cloudflare R2 bucket under *prefix*."""
  import boto3

  endpoint_url = os.environ.get("R2_ENDPOINT_URL")
  if not endpoint_url:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if account_id:
      endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    else:
      fatal("CLOUDFLARE_ACCOUNT_ID or R2_ENDPOINT_URL must be set for R2 "
            "uploads.", ValueError)

  s3_client = boto3.client(
      "s3",
      endpoint_url=endpoint_url,
      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
  )

  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  s3_client.upload_file(local_path, bucket_name, blob_name)
  return f"r2://{bucket_name}/{blob_name}"


def storage_upload(bucket_name, local_path, prefix="", scheme="gs"):
  """Upload *local_path* to cloud storage.

    Args:
        bucket_name: Target bucket name.
        local_path: Local file to upload.
        prefix: Optional key prefix inside the bucket.
        scheme: ``"gs"`` for Google Cloud Storage or ``"r2"`` for
            Cloudflare R2.

    Returns:
        The full remote URI of the uploaded object.
  """
  if scheme == "gs":
    return _upload_gcs(bucket_name, local_path, prefix)
  elif scheme == "r2":
    return _upload_r2(bucket_name, local_path, prefix)
  else:
    fatal(f"Unsupported storage scheme: {scheme!r}", ValueError)


def save_checkpoint(save_dict, path, remote_uri=None):
  """Save a checkpoint dict to disk and optionally sync to cloud storage.

    Args:
        save_dict: Dictionary to pass to ``torch.save``.
        path: Local file path for the checkpoint.
        remote_uri: Optional ``gs://`` or ``r2://`` URI to upload to.
    """
  dirname = os.path.dirname(path)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  torch.save(save_dict, path)
  if remote_uri:
    scheme, bucket, prefix = parse_storage_uri(remote_uri)
    remote = storage_upload(bucket, path, prefix, scheme=scheme)
    logging.info(f"  Synced to remote: {remote}")
  return path
