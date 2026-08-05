"""Google Cloud Storage helpers (copied from conv_vit)."""

import logging
import os

import torch

from scdiag.logging_utils import fatal


def parse_gcs_uri(gcs_uri):
  """Parse a ``gs://BUCKET/PREFIX`` URI into ``(bucket, prefix)``."""
  if not gcs_uri.startswith("gs://"):
    fatal(f"GCS URI must start with gs://, got: {gcs_uri}", ValueError)
  without_scheme = gcs_uri[len("gs://"):]
  parts = without_scheme.split("/", 1)
  bucket = parts[0]
  prefix = parts[1] if len(parts) > 1 else ""
  return bucket, prefix


def gcs_upload(bucket_name, local_path, gcs_prefix):
  """Upload a single file to GCS.  Requires *google-cloud-storage*."""
  try:
    from google.cloud import storage
  except ImportError:
    fatal(
        "google-cloud-storage is required for GCS sync. "
        "Install with: pip install skin-classifier[gcs]", ImportError)
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  filename = os.path.basename(local_path)
  blob_name = f"{gcs_prefix}/{filename}" if gcs_prefix else filename
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(local_path)
  return f"gs://{bucket_name}/{blob_name}"


def save_checkpoint(save_dict, path, gcs_uri=None):
  """Save a checkpoint dict to disk and optionally sync to GCS."""
  dirname = os.path.dirname(path)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  torch.save(save_dict, path)
  if gcs_uri:
    bucket, prefix = parse_gcs_uri(gcs_uri)
    remote = gcs_upload(bucket, path, prefix)
    logging.info(f"  Synced to GCS: {remote}")
  return path
