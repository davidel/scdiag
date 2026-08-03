"""Google Cloud Storage helpers (copied from conv_vit)."""

import logging
import os

import torch


def parse_gcs_uri(gcs_uri):
  """Parse a ``gs://BUCKET/PREFIX`` URI into ``(bucket, prefix)``."""
  if not gcs_uri.startswith("gs://"):
    raise ValueError(f"GCS URI must start with gs://, got: {gcs_uri}")
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
    raise ImportError("google-cloud-storage is required for GCS sync. "
                      "Install with: pip install skin-classifier[gcs]")
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


def checkpoint_dict(model, optimizer, scheduler, epoch,
                    states_to_save=None, scaler=None, **extra):
  """Build a standard checkpoint dict.

  ``states_to_save`` is a set like ``{"opt", "sched", "amp"}``.
  If ``None``, everything is saved (backward compat).
  Any additional keyword arguments are merged into the dict as-is.
  """
  d = {
      "model_state_dict": model.state_dict(),
      "epoch": epoch,
      "id2label": model.config.id2label,
  }
  if states_to_save is None or "opt" in states_to_save:
    d["optimizer_state_dict"] = optimizer.state_dict()
  if states_to_save is None or "sched" in states_to_save:
    d["scheduler_state_dict"] = scheduler.state_dict()
  if "amp" in states_to_save:
    d["scaler_state_dict"] = (
        scaler.state_dict() if scaler is not None else None
    )
  d.update(extra)
  return d
