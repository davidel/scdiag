"""Google Cloud Storage helpers (copied from conv_vit)."""

import logging
import os


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
        raise ImportError(
            "google-cloud-storage is required for GCS sync. "
            "Install with: pip install skin-classifier[gcs]"
        )
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
    import torch

    torch.save(save_dict, path)
    if gcs_uri:
        bucket, prefix = parse_gcs_uri(gcs_uri)
        remote = gcs_upload(bucket, path, prefix)
        logging.info(f"  Synced to GCS: {remote}")
    return path


def checkpoint_dict(model, optimizer, scheduler, epoch, extra=None):
    """Build a standard checkpoint dict."""
    d = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }
    if extra:
        d.update(extra)
    return d
