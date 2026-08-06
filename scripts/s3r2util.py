#!/usr/bin/env python3
import argparse
import os
import sys

import boto3
from botocore.exceptions import ClientError


def create_s3_client():
  account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
  access_key_id = os.getenv("R2_ACCESS_KEY_ID")
  secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

  if not all([account_id, access_key_id, secret_access_key]):
    print("Error: Missing credentials.", file=sys.stderr)
    print(
        "Please set CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY in your environment.",
        file=sys.stderr)
    sys.exit(1)

  return boto3.client(
      service_name="s3",
      endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
      aws_access_key_id=access_key_id,
      aws_secret_access_key=secret_access_key,
      region_name="auto",
  )


def _resolve_destination(source, destination):
  """If destination ends with '/', append the source basename to it."""
  if destination.endswith("/"):
    basename = os.path.basename(source.rstrip("/"))
    return destination + basename
  return destination


def parse_r2_path(path):
  """Parses 'r2://bucket-name/object/key' strings.

  Returns:
    A tuple of (bucket_name, object_key), or (None, None) if not an R2 path.
  """
  if path.startswith("r2://"):
    cleaned = path[5:]
    if "/" in cleaned:
      bucket, key = cleaned.split("/", 1)
      return bucket, key
    return cleaned, ""
  return None, None


def handle_ls(args, s3_client):
  bucket, prefix = parse_r2_path(args.path)
  if not bucket:
    print("Error: 'ls' target must start with r2://", file=sys.stderr)
    sys.exit(1)

  try:
    kwargs = {"Bucket": bucket}
    if prefix:
      kwargs["Prefix"] = prefix

    paginator = s3_client.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(**kwargs):
      if "Contents" in page:
        found = True
        for obj in page["Contents"]:
          print(f"{obj['LastModified']}  {obj['Size']:12} B  {obj['Key']}")

    if not found:
      print(f"No objects found in r2://{bucket}/{prefix}")

  except ClientError as e:
    print(f"Error executing ls: {e}", file=sys.stderr)


def handle_cp(args, s3_client):
  src_bucket, src_key = parse_r2_path(args.source)
  destination = _resolve_destination(args.source, args.destination)
  dst_bucket, dst_key = parse_r2_path(destination)

  try:
    if not src_bucket and dst_bucket:
      print(f"Uploading {args.source} -> r2://{dst_bucket}/{dst_key}")
      s3_client.upload_file(args.source, dst_bucket, dst_key)
      print("Upload complete.")

    elif src_bucket and not dst_bucket:
      print(f"Downloading r2://{src_bucket}/{src_key} -> {args.destination}")
      local_dir = os.path.dirname(args.destination)
      if local_dir:
        os.makedirs(local_dir, exist_ok=True)
      s3_client.download_file(src_bucket, src_key, args.destination)
      print("Download complete.")

    elif src_bucket and dst_bucket:
      print(f"Copying r2://{src_bucket}/{src_key} -> r2://{dst_bucket}/{dst_key}")
      copy_source = {"Bucket": src_bucket, "Key": src_key}
      s3_client.copy(copy_source, dst_bucket, dst_key)
      print("Remote copy complete.")

    else:
      print("Error: Both source and destination cannot be local files.",
            file=sys.stderr)
      sys.exit(1)

  except ClientError as e:
    print(f"Error executing cp: {e}", file=sys.stderr)
    raise


def handle_rm(args, s3_client):
  bucket, key = parse_r2_path(args.path)
  if not bucket or not key:
    print("Error: Target must be a specific object starting with r2://bucket/key",
          file=sys.stderr)
    sys.exit(1)

  try:
    print(f"Deleting r2://{bucket}/{key}")
    s3_client.delete_object(Bucket=bucket, Key=key)
    print("Delete complete.")
  except ClientError as e:
    print(f"Error executing rm: {e}", file=sys.stderr)


def handle_mv(args, s3_client):
  """Copies the source to destination, then deletes the source.

  Not atomic: if copy succeeds but deletion fails, a warning is printed.
  """
  src_bucket, src_key = parse_r2_path(args.source)

  try:
    handle_cp(args, s3_client)
  except ClientError:
    print("Error: Copy failed, aborting move.", file=sys.stderr)
    sys.exit(1)

  if src_bucket and src_key:
    try:
      print(f"Removing source file r2://{src_bucket}/{src_key}")
      s3_client.delete_object(Bucket=src_bucket, Key=src_key)
      print("Move process completed successfully.")
    except ClientError as e:
      print(f"Warning: File copied, but failed to delete original source: {e}",
            file=sys.stderr)
  elif not src_bucket:
    try:
      os.remove(args.source)
      print("Removed local source file.")
    except OSError as e:
      print(f"Warning: File uploaded, but failed to delete local file: {e}",
            file=sys.stderr)


def main():
  parser = argparse.ArgumentParser(
      description="Cloudflare R2 CLI tool mimicking AWS S3 commands.")
  subparsers = parser.add_subparsers(dest="command",
                                     required=True,
                                     help="Sub-command to run")

  parser_ls = subparsers.add_parser(
      "ls", help="List objects in a bucket prefix (Format: r2://bucket-name/prefix)")
  parser_ls.add_argument("path", help="Target bucket path starting with r2://")
  parser_ls.set_defaults(func=handle_ls)

  parser_cp = subparsers.add_parser("cp", help="Copy files locally or remotely")
  parser_cp.add_argument("source",
                         help="Source file path (local path or r2://bucket/key)")
  parser_cp.add_argument("destination",
                         help="Destination file path (local path or r2://bucket/key)")
  parser_cp.set_defaults(func=handle_cp)

  parser_rm = subparsers.add_parser(
      "rm", help="Remove an object from R2 (Format: r2://bucket-name/key)")
  parser_rm.add_argument("path", help="Target object key path starting with r2://")
  parser_rm.set_defaults(func=handle_rm)

  parser_mv = subparsers.add_parser(
      "mv", help="Move files locally or remotely (Copies then deletes source)")
  parser_mv.add_argument("source",
                         help="Source file path (local path or r2://bucket/key)")
  parser_mv.add_argument("destination",
                         help="Destination file path (local path or r2://bucket/key)")
  parser_mv.set_defaults(func=handle_mv)

  args = parser.parse_args()
  s3_client = create_s3_client()
  args.func(args, s3_client)


if __name__ == "__main__":
  main()
