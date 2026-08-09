#!/usr/bin/env python3
"""CloudFlare R2 CLI utility (S3-compatible).

Provides ls/cp/rm/mv commands for R2 buckets via the boto3 S3 API.

NOTE: This script could in theory be simplified by setting the standard
AWS environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_REGION) from the CloudFlare R2 ones and forwarding directly to the
`aws s3` CLI (e.g. `aws s3 ls --endpoint-url <r2-endpoint>`). R2
implements the S3 API, so the AWS CLI works as a drop-in client. That
approach would remove the boto3 dependency and most of this script.
However, keeping our own wrapper lets us tailor the interface (e.g.
trailing-slash destination semantics, r2:// path prefixes) without
depending on the `aws` CLI being installed.
"""
import argparse
import fnmatch
import os
import re
import sys

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def fatal(msg, code=1):
  print(f"Error: {msg}", file=sys.stderr)
  sys.exit(code)


def create_s3_client():
  account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
  access_key_id = os.getenv("R2_ACCESS_KEY_ID")
  secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

  if not all([account_id, access_key_id, secret_access_key]):
    fatal("Missing credentials. Please set CLOUDFLARE_ACCOUNT_ID, "
          "R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY in your environment.")

  return boto3.client(
      service_name="s3",
      endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
      aws_access_key_id=access_key_id,
      aws_secret_access_key=secret_access_key,
      region_name="auto",
  )


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


def _has_wildcard(path):
  """Check if path contains shell wildcard characters (* or ?)."""
  return "*" in path or "?" in path


def _get_static_prefix(pattern):
  """Return the longest prefix of *pattern* that contains no wildcard chars."""
  m = re.search(r"[*?]", pattern)
  return pattern[:m.start()] if m else pattern


def _expand_wildcard(s3_client, r2_path):
  """Expand a wildcard R2 path via server-side prefix listing + client-side fnmatch.

  Returns a list of (bucket, key) tuples for all matching objects.
  """
  bucket, pattern = parse_r2_path(r2_path)
  if not bucket:
    fatal(f"Wildcard path must start with r2://: {r2_path}")

  prefix = _get_static_prefix(pattern)

  # Try the prefix as-is first, then fall back to toggling leading '/'.
  # Some buckets store keys with a leading '/' (e.g. '/content/file.pt')
  # while parse_r2_path strips it (giving 'content/file.pt').
  prefixes_to_try = [prefix]
  if prefix.startswith("/"):
    prefixes_to_try.append(prefix[1:])
  elif prefix:
    prefixes_to_try.append("/" + prefix)

  matches = []
  try:
    paginator = s3_client.get_paginator("list_objects_v2")
    for try_prefix in prefixes_to_try:
      page_iter = paginator.paginate(Bucket=bucket, Prefix=try_prefix)
      for page in page_iter:
        for obj in page.get("Contents", []):
          key = obj["Key"].lstrip("/")
          if fnmatch.fnmatch(key, pattern):
            matches.append((bucket, key))
      if matches:
        break
  except ClientError as exc:
    fatal(f"Error listing objects: {exc}")

  if not matches:
    fatal(f"No objects matched pattern: {r2_path}")
  return matches


def _compute_dest_key(src_key, src_prefix, dst_prefix):
  """Map a matched source key to a destination key.

  Args:
    src_key: the full object key of the matched source object.
    src_prefix: the static (non-wildcard) prefix of the source pattern.
    dst_prefix: the destination prefix (bucket/key/path).
  """
  relative = src_key[len(src_prefix):]
  return dst_prefix + relative


def _format_size(num_bytes):
  """Format byte count as a human-readable string (KB, MB, GB, TB)."""
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if abs(num_bytes) < 1024:
      return f"{num_bytes:.2f} {unit}"
    num_bytes /= 1024
  return f"{num_bytes:.2f} PB"


def handle_ls(args, s3_client):
  bucket, prefix = parse_r2_path(args.path)
  if not bucket:
    fatal("'ls' target must start with r2://")

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
          local_time = obj["LastModified"].astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
          print(f"{local_time}  {_format_size(obj['Size']):>14s}  {obj['Key']}")

    if not found:
      print(f"No objects found in r2://{bucket}/{prefix}")

  except ClientError as e:
    fatal(f"ls failed: {e}")


def _is_dir_destination(destination):
  """Check if the destination should be treated as a directory.

  Returns True if it ends with '/' or is an existing local directory.
  """
  return destination.endswith("/") or os.path.isdir(destination)


def handle_cp(args, s3_client):
  sources = args.sources
  destination = args.destination
  is_dir = _is_dir_destination(destination) or len(sources) > 1

  if len(sources) > 1 and not is_dir:
    fatal("Multiple sources require a directory destination (existing "
          "directory or trailing '/').")

  for source in sources:
    # --- Wildcard R2 source ------------------------------------------
    if _has_wildcard(source):
      src_bucket, src_prefix = parse_r2_path(source)
      if not src_bucket:
        fatal(f"Wildcard source must start with r2://: {source}")
      matches = _expand_wildcard(s3_client, source)
      dst_bucket, dst_prefix = parse_r2_path(destination)

      if not dst_bucket:
        # R2 wildcard -> local directory
        local_dir = destination
        if not os.path.isdir(local_dir):
          fatal(f"Destination directory does not exist: {local_dir}")

        print(f"Wildcard downloading {len(matches)} object(s) -> {local_dir}/")
        total_size = 0
        for _, obj_key in matches:
          head = s3_client.head_object(Bucket=src_bucket, Key=obj_key)
          total_size += head["ContentLength"]
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar:
          for _, obj_key in matches:
            local_path = os.path.join(local_dir, os.path.basename(obj_key))
            s3_client.download_file(src_bucket,
                                    obj_key,
                                    local_path,
                                    Callback=pbar.update)
        print(f"Downloaded {len(matches)} object(s).")

      else:
        # R2 wildcard -> R2 directory
        print(f"Wildcard copying {len(matches)} object(s) -> "
              f"r2://{dst_bucket}/{dst_prefix}")
        with tqdm(total=len(matches), unit="file", desc="Copy") as pbar:
          for _, obj_key in matches:
            dest_key = _compute_dest_key(obj_key, src_prefix, dst_prefix)
            s3_client.copy(
                {
                    "Bucket": src_bucket,
                    "Key": obj_key
                },
                dst_bucket,
                dest_key,
            )
            pbar.update(1)
        print(f"Copied {len(matches)} object(s).")
      continue

    # --- Single source ------------------------------------------------
    src_bucket, src_key = parse_r2_path(source)

    if is_dir:
      if src_bucket:
        # R2 -> R2 dir: preserve full key structure
        dest_path = (destination +
                     src_key if destination.endswith("/") else destination + "/" +
                     src_key)
      else:
        # Local -> R2 dir: key is relative path from cwd (no leading '/')
        rel_key = os.path.relpath(source).lstrip("/")
        dest_path = (destination +
                     rel_key if destination.endswith("/") else destination + "/" +
                     rel_key)
    else:
      dest_path = destination

    dst_bucket, dst_key = parse_r2_path(dest_path)

    try:
      if not src_bucket and dst_bucket:
        print(f"Uploading {source} -> r2://{dst_bucket}/{dst_key}")
        if args.progress:
          file_size = os.path.getsize(source)
          with tqdm(total=file_size, unit="B", unit_scale=True, desc="Upload") as pbar:
            s3_client.upload_file(source, dst_bucket, dst_key, Callback=pbar.update)
        else:
          s3_client.upload_file(source, dst_bucket, dst_key)

      elif src_bucket and not dst_bucket:
        print(f"Downloading r2://{src_bucket}/{src_key} -> {dest_path}")
        local_dir = os.path.dirname(dest_path)
        if local_dir:
          os.makedirs(local_dir, exist_ok=True)
        if args.progress:
          head = s3_client.head_object(Bucket=src_bucket, Key=src_key)
          total = head["ContentLength"]
          with tqdm(total=total, unit="B", unit_scale=True, desc="Download") as pbar:
            s3_client.download_file(src_bucket,
                                    src_key,
                                    dest_path,
                                    Callback=pbar.update)
        else:
          s3_client.download_file(src_bucket, src_key, dest_path)
        print("Download complete.")

      elif src_bucket and dst_bucket:
        print(f"Copying r2://{src_bucket}/{src_key} -> "
              f"r2://{dst_bucket}/{dst_key}")
        copy_source = {"Bucket": src_bucket, "Key": src_key}
        if args.progress:
          head = s3_client.head_object(Bucket=src_bucket, Key=src_key)
          total = head["ContentLength"]
          with tqdm(total=total, unit="B", unit_scale=True, desc="Copy") as pbar:
            s3_client.copy(copy_source, dst_bucket, dst_key, Callback=pbar.update)
        else:
          s3_client.copy(copy_source, dst_bucket, dst_key)
        print("Remote copy complete.")

      else:
        fatal("Both source and destination cannot be local files.")

    except ClientError as e:
      fatal(f"cp failed: {e}")


def handle_rm(args, s3_client):
  bucket, key = parse_r2_path(args.path)
  if not bucket or not key:
    fatal("Target must be a specific object starting with r2://bucket/key")

  if args.recursive:
    # List all objects under the prefix and batch-delete them.
    try:
      paginator = s3_client.get_paginator("list_objects_v2")
      to_delete = []
      for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
          to_delete.append({"Key": obj["Key"]})

      if not to_delete:
        print(f"No objects found under r2://{bucket}/{key}")
        return

      if args.dry_run:
        for entry in to_delete:
          print(f"  would delete: {entry['Key']}")
        print(f"Would delete {len(to_delete)} object(s).")
        return

      print(f"Deleting {len(to_delete)} object(s) under r2://{bucket}/{key} ...")
      # delete_objects accepts up to 1000 keys per request.
      for i in range(0, len(to_delete), 1000):
        batch = to_delete[i:i + 1000]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": batch,
                "Quiet": True
            },
        )
      print(f"Deleted {len(to_delete)} object(s).")
    except ClientError as e:
      fatal(f"rm failed: {e}")
  else:
    try:
      print(f"Deleting r2://{bucket}/{key}")
      s3_client.delete_object(Bucket=bucket, Key=key)
      print("Delete complete.")
    except ClientError as e:
      fatal(f"rm failed: {e}")


def handle_du(args, s3_client):
  """Report total size of objects matching an R2 prefix or wildcard pattern."""
  path = args.path
  bucket, prefix = parse_r2_path(path)
  if not bucket:
    fatal("'du' target must start with r2://")

  # If no filter given, list the entire bucket.
  if not prefix:
    prefix = ""

  # Determine if the user passed a wildcard.
  if _has_wildcard(path):
    matches = _expand_wildcard(s3_client, path)
    total = 0
    for _, key in matches:
      head = s3_client.head_object(Bucket=bucket, Key=key)
      total += head["ContentLength"]
    print(f"{_format_size(total)}\t{len(matches)} object(s)\t{path}")
  else:
    # Plain prefix listing — accumulate sizes as we paginate.
    try:
      paginator = s3_client.get_paginator("list_objects_v2")
      total = 0
      count = 0
      for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
          total += obj["Size"]
          count += 1
      print(f"{_format_size(total)}\t{count} object(s)\tr2://{bucket}/{prefix}")
    except ClientError as e:
      fatal(f"du failed: {e}")


def handle_mv(args, s3_client):
  """Renames R2 objects (copy + delete). All paths must be r2://."""
  sources = args.sources
  destination = args.destination
  is_dir = destination.endswith("/") or len(sources) > 1

  dst_bucket, _ = parse_r2_path(destination)
  if not dst_bucket:
    fatal("Destination must be an R2 path (r2://bucket/key)")

  if len(sources) > 1 and not is_dir:
    fatal("Multiple sources require a directory destination "
          "(ending with '/').")

  for source in sources:
    src_bucket, _ = parse_r2_path(source)
    if not src_bucket:
      fatal(f"Source must be an R2 path: {source}")
    # Extract the key portion, stripping leading '/' to avoid
    # creating double-slash R2 URLs.
    src_key = source.split("/", 3)[3].lstrip("/")

    # --- Wildcard source ---------------------------------------------
    if _has_wildcard(source):
      src_prefix = src_key  # already normalized
      matches = _expand_wildcard(s3_client, source)
      _, dst_prefix = parse_r2_path(destination)

      print(f"Wildcard moving {len(matches)} object(s):")
      try:
        if args.progress:
          with tqdm(total=len(matches), unit="file", desc="Move") as pbar:
            for _, obj_key in matches:
              dest_key = _compute_dest_key(obj_key, src_prefix, dst_prefix)
              s3_client.copy(
                  {
                      "Bucket": src_bucket,
                      "Key": obj_key
                  },
                  dst_bucket,
                  dest_key,
              )
              s3_client.delete_object(Bucket=src_bucket, Key=obj_key)
              pbar.update(1)
        else:
          for _, obj_key in matches:
            dest_key = _compute_dest_key(obj_key, src_prefix, dst_prefix)
            s3_client.copy(
                {
                    "Bucket": src_bucket,
                    "Key": obj_key
                },
                dst_bucket,
                dest_key,
            )
            s3_client.delete_object(Bucket=src_bucket, Key=obj_key)
        print(f"Moved {len(matches)} object(s).")
      except ClientError as e:
        fatal(f"mv failed: {e}")
      continue

    # --- Single source ------------------------------------------------
    _, dst_prefix = parse_r2_path(destination)
    src_basename = os.path.basename(src_key)

    if is_dir:
      dest_key = dst_prefix + src_basename
    else:
      dest_key = dst_prefix

    print(f"Moving r2://{src_bucket}/{src_key} -> r2://{dst_bucket}/{dest_key}")
    try:
      s3_client.copy(
          {
              "Bucket": src_bucket,
              "Key": src_key
          },
          dst_bucket,
          dest_key,
      )
      s3_client.delete_object(Bucket=src_bucket, Key=src_key)
      print("Move complete.")
    except ClientError as e:
      fatal(f"mv failed: {e}")


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
  parser_cp.add_argument("sources",
                         nargs="+",
                         help="One or more source paths (local or r2://bucket/key). "
                         "Wildcards (*, ?) are supported for R2 sources.")
  parser_cp.add_argument("destination",
                         help="Destination path (local or r2://bucket/key). "
                         "When multiple sources are given, the destination is "
                         "treated as a directory.")
  parser_cp.add_argument("--progress",
                         action="store_true",
                         help="Show a progress bar during transfer")
  parser_cp.set_defaults(func=handle_cp)

  parser_rm = subparsers.add_parser("rm", help="Remove an object or directory from R2")
  parser_rm.add_argument("path", help="Target object key path starting with r2://")
  parser_rm.add_argument("-r",
                         "--recursive",
                         action="store_true",
                         help="Delete all objects under the given prefix")
  parser_rm.add_argument("--dry-run",
                         action="store_true",
                         help="Show what would be deleted without deleting")
  parser_rm.set_defaults(func=handle_rm)

  parser_mv = subparsers.add_parser("mv", help="Rename R2 objects (copy + delete)")
  parser_mv.add_argument("sources",
                         nargs="+",
                         help="One or more R2 source paths (r2://bucket/key). "
                         "Wildcards (*, ?) are supported.")
  parser_mv.add_argument("destination",
                         help="Destination R2 path (r2://bucket/key). "
                         "When multiple sources are given, the destination is "
                         "treated as a directory prefix.")
  parser_mv.set_defaults(func=handle_mv)

  parser_du = subparsers.add_parser(
      "du", help="Report total size of objects under a prefix or matching a wildcard")
  parser_du.add_argument(
      "path",
      help="R2 prefix or wildcard pattern (e.g. r2://bucket/prefix/ or "
      "r2://bucket/prefix/*.pt)")
  parser_du.set_defaults(func=handle_du)

  args = parser.parse_args()
  s3_client = create_s3_client()
  args.func(args, s3_client)


if __name__ == "__main__":
  main()
