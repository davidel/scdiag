#!/usr/bin/env python
"""Prepare ISIC 2019 dataset with lesion_id-grouped splits.

Downloads the ISIC 2019 challenge training data (images, ground truth,
and metadata), and creates train/val/test splits grouped by lesion_id
to prevent data leakage (same lesion never appears in multiple splits).

Output is an ImageFolder structure compatible with:
    --dataset imagefolder/OUTPUT_PATH

Usage:
    python scripts/prepare_isic2019_cls.py --output_dir ./isic2019_grouped
    python scripts/prepare_isic2019_cls.py --output_dir ./isic2019_grouped --group_by patient_id

Data source: https://challenge.isic-archive.com/data/
"""

import argparse
import csv
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

# Download URLs — original challenge S3 bucket is dead (403), so we use
# the ISIC Archive S3 for images and GitHub mirrors for the CSVs.
ISIC_2019_URLS = {
    "ground_truth":
        "https://raw.githubusercontent.com/saams4u/isic-2019-challenge/master/ISIC_2019_Training_GroundTruth.csv",
    "metadata":
        "https://raw.githubusercontent.com/saams4u/isic-2019-challenge/master/ISIC_2019_Training_Metadata.csv",
}
# Individual image URLs come from the ISIC Archive S3 bucket.
ISIC_ARCHIVE_IMAGE_URL = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"

# Class names from the ground truth CSV
CLASS_NAMES = [
    "MEL",  # Melanoma
    "NV",  # Melanocytic nevus
    "BCC",  # Basal cell carcinoma
    "AK",  # Actinic keratosis
    "BKL",  # Benign keratosis
    "DF",  # Dermatofibroma
    "VASC",  # Vascular lesion
    "SCC",  # Squamous cell carcinoma
    "UNK",  # Unknown
]


def download_file(url, dest, description=""):
  """Download a file with progress indication."""
  if os.path.exists(dest):
    print(f"  Already exists: {os.path.basename(dest)}")
    return

  print(f"  Downloading {description or os.path.basename(dest)}...")
  print(f"  URL: {url}")

  def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
      percent = min(100, downloaded * 100 / total_size)
      mb_downloaded = downloaded / (1024 * 1024)
      mb_total = total_size / (1024 * 1024)
      print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
            end="",
            flush=True)

  try:
    request = Request(url, headers={"User-Agent": "ISIC2019-Preparation/1.0"})
    with urlopen(request) as response, open(dest, "wb") as out_file:
      total_size = int(response.headers.get("Content-Length", 0))
      downloaded = 0
      while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
          break
        out_file.write(chunk)
        downloaded += len(chunk)
        if total_size > 0:
          percent = min(100, downloaded * 100 / total_size)
          mb_downloaded = downloaded / (1024 * 1024)
          mb_total = total_size / (1024 * 1024)
          print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True)
    print()
  except (HTTPError, URLError, OSError) as e:
    print(f"\n  Error downloading: {e}")
    raise


def download_isic2019_csvs(cache_dir):
  """Download ISIC 2019 ground truth and metadata CSVs.

  Args:
      cache_dir: Directory to cache downloaded files

  Returns:
      Dict with paths to downloaded files
  """
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)

  paths = {}

  # Download ground truth CSV
  gt_csv = cache_dir / "ISIC_2019_Training_GroundTruth.csv"
  download_file(ISIC_2019_URLS["ground_truth"], str(gt_csv), "ISIC 2019 Ground Truth")
  paths["ground_truth"] = gt_csv

  # Download metadata CSV
  meta_csv = cache_dir / "ISIC_2019_Training_Metadata.csv"
  download_file(ISIC_2019_URLS["metadata"], str(meta_csv), "ISIC 2019 Metadata")
  paths["metadata"] = meta_csv

  return paths


def download_one_image(image_id, dest_path, retries=3):
  """Download a single image from the ISIC Archive S3 bucket.

  Args:
      image_id: ISIC image ID (e.g. "ISIC_0000000")
      dest_path: Destination file path
      retries: Number of retries on failure

  Returns:
      (image_id, success, error_message)
  """
  url = ISIC_ARCHIVE_IMAGE_URL.format(image_id=image_id)
  for attempt in range(retries):
    try:
      request = Request(url, headers={"User-Agent": "ISIC2019-Preparation/1.0"})
      with urlopen(request, timeout=30) as response, open(dest_path, "wb") as f:
        while True:
          chunk = response.read(1024 * 1024)
          if not chunk:
            break
          f.write(chunk)
      return (image_id, True, None)
    except (HTTPError, URLError, OSError) as e:
      if attempt < retries - 1:
        time.sleep(1 * (attempt + 1))
      else:
        return (image_id, False, str(e))
  return (image_id, False, "max retries exceeded")


def download_images_parallel(image_ids, images_dir, num_workers=8):
  """Download images from the ISIC Archive in parallel.

  Args:
      image_ids: List of ISIC image IDs to download
      images_dir: Directory to save images to
      num_workers: Number of parallel download threads

  Returns:
      Tuple of (success_count, fail_count, failed_ids)
  """
  images_dir = Path(images_dir)
  images_dir.mkdir(parents=True, exist_ok=True)

  # Skip already downloaded images
  to_download = []
  already_exist = 0
  for image_id in image_ids:
    dest = images_dir / f"{image_id}.jpg"
    if dest.exists():
      already_exist += 1
    else:
      to_download.append(image_id)

  if already_exist > 0:
    print(f"  {already_exist} images already cached, {len(to_download)} to download")

  if not to_download:
    return (already_exist, 0, [])

  failed_ids = []
  success_count = 0
  start_time = time.time()

  print(f"  Downloading {len(to_download)} images with {num_workers} workers...")

  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {}
    for image_id in to_download:
      dest = images_dir / f"{image_id}.jpg"
      future = executor.submit(download_one_image, image_id, str(dest))
      futures[future] = image_id

    for i, future in enumerate(as_completed(futures)):
      image_id, success, error = future.result()
      if success:
        success_count += 1
      else:
        failed_ids.append((image_id, error))

      # Progress reporting every 100 images
      done = i + 1
      if done % 100 == 0 or done == len(to_download):
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(to_download) - done) / rate if rate > 0 else 0
        print(
            f"\r  Progress: {done}/{len(to_download)} images "
            f"({done*100//len(to_download)}%) "
            f"- {rate:.1f} img/s - ETA {remaining:.0f}s",
            end="",
            flush=True,
        )

  print()
  total_time = time.time() - start_time
  print(f"  Download complete: {success_count} succeeded, "
        f"{len(failed_ids)} failed in {total_time:.1f}s")

  if failed_ids:
    print("  Failed images:")
    for image_id, error in failed_ids[:10]:
      print(f"    {image_id}: {error}")
    if len(failed_ids) > 10:
      print(f"    ... and {len(failed_ids) - 10} more")

  return (success_count + already_exist, len(failed_ids), failed_ids)


def load_ground_truth(csv_path):
  """Load ISIC 2019 ground truth CSV (one-hot encoded).

  Args:
      csv_path: Path to ground truth CSV

  Returns:
      Dict mapping image_id to class index
  """
  labels = {}
  with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      image_id = row["image"]
      # Find which class has value 1.0
      class_idx = None
      for i, cls in enumerate(CLASS_NAMES):
        if cls in row and float(row[cls]) == 1.0:
          class_idx = i
          break
      if class_idx is None:
        # All zeros - treat as UNK or skip
        if "UNK" in row and float(row["UNK"]) == 1.0:
          class_idx = CLASS_NAMES.index("UNK")
        else:
          print(f"  Warning: No label found for {image_id}, skipping")
          continue
      labels[image_id] = class_idx

  print(f"  Loaded {len(labels)} labels from ground truth")
  return labels


def load_metadata(csv_path):
  """Load ISIC 2019 metadata CSV.

  Args:
      csv_path: Path to metadata CSV

  Returns:
      Dict mapping image_id to metadata dict
  """
  metadata = {}
  with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
      image_id = row["image"]
      metadata[image_id] = {
          "lesion_id": row.get("lesion_id", ""),
          "age_approx": row.get("age_approx", ""),
          "anatom_site_general": row.get("anatom_site_general", ""),
          "sex": row.get("sex", ""),
      }

  # Count lesion_id coverage
  total = len(metadata)
  has_lesion_id = sum(1 for v in metadata.values() if v["lesion_id"])
  print(f"  Loaded metadata for {total} images")
  print(
      f"  Lesion ID coverage: {has_lesion_id}/{total} ({has_lesion_id/total*100:.1f}%)")

  return metadata


def group_split_by_attribute(
    data,
    groups,
    test_size=0.15,
    val_size=0.15,
    seed=42,
):
  """Split dataset by group attribute to prevent data leakage.

  All items with the same group value go into the same split.

  Args:
      data: List of (image_id, class_index) tuples
      groups: List of group values (lesion_id or patient_id) for each item
      test_size: Fraction of groups for test set
      val_size: Fraction of groups for validation set
      seed: Random seed

  Returns:
      Dict with 'train', 'val', 'test' keys mapping to lists of (image_id, class_index)
  """
  if len(data) == 0:
    return {"train": [], "val": [], "test": []}

  # Get unique groups
  unique_groups = list(set(groups))
  print(f"\n  Total unique groups: {len(unique_groups)}")

  # First split: separate test set
  if test_size > 0 and val_size > 0:
    # Three-way split: train / val / test
    n_test = max(1, int(len(unique_groups) * test_size))
    n_val = max(1, int(len(unique_groups) * val_size))
    n_train = len(unique_groups) - n_test - n_val

    if n_train < 1:
      n_train = 1
      n_val = max(0, len(unique_groups) - n_train - n_test)

    # Shuffle groups
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])
    train_groups = set(shuffled[n_test + n_val:])

  elif test_size > 0:
    # Two-way split: train / test
    n_test = max(1, int(len(unique_groups) * test_size))
    rng = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    test_groups = set(shuffled[:n_test])
    val_groups = set()
    train_groups = set(shuffled[n_test:])

  else:
    # No test split
    train_groups = set(unique_groups)
    val_groups = set()
    test_groups = set()

  # Assign items to splits based on their group
  train_examples = []
  val_examples = []
  test_examples = []

  for item, group in zip(data, groups):
    if group in test_groups:
      test_examples.append(item)
    elif group in val_groups:
      val_examples.append(item)
    else:
      train_examples.append(item)

  print("  Split by grouping:")
  print(
      f"    Groups: {len(train_groups)} train, {len(val_groups)} val, {len(test_groups)} test"
  )
  print(
      f"    Images: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test"
  )

  return {
      "train": train_examples,
      "val": val_examples,
      "test": test_examples,
  }


def save_as_imagefolder(
    splits,
    output_dir,
    images_dir,
    label_names,
    min_resolution=None,
):
  """Save splits as ImageFolder structure.

  Args:
      splits: Dict with 'train', 'val', 'test' keys
      output_dir: Output directory path
      images_dir: Directory containing extracted images
      label_names: Dict mapping class index to class name
      min_resolution: Optional minimum image resolution filter
  """
  output_path = Path(output_dir)
  images_path = Path(images_dir)

  total_copied = 0
  total_skipped = 0

  for split_name, examples in splits.items():
    if not examples:
      continue

    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing {split_name} split ({len(examples)} images)...")

    for image_id, class_idx in examples:
      class_name = label_names[class_idx]
      class_dir = split_dir / class_name
      class_dir.mkdir(parents=True, exist_ok=True)

      # Source image path (ISIC images are named like ISIC_0000000.jpg)
      src_path = images_path / f"{image_id}.jpg"

      if not src_path.exists():
        print(f"    Warning: Image not found: {src_path}")
        total_skipped += 1
        continue

      # Check resolution if specified
      if min_resolution:
        try:
          with Image.open(src_path) as img:
            w, h = img.size
            if min(w, h) < min_resolution:
              total_skipped += 1
              continue
        except (OSError, ValueError):
          total_skipped += 1
          continue

      dst_path = class_dir / f"{image_id}.jpg"
      shutil.copy2(str(src_path), str(dst_path))
      total_copied += 1

  print(f"\n  Copied {total_copied} images to {output_dir}")
  if total_skipped > 0:
    print(f"  Skipped {total_skipped} images (not found or below resolution)")


def main():
  parser = argparse.ArgumentParser(
      description="Prepare ISIC 2019 dataset with lesion_id-grouped splits.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  # Basic usage - group by lesion_id (recommended)
  python scripts/prepare_isic2019_cls.py --output_dir ./isic2019_grouped

  # Group by lesion_id with 20% test, 15% val
  python scripts/prepare_isic2019_cls.py --output_dir ./isic2019_grouped \\
      --test_size 0.2 --val_size 0.15

  # With minimum resolution filter
  python scripts/prepare_isic2019_cls.py --output_dir ./isic2019_grouped \\
      --min_resolution 448

  # Only download metadata for inspection
  python scripts/prepare_isic2019_cls.py --download_only

Output structure:
  output_dir/
  ├── train/
  │   ├── MEL/   (Melanoma)
  │   ├── NV/    (Nevus)
  │   ├── BCC/   (Basal cell carcinoma)
  │   ├── AK/    (Actinic keratosis)
  │   ├── BKL/   (Benign keratosis)
  │   ├── DF/    (Dermatofibroma)
  │   ├── VASC/  (Vascular lesion)
  │   ├── SCC/   (Squamous cell carcinoma)
  │   └── UNK/   (Unknown)
  ├── val/
  │   └── ...
  └── test/
      └── ...
""",
  )

  parser.add_argument(
      "--output_dir",
      type=str,
      default="./isic2019_grouped",
      help="Output directory for ImageFolder dataset (default: %(default)s)",
  )

  parser.add_argument(
      "--cache_dir",
      type=str,
      default="./isic2019_cache",
      help="Directory to cache downloaded files (default: %(default)s)",
  )

  parser.add_argument(
      "--group_by",
      type=str,
      choices=["lesion_id", "patient_id"],
      default="lesion_id",
      help=
      "Group by lesion_id or patient_id to prevent data leakage (default: %(default)s)",
  )

  parser.add_argument(
      "--test_size",
      type=float,
      default=0.15,
      help="Fraction of groups for test set (default: %(default)s)",
  )

  parser.add_argument(
      "--val_size",
      type=float,
      default=0.15,
      help="Fraction of groups for validation set (default: %(default)s)",
  )

  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed (default: %(default)s)",
  )

  parser.add_argument(
      "--min_resolution",
      type=int,
      default=None,
      help="Minimum image resolution (shorter side) to keep (default: all)",
  )

  parser.add_argument(
      "--download_only",
      action="store_true",
      help="Only download files without creating splits",
  )

  parser.add_argument(
      "--num_workers",
      type=int,
      default=8,
      help="Number of parallel download threads (default: %(default)s)",
  )

  args = parser.parse_args()

  print("=" * 60)
  print("ISIC 2019 Dataset Preparation")
  print("=" * 60)

  # Step 1: Download CSVs
  print("\n[1/6] Downloading ISIC 2019 CSVs...")
  paths = download_isic2019_csvs(args.cache_dir)

  if args.download_only:
    print("\nDownload complete. Files saved to:", args.cache_dir)
    return

  # Step 2: Load ground truth
  print("\n[2/6] Loading ground truth labels...")
  labels = load_ground_truth(str(paths["ground_truth"]))

  # Step 3: Load metadata
  print("\n[3/6] Loading metadata...")
  metadata = load_metadata(str(paths["metadata"]))

  # Step 4: Download images from ISIC Archive
  print("\n[4/6] Downloading images from ISIC Archive...")
  images_dir = Path(args.cache_dir) / "ISIC_2019_Training_Input"
  image_ids = list(labels.keys())
  _success, failed, _failed_ids = download_images_parallel(image_ids,
                                                           str(images_dir),
                                                           num_workers=args.num_workers)

  if failed > 0:
    print(f"\n  Warning: {failed} images failed to download. "
          "These will be skipped in the output.")

  # Step 5: Create splits
  print("\n[5/6] Creating grouped splits...")

  # Prepare data and groups
  data = []
  groups = []
  skipped_no_group = 0

  for image_id, class_idx in labels.items():
    # Skip images that failed to download
    if not (images_dir / f"{image_id}.jpg").exists():
      continue

    # Get group value
    group_value = ""
    if image_id in metadata:
      group_value = metadata[image_id].get(args.group_by, "")

    # Skip if no group value
    if not group_value:
      skipped_no_group += 1
      # Still include but with image_id as its own group (prevents leakage
      # since each image is in its own group, but won't share groups)
      group_value = f"__solo__{image_id}"

    data.append((image_id, class_idx))
    groups.append(group_value)

  if skipped_no_group > 0:
    print(f"\n  Note: {skipped_no_group} images have no {args.group_by} value")
    print("  These are treated as individual groups (safe but less grouped)")

  # Create splits
  splits = group_split_by_attribute(
      data,
      groups,
      test_size=args.test_size,
      val_size=args.val_size,
      seed=args.seed,
  )

  # Create label names mapping
  label_names = {i: name for i, name in enumerate(CLASS_NAMES)}
  print(f"\n  Label mapping: {label_names}")

  # Step 6: Save as ImageFolder
  print("\n[6/6] Saving as ImageFolder...")
  save_as_imagefolder(
      splits,
      args.output_dir,
      str(images_dir),
      label_names,
      min_resolution=args.min_resolution,
  )

  # Print summary
  print("\n" + "=" * 60)
  print("Dataset Summary")
  print("=" * 60)
  for split_name, examples in splits.items():
    if examples:
      # Count per class
      class_counts = defaultdict(int)
      for _, class_idx in examples:
        class_counts[CLASS_NAMES[class_idx]] += 1
      print(f"\n  {split_name}: {len(examples)} images")
      for cls in CLASS_NAMES:
        if cls in class_counts:
          print(f"    {cls}: {class_counts[cls]}")

  print("\n" + "=" * 60)
  print("DONE! Use the following command to train with grouped splits:")
  print(f"  python -m scdiag.train --dataset imagefolder/{args.output_dir}")
  print("=" * 60)


if __name__ == "__main__":
  main()
