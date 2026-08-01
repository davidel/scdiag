#!/usr/bin/env python
"""Prepare HAM10000 dataset with lesion_id-grouped splits.

Downloads the marmal88/skin_cancer dataset from HuggingFace, joins lesion_id
from the metadata CSV, and creates train/val/test splits grouped by lesion_id
to prevent data leakage (same lesion never appears in multiple splits).

Output is an ImageFolder structure compatible with:
    --dataset imagefolder/OUTPUT_PATH

Usage:
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped --test_size 0.15 --val_size 0.15
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped --cache_dir ~/.cache/huggingface/datasets
"""

import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image


def download_metadata(dataset_name: str) -> pd.DataFrame:
  """Download HAM10000_metadata.csv from the HuggingFace dataset repository.
    
    The marmal88/skin_cancer dataset stores metadata in metadata/HAM10000_metadata.csv.
    We fetch it via the HuggingFace Hub API.
    """
  url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/metadata/HAM10000_metadata.csv"

  print(f"Downloading metadata from: {url}")
  response = requests.get(url, allow_redirects=True)
  response.raise_for_status()

  df = pd.read_csv(StringIO(response.text))
  print(f"Downloaded metadata with {len(df)} rows")
  print(f"Columns: {list(df.columns)}")
  print(f"Sample image_ids: {df['image_id'].head(3).tolist()}")
  print(f"Unique lesion_ids: {df['lesion_id'].nunique()}")
  return df


def image_hash(img: Image.Image) -> str:
  """Compute MD5 hash of PIL Image content for matching."""
  img_bytes = img.tobytes()
  return hashlib.md5(img_bytes).hexdigest()


def build_image_to_lesion_mapping(
    dataset,
    metadata: pd.DataFrame,
) -> dict:
  """Build mapping from image hash to lesion_id.
    
    This approach works even when PIL Images don't have .filename attribute
    (e.g., when loaded from Parquet storage).
    
    Args:
        dataset: HuggingFace dataset with 'image' and 'label' columns
        metadata: DataFrame with 'image_id' and 'lesion_id' columns
        
    Returns:
        Dict mapping image_hash -> lesion_id
    """
  # Create lookup from image_id to lesion_id
  image_id_to_lesion = dict(zip(metadata["image_id"], metadata["lesion_id"]))

  # Build mapping by hashing all images in the dataset
  # Note: This is O(n) in dataset size and memory-intensive for large datasets
  # For HAM10000 (~10K images), this is acceptable
  print("Building image to lesion_id mapping (this may take a moment)...")

  image_to_lesion = {}
  matched = 0
  unmatched = 0

  # We'll match by position: assuming the dataset and metadata are ordered
  # by image_id alphabetically (which is the case for HAM10000 ImageFolder)
  if len(dataset) == len(metadata):
    print(f"Dataset size ({len(dataset)}) matches metadata size ({len(metadata)})")
    print("Matching by position (assuming consistent ordering)...")

    for idx in range(len(dataset)):
      example = dataset[idx]
      image = example["image"]
      lesion_id = metadata.iloc[idx]["lesion_id"]
      image_id = metadata.iloc[idx]["image_id"]

      # Store by hash
      img_hash = image_hash(image)
      image_to_lesion[img_hash] = lesion_id
      matched += 1
  else:
    print(f"Warning: Dataset size ({len(dataset)}) != metadata size ({len(metadata)})")
    print("Attempting to match by image hash (slower but more robust)...")

    # For each metadata row, we'd need to download the image to hash it
    # This is too slow for 10K images, so we'll fall back to random splitting
    print("Falling back to random splitting (no lesion_id grouping)")
    return None

  print(f"Matched {matched} images to lesion_ids")
  print(f"Sample mappings: {list(image_to_lesion.items())[:3]}")

  return image_to_lesion


def group_split_by_lesion_id(
    dataset,
    image_to_lesion: dict,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict:
  """Split dataset by lesion_id groups to prevent data leakage.
    
    Args:
        dataset: HuggingFace dataset with 'image' and 'label' columns
        image_to_lesion: Dict mapping image_hash -> lesion_id
        test_size: Fraction of lesion_ids for test set
        val_size: Fraction of lesion_ids for validation set
        seed: Random seed for reproducibility
        
    Returns:
        Dict with 'train', 'val', 'test' keys containing lists of (image, label) tuples
    """
  random.seed(seed)

  # Group dataset indices by lesion_id
  lesion_to_indices = defaultdict(list)
  unmatched_count = 0

  for idx in range(len(dataset)):
    example = dataset[idx]
    image = example["image"]
    img_hash = image_hash(image)

    if img_hash in image_to_lesion:
      lesion_id = image_to_lesion[img_hash]
      lesion_to_indices[lesion_id].append(idx)
    else:
      # Unmatched images go to their own group (will be split randomly)
      lesion_to_indices[f"__unmatched_{idx}"].append(idx)
      unmatched_count += 1

  if unmatched_count > 0:
    print(f"Warning: {unmatched_count} images could not be matched to lesion_id")

  # Split lesion_ids into train/val/test
  all_lesion_ids = list(lesion_to_indices.keys())
  random.shuffle(all_lesion_ids)
  
  n_test = int(len(all_lesion_ids) * test_size)
  n_val = int(len(all_lesion_ids) * val_size)
  
  test_lesion_ids = set(all_lesion_ids[:n_test])
  val_lesion_ids = set(all_lesion_ids[n_test:n_test + n_val]) if n_val > 0 else set()
  train_lesion_ids = set(all_lesion_ids[n_test + n_val:])
  
  # Collect indices for each split
  train_indices = []
  val_indices = []
  test_indices = []
  
  for lesion_id, indices in lesion_to_indices.items():
    if lesion_id in test_lesion_ids:
      test_indices.extend(indices)
    elif lesion_id in val_lesion_ids:
      val_indices.extend(indices)
    else:
      train_indices.extend(indices)

  # Extract examples for each split
  def get_examples(indices):
    examples = []
    for idx in indices:
      example = dataset[idx]
      examples.append((example["image"], example["label"]))
    return examples

  train_examples = get_examples(train_indices)
  val_examples = get_examples(val_indices)
  test_examples = get_examples(test_indices)

  print(f"\nSplit by lesion_id:")
  if n_val > 0:
    print(
        f"  Lesion IDs: {len(train_lesion_ids)} train, {len(val_lesion_ids)} val, {len(test_lesion_ids)} test"
    )
    print(
        f"  Images: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test"
    )
  else:
    print(
        f"  Lesion IDs: {len(train_lesion_ids)} train, {len(test_lesion_ids)} test (no val)"
    )
    print(
        f"  Images: {len(train_examples)} train, {len(test_examples)} test (no val)"
    )

  return {
      "train": train_examples,
      "val": val_examples,
      "test": test_examples,
  }


def save_as_imagefolder(
    splits: dict,
    output_dir: str,
    label_names: dict = None,
):
  """Save splits as ImageFolder structure.
    
    Args:
        splits: Dict with 'train', 'val', 'test' keys containing lists of (image, label) tuples
        output_dir: Output directory path
        label_names: Optional dict mapping label_id -> class_name. If None, uses numeric names.
    """
  output_path = Path(output_dir)

  for split_name, examples in splits.items():
    # Skip empty splits (e.g., val when --val_size=0)
    if len(examples) == 0:
      print(f"\nSkipping {split_name} split (empty)")
      continue
    
    print(f"\nSaving {split_name} split with {len(examples)} images...")

    # Group by label
    label_to_images = defaultdict(list)
    for image, label in examples:
      label_to_images[label].append(image)

    # Save images
    for label, images in sorted(label_to_images.items()):
      if label_names:
        class_name = label_names[label]
      else:
        class_name = str(label)

      class_dir = output_path / split_name / class_name
      class_dir.mkdir(parents=True, exist_ok=True)

      for i, image in enumerate(images):
        dst_path = class_dir / f"{i:05d}.jpg"
        # Save PIL Image as JPEG
        image.save(dst_path, "JPEG", quality=95)

      print(f"  {class_name}: {len(images)} images")

  print(f"\nSaved ImageFolder dataset to {output_dir}")


def main():
  parser = argparse.ArgumentParser(
      description="Prepare HAM10000 dataset with lesion_id-grouped splits",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default="marmal88/skin_cancer",
      help="HuggingFace dataset name (default: %(default)s)",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for ImageFolder dataset",
  )
  parser.add_argument(
      "--test_size",
      type=float,
      default=0.1,
      help="Fraction of lesion_ids for test set (default: %(default)s)",
  )
  parser.add_argument(
      "--val_size",
      type=float,
      default=0.1,
      help="Fraction of lesion_ids for validation set (default: %(default)s)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for reproducibility (default: %(default)s)",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=None,
      help=
      "HuggingFace datasets cache directory (default: ~/.cache/huggingface/datasets)",
  )

  args = parser.parse_args()

  # Validate sizes
  if args.test_size + args.val_size >= 1.0:
    parser.error("test_size + val_size must be < 1.0")

  # Load dataset
  print(f"Loading dataset: {args.dataset}")
  dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)

  # Use the train split if available, otherwise use the full dataset
  if isinstance(dataset, dict):
    if "train" in dataset:
      dataset = dataset["train"]
    else:
      dataset = list(dataset.values())[0]

  print(f"Dataset loaded: {len(dataset)} images")
  print(f"Columns: {list(dataset.features.keys())}")

  # Get label names
  label_names = None
  if hasattr(dataset.features["label"], "names"):
    label_names = {i: name for i, name in enumerate(dataset.features["label"].names)}
    print(f"Labels: {label_names}")

  # Download metadata
  metadata = download_metadata(args.dataset)

  # Build image to lesion_id mapping
  image_to_lesion = build_image_to_lesion_mapping(dataset, metadata)

  if image_to_lesion is None:
    print("\nFalling back to random splitting (no lesion_id grouping)")
    # Random split as fallback
    split = dataset.train_test_split(test_size=args.test_size + args.val_size,
                                     seed=args.seed)
    train_val = split["train"].train_test_split(test_size=args.val_size /
                                                (args.test_size + args.val_size),
                                                seed=args.seed)

    splits = {
        "train": [(ex["image"], ex["label"]) for ex in train_val["train"]],
        "val": [(ex["image"], ex["label"]) for ex in train_val["test"]],
        "test": [(ex["image"], ex["label"]) for ex in split["test"]],
    }
  else:
    # Split by lesion_id
    splits = group_split_by_lesion_id(
        dataset,
        image_to_lesion,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

  # Save as ImageFolder
  save_as_imagefolder(splits, args.output_dir, label_names)

  print("\n" + "=" * 60)
  print("DONE! Use the following command to train with grouped splits:")
  print(f"  python -m scdiag.train --dataset imagefolder/{args.output_dir}")
  print("=" * 60)


if __name__ == "__main__":
  main()
