#!/usr/bin/env python
"""Prepare ISIC archive images for pretraining.

Downloads images from the ISIC (International Skin Imaging Collaboration)
archive using the isic-cli tool, and stores them in a target folder
compatible with scdiag-pretrain.

Usage:
    python scripts/prepare_isic.py --output_dir ./isic_images

    # Download with a specific query (e.g., only melanoma cases)
    python scripts/prepare_isic.py --output_dir ./isic_melanoma \
        --query 'diagnosis:"melanoma"'

    # Download the SIIM-ISIC 2020 Challenge subset
    python scripts/prepare_isic.py --output_dir ./isic_2020 \
        --query "anonymous:true"

    # Download the entire archive (may take several days)
    python scripts/prepare_isic.py --output_dir ./isic_full

Requirements:
    pip install isic-cli

Note:
    The ISIC archive contains dermatoscopic images of skin lesions.
    A full download can be very large (hundreds of GB) and may take
    several days. It is recommended to start with a specific query
    to benchmark your ConvViT pipeline first.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def download_isic_archive(target_dir, search_query=None, verbose=False, max_retries=3):
  """Invoke the isic-cli tool via Python to download images and metadata.

  Args:
      target_dir: Relative or absolute path where images will be saved.
      search_query: Lucene query string to filter downloads.
                    Leave None to attempt a full archive download.
      verbose: If True, print detailed progress information.
      max_retries: Number of retry attempts for failed downloads.
  """
  target_path = Path(target_dir)
  target_path.mkdir(parents=True, exist_ok=True)

  # Build the isic command
  cmd = [
      sys.executable, "-m", "isic", "archive", "download", "--output",
      str(target_path)
  ]

  if search_query:
    cmd.extend(["--query", search_query])

  if verbose:
    print(f"Running command: {' '.join(cmd)}")

  for attempt in range(1, max_retries + 1):
    try:
      if verbose:
        print(f"\nAttempt {attempt}/{max_retries}")

      # Run the command and stream output
      process = subprocess.run(
          cmd,
          check=True,
          capture_output=not verbose,
          text=True,
      )

      if verbose and process.stdout:
        print(process.stdout)

      print(f"Successfully downloaded images to {target_path}")
      return

    except subprocess.CalledProcessError as e:
      print(f"Attempt {attempt} failed with return code {e.returncode}")
      if e.stderr and not verbose:
        print(f"Error output: {e.stderr}")
      if attempt == max_retries:
        raise RuntimeError(f"Failed to download after {max_retries} attempts") from e
      print("Retrying in 5 seconds...")
      time.sleep(5)


def main():
  parser = argparse.ArgumentParser(
      description="Prepare ISIC archive images for pretraining",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for downloaded images",
  )
  parser.add_argument(
      "--query",
      type=str,
      default=None,
      help="Lucene query string to filter downloads "
      "(e.g., 'diagnosis:\"melanoma\"' or 'anonymous:true'). "
      "Leave empty to download the entire archive.",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print detailed progress information",
  )
  parser.add_argument(
      "--max_retries",
      type=int,
      default=3,
      help="Number of retry attempts for failed downloads (default: 3)",
  )
  parser.add_argument(
      "--check_cli",
      action="store_true",
      help="Check if isic-cli is installed and exit",
  )
  args = parser.parse_args()

  # Check if isic-cli is installed
  if args.check_cli:
    try:
      subprocess.run(
          [sys.executable, "-m", "isic", "--help"],
          check=True,
          capture_output=True,
      )
      print("isic-cli is installed and ready to use.")
      sys.exit(0)
    except (subprocess.CalledProcessError, FileNotFoundError):
      print("isic-cli is not installed. Please install it with:\n"
            "  pip install isic-cli")
      sys.exit(1)

  # Check if output directory already has images
  image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
  output_path = Path(args.output_dir)
  if output_path.exists() and any(
      f.suffix.lower() in image_exts for f in output_path.rglob("*") if f.is_file()):
    print(f"Output directory {output_path} already contains images.")
    print("Delete it and re-run to refresh, or use a different --output_dir.")
    return

  print("=" * 60)
  print("ISIC Archive Downloader")
  print("=" * 60)
  print(f"Output directory: {output_path}")
  if args.query:
    print(f"Search query: {args.query}")
  else:
    print("Search query: None (downloading entire archive)")
  print(f"Max retries: {args.max_retries}")
  print("=" * 60)

  try:
    download_isic_archive(
        target_dir=args.output_dir,
        search_query=args.query,
        verbose=args.verbose,
        max_retries=args.max_retries,
    )

    print("\n" + "=" * 60)
    print("DONE! Use the following command to pretrain:")
    print(f"  scdiag-pretrain --datasets {args.output_dir} --image_size 448 ...")
    print("=" * 60)

  except RuntimeError as e:
    print(f"\nError: {e}", file=sys.stderr)
    sys.exit(1)
  except KeyboardInterrupt:
    print("\nDownload interrupted by user.")
    sys.exit(1)


if __name__ == "__main__":
  main()
