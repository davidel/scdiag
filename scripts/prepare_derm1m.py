#!/usr/bin/env python
"""Prepare Derm1M dataset for pretraining by extracting images from zip archives.

The redlessone/Derm1M HuggingFace dataset contains CSV metadata with filenames
that reference images inside zip archives (IIYI.zip, ISIC.zip, etc.). This
script downloads and extracts those archives, then copies the images into a
flat ImageFolder directory compatible with scdiag-pretrain.

Usage:
    python scripts/prepare_derm1m.py --output_dir ./derm1m_images

    Then use with pretraining:
    scdiag-pretrain --datasets ./derm1m_images --image_size 448 ...
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

# Derm1M zip archives hosted on HuggingFace.
DERM1M_ZIPS = [
    "IIYI.zip",
    "edu.zip",
    "note.zip",
    "public.zip",
    "pubmed.zip",
    "reddit.zip",
    "twitter.zip",
    "validation_data.zip",
    "youtube.zip",
]

BASE_URL = "https://huggingface.co/datasets/redlessone/Derm1M/resolve/main"


def download_file(url, dest_path, token=None):
    """Download a file with progress reporting."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"  Downloading {os.path.basename(dest_path)} ...")
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (100 * 1024 * 1024) < 1024 * 1024:
                pct = (downloaded / total) * 100
                print(f"    {pct:5.1f}%  ({downloaded / 1024**3:.2f} GB / "
                      f"{total / 1024**3:.2f} GB)")

    print(f"    Done  ({downloaded / 1024**3:.2f} GB)")


def collect_images_from_dir(src_dir):
    """Recursively find all image files under *src_dir*."""
    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
    images = []
    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _EXTS:
            images.append(path)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Derm1M dataset for pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for extracted images",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip downloading zips (use if already extracted elsewhere)",
    )
    parser.add_argument(
        "--zip_dir", type=str, default=None,
        help="Directory containing pre-downloaded zip files (with --skip_download)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.skip_download and args.zip_dir:
        # Use pre-downloaded zips
        zip_dir = Path(args.zip_dir)
        print(f"Using pre-downloaded zips from {zip_dir}")
    else:
        # Download and extract into a temp directory
        zip_dir = Path(tempfile.mkdtemp(prefix="derm1m_"))
        print(f"Downloading Derm1M zip archives to {zip_dir} ...")

        for zip_name in DERM1M_ZIPS:
            url = f"{BASE_URL}/{zip_name}"
            zip_path = zip_dir / zip_name
            download_file(url, zip_path, token)

            print(f"  Extracting {zip_name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_dir)
            print(f"    Done")

    # Walk the extracted directories and copy images into the output folder.
    # The zip files extract into sub-directories (IIYI/, ISIC/, youtube/ etc.)
    # with image files inside them.  We copy everything into a flat output
    # directory preserving relative structure.
    image_count = 0
    for root, _dirs, files in os.walk(zip_dir):
        root_path = Path(root)
        for fname in sorted(files):
            fpath = root_path / fname
            if fpath.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
            }:
                continue
            # Compute relative path from zip_dir to preserve subdirectory structure
            rel = fpath.relative_to(zip_dir)
            dest = output_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest)
            image_count += 1
            if image_count % 10000 == 0:
                print(f"  Copied {image_count:,} images ...")

    print(f"\nTotal images copied: {image_count:,}")

    # Clean up temp dir if we downloaded
    if not (args.skip_download and args.zip_dir):
        shutil.rmtree(zip_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("DONE! Use the following command to pretrain:")
    print(f"  scdiag-pretrain --datasets {args.output_dir} --image_size 448 ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
