"""Download all image URLs from a metadata CSV into a local folder with retries and logging.

Usage:
  python examples/download_images.py --metadata-csv "path\to\AI_Classification-Project.csv" --images-dir "data\images" --workers 8

This script will:
- read the CSV
- attempt to download each image_url into images-dir using the image_id as filename
- retry transient failures
- produce downloads_failed.csv listing rows that couldn't be downloaded
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import requests


def download_one(row: Dict[str, str], images_dir: Path, timeout: int = 30, retries: int = 3) -> tuple:
    url = row.get("image_url")
    image_id = str(row.get("image_id"))
    if not url or not url.startswith(("http://", "https://")):
        return (image_id, False, "invalid_url")
    out_path = images_dir / f"{image_id}.jpg"
    if out_path.exists():
        return (image_id, True, "exists")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            return (image_id, True, "ok")
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * attempt)
    return (image_id, False, repr(last_exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.metadata_csv, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append(r)

    failed = []
    total = len(rows)
    print(f"Downloading {total} images to {images_dir} using {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, r, images_dir): r for r in rows}
        for fut in as_completed(futures):
            image_id, ok, reason = fut.result()
            if not ok:
                failed.append((image_id, reason))

    print(f"Done. {len(failed)} failed downloads.")
    failed_csv = images_dir.parent / "downloads_failed.csv"
    with open(failed_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image_id", "error"])
        writer.writerows(failed)
    print(f"Wrote failures to {failed_csv}")


if __name__ == "__main__":
    main()
