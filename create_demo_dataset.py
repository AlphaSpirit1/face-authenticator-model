from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_face_image(size: int, label: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    center = size // 2
    skin_color = (220, 190, 160) if label == 1 else (200, 180, 170)
    face_color = tuple(np.clip(np.array(skin_color) + rng.normal(0, 20, 3), 0, 255).astype(int))

    face_box = (size // 5, size // 7, 4 * size // 5, 6 * size // 7)
    draw.ellipse(face_box, fill=face_color)

    if label == 0:
        for _ in range(8):
            x = int(rng.integers(0, size))
            y = int(rng.integers(0, size))
            draw.ellipse((x, y, x + 12, y + 12), fill=(80, 80, 80))

    eye_y = center - 10
    draw.ellipse((center - 30, eye_y - 8, center - 10, eye_y + 8), fill=(0, 0, 0))
    draw.ellipse((center + 10, eye_y - 8, center + 30, eye_y + 8), fill=(0, 0, 0))
    draw.ellipse((center - 6, center - 10, center + 6, center + 10), fill=(160, 120, 100))
    draw.arc((center - 25, center + 18, center + 25, center + 42), start=200, end=340, fill=(180, 40, 40), width=4)

    if label == 0:
        for _ in range(25):
            x = int(rng.integers(0, size))
            y = int(rng.integers(0, size))
            draw.point((x, y), fill=(255, 0, 0))

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic demo dataset for face authenticity classification.")
    parser.add_argument("--output-dir", type=str, default="examples/demo_dataset")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng(42)

    for index in range(args.n_samples):
        label = 1 if index % 2 == 0 else 0
        seed = int(rng.integers(0, 1_000_000))
        image = generate_face_image(128, label, seed)
        image_name = f"face_{index:04d}.png"
        image_path = images_dir / image_name
        image.save(image_path)

        gender = "Male" if index % 2 == 0 else "Female"
        age_group = ["18-25", "25-34", "35-44", "45-54"][index % 4]
        quality = round(float(rng.uniform(0.7, 0.99)), 3)
        confidence = round(float(rng.uniform(0.6, 0.95)), 3)
        difficulty = round(float(rng.uniform(0.2, 0.8)), 3)
        dataset_split = "train" if index % 3 != 0 else "val" if index % 5 == 0 else "test"

        rows.append(
            {
                "image_id": f"face_{index:04d}",
                "image_path": str(image_path),
                "label": "REAL" if label == 1 else "FAKE",
                "gender": gender,
                "age_group": age_group,
                "quality_score": quality,
                "confidence": confidence,
                "difficulty": difficulty,
                "dataset_split": dataset_split,
            }
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["image_id", "image_path", "label", "gender", "age_group", "quality_score", "confidence", "difficulty", "dataset_split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {args.n_samples} demo samples at {output_dir}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
