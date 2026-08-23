from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_split_name(value: object) -> str:
    if value is None or pd.isna(value):
        return "train"
    return str(value).strip().lower().replace(" ", "_")


def _resolve_label(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"real", "authentic", "1", "true", "real_image"}:
            return 1.0
        if v in {"fake", "manipulated", "0", "false", "fake_image"}:
            return 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if numeric > 0.5 else 0.0


def _safe_numeric(value: object, default: float = 0.5) -> float:
    if value is None or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_resolution(value: object) -> Tuple[float, float]:
    if value is None or pd.isna(value):
        return (0.5, 0.5)
    text = str(value).strip().lower()
    if "x" in text:
        parts = text.split("x")
        if len(parts) == 2:
            try:
                width = float(parts[0].strip())
                height = float(parts[1].strip())
                return width, height
            except ValueError:
                pass
    return (0.5, 0.5)


def _stable_url_token(url: str) -> str:
    return hashlib.md5(url.strip().encode("utf-8")).hexdigest()[:16]


def _download_image(url: str, target_path: Path) -> bool:
    if not url or not str(url).startswith(("http://", "https://")):
        return False
    if target_path.exists():
        return True
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def _standardize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "label" not in result.columns:
        for candidate in ["class", "classification", "authenticity", "target", "label_numeric"]:
            if candidate in result.columns:
                result["label"] = result[candidate]
                break

    if "label_numeric" in result.columns and "label" not in result.columns:
        result["label"] = result["label_numeric"]

    if "label" in result.columns:
        result["label"] = result["label"].apply(_resolve_label)

    if "gender" in result.columns:
        mapping = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0, "unknown": 0.5, "u": 0.5}
        result["gender_num"] = result["gender"].map(lambda x: mapping.get(str(x).strip().lower(), 0.5))
    else:
        result["gender_num"] = 0.5

    if "age_group" in result.columns:
        age_map = {
            "0-18": 0.1,
            "18-25": 0.3,
            "26-35": 0.5,
            "35-44": 0.7,
            "36-50": 0.8,
            "45-54": 0.9,
            "50+": 1.0,
            "unknown": 0.5,
        }
        result["age_group_num"] = result["age_group"].map(lambda x: age_map.get(str(x).strip().lower(), 0.5))
    else:
        result["age_group_num"] = 0.5

    for col in ["quality_score", "confidence", "difficulty", "image_quality", "detection_difficulty", "confidence_score"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.5)

    if "quality_score" not in result.columns and "image_quality" in result.columns:
        result["quality_score"] = pd.to_numeric(result["image_quality"], errors="coerce").fillna(0.5)
    if "confidence" not in result.columns and "confidence_score" in result.columns:
        result["confidence"] = pd.to_numeric(result["confidence_score"], errors="coerce").fillna(0.5)
    if "confidence" not in result.columns:
        result["confidence"] = 0.5
    if "difficulty" not in result.columns and "detection_difficulty" in result.columns:
        result["difficulty"] = pd.to_numeric(result["detection_difficulty"], errors="coerce").fillna(0.5)
    if "difficulty" not in result.columns:
        result["difficulty"] = 0.5

    if "resolution" in result.columns:
        resolution_width = []
        resolution_height = []
        for value in result["resolution"]:
            w, h = _parse_resolution(value)
            resolution_width.append(w)
            resolution_height.append(h)
        result["resolution_width"] = resolution_width
        result["resolution_height"] = resolution_height
    else:
        result["resolution_width"] = 0.5
        result["resolution_height"] = 0.5

    if "source" in result.columns:
        result["source_num"] = result["source"].map(lambda x: 1.0 if str(x).strip().lower() in {"unsplash", "source"} else 0.5)
    else:
        result["source_num"] = 0.5

    return result


def _resolve_image_path(row: pd.Series, image_dir: Path) -> Path:
    candidates = [
        row.get("image_path"),
        row.get("path"),
        row.get("file_path"),
        row.get("filename"),
        row.get("file_name"),
        row.get("image_id"),
    ]

    for value in candidates:
        if value is None or pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute():
            if candidate.exists():
                return candidate
        else:
            path = image_dir / candidate
            if path.exists():
                return path
            if candidate.suffix.lower() == "":
                for suffix in sorted(IMAGE_EXTENSIONS):
                    alt = image_dir / f"{candidate}{suffix}"
                    if alt.exists():
                        return alt

    image_id = row.get("image_id")
    if image_id is not None and not pd.isna(image_id):
        image_id_str = str(image_id)
        for suffix in sorted(IMAGE_EXTENSIONS):
            alt = image_dir / f"{image_id_str}{suffix}"
            if alt.exists():
                return alt
            alt2 = image_dir / f"{image_id_str}.jpg"
            if alt2.exists():
                return alt2

        if image_dir.exists():
            for file in image_dir.iterdir():
                stem = file.stem.lower()
                if file.is_file() and stem == image_id_str.lower():
                    return file
                if file.is_file() and stem.startswith(f"{image_id_str.lower()}_"):
                    return file

    url = row.get("image_url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        if image_id is not None and not pd.isna(image_id):
            stem = f"{image_id}_{_stable_url_token(url)}"
            for suffix in sorted(IMAGE_EXTENSIONS):
                local_path = image_dir / f"{stem}{suffix}"
                if local_path.exists():
                    return local_path
            return image_dir / f"{stem}.jpg"
        file_name = f"image_{_stable_url_token(url)}.jpg"
        local_path = image_dir / file_name
        if local_path.exists():
            return local_path
        return local_path

    return image_dir / f"{row.get('image_id', 'unknown')}.jpg"


def build_splits(df: pd.DataFrame, split_column: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_column is not None and split_column in df.columns:
        normalized = df.copy()
        normalized["split"] = normalized[split_column].apply(_normalize_split_name)
    elif "dataset_split" in df.columns:
        normalized = df.copy()
        normalized["split"] = normalized["dataset_split"].apply(_normalize_split_name)
    elif "split" in df.columns:
        normalized = df.copy()
        normalized["split"] = normalized["split"].apply(_normalize_split_name)
    else:
        normalized = df.copy()
        train_df, temp_df = train_test_split(
            normalized,
            test_size=0.2,
            stratify=normalized["label"] if "label" in normalized.columns else None,
            random_state=42,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["label"] if "label" in temp_df.columns else None,
            random_state=42,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    train_df = normalized[normalized["split"] == "train"].copy()
    val_df = normalized[normalized["split"] == "val"].copy()
    test_df = normalized[normalized["split"] == "test"].copy()

    if train_df.empty:
        train_df, temp_df = train_test_split(
            normalized,
            test_size=0.2,
            stratify=normalized["label"] if "label" in normalized.columns else None,
            random_state=42,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["label"] if "label" in temp_df.columns else None,
            random_state=42,
        )
    elif val_df.empty and not test_df.empty:
        val_df = normalized[normalized["split"] == "validation"].copy()
    elif val_df.empty:
        train_df, temp_df = train_test_split(
            normalized,
            test_size=0.2,
            stratify=normalized["label"] if "label" in normalized.columns else None,
            random_state=42,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["label"] if "label" in temp_df.columns else None,
            random_state=42,
        )

    if test_df.empty:
        remaining = normalized[~normalized.index.isin(train_df.index.union(val_df.index))]
        if not remaining.empty:
            test_df = remaining.copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def select_metadata_columns(df: pd.DataFrame) -> List[str]:
    possible = [
        "gender_num",
        "age_group_num",
        "confidence",
        "quality_score",
        "difficulty",
        "image_quality",
        "detection_difficulty",
        "confidence_score",
        "resolution_width",
        "resolution_height",
        "source_num",
    ]
    present = [col for col in possible if col in df.columns]
    return present


def prepare_datasets(metadata_csv: str | Path, images_dir: str | Path, split_column: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    metadata = pd.read_csv(metadata_csv)
    metadata = _standardize_metadata(metadata)

    image_dir = Path(images_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing image dataset from {len(metadata)} records...")
    valid_rows = []
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Checking local images"):
        resolved_path = _resolve_image_path(row, image_dir)
        if not resolved_path.exists():
            remote_url = row.get("image_url")
            if isinstance(remote_url, str) and remote_url.startswith(("http://", "https://")):
                file_name = f"{row.get('image_id', idx)}.jpg"
                local_path = image_dir / file_name
                if _download_image(remote_url, local_path):
                    metadata.at[idx, "image_path"] = str(local_path)
                    valid_rows.append(idx)
                    continue
        if Path(str(resolved_path)).exists():
            metadata.at[idx, "image_path"] = str(resolved_path)
            valid_rows.append(idx)

    metadata = metadata.loc[valid_rows].reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No valid image files were found for training. Check image_url values or the images directory.")

    print(f"Loaded {len(metadata)} usable rows after filtering missing images.")
    train_df, val_df, test_df = build_splits(metadata, split_column)
    meta_cols = select_metadata_columns(metadata)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("After filtering missing files, some dataset splits are empty. Check dataset_split values or use the default split generation.")
    return train_df, val_df, test_df, meta_cols


class FaceImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        metadata_columns: Iterable[str],
        image_size: int = 128,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.metadata_columns = list(metadata_columns)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            image_path = self.image_dir / f"{row.get('image_id', 'unknown')}.jpg"

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError, ValueError):
            image = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))

        image = image.resize((self.image_size, self.image_size))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        metadata_values = []
        for col in self.metadata_columns:
            value = row.get(col, 0.5)
            if pd.isna(value):
                value = 0.5
            metadata_values.append(float(value))

        metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32)
        label_tensor = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": image_tensor,
            "metadata": metadata_tensor,
            "label": label_tensor,
            "sample_id": str(row.get("image_id", idx)),
        }


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_dir: str | Path,
    metadata_columns: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = FaceImageDataset(train_df, image_dir, metadata_columns, image_size=image_size)
    val_dataset = FaceImageDataset(val_df, image_dir, metadata_columns, image_size=image_size)
    test_dataset = FaceImageDataset(test_df, image_dir, metadata_columns, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
