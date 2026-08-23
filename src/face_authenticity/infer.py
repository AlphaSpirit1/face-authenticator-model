from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .data import FaceImageDataset, _resolve_image_path, _standardize_metadata
from .model import FaceAuthNet


def load_saved_model(model_path: str | Path) -> tuple[FaceAuthNet, list[str], torch.device]:
    checkpoint = torch.load(model_path, map_location="cpu")
    metadata_columns = checkpoint.get("metadata_columns", [])
    model = FaceAuthNet(metadata_dim=len(metadata_columns) if metadata_columns else 1)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, metadata_columns, device


def build_eval_dataframe(metadata_csv: str | Path, images_dir: str | Path, split_column: str | None = None) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)
    metadata = _standardize_metadata(metadata)
    image_dir = Path(images_dir)

    if split_column is not None and split_column in metadata.columns:
        subset = metadata[metadata[split_column].astype(str).str.strip().str.lower().isin({"test", "validation", "val"})].copy()
        if subset.empty:
            subset = metadata.copy()
    elif "dataset_split" in metadata.columns:
        subset = metadata[metadata["dataset_split"].astype(str).str.strip().str.lower().isin({"test", "validation", "val"})].copy()
        if subset.empty:
            subset = metadata.copy()
    else:
        subset = metadata.copy()

    subset = subset.reset_index(drop=True)
    for idx, row in subset.iterrows():
        resolved = _resolve_image_path(row, image_dir)
        subset.at[idx, "image_path"] = str(resolved)

    return subset


def evaluate_saved_model(
    model_path: str | Path,
    metadata_csv: str | Path,
    images_dir: str | Path,
    split_column: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> dict:
    model, metadata_columns, device = load_saved_model(model_path)
    df = build_eval_dataframe(metadata_csv, images_dir, split_column)
    if df.empty:
        raise ValueError("No rows available for evaluation. Check the metadata file and image directory.")

    dataset = FaceImageDataset(df, image_dir=images_dir, metadata_columns=metadata_columns, image_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_prob = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)
            logits = model(images, metadata)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

    predictions = np.array([1 if p >= threshold else 0 for p in y_prob])
    y_true = np.array(y_true)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    return metrics


def _metadata_overrides_from_args(metadata_columns: list[str], values: dict[str, object]) -> list[float]:
    mapping = {
        "gender": {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0, "unknown": 0.5, "u": 0.5},
        "age_group": {
            "0-18": 0.1,
            "18-25": 0.3,
            "26-35": 0.5,
            "35-44": 0.7,
            "36-50": 0.8,
            "45-54": 0.9,
            "50+": 1.0,
            "unknown": 0.5,
        },
    }
    result: list[float] = []
    for column in metadata_columns:
        v = values.get(column)
        if v is None and column == "gender_num":
            v = values.get("gender")
            if isinstance(v, str):
                v = mapping["gender"].get(v.strip().lower(), 0.5)
        elif v is None and column == "age_group_num":
            v = values.get("age_group")
            if isinstance(v, str):
                v = mapping["age_group"].get(v.strip().lower(), 0.5)
        elif v is None and column == "resolution_width":
            resolution = values.get("resolution")
            if isinstance(resolution, str) and "x" in resolution.lower():
                w, _ = resolution.lower().split("x", 1)
                try:
                    v = float(w.strip())
                except ValueError:
                    v = 0.5
        elif v is None and column == "resolution_height":
            resolution = values.get("resolution")
            if isinstance(resolution, str) and "x" in resolution.lower():
                _, h = resolution.lower().split("x", 1)
                try:
                    v = float(h.strip())
                except ValueError:
                    v = 0.5
        if v is None:
            result.append(0.5)
            continue
        try:
            result.append(float(v))
        except (TypeError, ValueError):
            result.append(0.5)
    return result


def predict_single_image(
    model_path: str | Path,
    image_path: str | Path,
    threshold: float = 0.5,
    metadata_overrides: dict[str, object] | None = None,
) -> dict:
    model, metadata_columns, device = load_saved_model(model_path)
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    image_array = np.asarray(img, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)

    override_values = metadata_overrides or {}
    metadata_values = _metadata_overrides_from_args(metadata_columns, override_values)
    metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor, metadata_tensor)
        probability = float(torch.sigmoid(logits[0]).cpu().item())

    pred_label = "REAL" if probability >= threshold else "FAKE"
    return {
        "prediction": pred_label,
        "real_probability": probability,
        "fake_probability": 1.0 - probability,
        "threshold": threshold,
        "metadata_columns": metadata_columns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or predict with a trained face-authenticity model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pt).")
    parser.add_argument("--image-path", type=str, default=None, help="Single image to classify in REAL/FAKE mode.")
    parser.add_argument("--metadata-csv", type=str, default=None, help="CSV containing rows to evaluate.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory containing the evaluation images.")
    parser.add_argument("--split-column", type=str, default=None, help="Optional split column to filter evaluation rows by test/val.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for REAL vs FAKE.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save metrics as JSON.")
    parser.add_argument("--gender", type=str, default=None, help="Optional gender override: male, female, unknown.")
    parser.add_argument("--age-group", type=str, default=None, help="Optional age bucket override such as 18-25 or 50+.")
    parser.add_argument("--quality-score", type=float, default=None, help="Optional numeric quality score.")
    parser.add_argument("--confidence", type=float, default=None, help="Optional numeric confidence score.")
    parser.add_argument("--difficulty", type=float, default=None, help="Optional numeric detection difficulty.")
    parser.add_argument("--resolution", type=str, default=None, help="Optional image resolution like 1080x1080.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.image_path:
        metadata_overrides = {}
        if args.gender is not None:
            metadata_overrides["gender"] = args.gender
        if args.age_group is not None:
            metadata_overrides["age_group"] = args.age_group
        if args.quality_score is not None:
            metadata_overrides["quality_score"] = args.quality_score
        if args.confidence is not None:
            metadata_overrides["confidence"] = args.confidence
        if args.difficulty is not None:
            metadata_overrides["difficulty"] = args.difficulty
        if args.resolution is not None:
            metadata_overrides["resolution"] = args.resolution

        result = predict_single_image(
            model_path=args.model_path,
            image_path=args.image_path,
            threshold=args.threshold,
            metadata_overrides=metadata_overrides,
        )
        print(json.dumps(result, indent=2))
    else:
        if args.metadata_csv is None or args.images_dir is None:
            raise ValueError("For evaluation mode, pass both --metadata-csv and --images-dir.")
        metrics = evaluate_saved_model(
            model_path=args.model_path,
            metadata_csv=args.metadata_csv,
            images_dir=args.images_dir,
            split_column=args.split_column,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        print(json.dumps(metrics, indent=2))
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(metrics, fp, indent=2)
