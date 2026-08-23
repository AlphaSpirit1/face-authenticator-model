from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .data import make_loaders, prepare_datasets
from .model import FaceAuthNet, evaluate_model, save_confusion_matrix


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module, loader, optimizer: AdamW, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    progress = tqdm(loader, desc="Training batches", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, metadata)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def run_training(
    metadata_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    split_column: str | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, meta_cols = prepare_datasets(metadata_csv, images_dir, split_column)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more dataset splits are empty. Check the metadata or the split column.")

    train_loader, val_loader, test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        image_dir=images_dir,
        metadata_columns=meta_cols,
        batch_size=batch_size,
        image_size=128,
    )

    metadata_dim = len(meta_cols) if meta_cols else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceAuthNet(metadata_dim=metadata_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_state = None

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": []}

    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting training for {epochs} epochs on {len(train_loader.dataset)} samples...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        model.eval()
        total_loss = 0.0
        total_samples = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_progress:
                images = batch["image"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device)
                logits = model(images, metadata)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                val_predictions.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
                val_labels.extend(labels.cpu().numpy().astype(int).tolist())

        val_loss = total_loss / total_samples
        val_accuracy = float(np.mean(np.array(val_predictions) == np.array(val_labels)))
        val_precision = float(np.sum((np.array(val_predictions) == 1) & (np.array(val_labels) == 1)) / max(np.sum(np.array(val_predictions) == 1), 1))
        val_recall = float(np.sum((np.array(val_predictions) == 1) & (np.array(val_labels) == 1)) / max(np.sum(np.array(val_labels) == 1), 1))
        val_f1 = float(2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f} - val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device)
    confusion = test_metrics["confusion_matrix"]
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.png")

    metrics_payload = {
        "test": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
        },
        "val_history": history,
    }

    model_path = output_dir / "face_auth_model.pt"
    torch.save({"model_state": model.state_dict(), "metadata_columns": meta_cols}, model_path)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print("\nTest set metrics:")
    print(json.dumps(metrics_payload["test"], indent=2))
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face-authenticity classifier.")
    parser.add_argument("--metadata-csv", type=str, required=True, help="Path to CSV metadata file.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output-dir", type=str, default="artifacts/run", help="Directory to save model and metrics.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--split-column", type=str, default=None, help="Optional dataset split column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run_training(
        metadata_csv=args.metadata_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        split_column=args.split_column,
    )
