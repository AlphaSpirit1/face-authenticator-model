from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn


class FaceAuthNet(nn.Module):
    def __init__(self, metadata_dim: int, num_filters: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters * 2),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters * 4),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear((num_filters * 4) * 4 * 4 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(image)
        image_features = image_features.flatten(start_dim=1)
        meta_features = self.metadata_branch(metadata)
        combined = torch.cat([image_features, meta_features], dim=1)
        return self.classifier(combined).squeeze(1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float | np.ndarray]:
    model.eval()
    all_labels: List[int] = []
    all_logits: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, metadata)
            probabilities = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_logits.extend(probabilities.cpu().numpy().tolist())

    predictions = np.array([1 if p >= threshold else 0 for p in all_logits])
    y_true = np.array(all_labels)

    metrics = {
        "accuracy": accuracy(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions),
        **precision_recall_f1(y_true, predictions),
    }
    return metrics


def save_confusion_matrix(confusion: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticklabels(["Fake", "Real"])

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
