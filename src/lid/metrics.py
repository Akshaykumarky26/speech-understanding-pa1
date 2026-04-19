"""
Evaluation utilities for frame-level Hindi-English LID.

This script loads the trained LID CNN and evaluates it on the validation split.
It generates:
1. confusion matrix figure
2. confusion matrix CSV
3. classification report CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

from src.lid.dataset import ID_TO_LABEL, LIDSegmentDataset
from src.lid.model import load_model


MANIFEST_PATH = Path("data/manifests/lid_segments.csv")
MODEL_PATH = Path("models/lid/lid_cnn.pt")

CONFUSION_MATRIX_FIG = Path("report/figures/lid_confusion_matrix.png")
CONFUSION_MATRIX_CSV = Path("report/tables/lid_confusion_matrix.csv")
CLASSIFICATION_REPORT_CSV = Path("report/tables/lid_classification_report.csv")

BATCH_SIZE = 8
TRAIN_SPLIT = 0.8
SEED = 42


def get_device() -> torch.device:
    """
    Select the best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_validation_loader() -> DataLoader:
    """
    Recreate the same validation split used during training.
    """
    dataset = LIDSegmentDataset(MANIFEST_PATH)

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return val_loader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[List[int], List[int]]:
    """
    Collect labels and predictions.
    """
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return all_labels, all_preds


def save_confusion_matrix(y_true: List[int], y_pred: List[int]) -> None:
    """
    Save confusion matrix as both CSV and PNG.
    """
    labels = [0, 1]
    label_names = [ID_TO_LABEL[label_id] for label_id in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    CONFUSION_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_FIG.parent.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names],
    )
    cm_df.to_csv(CONFUSION_MATRIX_CSV)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("LID Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(label_names)), label_names)
    plt.yticks(range(len(label_names)), label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FIG, dpi=300)
    plt.close()

    print(f"Confusion matrix CSV saved to: {CONFUSION_MATRIX_CSV}")
    print(f"Confusion matrix figure saved to: {CONFUSION_MATRIX_FIG}")


def save_classification_report(y_true: List[int], y_pred: List[int]) -> None:
    """
    Save precision, recall, and F1-score report as CSV.
    """
    target_names = [ID_TO_LABEL[0], ID_TO_LABEL[1]]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    CLASSIFICATION_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_CSV)

    print(f"Classification report saved to: {CLASSIFICATION_REPORT_CSV}")


def main() -> None:
    """
    Main evaluation function.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained LID model not found: {MODEL_PATH}. "
            "Run python -m src.lid.train first."
        )

    device = get_device()
    print(f"Using device: {device}")

    val_loader = create_validation_loader()
    model = load_model(MODEL_PATH, device=str(device))

    y_true, y_pred = collect_predictions(model, val_loader, device)

    save_confusion_matrix(y_true, y_pred)
    save_classification_report(y_true, y_pred)

    print("\nLID evaluation completed.")


if __name__ == "__main__":
    main()