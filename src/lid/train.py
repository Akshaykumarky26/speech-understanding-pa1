"""
Training script for frame-level Hindi-English Language Identification.

This script trains a lightweight CNN on log-mel windows created from
manually annotated language segments.

Outputs:
- trained LID weights
- metrics CSV
- training curve figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split

from src.lid.dataset import LIDSegmentDataset
from src.lid.model import LIDCNN, save_model


MANIFEST_PATH = Path("data/manifests/lid_segments.csv")
MODEL_PATH = Path("models/lid/lid_cnn.pt")
METRICS_PATH = Path("report/tables/lid_training_metrics.csv")
FIGURE_PATH = Path("report/figures/lid_training_curve.png")

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
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


def create_loaders() -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    dataset = LIDSegmentDataset(MANIFEST_PATH)

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation data.
    """
    model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * features.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_f1_macro": f1,
    }


def save_training_curve(history: List[Dict[str, float]]) -> None:
    """
    Save loss curve for the report.
    """
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LID Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    print(f"Training curve saved to: {FIGURE_PATH}")


def main() -> None:
    """
    Main training function.
    """
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders()

    model = LIDCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: List[Dict[str, float]] = []

    print("Starting LID training...")
    print("=" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Acc: {metrics['val_accuracy']:.4f} | "
            f"Val F1: {metrics['val_f1_macro']:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_model(model, MODEL_PATH)

    pd.DataFrame(history).to_csv(METRICS_PATH, index=False)
    print(f"Training metrics saved to: {METRICS_PATH}")

    save_training_curve(history)

    print("\nLID training completed.")


if __name__ == "__main__":
    main()