"""
CNN model for frame/window-level Hindi-English Language Identification.

Input:
    log-mel spectrogram of shape [batch, n_mels, time_frames]

Output:
    logits for two classes:
    0 = English
    1 = Hindi
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LIDCNN(nn.Module):
    """
    Lightweight CNN for language identification from log-mel features.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch, n_mels, time_frames]

        Returns
        -------
        torch.Tensor
            Logits of shape [batch, num_classes]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [batch, n_mels, time_frames], got {x.shape}"
            )

        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits


def save_model(model: nn.Module, path: str | Path) -> None:
    """
    Save model weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(path: str | Path, device: str = "cpu") -> LIDCNN:
    """
    Load model weights.
    """
    model = LIDCNN()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def inspect_model() -> None:
    """
    Quick sanity check for model input/output shapes.
    """
    model = LIDCNN()
    dummy_batch = torch.randn(4, 64, 101)

    logits = model(dummy_batch)

    print("LID Model Inspection")
    print("=" * 40)
    print(f"Input shape : {dummy_batch.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Classes     : 2")


if __name__ == "__main__":
    inspect_model()