"""
Dataset utilities for frame-level Hindi-English Language Identification.

This module reads manually annotated language segments and converts them into
fixed-duration audio windows with log-mel features.

The current setup uses two labels:
0 = English
1 = Hindi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_TO_ID: Dict[str, int] = {
    "english": 0,
    "hindi": 1,
}

ID_TO_LABEL: Dict[int, str] = {
    0: "english",
    1: "hindi",
}


@dataclass
class LIDWindow:
    audio_path: Path
    start_sec: float
    end_sec: float
    label: int


class LIDSegmentDataset(Dataset):
    """
    PyTorch Dataset for Hindi-English frame/window-level LID.

    Each item returns:
    - log-mel feature tensor of shape [n_mels, time_frames]
    - integer language label
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.windows = self._build_windows()

    def _build_windows(self) -> List[LIDWindow]:
        df = pd.read_csv(self.manifest_path)
        windows: List[LIDWindow] = []

        required_cols = {"audio_path", "start_sec", "end_sec", "language"}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing columns in manifest: {missing_cols}")

        for _, row in df.iterrows():
            language = str(row["language"]).strip().lower()

            if language not in LABEL_TO_ID:
                raise ValueError(
                    f"Unsupported language label: {language}. "
                    f"Allowed labels: {list(LABEL_TO_ID.keys())}"
                )

            audio_path = Path(row["audio_path"])
            segment_start = float(row["start_sec"])
            segment_end = float(row["end_sec"])
            label = LABEL_TO_ID[language]

            current_start = segment_start

            while current_start + self.window_sec <= segment_end:
                windows.append(
                    LIDWindow(
                        audio_path=audio_path,
                        start_sec=current_start,
                        end_sec=current_start + self.window_sec,
                        label=label,
                    )
                )
                current_start += self.hop_sec

        if not windows:
            raise ValueError("No LID windows were created. Check manifest durations.")

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _extract_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        log_mel = (log_mel - mean) / std

        return log_mel.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]

        waveform, _ = librosa.load(
            str(window.audio_path),
            sr=self.sample_rate,
            mono=True,
            offset=window.start_sec,
            duration=self.window_sec,
        )

        expected_len = int(self.window_sec * self.sample_rate)

        if len(waveform) < expected_len:
            waveform = np.pad(waveform, (0, expected_len - len(waveform)))

        waveform = waveform[:expected_len]

        log_mel = self._extract_log_mel(waveform)

        features = torch.tensor(log_mel, dtype=torch.float32)
        label = torch.tensor(window.label, dtype=torch.long)

        return features, label


def inspect_dataset(manifest_path: str | Path) -> None:
    dataset = LIDSegmentDataset(manifest_path)

    print("LID Dataset Inspection")
    print("=" * 40)
    print(f"Manifest      : {manifest_path}")
    print(f"Total windows : {len(dataset)}")
    print(f"Feature shape : {dataset[0][0].shape}")

    labels = [window.label for window in dataset.windows]
    unique, counts = np.unique(labels, return_counts=True)

    print("\nLabel Distribution")
    print("-" * 40)
    for label_id, count in zip(unique, counts):
        print(f"{ID_TO_LABEL[int(label_id)]}: {int(count)} windows")


if __name__ == "__main__":
    inspect_dataset("data/manifests/lid_segments.csv")