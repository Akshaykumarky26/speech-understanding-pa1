"""
Inference script for frame-level Hindi-English Language Identification.

This script applies the trained LID model over the full lecture segment
using sliding windows and produces:
1. timestamp-level language predictions
2. timeline CSV
3. language timeline plot
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.lid.dataset import ID_TO_LABEL
from src.lid.model import load_model


AUDIO_PATH = Path("audio/original_segment.wav")
MODEL_PATH = Path("models/lid/lid_cnn.pt")

TIMELINE_CSV = Path("outputs/transcripts/lid_timeline.csv")
TIMELINE_FIG = Path("report/figures/lid_timeline.png")

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.5
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def extract_log_mel(waveform: np.ndarray) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-8
    log_mel = (log_mel - mean) / std

    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def predict_window(
    model: torch.nn.Module,
    waveform: np.ndarray,
    device: torch.device,
) -> tuple[int, float]:
    features = extract_log_mel(waveform).to(device)
    logits = model(features)
    probs = torch.softmax(logits, dim=1)
    pred_id = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_id].item())
    return pred_id, confidence


def run_inference() -> pd.DataFrame:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio not found: {AUDIO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LID model not found: {MODEL_PATH}. Run python -m src.lid.train first."
        )

    device = get_device()
    print(f"Using device: {device}")

    waveform, _ = librosa.load(str(AUDIO_PATH), sr=SAMPLE_RATE, mono=True)
    total_duration = len(waveform) / SAMPLE_RATE

    model = load_model(MODEL_PATH, device=str(device))

    rows: List[dict] = []

    current_start = 0.0

    while current_start + WINDOW_SEC <= total_duration:
        start_sample = int(current_start * SAMPLE_RATE)
        end_sample = int((current_start + WINDOW_SEC) * SAMPLE_RATE)

        chunk = waveform[start_sample:end_sample]

        pred_id, confidence = predict_window(model, chunk, device)
        language = ID_TO_LABEL[pred_id]

        rows.append(
            {
                "start_sec": round(current_start, 3),
                "end_sec": round(current_start + WINDOW_SEC, 3),
                "language": language,
                "confidence": round(confidence, 4),
            }
        )

        current_start += HOP_SEC

    df = pd.DataFrame(rows)

    TIMELINE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TIMELINE_CSV, index=False)

    print(f"LID timeline saved to: {TIMELINE_CSV}")
    return df


def save_timeline_plot(df: pd.DataFrame) -> None:
    label_to_value = {
        "english": 0,
        "hindi": 1,
    }

    y = [label_to_value[label] for label in df["language"]]
    x = df["start_sec"].values

    TIMELINE_FIG.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.step(x, y, where="post")
    plt.yticks([0, 1], ["English", "Hindi"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Predicted Language")
    plt.title("Frame-Level LID Timeline")
    plt.tight_layout()
    plt.savefig(TIMELINE_FIG, dpi=300)
    plt.close()

    print(f"LID timeline plot saved to: {TIMELINE_FIG}")


def main() -> None:
    df = run_inference()
    save_timeline_plot(df)

    print("\nSample predictions:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()