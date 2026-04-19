"""
Generate preprocessing comparison figures for the PA1 report.

This script compares:
1. raw original lecture audio
2. denoised + normalized lecture audio

It saves waveform and spectrogram figures inside report/figures/.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


RAW_AUDIO = Path("audio/original_segment.wav")
PROCESSED_AUDIO = Path("data/processed/original_segment_denoised_normalized.wav")
FIGURE_DIR = Path("report/figures")


def load_audio(path: Path, sr: int = 16000):
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = librosa.load(str(path), sr=sr, mono=True)
    return waveform, sr


def save_waveform_plot(waveform, sr, title: str, output_path: Path):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(waveform, sr=sr)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_spectrogram_plot(waveform, sr, title: str, output_path: Path):
    stft = librosa.stft(waveform, n_fft=512, hop_length=160, win_length=400)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="hz",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_waveform, sr = load_audio(RAW_AUDIO)
    processed_waveform, _ = load_audio(PROCESSED_AUDIO, sr=sr)

    save_waveform_plot(
        raw_waveform,
        sr,
        "Raw Lecture Audio Waveform",
        FIGURE_DIR / "preprocessing_raw_waveform.png",
    )

    save_waveform_plot(
        processed_waveform,
        sr,
        "Denoised and Normalized Lecture Audio Waveform",
        FIGURE_DIR / "preprocessing_processed_waveform.png",
    )

    save_spectrogram_plot(
        raw_waveform,
        sr,
        "Raw Lecture Audio Spectrogram",
        FIGURE_DIR / "preprocessing_raw_spectrogram.png",
    )

    save_spectrogram_plot(
        processed_waveform,
        sr,
        "Denoised and Normalized Lecture Audio Spectrogram",
        FIGURE_DIR / "preprocessing_processed_spectrogram.png",
    )

    print("Preprocessing figures saved in report/figures/")


if __name__ == "__main__":
    main()