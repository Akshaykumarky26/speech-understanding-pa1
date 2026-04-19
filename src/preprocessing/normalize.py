"""
Audio normalization utilities for Programming Assignment 1.

This module prepares audio for downstream speech tasks by:
1. converting to mono
2. resampling to a fixed sampling rate
3. removing DC offset
4. applying peak normalization
5. saving normalized audio files
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(audio_path: str | Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono at a fixed sampling rate.
    """
    waveform, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    return waveform.astype(np.float32), sr


def remove_dc_offset(waveform: np.ndarray) -> np.ndarray:
    """
    Remove DC offset by subtracting the mean amplitude.
    """
    return waveform - np.mean(waveform)


def peak_normalize(waveform: np.ndarray, peak_value: float = 0.95) -> np.ndarray:
    """
    Normalize waveform so that its absolute peak reaches the target value.
    """
    max_amp = np.max(np.abs(waveform))

    if max_amp < 1e-8:
        return waveform

    return waveform / max_amp * peak_value


def rms_normalize(waveform: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Normalize waveform to a target RMS energy.
    """
    rms = np.sqrt(np.mean(waveform ** 2))

    if rms < 1e-8:
        return waveform

    normalized = waveform * (target_rms / rms)
    return np.clip(normalized, -1.0, 1.0)


def normalize_waveform(
    waveform: np.ndarray,
    use_rms: bool = True,
    target_rms: float = 0.05,
    peak_value: float = 0.95,
) -> np.ndarray:
    """
    Apply full normalization pipeline.
    """
    waveform = remove_dc_offset(waveform)

    if use_rms:
        waveform = rms_normalize(waveform, target_rms=target_rms)

    waveform = peak_normalize(waveform, peak_value=peak_value)
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform.astype(np.float32)


def normalize_file(
    input_path: str | Path,
    output_path: str | Path,
    sr: int = 16000,
) -> None:
    """
    Load, normalize, and save an audio file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    waveform, sr = load_audio(input_path, sr=sr)
    normalized = normalize_waveform(waveform)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), normalized, sr)

    print(f"Normalized audio saved to: {output_path}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(normalized) / sr:.2f} seconds")


def main():
    normalize_file(
        input_path="audio/original_segment.wav",
        output_path="data/processed/original_segment_normalized.wav",
        sr=16000,
    )

    normalize_file(
        input_path="audio/student_voice_ref.wav",
        output_path="data/processed/student_voice_ref_normalized.wav",
        sr=16000,
    )


if __name__ == "__main__":
    main()