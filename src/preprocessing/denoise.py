"""
Denoising utilities for Programming Assignment 1.

This module implements a simple spectral subtraction baseline for reducing
classroom background noise in lecture recordings.

Why this baseline?
- easy to explain in the report
- deterministic and reproducible
- useful as an interpretable first preprocessing stage
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(audio_path: str | Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load mono audio at a fixed sampling rate.
    """
    waveform, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    return waveform, sr


def save_audio(audio_path: str | Path, waveform: np.ndarray, sr: int) -> None:
    """
    Save waveform to disk.
    """
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), waveform, sr)


def estimate_noise_profile(
    stft_magnitude: np.ndarray,
    num_noise_frames: int = 20,
) -> np.ndarray:
    """
    Estimate a noise profile from the first few STFT frames.

    Assumption:
    The recording starts with a short low-speech / low-energy region.
    """
    num_noise_frames = min(num_noise_frames, stft_magnitude.shape[1])
    noise_profile = np.mean(stft_magnitude[:, :num_noise_frames], axis=1, keepdims=True)
    return noise_profile


def spectral_subtraction(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    alpha: float = 1.5,
    beta: float = 0.02,
    num_noise_frames: int = 20,
) -> np.ndarray:
    """
    Apply spectral subtraction to a noisy waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Input noisy audio.
    sr : int
        Sampling rate.
    n_fft, hop_length, win_length : int
        STFT parameters.
    alpha : float
        Over-subtraction factor.
    beta : float
        Spectral floor factor.
    num_noise_frames : int
        Number of initial frames used for noise estimation.

    Returns
    -------
    np.ndarray
        Denoised waveform.
    """
    stft = librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_profile = estimate_noise_profile(
        magnitude,
        num_noise_frames=num_noise_frames,
    )

    subtracted_mag = magnitude - alpha * noise_profile
    floored_mag = np.maximum(subtracted_mag, beta * magnitude)

    reconstructed_stft = floored_mag * np.exp(1j * phase)
    denoised = librosa.istft(
        reconstructed_stft,
        hop_length=hop_length,
        win_length=win_length,
        length=len(waveform),
    )

    denoised = np.clip(denoised, -1.0, 1.0)
    return denoised.astype(np.float32)


def denoise_file(
    input_path: str | Path,
    output_path: str | Path,
    sr: int = 16000,
) -> None:
    """
    Load an audio file, denoise it, and save the result.
    """
    waveform, sr = load_audio(input_path, sr=sr)
    denoised = spectral_subtraction(waveform, sr=sr)
    save_audio(output_path, denoised, sr)
    print(f"Denoised audio saved to: {output_path}")


if __name__ == "__main__":
    input_path = "audio/original_segment.wav"
    output_path = "outputs/tts/original_segment_denoised.wav"
    denoise_file(input_path, output_path)