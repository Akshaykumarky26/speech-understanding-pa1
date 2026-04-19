"""
Prepare required audio files for Programming Assignment 1.

This script:
1. extracts a 10-minute lecture segment
2. extracts exactly 60 seconds of student voice
3. converts both files to mono
4. resamples both files to 16 kHz
5. saves them with assignment-required names
"""

from pathlib import Path

import librosa
import soundfile as sf


def extract_segment(
    input_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    target_sr: int = 16000,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    waveform, sr = librosa.load(
        str(input_path),
        sr=target_sr,
        mono=True,
        offset=start_sec,
        duration=duration_sec,
    )

    expected_samples = int(duration_sec * target_sr)

    if len(waveform) < expected_samples:
        raise ValueError(
            f"Extracted audio is shorter than expected. "
            f"Expected {duration_sec}s but got {len(waveform) / target_sr:.2f}s"
        )

    waveform = waveform[:expected_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, target_sr)

    print(f"Saved: {output_path}")
    print(f"Duration: {len(waveform) / target_sr:.2f} seconds")
    print(f"Sample Rate: {target_sr} Hz")
    print("Channels: mono")


def main():
    # Change these start times if you want a different portion.
    lecture_start_sec = 0
    voice_start_sec = 0

    extract_segment(
        input_path="data/raw/lecture_full.wav",
        output_path="audio/original_segment.wav",
        start_sec=lecture_start_sec,
        duration_sec=600,
        target_sr=16000,
    )

    extract_segment(
        input_path="data/raw/student_voice_full.wav",
        output_path="audio/student_voice_ref.wav",
        start_sec=voice_start_sec,
        duration_sec=60,
        target_sr=16000,
    )


if __name__ == "__main__":
    main()