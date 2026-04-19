"""
Preprocessing pipeline for Programming Assignment 1.

This script prepares the required audio files for downstream modules.

Steps:
1. Normalize the original lecture segment.
2. Denoise the original lecture segment using spectral subtraction.
3. Normalize the denoised lecture audio.
4. Normalize the student reference voice.

The lecture audio is denoised because it may contain classroom noise/reverb.
The student voice reference is only normalized to preserve speaker identity.
"""

from pathlib import Path

from src.preprocessing.denoise import denoise_file
from src.preprocessing.normalize import normalize_file


RAW_LECTURE = Path("audio/original_segment.wav")
RAW_STUDENT_VOICE = Path("audio/student_voice_ref.wav")

PROCESSED_DIR = Path("data/processed")

LECTURE_NORMALIZED = PROCESSED_DIR / "original_segment_normalized.wav"
LECTURE_DENOISED = PROCESSED_DIR / "original_segment_denoised.wav"
LECTURE_DENOISED_NORMALIZED = PROCESSED_DIR / "original_segment_denoised_normalized.wav"

STUDENT_VOICE_NORMALIZED = PROCESSED_DIR / "student_voice_ref_normalized.wav"


def main() -> None:
    print("Starting preprocessing pipeline...\n")

    print("[1/4] Normalizing original lecture segment...")
    normalize_file(
        input_path=RAW_LECTURE,
        output_path=LECTURE_NORMALIZED,
        sr=16000,
    )

    print("\n[2/4] Denoising original lecture segment...")
    denoise_file(
        input_path=RAW_LECTURE,
        output_path=LECTURE_DENOISED,
        sr=16000,
    )

    print("\n[3/4] Normalizing denoised lecture segment...")
    normalize_file(
        input_path=LECTURE_DENOISED,
        output_path=LECTURE_DENOISED_NORMALIZED,
        sr=16000,
    )

    print("\n[4/4] Normalizing student reference voice...")
    normalize_file(
        input_path=RAW_STUDENT_VOICE,
        output_path=STUDENT_VOICE_NORMALIZED,
        sr=16000,
    )

    print("\nPreprocessing completed successfully.")
    print(f"Processed files saved in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()