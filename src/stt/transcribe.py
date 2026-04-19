"""
Baseline STT transcription for Programming Assignment 1.

This script uses a pretrained Whisper model through Hugging Face Transformers
to generate a baseline transcript for the 10-minute code-switched lecture.

The output of this script is later refined using syllabus-aware N-gram biasing.
"""

from __future__ import annotations

from pathlib import Path
import json

import torch
from transformers import pipeline


AUDIO_PATH = Path("audio/original_segment.wav")

OUTPUT_TXT = Path("outputs/transcripts/baseline_transcript.txt")
OUTPUT_JSON = Path("outputs/transcripts/baseline_transcript.json")

# Good starting model for local testing.
# Later, this can be changed to openai/whisper-small or openai/whisper-large-v3
# depending on system capacity.
MODEL_NAME = "openai/whisper-tiny"


def get_device_id() -> int:
    """
    Hugging Face pipeline expects:
    - device = 0 for CUDA
    - device = -1 for CPU

    MPS is not always handled cleanly by the pipeline API,
    so for reliability on Mac, CPU is used here.
    """
    if torch.cuda.is_available():
        return 0

    return -1


def transcribe_audio(audio_path: Path = AUDIO_PATH) -> dict:
    """
    Generate baseline transcript using Whisper.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = get_device_id()
    print(f"Loading model: {MODEL_NAME}")
    print(f"Pipeline device: {'cuda' if device == 0 else 'cpu'}")

    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
    )

    print(f"Transcribing: {audio_path}")

    result = asr(
        str(audio_path),
        generate_kwargs={
            "task": "transcribe",
        },
    )

    return result


def save_outputs(result: dict) -> None:
    """
    Save transcript as plain text and JSON.
    """
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    transcript = result.get("text", "").strip()

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write(transcript + "\n")

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Transcript TXT saved to: {OUTPUT_TXT}")
    print(f"Transcript JSON saved to: {OUTPUT_JSON}")


def main() -> None:
    result = transcribe_audio()
    save_outputs(result)

    print("\nTranscript preview:")
    print("-" * 60)
    print(result.get("text", "")[:1000])


if __name__ == "__main__":
    main()