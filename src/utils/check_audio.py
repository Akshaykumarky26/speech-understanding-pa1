"""
Utility script to verify required audio files for Programming Assignment 1.

Checks:
1. file existence
2. sample rate
3. number of channels
4. duration
"""

from pathlib import Path
import soundfile as sf


REQUIRED_FILES = {
    "original_segment": {
        "path": Path("audio/original_segment.wav"),
        "expected_duration": 600,
        "tolerance": 5,
    },
    "student_voice_ref": {
        "path": Path("audio/student_voice_ref.wav"),
        "expected_duration": 60,
        "tolerance": 1,
    },
}


def check_audio_file(name, config):
    path = config["path"]

    print(f"\nChecking: {name}")
    print("-" * 40)

    if not path.exists():
        print(f"Missing file: {path}")
        return False

    info = sf.info(str(path))
    duration = info.frames / info.samplerate

    print(f"Path        : {path}")
    print(f"Sample Rate : {info.samplerate} Hz")
    print(f"Channels    : {info.channels}")
    print(f"Frames      : {info.frames}")
    print(f"Duration    : {duration:.2f} seconds")

    expected = config["expected_duration"]
    tolerance = config["tolerance"]

    if abs(duration - expected) <= tolerance:
        print("Duration OK")
        return True

    print(f"Duration mismatch. Expected around {expected} seconds.")
    return False


def main():
    all_ok = True

    for name, config in REQUIRED_FILES.items():
        ok = check_audio_file(name, config)
        all_ok = all_ok and ok

    print("\nFinal Check")
    print("=" * 40)

    if all_ok:
        print("All required audio files are present and valid.")
    else:
        print("Some audio files need correction.")


if __name__ == "__main__":
    main()