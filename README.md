# Programming Assignment 1: Code-Switched Speech Understanding Pipeline

This repository contains the implementation for **Programming Assignment 1** of the **Speech Understanding** course.

The goal of this project is to build a complete speech pipeline for **code-switched Hindi-English academic lectures**, including:

- frame-level Language Identification (LID)
- constrained Automatic Speech Recognition (ASR / STT)
- denoising and normalization
- Hinglish-to-IPA phonetic mapping
- translation into a Low-Resource Language (LRL)
- zero-shot cross-lingual voice cloning
- DTW-based prosody warping
- anti-spoofing detection
- adversarial robustness evaluation

## Repository Structure

```text
configs/               # YAML config files
src/                   # Source code for all modules
data/                  # Raw, processed, and manifest files
models/                # Saved weights and pretrained checkpoints
outputs/               # Generated outputs, predictions, plots
report/                # Report assets, screenshots, figures, tables
implementation_note/   # One-page implementation note
audio/                 # Input/output audio files
pipeline.py            # Main pipeline entry point
requirements.txt       # Python dependencies
environment.yml        # Conda environment file
run_all.sh             # Helper script to run the full pipeline