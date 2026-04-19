# PA1 Report Notes

## Project Title
Code-Switched Speech Understanding Pipeline for Indian Classroom Lectures

## Target Low-Resource Language
Maithili

## Completed Steps

### 1. Repository Setup
- GitHub repository created: speech-understanding-pa1
- SSH authentication configured
- Modular Python project structure created
- README, requirements, gitignore, environment file, and pipeline entry point added

### 2. Audio Preparation
Required files prepared:
- audio/original_segment.wav
  - Duration: 600 seconds
  - Sample rate: 16 kHz
  - Channels: mono
- audio/student_voice_ref.wav
  - Duration: 60 seconds
  - Sample rate: 16 kHz
  - Channels: mono

### 3. Preprocessing
Implemented:
- audio normalization
- DC offset removal
- RMS normalization
- peak normalization
- spectral subtraction denoising

Generated figures:
- preprocessing_raw_waveform.png
- preprocessing_processed_waveform.png
- preprocessing_raw_spectrogram.png
- preprocessing_processed_spectrogram.png

## Important Design Notes

### Preprocessing Design Choice
A spectral subtraction baseline was selected because it is deterministic, explainable, and suitable for classroom-noise reduction without hiding the core logic behind an API wrapper.

## Pending Sections
- Frame-level Hindi-English LID
- Constrained STT decoding
- Hinglish-to-IPA mapping
- Maithili translation dictionary
- Speaker embedding extraction
- Prosody extraction and DTW warping
- TTS synthesis
- Anti-spoofing classifier
- Adversarial FGSM robustness

### 4. Initial LID Training and Evaluation
Implemented:
- log-mel feature extraction
- CNN-based Hindi-English LID model
- training loop with validation accuracy and macro F1
- confusion matrix generation
- classification report generation

Generated files:
- lid_training_curve.png
- lid_training_metrics.csv
- lid_confusion_matrix.png
- lid_confusion_matrix.csv
- lid_classification_report.csv

Note:
The current LID labels are based on a temporary annotation manifest. Final LID results will be obtained after replacing the template labels with real manually verified Hindi-English segment annotations.