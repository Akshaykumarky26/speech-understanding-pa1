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

### 5. LID Inference Timeline

Implemented:
- sliding-window LID inference over the full 10-minute lecture
- 1-second analysis window with 0.5-second hop
- timestamp-level Hindi/English prediction
- confidence score for each prediction
- timeline CSV and plot generation

Generated files:
- outputs/transcripts/lid_timeline.csv
- report/figures/lid_timeline.png

Note:
The current timeline is generated using the initial CNN model trained on temporary annotation labels. Final inference will be repeated after manually verified LID segment annotations are added.
### 5. LID Inference Timeline

Implemented:
- sliding-window LID inference over the full 10-minute lecture
- 1-second analysis window with 0.5-second hop
- timestamp-level Hindi/English prediction
- confidence score for each prediction
- timeline CSV and plot generation

Generated files:
- outputs/transcripts/lid_timeline.csv
- report/figures/lid_timeline.png

Note:
The current timeline is generated using the initial CNN model trained on temporary annotation labels. Final inference will be repeated after manually verified LID segment annotations are added.

cat >> report/report_notes.md <<'EOF'

### 6. Baseline STT Transcription

Implemented:
- baseline Whisper transcription using Hugging Face Transformers
- transcript output as TXT and JSON
- 30-second chunking with overlap for long-form lecture audio

Generated files:
- outputs/transcripts/baseline_transcript.txt
- outputs/transcripts/baseline_transcript.json

Note:
This is the baseline transcription before applying syllabus-aware N-gram biasing.
EOF

cat >> report/report_notes.md <<'EOF'

### 7. Constrained STT Decoding with N-gram Logit Biasing

Implemented:
- syllabus-based technical term vocabulary
- unigram, bigram, and trigram language model artifact
- custom Whisper logits processor
- token-level technical term biasing during generation
- biased transcript generation in chunked mode

Generated local files:
- outputs/transcripts/ngram_lm.json
- outputs/transcripts/biased_transcript.txt
- outputs/transcripts/biased_transcript.json

Design note:
The N-gram bias is applied during decoding by increasing the logits of tokens that continue syllabus-relevant technical terms. This makes the method closer to constrained decoding than simple transcript post-processing.
EOF