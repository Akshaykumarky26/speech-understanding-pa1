"""
Constrained STT decoding with syllabus-aware N-gram logit biasing.

This module implements a custom token-level LogitsProcessor for Whisper.
The processor biases the decoder toward technical terms from the speech
course syllabus such as "spectrogram", "cepstrum", "MFCC", "DTW", etc.

This is different from simple post-processing:
- the token probabilities are modified during generation
- candidate technical terms are encouraged before final text is produced
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import librosa
import numpy as np
import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.stt.ngram_lm import load_terms, normalize_text


AUDIO_PATH = Path("audio/original_segment.wav")
TERMS_PATH = Path("data/manifests/speech_course_terms.txt")

OUTPUT_TXT = Path("outputs/transcripts/biased_transcript.txt")
OUTPUT_JSON = Path("outputs/transcripts/biased_transcript.json")

MODEL_NAME = "openai/whisper-tiny"

SAMPLE_RATE = 16000
CHUNK_SEC = 30
BIAS_VALUE = 4.0
MAX_NEW_TOKENS = 256


class NGramLogitBiasProcessor(LogitsProcessor):
    """
    Custom logits processor for syllabus-aware technical term biasing.

    The processor stores tokenized technical terms. During decoding:
    1. If the generated suffix matches the prefix of a technical term,
       the next token in that term is boosted.
    2. First tokens of technical terms also receive a small boost so the
       decoder can start producing syllabus terms.
    """

    def __init__(
        self,
        term_token_sequences: List[List[int]],
        bias_value: float = 4.0,
        start_bias_value: float = 1.0,
    ) -> None:
        self.term_token_sequences = [
            seq for seq in term_token_sequences if len(seq) > 0
        ]
        self.bias_value = bias_value
        self.start_bias_value = start_bias_value

        self.first_tokens = {
            seq[0] for seq in self.term_token_sequences if len(seq) > 0
        }

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Modify next-token logits.

        input_ids shape: [batch, current_sequence_length]
        scores shape: [batch, vocab_size]
        """
        for batch_idx in range(input_ids.shape[0]):
            generated = input_ids[batch_idx].tolist()

            # Small general boost for beginning technical terms.
            for token_id in self.first_tokens:
                if 0 <= token_id < scores.shape[-1]:
                    scores[batch_idx, token_id] += self.start_bias_value

            # Stronger contextual boost for continuing technical terms.
            for seq in self.term_token_sequences:
                if len(seq) < 2:
                    continue

                for prefix_len in range(1, len(seq)):
                    prefix = seq[:prefix_len]
                    next_token = seq[prefix_len]

                    if len(generated) >= prefix_len and generated[-prefix_len:] == prefix:
                        if 0 <= next_token < scores.shape[-1]:
                            scores[batch_idx, next_token] += self.bias_value

        return scores


def get_device() -> torch.device:
    """
    Select device.

    MPS is supported here because we call the model directly rather than
    using the high-level pipeline.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_audio_chunks(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: int = CHUNK_SEC,
) -> List[Tuple[float, np.ndarray]]:
    """
    Load full audio and split into fixed-length chunks.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    chunk_samples = int(chunk_sec * sample_rate)
    chunks: List[Tuple[float, np.ndarray]] = []

    for start_sample in range(0, len(waveform), chunk_samples):
        end_sample = min(start_sample + chunk_samples, len(waveform))
        chunk = waveform[start_sample:end_sample]

        if len(chunk) < sample_rate:
            continue

        start_sec = start_sample / sample_rate
        chunks.append((start_sec, chunk))

    return chunks


def build_term_token_sequences(
    processor: WhisperProcessor,
    terms: List[str],
) -> List[List[int]]:
    """
    Convert technical terms into Whisper decoder token ID sequences.
    """
    sequences: List[List[int]] = []

    for term in terms:
        normalized = normalize_text(term)

        if not normalized:
            continue

        token_ids = processor.tokenizer(
            normalized,
            add_special_tokens=False,
        ).input_ids

        if token_ids:
            sequences.append(token_ids)

    return sequences


def transcribe_chunk(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    chunk: np.ndarray,
    logits_processor: LogitsProcessorList,
    device: torch.device,
) -> str:
    """
    Transcribe a single audio chunk using biased Whisper decoding.
    """
    inputs = processor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",
        task="transcribe",
    )

    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        logits_processor=logits_processor,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=3,
        do_sample=False,
    )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()


def transcribe_with_bias() -> dict:
    """
    Run full biased transcription.
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    terms = load_terms(TERMS_PATH)
    term_token_sequences = build_term_token_sequences(processor, terms)

    logits_processor = LogitsProcessorList(
        [
            NGramLogitBiasProcessor(
                term_token_sequences=term_token_sequences,
                bias_value=BIAS_VALUE,
                start_bias_value=1.0,
            )
        ]
    )

    chunks = load_audio_chunks(AUDIO_PATH)

    print(f"Loaded terms: {len(terms)}")
    print(f"Tokenized term sequences: {len(term_token_sequences)}")
    print(f"Audio chunks: {len(chunks)}")
    print("Starting biased transcription...")

    transcript_rows = []
    full_text_parts = []

    for idx, (start_sec, chunk) in enumerate(chunks, start=1):
        print(f"Chunk {idx}/{len(chunks)} | start={start_sec:.2f}s")

        text = transcribe_chunk(
            model=model,
            processor=processor,
            chunk=chunk,
            logits_processor=logits_processor,
            device=device,
        )

        transcript_rows.append(
            {
                "chunk_id": idx,
                "start_sec": round(start_sec, 3),
                "end_sec": round(start_sec + len(chunk) / SAMPLE_RATE, 3),
                "text": text,
            }
        )

        full_text_parts.append(text)

    full_text = " ".join(full_text_parts).strip()

    return {
        "model": MODEL_NAME,
        "bias_value": BIAS_VALUE,
        "terms_path": str(TERMS_PATH),
        "num_terms": len(terms),
        "num_chunks": len(chunks),
        "chunks": transcript_rows,
        "text": full_text,
    }


def save_outputs(result: dict) -> None:
    """
    Save biased transcript as TXT and JSON.
    """
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write(result["text"] + "\n")

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Biased transcript TXT saved to: {OUTPUT_TXT}")
    print(f"Biased transcript JSON saved to: {OUTPUT_JSON}")


def main() -> None:
    result = transcribe_with_bias()
    save_outputs(result)

    print("\nBiased transcript preview:")
    print("-" * 60)
    print(result["text"][:1000])


if __name__ == "__main__":
    main()