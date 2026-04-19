"""
Simple N-gram language model for syllabus-aware STT biasing.

This module builds unigram, bigram, and trigram counts from a curated
speech-course technical vocabulary. The resulting scores are later used
to bias or correct STT output toward assignment-relevant technical terms.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import re
import json


TERMS_PATH = Path("data/manifests/speech_course_terms.txt")
OUTPUT_PATH = Path("outputs/transcripts/ngram_lm.json")


def normalize_text(text: str) -> str:
    """
    Lowercase and remove unnecessary punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Basic whitespace tokenizer.
    """
    return normalize_text(text).split()


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from tokens.
    """
    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def load_terms(path: Path = TERMS_PATH) -> List[str]:
    """
    Load technical terms from text file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Term file not found: {path}")

    terms = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            term = normalize_text(line)
            if term:
                terms.append(term)

    return terms


def build_ngram_lm(terms: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build unigram, bigram, and trigram count dictionaries.
    """
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()

    for term in terms:
        tokens = tokenize(term)

        unigram_counts.update(generate_ngrams(tokens, 1))
        bigram_counts.update(generate_ngrams(tokens, 2))
        trigram_counts.update(generate_ngrams(tokens, 3))

    lm = {
        "unigrams": {" ".join(k): v for k, v in unigram_counts.items()},
        "bigrams": {" ".join(k): v for k, v in bigram_counts.items()},
        "trigrams": {" ".join(k): v for k, v in trigram_counts.items()},
    }

    return lm


def score_phrase(phrase: str, lm: Dict[str, Dict[str, int]]) -> float:
    """
    Score a phrase using weighted n-gram counts.

    Trigrams receive the highest weight because they represent stronger
    technical phrase evidence.
    """
    tokens = tokenize(phrase)

    score = 0.0

    for unigram in generate_ngrams(tokens, 1):
        score += 1.0 * lm["unigrams"].get(" ".join(unigram), 0)

    for bigram in generate_ngrams(tokens, 2):
        score += 2.0 * lm["bigrams"].get(" ".join(bigram), 0)

    for trigram in generate_ngrams(tokens, 3):
        score += 3.0 * lm["trigrams"].get(" ".join(trigram), 0)

    return score


def save_lm(lm: Dict[str, Dict[str, int]], path: Path = OUTPUT_PATH) -> None:
    """
    Save language model counts as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(lm, f, indent=2, ensure_ascii=False)

    print(f"N-gram LM saved to: {path}")


def main() -> None:
    terms = load_terms()
    lm = build_ngram_lm(terms)
    save_lm(lm)

    print("N-gram LM Summary")
    print("=" * 40)
    print(f"Terms    : {len(terms)}")
    print(f"Unigrams : {len(lm['unigrams'])}")
    print(f"Bigrams  : {len(lm['bigrams'])}")
    print(f"Trigrams : {len(lm['trigrams'])}")

    examples = [
        "short time fourier transform",
        "dynamic time warping",
        "hidden markov model",
        "random unrelated phrase",
    ]

    print("\nExample Scores")
    print("-" * 40)
    for phrase in examples:
        print(f"{phrase}: {score_phrase(phrase, lm)}")


if __name__ == "__main__":
    main()