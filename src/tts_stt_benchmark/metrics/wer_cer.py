"""
WER (Word Error Rate) and CER (Character Error Rate) computation.
Uses the `jiwer` library.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer


def _normalise(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    # Unicode normalisation (NFD → remove combining marks)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])

_CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfChars(),
])


def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """
    Return (WER, CER) as values between 0 and 1 (inclusive).
    Both reference and hypothesis are normalised before computing.
    """
    ref = _normalise(reference)
    hyp = _normalise(hypothesis)

    if not ref:
        return 0.0, 0.0

    try:
        wer = jiwer.wer(
            ref, hyp,
            reference_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        )
    except Exception:
        wer = 1.0

    try:
        cer = jiwer.cer(
            ref, hyp,
            reference_transform=_CER_TRANSFORM,
            hypothesis_transform=_CER_TRANSFORM,
        )
    except Exception:
        cer = 1.0

    # Clamp to [0, 1] (substitution-heavy transcripts can exceed 1 in jiwer)
    return min(float(wer), 1.0), min(float(cer), 1.0)
