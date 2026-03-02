"""
Unit tests for WER/CER computation utilities.
"""

import pytest
from tts_stt_benchmark.metrics.wer_cer import compute_wer_cer, _normalise


class TestNormalise:
    def test_lowercase(self):
        assert _normalise("Hello World") == "hello world"

    def test_removes_accents(self):
        result = _normalise("niño café")
        assert "n" in result
        assert "cafe" in result

    def test_removes_punctuation(self):
        assert "." not in _normalise("Hello. World!")
        assert "," not in _normalise("one, two, three")

    def test_collapses_whitespace(self):
        assert _normalise("  hello   world  ") == "hello world"


class TestWerCer:
    def test_perfect_match(self):
        wer, cer = compute_wer_cer("hello world", "hello world")
        assert wer == pytest.approx(0.0)
        assert cer == pytest.approx(0.0)

    def test_completely_wrong(self):
        wer, cer = compute_wer_cer("hello world", "foo bar")
        assert wer > 0.5

    def test_one_word_wrong(self):
        wer, cer = compute_wer_cer("the cat sat on the mat", "the cat sat on the hat")
        assert 0 < wer <= 0.20

    def test_empty_hypothesis(self):
        wer, cer = compute_wer_cer("hello world", "")
        assert wer == pytest.approx(1.0)
        assert cer == pytest.approx(1.0)

    def test_empty_reference_returns_zero(self):
        wer, cer = compute_wer_cer("", "hello")
        assert wer == 0.0
        assert cer == 0.0

    def test_spanish_text(self):
        ref = "buenos días, su cita está confirmada"
        hyp = "buenos días su cita está confirmada"
        wer, cer = compute_wer_cer(ref, hyp)
        assert wer == pytest.approx(0.0)   # punctuation stripped → identical

    def test_wer_clamped_to_one(self):
        """WER should never exceed 1.0 in our implementation."""
        wer, cer = compute_wer_cer("a b", "x y z w v")
        assert wer <= 1.0
        assert cer <= 1.0

    def test_case_insensitive(self):
        wer, cer = compute_wer_cer("Hello World", "hello world")
        assert wer == pytest.approx(0.0)
        assert cer == pytest.approx(0.0)
