"""
Unit tests for audio quality checks (no real audio required — uses synthetic signals).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_stt_benchmark.metrics.audio_checks import analyse_audio, _rms_dbfs


def _write_wav(path: Path, signal: np.ndarray, sr: int = 24000) -> None:
    sf.write(str(path), signal, sr, subtype="PCM_16")


class TestRmsDbfs:
    def test_full_scale_sine(self):
        sr = 24000
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        rms = _rms_dbfs(sig)
        # Full-scale sine RMS ≈ -3 dBFS
        assert -4.0 < rms < -2.0

    def test_silence_is_negative_inf(self):
        import math
        sig = np.zeros(1000, dtype=np.int16)
        assert math.isinf(_rms_dbfs(sig))


class TestAnalyseAudio:
    def test_normal_speech_no_clipping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            sr = 24000
            t = np.linspace(0, 2, 2 * sr, endpoint=False)
            # Mid-level sine (well below clipping)
            sig = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            _write_wav(path, sig, sr)

            quality = analyse_audio(path)
            assert not quality.clipping_detected
            assert quality.rms_dbfs < -3.0   # not full scale

    def test_clipped_signal_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clipped.wav"
            sr = 24000
            # Create a signal that clips (saturated at max int16)
            sig = np.full(sr, 32767, dtype=np.int16)
            _write_wav(path, sig, sr)

            quality = analyse_audio(path)
            assert quality.clipping_detected

    def test_silence_at_start_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "silence_start.wav"
            sr = 24000
            # 0.5s silence then speech
            silence = np.zeros(int(0.5 * sr), dtype=np.int16)
            speech = (np.random.randn(sr) * 10000).astype(np.int16)
            sig = np.concatenate([silence, speech])
            _write_wav(path, sig, sr)

            quality = analyse_audio(path)
            assert quality.silence_at_start_s >= 0.4

    def test_missing_file_returns_empty_quality(self):
        quality = analyse_audio(Path("/nonexistent/file.wav"))
        assert quality.clipping_detected is False
        assert quality.rms_dbfs == 0.0
