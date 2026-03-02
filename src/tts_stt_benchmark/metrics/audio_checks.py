"""
Objective audio quality checks for synthesised TTS output.

Checks:
  - Clipping (samples at ±full scale)
  - Silence at start / end
  - RMS loudness (dBFS)
  - Abrupt cut at the end (hard clipping of the waveform)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_stt_benchmark.models import TTSQuality

# Thresholds
_CLIPPING_THRESHOLD = 0.99        # fraction of max amplitude considered clipping
_SILENCE_THRESHOLD_DBFS = -50.0   # below this → silence
_SILENCE_MIN_DURATION_S = 0.05    # minimum silence block to report (50 ms)
_ABRUPT_CUT_WINDOW_S = 0.05       # last N seconds to check for abrupt cut
_ABRUPT_CUT_AMPLITUDE_RATIO = 0.3 # if last window is >30% of mean, it's abrupt


def _rms_dbfs(signal: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    if rms == 0:
        return -math.inf
    return 20.0 * math.log10(rms / 32768.0)  # relative to 16-bit full scale


def _silence_duration(signal: np.ndarray, sr: int, from_start: bool) -> float:
    """Count leading or trailing silence duration in seconds."""
    threshold = 32768 * (10 ** (_SILENCE_THRESHOLD_DBFS / 20.0))
    chunk = signal if from_start else signal[::-1]
    for i, sample in enumerate(chunk):
        if abs(sample) > threshold:
            return i / sr
    return len(signal) / sr   # entire signal is silence


def analyse_audio(audio_path: Path) -> TTSQuality:
    """Run all objective checks on a WAV file and return a TTSQuality instance."""
    quality = TTSQuality()
    try:
        data, sr = sf.read(str(audio_path), dtype="int16", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]  # use first channel

        # RMS loudness
        quality.rms_dbfs = _rms_dbfs(data)

        # Clipping
        max_amplitude = np.max(np.abs(data))
        quality.clipping_detected = bool(max_amplitude >= int(_CLIPPING_THRESHOLD * 32767))

        # Silence at start/end
        quality.silence_at_start_s = _silence_duration(data, sr, from_start=True)
        quality.silence_at_end_s = _silence_duration(data, sr, from_start=False)

        # Abrupt cut: check if the last window has high amplitude
        window_samples = max(1, int(_ABRUPT_CUT_WINDOW_S * sr))
        if len(data) > window_samples * 2:
            last_rms = float(np.sqrt(np.mean(data[-window_samples:].astype(np.float64) ** 2)))
            mean_rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            quality.has_abrupt_cut = (
                mean_rms > 0 and (last_rms / mean_rms) > _ABRUPT_CUT_AMPLITUDE_RATIO
            )

    except Exception:
        pass  # Return empty quality on any read error

    return quality
