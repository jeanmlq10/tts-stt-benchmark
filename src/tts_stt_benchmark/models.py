"""
Shared data models (dataclasses) for benchmark results.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ─── TTS ─────────────────────────────────────────────────────────────────────

@dataclass
class TTSLatency:
    """Raw latency measurements for a single TTS request."""
    time_to_first_byte_s: Optional[float] = None   # TTFB / time to first audio chunk
    time_to_first_chunk_s: Optional[float] = None  # alias kept for streaming providers
    total_synthesis_s: float = 0.0                  # wall-clock from request to last byte
    audio_duration_s: float = 0.0                   # duration of the synthesised audio


@dataclass
class TTSQuality:
    """Objective quality checks for a synthesised audio file."""
    clipping_detected: bool = False
    silence_at_start_s: float = 0.0
    silence_at_end_s: float = 0.0
    rms_dbfs: float = 0.0            # RMS loudness in dBFS
    has_abrupt_cut: bool = False      # detected hard cut at the end
    # MOS is filled in manually via the checklist; None = not evaluated
    mos_score: Optional[float] = None


@dataclass
class TTSResult:
    """Full result for one TTS provider / text combination."""
    provider: str
    model: str
    language: str
    text_id: str
    text_chars: int
    repetition: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency: TTSLatency = field(default_factory=TTSLatency)
    quality: TTSQuality = field(default_factory=TTSQuality)
    audio_path: Optional[str] = None
    error: Optional[str] = None
    streaming: bool = False


# ─── STT ─────────────────────────────────────────────────────────────────────

@dataclass
class STTLatency:
    """Raw latency measurements for a single STT request."""
    time_to_first_transcript_s: Optional[float] = None
    total_transcription_s: float = 0.0
    audio_duration_s: float = 0.0


@dataclass
class STTQuality:
    """Quality metrics for a transcription result."""
    wer: Optional[float] = None   # Word Error Rate  (0–1)
    cer: Optional[float] = None   # Character Error Rate (0–1)
    hypothesis: str = ""
    reference: str = ""


@dataclass
class STTResult:
    """Full result for one STT provider / audio combination."""
    provider: str
    model: str
    language: str
    audio_id: str
    audio_duration_s: float
    repetition: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency: STTLatency = field(default_factory=STTLatency)
    quality: STTQuality = field(default_factory=STTQuality)
    transcript: str = ""
    error: Optional[str] = None
    mode: str = "batch"   # "batch" | "streaming"


# ─── Aggregated stats ────────────────────────────────────────────────────────

@dataclass
class PercentileStats:
    p50: float
    p90: float
    mean: float
    min: float
    max: float
    n: int


# ─── Serialisation helpers ───────────────────────────────────────────────────

def result_to_dict(result: TTSResult | STTResult) -> dict:
    """Recursively convert a result dataclass to a plain dict."""
    return dataclasses.asdict(result)
