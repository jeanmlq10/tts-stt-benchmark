"""
Abstract base classes for TTS and STT adapters.
All provider adapters must inherit from these classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from tts_stt_benchmark.models import TTSResult, STTResult


class TTSAdapter(ABC):
    provider: str = ""
    model: str = ""
    supports_streaming: bool = False

    @abstractmethod
    async def synthesise(
        self,
        text: str,
        language: str,
        output_path: Path,
        streaming: bool = False,
        voice: str | None = None,
    ) -> TTSResult:
        """Synthesise speech and return a TTSResult with latency and quality."""
        ...


class STTAdapter(ABC):
    provider: str = ""
    model: str = ""
    supports_streaming: bool = False

    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        language: str,
        mode: str = "batch",
        reference: str = "",
    ) -> STTResult:
        """Transcribe audio and return an STTResult with latency and quality."""
        ...
