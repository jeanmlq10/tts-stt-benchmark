"""
OpenAI Whisper STT adapter.
Supports whisper-1 (standard) and whisper-1 with "mini" quality preset.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import soundfile as sf
from openai import AsyncOpenAI

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import STTAdapter
from tts_stt_benchmark.models import STTResult, STTLatency, STTQuality
from tts_stt_benchmark.metrics.wer_cer import compute_wer_cer

# Cost per minute of audio (USD) as of 2025-03
COST_PER_MINUTE: dict[str, float] = {
    "whisper-1": 0.006,
}


class OpenAISTTAdapter(STTAdapter):
    provider = "openai_whisper"
    supports_streaming = False   # Whisper REST API is batch-only

    def __init__(self, model: str = "whisper-1", quality: str = "standard"):
        """
        Parameters
        ----------
        model   : whisper-1
        quality : "standard" (default) | "mini"
                  "mini" uses a smaller response_format and prompt to reduce latency.
        """
        self.model = model
        self.quality = quality
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=config.openai_api_key())
        return self._client

    @property
    def provider_label(self) -> str:
        return f"{self.provider}_{self.quality}"

    async def transcribe(
        self,
        audio_path: Path,
        language: str,
        mode: str = "batch",
        reference: str = "",
    ) -> STTResult:
        latency = STTLatency()
        error: str | None = None
        transcript = ""
        quality = STTQuality()

        # Measure audio duration
        try:
            info = sf.info(str(audio_path))
            latency.audio_duration_s = info.duration
        except Exception:
            pass

        try:
            # "mini" variant: use text format (faster) and no prompt
            response_format = "text" if self.quality == "mini" else "verbose_json"

            with open(audio_path, "rb") as audio_file:
                t_start = time.perf_counter()

                if response_format == "text":
                    response = await self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language,
                        response_format="text",  # type: ignore[arg-type]
                    )
                    latency.time_to_first_transcript_s = time.perf_counter() - t_start
                    latency.total_transcription_s = latency.time_to_first_transcript_s
                    transcript = str(response).strip()
                else:
                    response = await self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language,
                        response_format="verbose_json",  # type: ignore[arg-type]
                        timestamp_granularities=["segment"],
                    )
                    latency.time_to_first_transcript_s = time.perf_counter() - t_start
                    latency.total_transcription_s = latency.time_to_first_transcript_s
                    transcript = response.text.strip()

            if reference:
                wer, cer = compute_wer_cer(reference, transcript)
                quality = STTQuality(wer=wer, cer=cer, hypothesis=transcript, reference=reference)

        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        return STTResult(
            provider=self.provider_label,
            model=self.model,
            language=language,
            audio_id="",       # filled in by runner
            audio_duration_s=latency.audio_duration_s,
            repetition=0,      # filled in by runner
            latency=latency,
            quality=quality,
            transcript=transcript,
            error=error,
            mode="batch",
        )
