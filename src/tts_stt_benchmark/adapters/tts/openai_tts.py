"""
OpenAI TTS adapter – supports tts-1, tts-1-hd, gpt-4o-mini-tts.
Measures TTFB via streaming and total synthesis time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import numpy as np

from openai import AsyncOpenAI

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import TTSAdapter
from tts_stt_benchmark.models import TTSResult, TTSLatency, TTSQuality
from tts_stt_benchmark.metrics.audio_checks import analyse_audio


# Mapping: language code → recommended voice
_VOICE_MAP: dict[str, str] = {
    "es": "alloy",
    "en": "nova",
}

# Cost per 1 000 000 characters (USD) as of 2025-03
COST_PER_1M_CHARS: dict[str, float] = {
    "tts-1": 15.0,
    "tts-1-hd": 30.0,
    "gpt-4o-mini-tts": 12.0,
}


class OpenAITTSAdapter(TTSAdapter):
    provider = "openai"
    supports_streaming = True

    def __init__(self, model: str = "tts-1-hd"):
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=config.openai_api_key())
        return self._client

    async def synthesise(
        self,
        text: str,
        language: str,
        output_path: Path,
        streaming: bool = True,
        voice: str | None = None,
    ) -> TTSResult:
        voice = voice or _VOICE_MAP.get(language, "alloy")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        latency = TTSLatency()
        error: str | None = None

        try:
            t_start = time.perf_counter()

            if streaming:
                first_chunk_received = False
                audio_bytes = bytearray()

                async with self.client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=voice,          # type: ignore[arg-type]
                    input=text,
                    response_format="wav",
                ) as response:
                    async for chunk in response.iter_bytes(chunk_size=4096):
                        if not first_chunk_received and chunk:
                            latency.time_to_first_byte_s = time.perf_counter() - t_start
                            latency.time_to_first_chunk_s = latency.time_to_first_byte_s
                            first_chunk_received = True
                        audio_bytes.extend(chunk)

                output_path.write_bytes(bytes(audio_bytes))
            else:
                response = await self.client.audio.speech.create(
                    model=self.model,
                    voice=voice,          # type: ignore[arg-type]
                    input=text,
                    response_format="wav",
                )
                output_path.write_bytes(response.content)

            latency.total_synthesis_s = time.perf_counter() - t_start

            # Measure audio duration
            info = sf.info(str(output_path))
            latency.audio_duration_s = info.duration

            quality = analyse_audio(output_path)

        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            quality = TTSQuality()

        return TTSResult(
            provider=self.provider,
            model=self.model,
            language=language,
            text_id="",          # filled in by runner
            text_chars=len(text),
            repetition=0,        # filled in by runner
            latency=latency,
            quality=quality,
            audio_path=str(output_path) if error is None else None,
            error=error,
            streaming=streaming,
        )
