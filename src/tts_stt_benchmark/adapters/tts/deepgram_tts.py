"""
Deepgram Aura TTS adapter.
Supports Aura-2 models via the Deepgram REST API.
Streaming: Deepgram Aura supports chunked HTTP streaming.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import TTSAdapter
from tts_stt_benchmark.models import TTSResult, TTSLatency, TTSQuality
from tts_stt_benchmark.metrics.audio_checks import analyse_audio

_BASE_URL = "https://api.deepgram.com/v1/speak"

# Deepgram Aura voices with language affinity
_VOICE_MAP: dict[str, str] = {
    "en": "aura-2-thalia-en",
    "es": "aura-2-andromeda-en",  # closest; Aura-2 ES voices TBD
}

# Cost per 1k characters (USD) as of 2025-03 — Deepgram Pay-as-you-go
COST_PER_1K_CHARS: dict[str, float] = {
    "aura-2": 0.0150,   # $15 / 1M chars
}


class DeepgramTTSAdapter(TTSAdapter):
    provider = "deepgram"
    model = "aura-2"
    supports_streaming = True

    def __init__(self, model: str = "aura-2"):
        self.model = model

    async def synthesise(
        self,
        text: str,
        language: str,
        output_path: Path,
        streaming: bool = True,
        voice: str | None = None,
    ) -> TTSResult:
        voice = voice or _VOICE_MAP.get(language, "aura-2-thalia-en")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        latency = TTSLatency()
        error: str | None = None

        headers = {
            "Authorization": f"Token {config.deepgram_api_key()}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}
        params = {"model": voice, "encoding": "linear16", "sample_rate": "24000"}

        try:
            t_start = time.perf_counter()
            audio_bytes = bytearray()
            first_chunk_received = False

            async with httpx.AsyncClient(timeout=config.timeout_seconds()) as client:
                async with client.stream(
                    "POST",
                    _BASE_URL,
                    headers=headers,
                    json=payload,
                    params=params,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if not first_chunk_received and chunk:
                            latency.time_to_first_byte_s = time.perf_counter() - t_start
                            latency.time_to_first_chunk_s = latency.time_to_first_byte_s
                            first_chunk_received = True
                        audio_bytes.extend(chunk)

            latency.total_synthesis_s = time.perf_counter() - t_start

            # Deepgram returns raw PCM; wrap into WAV using soundfile
            import numpy as np
            pcm = np.frombuffer(bytes(audio_bytes), dtype=np.int16)
            sf.write(str(output_path), pcm, samplerate=24000, subtype="PCM_16")

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
            text_id="",
            text_chars=len(text),
            repetition=0,
            latency=latency,
            quality=quality,
            audio_path=str(output_path) if error is None else None,
            error=error,
            streaming=streaming,
        )
