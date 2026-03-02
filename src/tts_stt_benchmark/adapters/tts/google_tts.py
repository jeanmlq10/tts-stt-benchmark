"""
Google Cloud Gemini TTS adapter.
Uses the google-cloud-texttospeech SDK with the gemini-2.5-flash-preview-tts model
(or cloud text-to-speech Neural2 / Chirp as fallback).

Streaming: GCP TTS does not expose true streaming; we measure TTFB as the time
until the first byte is available in the response (non-streaming HTTP round-trip).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import soundfile as sf

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import TTSAdapter
from tts_stt_benchmark.models import TTSResult, TTSLatency, TTSQuality
from tts_stt_benchmark.metrics.audio_checks import analyse_audio

# Language → voice name mapping for Gemini-TTS / Neural2 fallback
_VOICE_MAP: dict[str, dict[str, str]] = {
    "gemini-2.5-flash-preview-tts": {
        "en": "en-US-Chirp3-HD-Charon",
        "es": "es-ES-Chirp3-HD-Charon",
    },
    "neural2": {
        "en": "en-US-Neural2-J",
        "es": "es-ES-Neural2-B",
    },
}

# Approximate cost per 1M characters (USD) for Neural2/Studio; Gemini-TTS pricing TBD
COST_PER_1M_CHARS: dict[str, float] = {
    "gemini-2.5-flash-preview-tts": 12.0,  # placeholder; check GCP pricing
    "neural2": 16.0,
}


class GoogleTTSAdapter(TTSAdapter):
    provider = "google"
    supports_streaming = False   # GCP TTS is batch-only

    def __init__(self, model: str = "gemini-2.5-flash-preview-tts"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Import here to allow the module to load without credentials
            from google.cloud import texttospeech_v1beta1 as tts  # type: ignore[import-untyped]
            self._client = tts.TextToSpeechAsyncClient()
        return self._client

    async def synthesise(
        self,
        text: str,
        language: str,
        output_path: Path,
        streaming: bool = False,   # GCP doesn't support streaming
        voice: str | None = None,
    ) -> TTSResult:
        from google.cloud import texttospeech_v1beta1 as tts  # type: ignore[import-untyped]

        voice_name = voice or _VOICE_MAP.get(self.model, _VOICE_MAP["neural2"]).get(
            language, "en-US-Neural2-J"
        )
        lang_code = f"{language}-ES" if language == "es" else f"{language}-US"
        # Fine-tune lang_code from voice name if available
        if "-" in voice_name:
            lang_code = "-".join(voice_name.split("-")[:2])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        latency = TTSLatency()
        error: str | None = None

        try:
            synthesis_input = tts.SynthesisInput(text=text)
            voice_params = tts.VoiceSelectionParams(
                language_code=lang_code,
                name=voice_name,
            )
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            )

            t_start = time.perf_counter()
            response = await self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
            # For GCP non-streaming: TTFB ≈ total time (no chunked delivery)
            latency.time_to_first_byte_s = time.perf_counter() - t_start
            latency.total_synthesis_s = latency.time_to_first_byte_s

            import numpy as np
            pcm = np.frombuffer(response.audio_content, dtype=np.int16)
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
            streaming=False,
        )
