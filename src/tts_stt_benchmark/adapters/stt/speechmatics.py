"""
Speechmatics STT adapter – supports both batch and real-time (streaming) transcription.

Batch: uses the Speechmatics Batch ASR API.
Streaming: uses the Speechmatics RT WebSocket API.
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from typing import Optional

import soundfile as sf

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import STTAdapter
from tts_stt_benchmark.models import STTResult, STTLatency, STTQuality
from tts_stt_benchmark.metrics.wer_cer import compute_wer_cer

# Cost per hour of audio (USD) as of 2025-03
COST_PER_HOUR: dict[str, float] = {
    "batch": 1.50,       # Standard batch
    "enhanced": 2.40,    # Enhanced model
    "realtime": 1.80,    # Real-time
}

_LANGUAGE_MAP: dict[str, str] = {
    "en": "en",
    "es": "es",
}


class SpeechmaticsSTTAdapter(STTAdapter):
    provider = "speechmatics"
    supports_streaming = True

    def __init__(self, model: str = "default"):
        """
        Parameters
        ----------
        model : "default" | "enhanced"
        """
        self.model = model

    async def transcribe(
        self,
        audio_path: Path,
        language: str,
        mode: str = "batch",
        reference: str = "",
    ) -> STTResult:
        if mode == "streaming":
            return await self._transcribe_streaming(audio_path, language, reference)
        return await self._transcribe_batch(audio_path, language, reference)

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def _transcribe_batch(
        self, audio_path: Path, language: str, reference: str
    ) -> STTResult:
        import httpx

        latency = STTLatency()
        error: str | None = None
        transcript = ""
        quality = STTQuality()

        try:
            info = sf.info(str(audio_path))
            latency.audio_duration_s = info.duration
        except Exception:
            pass

        try:
            api_key = config.speechmatics_api_key()
            headers = {"Authorization": f"Bearer {api_key}"}

            transcription_config = {
                "language": _LANGUAGE_MAP.get(language, language),
                "operating_point": self.model if self.model != "default" else "standard",
                "diarization": "none",
            }

            async with httpx.AsyncClient(timeout=config.timeout_seconds()) as client:
                t_start = time.perf_counter()

                # 1. Submit job
                with open(audio_path, "rb") as af:
                    submit_resp = await client.post(
                        "https://asr.api.speechmatics.com/v2/jobs/",
                        headers=headers,
                        data={"config": __import__("json").dumps({
                            "type": "transcription",
                            "transcription_config": transcription_config,
                        })},
                        files={"data_file": (audio_path.name, af, "audio/wav")},
                    )
                submit_resp.raise_for_status()
                job_id = submit_resp.json()["id"]

                # 2. Poll until done
                while True:
                    status_resp = await client.get(
                        f"https://asr.api.speechmatics.com/v2/jobs/{job_id}",
                        headers=headers,
                    )
                    status_resp.raise_for_status()
                    status = status_resp.json()["job"]["status"]
                    if status == "done":
                        break
                    if status == "rejected":
                        raise RuntimeError(f"Speechmatics job {job_id} rejected")
                    await asyncio.sleep(1.0)

                latency.time_to_first_transcript_s = time.perf_counter() - t_start

                # 3. Retrieve transcript
                transcript_resp = await client.get(
                    f"https://asr.api.speechmatics.com/v2/jobs/{job_id}/transcript",
                    headers=headers,
                    params={"format": "txt"},
                )
                transcript_resp.raise_for_status()
                transcript = transcript_resp.text.strip()
                latency.total_transcription_s = time.perf_counter() - t_start

            if reference:
                wer, cer = compute_wer_cer(reference, transcript)
                quality = STTQuality(wer=wer, cer=cer, hypothesis=transcript, reference=reference)

        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        return STTResult(
            provider=self.provider,
            model=self.model,
            language=language,
            audio_id="",
            audio_duration_s=latency.audio_duration_s,
            repetition=0,
            latency=latency,
            quality=quality,
            transcript=transcript,
            error=error,
            mode="batch",
        )

    # ── Streaming (Real-Time WebSocket) ───────────────────────────────────────

    async def _transcribe_streaming(
        self, audio_path: Path, language: str, reference: str
    ) -> STTResult:
        """
        Uses the Speechmatics real-time WebSocket API to stream audio and
        capture the first partial transcript as TTFB metric.
        """
        latency = STTLatency()
        error: str | None = None
        transcript_parts: list[str] = []
        first_received = False

        try:
            info = sf.info(str(audio_path))
            latency.audio_duration_s = info.duration
        except Exception:
            pass

        try:
            from speechmatics.models import ConnectionSettings, TranscriptionConfig
            from speechmatics.client import WebsocketClient

            api_key = config.speechmatics_api_key()
            settings = ConnectionSettings(
                url="wss://eu2.rt.speechmatics.com/v2",
                auth_token=api_key,
            )
            transcription_config = TranscriptionConfig(
                language=_LANGUAGE_MAP.get(language, language),
                operating_point=self.model if self.model not in ("default", "enhanced") else "standard",
                enable_partials=True,
            )

            t_start = time.perf_counter()

            async with WebsocketClient(settings) as client:
                async def on_partial(msg):
                    nonlocal first_received
                    if not first_received:
                        latency.time_to_first_transcript_s = time.perf_counter() - t_start
                        first_received = True

                async def on_final(msg):
                    transcript_parts.append(msg["metadata"]["transcript"])

                client.add_event_handler("AddPartialTranscript", on_partial)
                client.add_event_handler("AddTranscript", on_final)

                with open(audio_path, "rb") as af:
                    await client.run(
                        af,
                        transcription_config,
                        audio_settings=None,
                    )

            latency.total_transcription_s = time.perf_counter() - t_start
            transcript = " ".join(transcript_parts).strip()

            quality = STTQuality()
            if reference:
                wer, cer = compute_wer_cer(reference, transcript)
                quality = STTQuality(wer=wer, cer=cer, hypothesis=transcript, reference=reference)

        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            transcript = ""
            quality = STTQuality()

        return STTResult(
            provider=self.provider,
            model=self.model,
            language=language,
            audio_id="",
            audio_duration_s=latency.audio_duration_s,
            repetition=0,
            latency=latency,
            quality=STTQuality() if error else quality,
            transcript=transcript,
            error=error,
            mode="streaming",
        )
