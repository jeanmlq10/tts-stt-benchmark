"""
Core benchmark runner: iterates over providers × test cases × repetitions,
collects results, and writes incremental JSONL + final JSON/CSV.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from tts_stt_benchmark import config
from tts_stt_benchmark.adapters import TTSAdapter, STTAdapter
from tts_stt_benchmark.models import TTSResult, STTResult
from tts_stt_benchmark.storage import append_result_json, save_results_json, save_results_csv

logger = logging.getLogger(__name__)
console = Console()


# ─── TTS runner ──────────────────────────────────────────────────────────────

async def run_tts_benchmark(
    adapters: Sequence[TTSAdapter],
    texts: list[dict],
    repetitions: int,
    output_dir: Path,
    streaming: bool = True,
) -> list[TTSResult]:
    """
    Run all TTS adapters over all texts for N repetitions.

    Parameters
    ----------
    adapters     : list of TTSAdapter instances
    texts        : list of dicts from texts.json (keys: id, language, text, ...)
    repetitions  : number of repetitions per (adapter, text) pair
    output_dir   : base directory for audio output and results
    streaming    : whether to request streaming from adapters that support it
    """
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    results_dir = output_dir / "tts" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "results.jsonl"

    all_results: list[TTSResult] = []
    total = len(adapters) * len(texts) * repetitions

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running TTS benchmark…", total=total)

        for adapter in adapters:
            for text_entry in texts:
                for rep in range(1, repetitions + 1):
                    text_id = text_entry["id"]
                    language = text_entry["language"]
                    text = text_entry["text"]

                    audio_out = (
                        results_dir
                        / adapter.provider
                        / adapter.model
                        / language
                        / f"{text_id}_rep{rep:02d}.wav"
                    )

                    progress.update(
                        task,
                        description=(
                            f"TTS {adapter.provider}/{adapter.model} "
                            f"{language} {text_id} rep{rep}"
                        ),
                    )

                    try:
                        result = await asyncio.wait_for(
                            adapter.synthesise(
                                text=text,
                                language=language,
                                output_path=audio_out,
                                streaming=streaming and adapter.supports_streaming,
                            ),
                            timeout=config.timeout_seconds(),
                        )
                    except asyncio.TimeoutError:
                        from tts_stt_benchmark.models import TTSLatency, TTSQuality
                        result = TTSResult(
                            provider=adapter.provider,
                            model=adapter.model,
                            language=language,
                            text_id=text_id,
                            text_chars=len(text),
                            repetition=rep,
                            error="TimeoutError",
                            latency=TTSLatency(),
                            quality=TTSQuality(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        from tts_stt_benchmark.models import TTSLatency, TTSQuality
                        result = TTSResult(
                            provider=adapter.provider,
                            model=adapter.model,
                            language=language,
                            text_id=text_id,
                            text_chars=len(text),
                            repetition=rep,
                            error=str(exc),
                            latency=TTSLatency(),
                            quality=TTSQuality(),
                        )

                    result.text_id = text_id
                    result.repetition = rep
                    all_results.append(result)
                    append_result_json(result, jsonl_path)
                    progress.advance(task)

    # Save consolidated results
    save_results_json(all_results, results_dir / "results.json")
    save_results_csv(all_results, results_dir / "results.csv")
    console.print(f"[green]TTS results saved to {results_dir}[/green]")
    return all_results


# ─── STT runner ──────────────────────────────────────────────────────────────

async def run_stt_benchmark(
    adapters: Sequence[STTAdapter],
    audio_entries: list[dict],
    audio_base_dir: Path,
    repetitions: int,
    output_dir: Path,
    mode: str = "batch",
) -> list[STTResult]:
    """
    Run all STT adapters over all audio entries for N repetitions.

    Parameters
    ----------
    adapters        : list of STTAdapter instances
    audio_entries   : list of dicts from manifest.json
    audio_base_dir  : directory containing the audio files
    repetitions     : number of repetitions
    output_dir      : base directory for results
    mode            : "batch" | "streaming"
    """
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    results_dir = output_dir / "stt" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "results.jsonl"

    all_results: list[STTResult] = []
    total = len(adapters) * len(audio_entries) * repetitions

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running STT benchmark…", total=total)

        for adapter in adapters:
            for entry in audio_entries:
                audio_id = entry["id"]
                language = entry["language"]
                audio_file = audio_base_dir / entry["language"] / entry["audio_file"]
                reference = entry.get("reference", "")

                if not audio_file.exists():
                    console.print(
                        f"[yellow]Skipping {audio_id}: file not found at {audio_file}[/yellow]"
                    )
                    progress.advance(task, repetitions)
                    continue

                for rep in range(1, repetitions + 1):
                    progress.update(
                        task,
                        description=(
                            f"STT {adapter.provider}/{adapter.model} "
                            f"{language} {audio_id} rep{rep}"
                        ),
                    )

                    effective_mode = mode if adapter.supports_streaming else "batch"

                    try:
                        result = await asyncio.wait_for(
                            adapter.transcribe(
                                audio_path=audio_file,
                                language=language,
                                mode=effective_mode,
                                reference=reference,  # type: ignore[call-arg]
                            ),
                            timeout=config.timeout_seconds(),
                        )
                    except asyncio.TimeoutError:
                        from tts_stt_benchmark.models import STTLatency, STTQuality
                        result = STTResult(
                            provider=adapter.provider,
                            model=adapter.model,
                            language=language,
                            audio_id=audio_id,
                            audio_duration_s=entry.get("duration_s", 0.0),
                            repetition=rep,
                            error="TimeoutError",
                            latency=STTLatency(),
                            quality=STTQuality(),
                            mode=effective_mode,
                        )
                    except Exception as exc:  # noqa: BLE001
                        from tts_stt_benchmark.models import STTLatency, STTQuality
                        result = STTResult(
                            provider=adapter.provider,
                            model=adapter.model,
                            language=language,
                            audio_id=audio_id,
                            audio_duration_s=entry.get("duration_s", 0.0),
                            repetition=rep,
                            error=str(exc),
                            latency=STTLatency(),
                            quality=STTQuality(),
                            mode=effective_mode,
                        )

                    result.audio_id = audio_id
                    result.repetition = rep
                    all_results.append(result)
                    append_result_json(result, jsonl_path)
                    progress.advance(task)

    save_results_json(all_results, results_dir / "results.json")
    save_results_csv(all_results, results_dir / "results.csv")
    console.print(f"[green]STT results saved to {results_dir}[/green]")
    return all_results
