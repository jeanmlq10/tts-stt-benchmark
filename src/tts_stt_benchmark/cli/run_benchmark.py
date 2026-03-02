"""
CLI: run_benchmark

Convenience command that runs both TTS and STT benchmarks in sequence.

Usage
-----
run_benchmark --tts --stt --lang es --lang en --repetitions 5
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console

from tts_stt_benchmark import config

console = Console()


@click.command("run_benchmark")
@click.option("--tts/--no-tts", default=True, help="Run TTS benchmark.")
@click.option("--stt/--no-stt", default=True, help="Run STT benchmark.")
@click.option("--lang", "-l", multiple=True, default=("es", "en"), show_default=True)
@click.option(
    "--tts_providers",
    multiple=True,
    default=("openai", "deepgram", "google"),
    type=click.Choice(["openai", "deepgram", "google"]),
    show_default=True,
)
@click.option(
    "--stt_providers",
    multiple=True,
    default=("speechmatics", "openai_whisper_standard", "openai_whisper_mini"),
    type=click.Choice(["speechmatics", "openai_whisper_standard", "openai_whisper_mini"]),
    show_default=True,
)
@click.option("--streaming/--no-streaming", default=True, show_default=True)
@click.option("--stt_mode", type=click.Choice(["batch", "streaming"]), default="batch", show_default=True)
@click.option("--repetitions", "-n", default=None, type=int)
@click.option("--output_dir", "-o", default=None, type=click.Path(path_type=Path))
def main(
    tts: bool,
    stt: bool,
    lang: tuple[str, ...],
    tts_providers: tuple[str, ...],
    stt_providers: tuple[str, ...],
    streaming: bool,
    stt_mode: str,
    repetitions: int | None,
    output_dir: Path | None,
):
    """Run TTS and/or STT benchmarks and then generate a report."""
    from tts_stt_benchmark.cli.run_tts import main as tts_main
    from tts_stt_benchmark.cli.run_stt import main as stt_main
    from tts_stt_benchmark.cli.generate_report import main as report_main

    ctx = click.get_current_context()

    if tts:
        ctx.invoke(
            tts_main,
            provider=tts_providers,
            model=None,
            lang=lang,
            text_file=(),
            streaming=streaming,
            repetitions=repetitions,
            output_dir=output_dir,
        )

    if stt:
        ctx.invoke(
            stt_main,
            provider=stt_providers,
            lang=lang,
            audio_dir=None,
            mode=stt_mode,
            repetitions=repetitions,
            output_dir=output_dir,
        )

    ctx.invoke(report_main, results_dir=output_dir or config.output_dir())
