"""
CLI: run_stt

Usage examples
--------------
# Run all STT providers on all ES/EN audio files, batch mode, 5 repetitions
run_stt --provider speechmatics --provider openai_whisper_standard \
        --provider openai_whisper_mini \
        --lang es --lang en --mode batch --repetitions 5

# Single provider, streaming
run_stt --provider speechmatics --lang en --mode streaming
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console

from tts_stt_benchmark import config
from tts_stt_benchmark.runner import run_stt_benchmark

console = Console()

_PROVIDER_MAP = {
    "speechmatics": (
        "tts_stt_benchmark.adapters.stt.speechmatics.SpeechmaticsSTTAdapter",
        {"model": "default"},
    ),
    "openai_whisper_standard": (
        "tts_stt_benchmark.adapters.stt.openai_whisper.OpenAISTTAdapter",
        {"model": "whisper-1", "quality": "standard"},
    ),
    "openai_whisper_mini": (
        "tts_stt_benchmark.adapters.stt.openai_whisper.OpenAISTTAdapter",
        {"model": "whisper-1", "quality": "mini"},
    ),
}


def _load_adapter(provider: str):
    import importlib
    fqn, kwargs = _PROVIDER_MAP[provider]
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _load_manifests(langs: tuple[str, ...], audio_base_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for lang in langs:
        manifest_path = audio_base_dir / lang / "manifest.json"
        if not manifest_path.exists():
            console.print(f"[yellow]Warning: manifest not found: {manifest_path}[/yellow]")
            continue
        with open(manifest_path, encoding="utf-8") as fh:
            entries.extend(json.load(fh))
    return entries


@click.command("run_stt")
@click.option(
    "--provider", "-p",
    multiple=True,
    type=click.Choice(list(_PROVIDER_MAP.keys())),
    default=list(_PROVIDER_MAP.keys()),
    show_default=True,
    help="STT provider(s) to benchmark.",
)
@click.option(
    "--lang", "-l",
    multiple=True,
    default=("es", "en"),
    show_default=True,
    help="Language(s) to test.",
)
@click.option(
    "--audio_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Base directory containing <lang>/manifest.json and audio files.",
)
@click.option(
    "--mode",
    type=click.Choice(["batch", "streaming"]),
    default="batch",
    show_default=True,
    help="Transcription mode.",
)
@click.option(
    "--repetitions", "-n",
    default=None,
    type=int,
    help="Number of repetitions per test case.",
)
@click.option(
    "--output_dir", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for results.",
)
def main(
    provider: tuple[str, ...],
    lang: tuple[str, ...],
    audio_dir: Path | None,
    mode: str,
    repetitions: int | None,
    output_dir: Path | None,
):
    """Run STT benchmark across selected providers, languages, and audio files."""
    reps = repetitions or config.repetitions()
    out = output_dir or config.output_dir()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    base_dir = audio_dir or (repo_root / "dataset" / "stt")

    entries = _load_manifests(lang, base_dir)
    if not entries:
        console.print("[red]No audio entries found. Check --lang and --audio_dir options.[/red]")
        raise SystemExit(1)

    adapters = [_load_adapter(p) for p in provider]

    console.print(
        f"[bold]STT Benchmark[/bold]: {len(adapters)} provider(s), "
        f"{len(entries)} audio file(s), {reps} repetition(s), mode={mode}"
    )

    asyncio.run(
        run_stt_benchmark(
            adapters=adapters,
            audio_entries=entries,
            audio_base_dir=base_dir,
            repetitions=reps,
            output_dir=out,
            mode=mode,
        )
    )
