"""
CLI: run_tts

Usage examples
--------------
# Run all TTS providers with all ES/EN texts, streaming, 5 repetitions
run_tts --provider openai --provider deepgram --provider google \
        --lang es --lang en --streaming --repetitions 5

# Single provider, single language, specific text file
run_tts --provider openai --lang en \
        --text_file dataset/tts/en/texts.json --model tts-1-hd
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console

from tts_stt_benchmark import config
from tts_stt_benchmark.runner import run_tts_benchmark

console = Console()

_PROVIDER_MAP = {
    "openai": "tts_stt_benchmark.adapters.tts.openai_tts.OpenAITTSAdapter",
    "deepgram": "tts_stt_benchmark.adapters.tts.deepgram_tts.DeepgramTTSAdapter",
    "google": "tts_stt_benchmark.adapters.tts.google_tts.GoogleTTSAdapter",
}

_DEFAULT_MODELS = {
    "openai": "tts-1-hd",
    "deepgram": "aura-2",
    "google": "gemini-2.5-flash-preview-tts",
}


def _load_adapter(provider: str, model: str | None):
    """Dynamically import and instantiate the adapter class.

    Returns None (and prints a warning) if the provider cannot be used due to
    missing credentials, so the benchmark can continue with the remaining providers.
    """
    import importlib
    from tts_stt_benchmark import config as _cfg

    # Fast-fail check for Google before attempting to import the heavy SDK
    if provider == "google" and not _cfg.google_credentials_available():
        console.print(
            "[yellow]⚠  Google TTS skipped: GOOGLE_APPLICATION_CREDENTIALS is not set "
            "or the file does not exist.[/yellow]"
        )
        return None

    fqn = _PROVIDER_MAP[provider]
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    m = model or _DEFAULT_MODELS[provider]
    return cls(model=m)


def _load_texts(text_files: list[Path]) -> list[dict]:
    texts: list[dict] = []
    for tf in text_files:
        with open(tf, encoding="utf-8") as fh:
            data = json.load(fh)
        texts.extend(data)
    return texts


@click.command("run_tts")
@click.option(
    "--provider", "-p",
    multiple=True,
    type=click.Choice(list(_PROVIDER_MAP.keys())),
    default=list(_PROVIDER_MAP.keys()),
    show_default=True,
    help="TTS provider(s) to benchmark.",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Override model name (applies to all selected providers).",
)
@click.option(
    "--lang", "-l",
    multiple=True,
    default=("es", "en"),
    show_default=True,
    help="Language(s) to test.",
)
@click.option(
    "--text_file",
    multiple=True,
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path(s) to a texts JSON file. "
        "Defaults to dataset/tts/<lang>/texts.json for each selected language."
    ),
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    show_default=True,
    help="Use streaming synthesis where supported.",
)
@click.option(
    "--repetitions", "-n",
    default=None,
    type=int,
    help="Number of repetitions per test case (default from env/config).",
)
@click.option(
    "--output_dir", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for results (default from env/config).",
)
def main(
    provider: tuple[str, ...],
    model: str | None,
    lang: tuple[str, ...],
    text_file: tuple[Path, ...],
    streaming: bool,
    repetitions: int | None,
    output_dir: Path | None,
):
    """Run TTS benchmark across selected providers, languages, and texts."""
    reps = repetitions or config.repetitions()
    out = output_dir or config.output_dir()

    # Resolve text files
    if text_file:
        text_files = list(text_file)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        text_files = []
        for l in lang:
            candidate = repo_root / "dataset" / "tts" / l / "texts.json"
            if candidate.exists():
                text_files.append(candidate)
            else:
                console.print(f"[yellow]Warning: text file not found: {candidate}[/yellow]")

    texts = _load_texts(text_files)
    # Filter by selected languages
    texts = [t for t in texts if t["language"] in lang]

    if not texts:
        console.print("[red]No texts found. Check --lang and --text_file options.[/red]")
        raise SystemExit(1)

    adapters = [a for p in provider if (a := _load_adapter(p, model)) is not None]

    if not adapters:
        console.print("[red]No TTS adapters available. Check your API keys and credentials.[/red]")
        raise SystemExit(1)

    provider_names = ", ".join(a.provider for a in adapters)
    console.print(
        f"[bold]TTS Benchmark[/bold]: {len(adapters)} provider(s) [{provider_names}], "
        f"{len(texts)} text(s), {reps} repetition(s), streaming={streaming}"
    )

    asyncio.run(
        run_tts_benchmark(
            adapters=adapters,
            texts=texts,
            repetitions=reps,
            output_dir=out,
            streaming=streaming,
        )
    )
