"""
scripts/generate_stt_audio.py
─────────────────────────────
Generates the 8 ground-truth WAV files required by the STT benchmark.

For each entry in dataset/stt/<lang>/manifest.json the script:
  1. Synthesises the `reference` text via OpenAI TTS (tts-1, alloy voice).
  2. Resamples to 16 kHz mono PCM — the format expected by all STT adapters.
  3. For entries with condition == "light_noise", adds synthetic Gaussian noise
     at -30 dBFS to simulate a light background environment.
  4. Saves the result as dataset/stt/<lang>/<audio_file>.

Usage
─────
    python scripts/generate_stt_audio.py
    python scripts/generate_stt_audio.py --lang es   # only Spanish
    python scripts/generate_stt_audio.py --dry-run   # show plan, no API calls

Requirements: package installed in editable mode (pip install -e ".[dev]")
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# ── repo root & env ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

DATASET_STT = REPO_ROOT / "dataset" / "stt"
TARGET_SR = 16_000          # Hz — required by all STT adapters
NOISE_DBFS = -30.0          # dBFS for light_noise condition
TTS_MODEL = "tts-1"         # cheaper model is fine for ground-truth audio
VOICE_MAP = {"es": "alloy", "en": "alloy"}   # neutral voice for both langs


# ── helpers ──────────────────────────────────────────────────────────────────

def _dbfs_to_linear(dbfs: float) -> float:
    return 10 ** (dbfs / 20.0)


def _add_noise(pcm: np.ndarray, noise_dbfs: float) -> np.ndarray:
    """Add Gaussian white noise at noise_dbfs level relative to signal RMS."""
    signal_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    noise_rms = signal_rms * _dbfs_to_linear(noise_dbfs)
    noise = np.random.default_rng(seed=42).normal(0.0, noise_rms, size=pcm.shape)
    mixed = pcm.astype(np.float64) + noise
    # Clip to int16 range
    mixed = np.clip(mixed, -32768, 32767)
    return mixed.astype(np.int16)


def _resample(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear interpolation resampler (no extra deps)."""
    if src_sr == dst_sr:
        return pcm
    duration = len(pcm) / src_sr
    n_samples = int(duration * dst_sr)
    old_indices = np.linspace(0, len(pcm) - 1, n_samples)
    return np.interp(old_indices, np.arange(len(pcm)), pcm.astype(np.float64)).astype(np.int16)


def _synthesise(text: str, lang: str, client) -> np.ndarray:
    """Call OpenAI TTS and return int16 PCM at the API's native sample rate (24kHz)."""
    voice = VOICE_MAP.get(lang, "alloy")
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="pcm",   # raw 16-bit little-endian PCM at 24kHz
    )
    raw = response.read()
    pcm = np.frombuffer(raw, dtype=np.int16)
    return pcm   # 24 000 Hz mono


def _process_entry(entry: dict, lang_dir: Path, client, dry_run: bool) -> bool:
    out_path = lang_dir / entry["audio_file"]
    label = f"  [{entry['id']}] {entry['condition']}"

    if out_path.exists():
        print(f"{label} → already exists, skipping")
        return True

    if dry_run:
        print(f"{label} → would synthesise: \"{entry['reference'][:60]}...\"")
        return True

    print(f"{label} → synthesising via OpenAI TTS …", end=" ", flush=True)
    try:
        pcm_24k = _synthesise(entry["reference"], lang_dir.name, client)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False

    # Resample 24kHz → 16kHz
    pcm_16k = _resample(pcm_24k, src_sr=24_000, dst_sr=TARGET_SR)

    # Add noise for noisy conditions
    if entry.get("condition", "clean") == "light_noise":
        pcm_16k = _add_noise(pcm_16k, NOISE_DBFS)
        print("(noise added)", end=" ", flush=True)

    # Save as WAV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm_16k, samplerate=TARGET_SR, subtype="PCM_16")
    duration = len(pcm_16k) / TARGET_SR
    print(f"saved ({duration:.1f}s) → {out_path.relative_to(REPO_ROOT)}")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate STT ground-truth WAV files.")
    parser.add_argument("--lang", choices=["es", "en"], default=None,
                        help="Generate only for this language (default: both).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making API calls.")
    args = parser.parse_args()

    # Import here so the script fails fast if the package is not installed
    try:
        from openai import OpenAI
        from tts_stt_benchmark import config as cfg
    except ImportError:
        print("ERROR: package not installed. Run: pip install -e '.[dev]'", file=sys.stderr)
        return 1

    if not args.dry_run:
        api_key = cfg.openai_api_key()
        client = OpenAI(api_key=api_key)
    else:
        client = None

    langs = [args.lang] if args.lang else ["es", "en"]
    total = ok = 0

    for lang in langs:
        manifest_path = DATASET_STT / lang / "manifest.json"
        if not manifest_path.exists():
            print(f"WARNING: manifest not found: {manifest_path}")
            continue

        with open(manifest_path, encoding="utf-8") as fh:
            entries = json.load(fh)

        print(f"\n── {lang.upper()} ({len(entries)} entries) ──────────────────────")
        for entry in entries:
            total += 1
            success = _process_entry(entry, manifest_path.parent, client, args.dry_run)
            if success:
                ok += 1

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Done: {ok}/{total} files ready.")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
