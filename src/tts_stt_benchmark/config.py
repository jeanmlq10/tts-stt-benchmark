"""
Central configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)


def _get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value


# ── Provider keys (lazy – only required when the provider is used) ───────────

def openai_api_key() -> str:
    return _require("OPENAI_API_KEY")


def deepgram_api_key() -> str:
    return _require("DEEPGRAM_API_KEY")


def google_credentials_path() -> str | None:
    return _get("GOOGLE_APPLICATION_CREDENTIALS")


def speechmatics_api_key() -> str:
    return _require("SPEECHMATICS_API_KEY")


# ── Benchmark settings ───────────────────────────────────────────────────────

def repetitions() -> int:
    return int(_get("BENCHMARK_REPETITIONS", "5"))


def output_dir() -> Path:
    d = Path(_get("BENCHMARK_OUTPUT_DIR", "results"))
    if not d.is_absolute():
        d = _REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def timeout_seconds() -> float:
    return float(_get("BENCHMARK_TIMEOUT_SECONDS", "120"))
