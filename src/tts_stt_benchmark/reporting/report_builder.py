"""
Report generator: reads all results JSON files under a results directory,
aggregates statistics, and produces a Markdown report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from tts_stt_benchmark.metrics.stats import compute_stats_dict


# ─── Cost tables (USD) ───────────────────────────────────────────────────────

TTS_COST_PER_1M_CHARS = {
    "openai/tts-1": 15.0,
    "openai/tts-1-hd": 30.0,
    "openai/gpt-4o-mini-tts": 12.0,
    "deepgram/aura-2": 15.0,
    "google/gemini-2.5-flash-preview-tts": 12.0,
    "google/neural2": 16.0,
}

STT_COST_PER_MINUTE = {
    "openai_whisper_standard/whisper-1": 0.006,
    "openai_whisper_mini/whisper-1": 0.006,
    "speechmatics/default": 0.025,     # $1.50/hr ÷ 60
    "speechmatics/enhanced": 0.040,    # $2.40/hr ÷ 60
}


def _fmt(value: float | None, decimals: int = 3) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.{decimals}f}"


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value * 100:.1f}%"


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_all_results(results_dir: Path, kind: str) -> list[dict]:
    """Load all results.json files under results_dir/kind/*/results.json."""
    rows: list[dict] = []
    pattern = results_dir / kind / "**" / "results.json"
    for path in sorted(results_dir.glob(f"{kind}/**/results.json")):
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    return rows


# ─── TTS aggregation ─────────────────────────────────────────────────────────

def _aggregate_tts(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # Derived columns
    df["provider_model"] = df["provider"] + "/" + df["model"]
    df["ttfb"] = df.get("latency.time_to_first_byte_s", pd.Series([None] * len(df)))
    df["total"] = df.get("latency.total_synthesis_s", pd.Series([None] * len(df)))
    df["audio_dur"] = df.get("latency.audio_duration_s", pd.Series([None] * len(df)))
    df["rms_dbfs"] = df.get("quality.rms_dbfs", pd.Series([None] * len(df)))
    df["clipping"] = df.get("quality.clipping_detected", pd.Series([False] * len(df)))
    df["abrupt_cut"] = df.get("quality.has_abrupt_cut", pd.Series([False] * len(df)))

    # Group by provider_model + language + text_id
    groups = df.groupby(["provider_model", "language", "text_id"])

    summary_rows: list[dict] = []
    for (pm, lang, tid), grp in groups:
        ttfb_stats = compute_stats_dict(grp["ttfb"].tolist())
        total_stats = compute_stats_dict(grp["total"].tolist())
        clipping_count = int(grp["clipping"].sum())
        abrupt_count = int(grp["abrupt_cut"].sum())
        n = len(grp)
        # Cost estimate (per call, based on char count)
        chars = grp["text_chars"].iloc[0] if "text_chars" in grp.columns else 0
        cost_per_1m = TTS_COST_PER_1M_CHARS.get(pm, 0.0)
        cost_per_call = (chars / 1_000_000) * cost_per_1m

        summary_rows.append({
            "provider": pm,
            "lang": lang,
            "text_id": tid,
            "n": n,
            "ttfb_p50_s": ttfb_stats.p50,
            "ttfb_p90_s": ttfb_stats.p90,
            "total_p50_s": total_stats.p50,
            "total_p90_s": total_stats.p90,
            "clipping_pct": clipping_count / n if n else 0,
            "abrupt_cut_pct": abrupt_count / n if n else 0,
            "chars": chars,
            "cost_per_call_usd": cost_per_call,
            "errors": int(grp["error"].notna().sum()) if "error" in grp.columns else 0,
        })

    return pd.DataFrame(summary_rows)


# ─── STT aggregation ─────────────────────────────────────────────────────────

def _aggregate_stt(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    df["provider_model"] = df["provider"] + "/" + df["model"]
    df["ttft"] = df.get("latency.time_to_first_transcript_s", pd.Series([None] * len(df)))
    df["total"] = df.get("latency.total_transcription_s", pd.Series([None] * len(df)))
    df["audio_dur"] = df.get("latency.audio_duration_s", pd.Series([None] * len(df)))
    df["wer"] = df.get("quality.wer", pd.Series([None] * len(df)))
    df["cer"] = df.get("quality.cer", pd.Series([None] * len(df)))

    groups = df.groupby(["provider_model", "language", "audio_id"])

    summary_rows: list[dict] = []
    for (pm, lang, aid), grp in groups:
        ttft_stats = compute_stats_dict(grp["ttft"].tolist())
        total_stats = compute_stats_dict(grp["total"].tolist())
        wer_mean = grp["wer"].dropna().mean() if not grp["wer"].dropna().empty else float("nan")
        cer_mean = grp["cer"].dropna().mean() if not grp["cer"].dropna().empty else float("nan")
        n = len(grp)
        audio_dur_s = grp["audio_dur"].iloc[0] if "audio_dur" in grp.columns else 0
        cost_per_min = STT_COST_PER_MINUTE.get(pm, 0.0)
        cost_per_call = (audio_dur_s / 60.0) * cost_per_min

        summary_rows.append({
            "provider": pm,
            "lang": lang,
            "audio_id": aid,
            "n": n,
            "ttft_p50_s": ttft_stats.p50,
            "ttft_p90_s": ttft_stats.p90,
            "total_p50_s": total_stats.p50,
            "total_p90_s": total_stats.p90,
            "wer_mean": wer_mean,
            "cer_mean": cer_mean,
            "audio_dur_s": audio_dur_s,
            "cost_per_call_usd": cost_per_call,
            "errors": int(grp["error"].notna().sum()) if "error" in grp.columns else 0,
        })

    return pd.DataFrame(summary_rows)


# ─── Markdown builder ────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._\n"
    return df.to_markdown(index=False, floatfmt=".4f")


def build_report(results_dir: Path) -> str:
    tts_rows = _load_all_results(results_dir, "tts")
    stt_rows = _load_all_results(results_dir, "stt")

    tts_df = _aggregate_tts(tts_rows)
    stt_df = _aggregate_stt(stt_rows)

    lines: list[str] = []

    lines += [
        "# TTS / STT Benchmark Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Results directory:** `{results_dir}`  ",
        f"**TTS samples:** {len(tts_rows)} · **STT samples:** {len(stt_rows)}",
        "",
        "---",
        "",
    ]

    # ── TTS Section ──────────────────────────────────────────────────────────
    lines += ["## 1. TTS Results", ""]

    if not tts_df.empty:
        # Overview table: per provider × language
        overview_cols = [
            "provider", "lang", "n",
            "ttfb_p50_s", "ttfb_p90_s",
            "total_p50_s", "total_p90_s",
            "clipping_pct", "abrupt_cut_pct",
            "cost_per_call_usd", "errors",
        ]
        overview = tts_df[overview_cols].groupby(["provider", "lang"]).agg({
            "n": "sum",
            "ttfb_p50_s": "median",
            "ttfb_p90_s": "median",
            "total_p50_s": "median",
            "total_p90_s": "median",
            "clipping_pct": "mean",
            "abrupt_cut_pct": "mean",
            "cost_per_call_usd": "mean",
            "errors": "sum",
        }).reset_index()

        lines += [
            "### 1.1 Latency & Cost Overview (per provider × language)",
            "",
            "> TTFB = Time to First Byte/Chunk · Total = Wall-clock synthesis time · "
            "Cost = per-call estimate based on avg text length",
            "",
            _df_to_md(overview),
            "",
        ]

        # Detail per text
        detail_cols = [
            "provider", "lang", "text_id", "chars",
            "ttfb_p50_s", "ttfb_p90_s", "total_p50_s", "total_p90_s",
        ]
        lines += [
            "### 1.2 Latency Detail (per text)",
            "",
            _df_to_md(tts_df[detail_cols]),
            "",
        ]

        # Quality
        quality_cols = [
            "provider", "lang", "text_id",
            "clipping_pct", "abrupt_cut_pct",
        ]
        lines += [
            "### 1.3 Quality Checks",
            "",
            _df_to_md(tts_df[quality_cols]),
            "",
        ]

        # Cost projection
        lines += [
            "### 1.4 Cost Projection",
            "",
            "| Provider | Cost per 1M chars (USD) | Projected cost / 1k calls @ 200 chars avg |",
            "|---|---|---|",
        ]
        for pm, cost in TTS_COST_PER_1M_CHARS.items():
            proj = (200 / 1_000_000) * cost * 1000
            lines.append(f"| {pm} | ${cost:.2f} | ${proj:.4f} |")
        lines.append("")
    else:
        lines += ["_No TTS results found._", ""]

    # ── MOS Checklist ────────────────────────────────────────────────────────
    lines += [
        "### 1.5 MOS / Subjective Quality Checklist",
        "",
        "Rate each provider on a 1–5 scale after listening to the generated samples.",
        "",
        "| Provider | Model | Lang | Naturalness (1–5) | Intelligibility (1–5) "
        "| Prosody (1–5) | Accent (1–5) | Notes |",
        "|---|---|---|---|---|---|---|---|",
        "| openai | tts-1-hd | es | | | | | |",
        "| openai | tts-1-hd | en | | | | | |",
        "| deepgram | aura-2 | es | | | | | |",
        "| deepgram | aura-2 | en | | | | | |",
        "| google | gemini-2.5-flash | es | | | | | |",
        "| google | gemini-2.5-flash | en | | | | | |",
        "",
    ]

    # ── STT Section ──────────────────────────────────────────────────────────
    lines += ["## 2. STT Results", ""]

    if not stt_df.empty:
        overview_cols = [
            "provider", "lang", "n",
            "ttft_p50_s", "ttft_p90_s",
            "total_p50_s", "total_p90_s",
            "wer_mean", "cer_mean",
            "cost_per_call_usd", "errors",
        ]
        overview = stt_df[overview_cols].groupby(["provider", "lang"]).agg({
            "n": "sum",
            "ttft_p50_s": "median",
            "ttft_p90_s": "median",
            "total_p50_s": "median",
            "total_p90_s": "median",
            "wer_mean": "mean",
            "cer_mean": "mean",
            "cost_per_call_usd": "mean",
            "errors": "sum",
        }).reset_index()

        lines += [
            "### 2.1 Latency & Quality Overview (per provider × language)",
            "",
            "> TTFT = Time to First Transcript · WER/CER = against ground-truth transcripts",
            "",
            _df_to_md(overview),
            "",
        ]

        detail_cols = [
            "provider", "lang", "audio_id", "audio_dur_s",
            "ttft_p50_s", "ttft_p90_s", "total_p50_s", "total_p90_s",
            "wer_mean", "cer_mean",
        ]
        lines += [
            "### 2.2 Detail per Audio File",
            "",
            _df_to_md(stt_df[detail_cols]),
            "",
        ]

        # Cost projection
        lines += [
            "### 2.3 Cost Projection",
            "",
            "| Provider | Cost per minute (USD) | Cost per hour | Projected / 1k min |",
            "|---|---|---|---|",
        ]
        for pm, cpm in STT_COST_PER_MINUTE.items():
            lines.append(f"| {pm} | ${cpm:.4f} | ${cpm * 60:.2f} | ${cpm * 1000:.2f} |")
        lines.append("")
    else:
        lines += ["_No STT results found._", ""]

    # ── Recommendations ──────────────────────────────────────────────────────
    lines += [
        "## 3. Executive Summary & Recommendations",
        "",
        "### 3.1 Key Findings",
        "",
        "<!-- TODO: Fill in after reviewing results -->",
        "",
        "| Scenario | Recommended Provider | Rationale |",
        "|---|---|---|",
        "| Real-time conversation (EN) | _TBD_ | Lowest TTFB + acceptable quality |",
        "| Real-time conversation (ES) | _TBD_ | |",
        "| Offline / batch TTS (EN) | _TBD_ | Best MOS + cost |",
        "| Offline / batch TTS (ES) | _TBD_ | |",
        "| STT dictation (EN, clean) | _TBD_ | Lowest WER |",
        "| STT dictation (ES, clean) | _TBD_ | |",
        "| STT streaming (EN) | _TBD_ | Lowest TTFT |",
        "",
        "### 3.2 Assumptions & Limitations",
        "",
        "- All tests run from a single geographic location; latency may differ across regions.",
        "- Network jitter is not controlled; results represent best-effort measurements.",
        "- TTS quality is partially subjective (MOS checklist must be completed manually).",
        "- Audio files for STT must be recorded or sourced separately and placed in `dataset/stt/<lang>/`.",
        "- Costs are estimated based on public pricing as of 2026-03; verify before budgeting.",
        "- Speechmatics streaming results depend on the RT endpoint region selected.",
        "",
        "### 3.3 Next Steps",
        "",
        "1. Complete MOS checklist (§ 1.5) by listening to generated audio samples.",
        "2. Record or source real STT audio files and populate ground-truth manifests.",
        "3. Re-run benchmark from a staging environment (closer to production network).",
        "4. Integrate winning provider(s) into the voice feature pipeline.",
        "",
        "---",
        "_Report generated by [tts-stt-benchmark](https://github.com/jeanmlq10/tts-stt-benchmark)_",
    ]

    return "\n".join(lines) + "\n"


def save_report(results_dir: Path, report_path: Path | None = None) -> Path:
    content = build_report(results_dir)
    out = report_path or results_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
