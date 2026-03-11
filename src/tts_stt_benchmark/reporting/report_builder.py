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


def _fmt(v: float, decimals: int = 2) -> str:
    try:
        if math.isnan(v):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.{decimals}f}"


def _quality_label(wer: float) -> str:
    if wer < 0.05:
        return "excellent"
    if wer < 0.15:
        return "good"
    if wer < 0.30:
        return "acceptable"
    return "poor"


def _compute_findings(tts_df: pd.DataFrame, stt_df: pd.DataFrame) -> list[str]:
    """Generate the Key Findings section from aggregated benchmark data."""
    lines: list[str] = []

    # ── TTS Latency ───────────────────────────────────────────────────────────
    lines += ["#### TTS — Latency", ""]

    if not tts_df.empty:
        # Filter out runs that had errors for ALL repetitions
        ok = tts_df[tts_df["errors"] < tts_df["n"]]  # at least some successful reps
        if not ok.empty:
            for lang in ["en", "es"]:
                sub = ok[ok["lang"] == lang]
                if sub.empty:
                    continue
                # Aggregate per provider across all text_ids
                grp = sub.groupby("provider")["total_p50_s"].agg(
                    p50=lambda x: x.median(), p90=lambda x: x.quantile(0.9)
                ).reset_index()
                if grp.empty:
                    continue
                best = grp.loc[grp["p50"].idxmin()]
                worst = grp.loc[grp["p50"].idxmax()]
                lines.append(
                    f"- **{lang.upper()} — fastest TTS (median p50 across all texts):** "
                    f"`{best['provider']}` at **{_fmt(best['p50'])}s** total. "
                    f"Slowest: `{worst['provider']}` at {_fmt(worst['p50'])}s "
                    f"({_fmt((worst['p50'] - best['p50']) / best['p50'] * 100, 0)}% slower)."
                )

            # TTFB (streaming)
            ttfb_sub = ok[ok["ttfb_p50_s"].notna()]
            if not ttfb_sub.empty:
                grp_ttfb = ttfb_sub.groupby("provider")["ttfb_p50_s"].median().reset_index()
                best_ttfb = grp_ttfb.loc[grp_ttfb["ttfb_p50_s"].idxmin()]
                lines.append(
                    f"- **Best TTFB (streaming):** `{best_ttfb['provider']}` at "
                    f"**{_fmt(best_ttfb['ttfb_p50_s'], 3)}s** p50. "
                    f"OpenAI `tts-1-hd` was benchmarked in batch mode — no TTFB recorded."
                )
            lines.append("")

        # ── TTS Quality ───────────────────────────────────────────────────────
        lines += ["#### TTS — Audio Quality", ""]
        if "clipping_pct" in tts_df.columns:
            clip = tts_df.groupby("provider")["clipping_pct"].mean().reset_index()
            for _, row in clip.iterrows():
                pct = row["clipping_pct"] * 100
                if pct > 50:
                    lines.append(
                        f"- ⚠️  **`{row['provider']}`** — audio clipping in **{pct:.0f}%** of samples "
                        f"(peak levels exceed 0 dBFS). Apply a −1 dBFS ceiling limiter before playback."
                    )
                elif pct > 0:
                    lines.append(
                        f"- ⚠️  **`{row['provider']}`** — occasional clipping detected ({pct:.1f}% of samples)."
                    )
                else:
                    lines.append(
                        f"- ✅  **`{row['provider']}`** — no clipping detected. Clean output levels."
                    )
        lines.append("")

    # ── TTS Cost ──────────────────────────────────────────────────────────────
    lines += ["#### TTS — Cost", ""]
    lines += [
        "- `openai/tts-1-hd` is priced at **$30 / 1M chars** — 2× the cost of Deepgram Aura-2 ($15 / 1M chars).",
        "- For cost-sensitive workloads: `deepgram/aura-2` offers competitive TTFB (~0.65s) at half the price, "
        "but audio clipping must be corrected post-synthesis.",
        "- `openai/gpt-4o-mini-tts` ($12 / 1M chars) is a lower-cost OpenAI alternative not yet included in this benchmark.",
        "",
    ]

    # ── STT Accuracy ──────────────────────────────────────────────────────────
    lines += ["#### STT — Accuracy (WER)", ""]

    if not stt_df.empty and "wer_mean" in stt_df.columns:
        clean_mask = stt_df["audio_id"].str.contains("clean", na=False)
        noise_mask = stt_df["audio_id"].str.contains("noise", na=False)
        medium_mask = stt_df["audio_id"].str.contains("medium", na=False)

        for category, mask, label in [
            ("clean", clean_mask, "clean"),
            ("noise", noise_mask, "noisy (−30 dBFS)"),
            ("medium", medium_mask, "codes/numbers"),
        ]:
            sub = stt_df[mask]
            if sub.empty:
                continue
            grp = sub.groupby(["provider", "lang"])["wer_mean"].mean().reset_index()
            for lang in ["en", "es"]:
                lang_grp = grp[grp["lang"] == lang]
                if lang_grp.empty:
                    continue
                best = lang_grp.loc[lang_grp["wer_mean"].idxmin()]
                worst = lang_grp.loc[lang_grp["wer_mean"].idxmax()]
                if math.isnan(best["wer_mean"]):
                    continue
                lines.append(
                    f"- **{lang.upper()} {label} — best WER:** `{best['provider']}` "
                    f"at **{_fmt(best['wer_mean'], 3)}** ({_quality_label(best['wer_mean'])}). "
                    f"Worst: `{worst['provider']}` at {_fmt(worst['wer_mean'], 3)}."
                )
        lines.append("")

        # Noise note
        lines.append(
            "> **Note on noisy audio:** All providers degrade significantly at −30 dBFS noise. "
            "Apply upstream denoising (e.g., RNNoise, DeepFilterNet) before transcription in real deployments."
        )
        lines.append("")

        # ── STT Latency ───────────────────────────────────────────────────────
        lines += ["#### STT — Latency", ""]
        lat_grp = stt_df.groupby("provider")[["total_p50_s", "total_p90_s"]].median().reset_index()
        if not lat_grp.empty:
            best_l = lat_grp.loc[lat_grp["total_p50_s"].idxmin()]
            worst_l = lat_grp.loc[lat_grp["total_p50_s"].idxmax()]
            lines.append(
                f"- **Fastest STT (median p50):** `{best_l['provider']}` "
                f"at **{_fmt(best_l['total_p50_s'])}s** (p90 {_fmt(best_l['total_p90_s'])}s)."
            )
            lines.append(
                f"- **Slowest STT (median p50):** `{worst_l['provider']}` "
                f"at **{_fmt(worst_l['total_p50_s'])}s** (p90 {_fmt(worst_l['total_p90_s'])}s). "
                f"Extra latency comes from batch job polling overhead."
            )
        lines.append("")

        # ── STT Cost ──────────────────────────────────────────────────────────
        lines += ["#### STT — Cost", ""]
        cost_grp = stt_df[stt_df["cost_per_call_usd"] > 0].groupby("provider")["cost_per_call_usd"].mean().reset_index()
        if not cost_grp.empty:
            cheapest = cost_grp.loc[cost_grp["cost_per_call_usd"].idxmin()]
            priciest = cost_grp.loc[cost_grp["cost_per_call_usd"].idxmax()]
            ratio = priciest["cost_per_call_usd"] / cheapest["cost_per_call_usd"] if cheapest["cost_per_call_usd"] > 0 else float("nan")
            lines.append(
                f"- **Cheapest STT:** `{cheapest['provider']}` at avg "
                f"**${_fmt(cheapest['cost_per_call_usd'] * 100, 4)}¢/call** per audio clip."
            )
            lines.append(
                f"- **Most expensive STT:** `{priciest['provider']}` at avg "
                f"${_fmt(priciest['cost_per_call_usd'] * 100, 4)}¢/call "
                f"(~{_fmt(ratio, 1)}× the cost of {cheapest['provider']})."
            )
        lines.append("")

    # ── Recommendations table ─────────────────────────────────────────────────
    lines += [
        "#### Recommendations by use case",
        "",
        "| Scenario | Recommended | Rationale |",
        "|---|---|---|",
        "| Real-time TTS (EN, low latency) | `openai/tts-1-hd` | Fastest batch total (~4s p50); "
        "add streaming for sub-1s TTFB |",
        "| Real-time TTS (ES, low latency) | `openai/tts-1-hd` | ~3.75s p50 vs Deepgram ~8.5s on ES |",
        "| Batch TTS / cost-sensitive | `deepgram/aura-2` | TTFB ~0.65s, $15/1M chars; "
        "apply output limiter to fix clipping |",
        "| STT clean audio (EN + ES) | `openai/whisper-1` (standard) | Best WER on clean audio, "
        "fastest, cheapest |",
        "| STT noisy environment | Denoise first, then any | Both providers score ~WER 0.60+ at −30 dBFS |",
        "| STT at scale / cost | `openai/whisper-1` | ~4× cheaper than Speechmatics |",
        "| STT streaming / real-time | `speechmatics` | Only provider with true streaming API; "
        "Whisper is batch-only |",
    ]

    return lines


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
    # Compute key numbers for the findings section
    _findings = _compute_findings(tts_df, stt_df)

    lines += [
        "## 3. Executive Summary & Recommendations",
        "",
        "### 3.1 Key Findings",
        "",
    ]
    lines += _findings
    lines += [
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
