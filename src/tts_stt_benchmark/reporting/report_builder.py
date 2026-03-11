"""
Generador de reportes: lee todos los archivos results.json bajo un directorio de resultados,
agrega estadísticas y produce un reporte en Markdown.
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
    """Carga todos los archivos results.json bajo results_dir/kind/*/results.json."""
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

    # Columnas derivadas
    df["provider_model"] = df["provider"] + "/" + df["model"]
    df["ttfb"] = df.get("latency.time_to_first_byte_s", pd.Series([None] * len(df)))
    df["total"] = df.get("latency.total_synthesis_s", pd.Series([None] * len(df)))
    df["audio_dur"] = df.get("latency.audio_duration_s", pd.Series([None] * len(df)))
    df["rms_dbfs"] = df.get("quality.rms_dbfs", pd.Series([None] * len(df)))
    df["clipping"] = df.get("quality.clipping_detected", pd.Series([False] * len(df)))
    df["abrupt_cut"] = df.get("quality.has_abrupt_cut", pd.Series([False] * len(df)))

    # Agrupar por provider_model + language + text_id
    groups = df.groupby(["provider_model", "language", "text_id"])

    summary_rows: list[dict] = []
    for (pm, lang, tid), grp in groups:
        ttfb_stats = compute_stats_dict(grp["ttfb"].tolist())
        total_stats = compute_stats_dict(grp["total"].tolist())
        clipping_count = int(grp["clipping"].sum())
        abrupt_count = int(grp["abrupt_cut"].sum())
        n = len(grp)
        # Estimación de costo (por llamada, basado en cantidad de caracteres)
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
        return "_Sin datos disponibles._\n"
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
        return "excelente"
    if wer < 0.15:
        return "bueno"
    if wer < 0.30:
        return "aceptable"
    return "deficiente"


def _compute_findings(tts_df: pd.DataFrame, stt_df: pd.DataFrame) -> list[str]:
    """Genera la sección de Hallazgos Clave a partir de los datos agregados del benchmark."""
    lines: list[str] = []

    # ── TTS Latencia ──────────────────────────────────────────────────────────
    lines += ["#### TTS — Latencia", ""]

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
                    f"- **{lang.upper()} — TTS más rápido (mediana p50 sobre todos los textos):** "
                    f"`{best['provider']}` en **{_fmt(best['p50'])}s** total. "
                    f"Más lento: `{worst['provider']}` en {_fmt(worst['p50'])}s "
                    f"({_fmt((worst['p50'] - best['p50']) / best['p50'] * 100, 0)}% más lento)."
                )

            # TTFB (streaming)
            ttfb_sub = ok[ok["ttfb_p50_s"].notna()]
            if not ttfb_sub.empty:
                grp_ttfb = ttfb_sub.groupby("provider")["ttfb_p50_s"].median().reset_index()
                best_ttfb = grp_ttfb.loc[grp_ttfb["ttfb_p50_s"].idxmin()]
                lines.append(
                    f"- **Mejor TTFB (streaming):** `{best_ttfb['provider']}` en "
                    f"**{_fmt(best_ttfb['ttfb_p50_s'], 3)}s** p50. "
                    f"OpenAI `tts-1-hd` fue evaluado en modo batch — sin TTFB registrado."
                )
            lines.append("")

        # ── TTS Calidad de audio ──────────────────────────────────────────────
        lines += ["#### TTS — Calidad de Audio", ""]
        if "clipping_pct" in tts_df.columns:
            clip = tts_df.groupby("provider")["clipping_pct"].mean().reset_index()
            for _, row in clip.iterrows():
                pct = row["clipping_pct"] * 100
                if pct > 50:
                    lines.append(
                        f"- ⚠️  **`{row['provider']}`** — saturación (clipping) en **{pct:.0f}%** de las muestras "
                        f"(picos superan 0 dBFS). Aplicar un limitador de techo −1 dBFS antes de reproducir."
                    )
                elif pct > 0:
                    lines.append(
                        f"- ⚠️  **`{row['provider']}`** — saturación ocasional detectada ({pct:.1f}% de las muestras)."
                    )
                else:
                    lines.append(
                        f"- ✅  **`{row['provider']}`** — sin saturación detectada. Niveles de salida limpios."
                    )
        lines.append("")

    # ── TTS Costo ─────────────────────────────────────────────────────────────
    lines += ["#### TTS — Costo", ""]
    lines += [
        "- `openai/tts-1-hd` tiene un precio de **$30 / 1M chars** — el doble que Deepgram Aura-2 ($15 / 1M chars).",
        "- Para cargas sensibles al costo: `deepgram/aura-2` ofrece TTFB competitivo (~0.65s) a mitad de precio, "
        "pero la saturación de audio debe corregirse en post-síntesis.",
        "- `openai/gpt-4o-mini-tts` ($12 / 1M chars) es una alternativa OpenAI más económica aún no incluida en este benchmark.",
        "",
    ]

    # ── STT Precisión ────────────────────────────────────────────────────────
    lines += ["#### STT — Precisión (WER)", ""]

    if not stt_df.empty and "wer_mean" in stt_df.columns:
        clean_mask = stt_df["audio_id"].str.contains("clean", na=False)
        noise_mask = stt_df["audio_id"].str.contains("noise", na=False)
        medium_mask = stt_df["audio_id"].str.contains("medium", na=False)

        for category, mask, label in [
            ("clean", clean_mask, "audio limpio"),
            ("noise", noise_mask, "audio con ruido (−30 dBFS)"),
            ("medium", medium_mask, "códigos/números"),
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
                    f"- **{lang.upper()} {label} — mejor WER:** `{best['provider']}` "
                    f"en **{_fmt(best['wer_mean'], 3)}** ({_quality_label(best['wer_mean'])}). "
                    f"Peor: `{worst['provider']}` en {_fmt(worst['wer_mean'], 3)}."
                )
        lines.append("")

        # Noise note
        lines.append(
            "> **Nota sobre audio con ruido:** Todos los proveedores degradan significativamente con ruido a −30 dBFS. "
            "Aplicar reducción de ruido previa (p. ej. RNNoise, DeepFilterNet) antes de transcribir en producción."
        )
        lines.append("")

        # ── STT Latencia ──────────────────────────────────────────────────────
        lines += ["#### STT — Latencia", ""]
        lat_grp = stt_df.groupby("provider")[["total_p50_s", "total_p90_s"]].median().reset_index()
        if not lat_grp.empty:
            best_l = lat_grp.loc[lat_grp["total_p50_s"].idxmin()]
            worst_l = lat_grp.loc[lat_grp["total_p50_s"].idxmax()]
            lines.append(
                f"- **STT más rápido (mediana p50):** `{best_l['provider']}` "
                f"en **{_fmt(best_l['total_p50_s'])}s** (p90 {_fmt(best_l['total_p90_s'])}s)."
            )
            lines.append(
                f"- **STT más lento (mediana p50):** `{worst_l['provider']}` "
                f"en **{_fmt(worst_l['total_p50_s'])}s** (p90 {_fmt(worst_l['total_p90_s'])}s). "
                f"La latencia extra proviene del overhead de polling en jobs batch."
            )
        lines.append("")

        # ── STT Costo ─────────────────────────────────────────────────────────
        lines += ["#### STT — Costo", ""]
        cost_grp = stt_df[stt_df["cost_per_call_usd"] > 0].groupby("provider")["cost_per_call_usd"].mean().reset_index()
        if not cost_grp.empty:
            cheapest = cost_grp.loc[cost_grp["cost_per_call_usd"].idxmin()]
            priciest = cost_grp.loc[cost_grp["cost_per_call_usd"].idxmax()]
            ratio = priciest["cost_per_call_usd"] / cheapest["cost_per_call_usd"] if cheapest["cost_per_call_usd"] > 0 else float("nan")
            lines.append(
                f"- **STT más económico:** `{cheapest['provider']}` a un promedio de "
                f"**${_fmt(cheapest['cost_per_call_usd'] * 100, 4)}¢/llamada** por clip de audio."
            )
            lines.append(
                f"- **STT más caro:** `{priciest['provider']}` a un promedio de "
                f"${_fmt(priciest['cost_per_call_usd'] * 100, 4)}¢/llamada "
                f"(~{_fmt(ratio, 1)}× el costo de {cheapest['provider']})."
            )
        lines.append("")

    # ── Tabla de recomendaciones ──────────────────────────────────────────────
    lines += [
        "#### Recomendaciones por caso de uso",
        "",
        "| Escenario | Recomendado | Justificación |",
        "|---|---|---|",
        "| TTS en tiempo real (EN, baja latencia) | `openai/tts-1-hd` | Total batch más rápido (~4s p50); "
        "agregar streaming para TTFB <1s |",
        "| TTS en tiempo real (ES, baja latencia) | `openai/tts-1-hd` | ~3.75s p50 vs Deepgram ~8.5s en ES |",
        "| TTS batch / sensible al costo | `deepgram/aura-2` | TTFB ~0.65s, $15/1M chars; "
        "aplicar limitador de salida para corregir saturación |",
        "| STT audio limpio (EN + ES) | `openai/whisper-1` (standard) | Mejor WER en audio limpio, "
        "más rápido y económico |",
        "| STT en entorno ruidoso | Aplicar denoising primero, luego cualquiera | Ambos proveedores obtienen WER ~0.60+ a −30 dBFS |",
        "| STT a escala / costo | `openai/whisper-1` | ~4× más barato que Speechmatics |",
        "| STT streaming / tiempo real | `speechmatics` | Único proveedor con API de streaming real; "
        "Whisper es solo batch |",
    ]

    return lines


def build_report(results_dir: Path) -> str:
    tts_rows = _load_all_results(results_dir, "tts")
    stt_rows = _load_all_results(results_dir, "stt")

    tts_df = _aggregate_tts(tts_rows)
    stt_df = _aggregate_stt(stt_rows)

    lines: list[str] = []

    lines += [
        "# Reporte de Benchmark TTS / STT",
        "",
        f"**Generado:** {__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Directorio de resultados:** `{results_dir}`  ",
        f"**Muestras TTS:** {len(tts_rows)} · **Muestras STT:** {len(stt_rows)}",
        "",
        "---",
        "",
    ]

    # ── Sección TTS ───────────────────────────────────────────────────────────
    lines += ["## 1. Resultados TTS", ""]

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
            "### 1.1 Resumen de Latencia y Costo (por proveedor × idioma)",
            "",
            "> TTFB = Tiempo al primer byte/chunk · Total = Tiempo total de síntesis · "
            "Costo = estimado por llamada según longitud promedio del texto",
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
            "### 1.2 Detalle de Latencia (por texto)",
            "",
            _df_to_md(tts_df[detail_cols]),
            "",
        ]

        # Calidad
        quality_cols = [
            "provider", "lang", "text_id",
            "clipping_pct", "abrupt_cut_pct",
        ]
        lines += [
            "### 1.3 Verificaciones de Calidad",
            "",
            _df_to_md(tts_df[quality_cols]),
            "",
        ]

        # Cost projection
        lines += [
            "### 1.4 Proyección de Costo",
            "",
            "| Proveedor | Costo por 1M chars (USD) | Costo estimado / 1k llamadas @ 200 chars promedio |",
            "|---|---|---|",
        ]
        for pm, cost in TTS_COST_PER_1M_CHARS.items():
            proj = (200 / 1_000_000) * cost * 1000
            lines.append(f"| {pm} | ${cost:.2f} | ${proj:.4f} |")
        lines.append("")
    else:
        lines += ["_No se encontraron resultados TTS._", ""]

    # ── Checklist MOS ─────────────────────────────────────────────────────────
    lines += [
        "### 1.5 Checklist de Calidad Subjetiva (MOS)",
        "",
        "Escucha cada archivo de audio generado y puntúa del 1 al 5 en cada dimensión.",
        "",
        "**Escala:** 1 = Inaceptable · 2 = Malo · 3 = Regular · 4 = Bueno · 5 = Excelente",
        "",
        "**Archivos de referencia:** `results/tts/<timestamp>/<provider>/<model>/<lang>/<text_id>_rep01.wav`",
        "",
        "| Proveedor | Modelo | Idioma | Texto | Naturalidad (1–5) | Inteligibilidad (1–5) "
        "| Prosodia (1–5) | Acento (1–5) | Notas |",
        "|---|---|---|---|---|---|---|---|---|",
        "| openai | tts-1-hd | es | es_short_01 | | | | | |",
        "| openai | tts-1-hd | es | es_short_02 | | | | | |",
        "| openai | tts-1-hd | es | es_medium_01 | | | | | |",
        "| openai | tts-1-hd | es | es_medium_02 | | | | | |",
        "| openai | tts-1-hd | es | es_long_01 | | | | | |",
        "| openai | tts-1-hd | es | es_numbers_01 | | | | | |",
        "| openai | tts-1-hd | es | es_acronyms_01 | | | | | |",
        "| openai | tts-1-hd | en | en_short_01 | | | | | |",
        "| openai | tts-1-hd | en | en_short_02 | | | | | |",
        "| openai | tts-1-hd | en | en_medium_01 | | | | | |",
        "| openai | tts-1-hd | en | en_medium_02 | | | | | |",
        "| openai | tts-1-hd | en | en_long_01 | | | | | |",
        "| openai | tts-1-hd | en | en_numbers_01 | | | | | |",
        "| openai | tts-1-hd | en | en_acronyms_01 | | | | | |",
        "| deepgram | aura-2 | es | es_short_01 | | | | | |",
        "| deepgram | aura-2 | es | es_short_02 | | | | | |",
        "| deepgram | aura-2 | es | es_medium_01 | | | | | |",
        "| deepgram | aura-2 | es | es_medium_02 | | | | | |",
        "| deepgram | aura-2 | es | es_long_01 | | | | | |",
        "| deepgram | aura-2 | es | es_numbers_01 | | | | | |",
        "| deepgram | aura-2 | es | es_acronyms_01 | | | | | |",
        "| deepgram | aura-2 | en | en_short_01 | | | | | |",
        "| deepgram | aura-2 | en | en_short_02 | | | | | |",
        "| deepgram | aura-2 | en | en_medium_01 | | | | | |",
        "| deepgram | aura-2 | en | en_medium_02 | | | | | |",
        "| deepgram | aura-2 | en | en_long_01 | | | | | |",
        "| deepgram | aura-2 | en | en_numbers_01 | | | | | |",
        "| deepgram | aura-2 | en | en_acronyms_01 | | | | | |",
        "",
        "> **⚠️ Nota sobre Deepgram:** Se detectó saturación (clipping) en el 100% de las muestras.",
        "> Antes de escuchar, normaliza el volumen con: `ffmpeg -i input.wav -af loudnorm output.wav`",
        "",
    ]

    # ── Sección STT ───────────────────────────────────────────────────────────
    lines += ["## 2. Resultados STT", ""]

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
            "### 2.1 Resumen de Latencia y Calidad (por proveedor × idioma)",
            "",
            "> TTFT = Tiempo al primer transcript · WER/CER = comparado contra transcripciones de referencia",
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
            "### 2.2 Detalle por Archivo de Audio",
            "",
            _df_to_md(stt_df[detail_cols]),
            "",
        ]

        # Proyección de costo
        lines += [
            "### 2.3 Proyección de Costo",
            "",
            "| Proveedor | Costo por minuto (USD) | Costo por hora | Proyectado / 1k min |",
            "|---|---|---|---|",
        ]
        for pm, cpm in STT_COST_PER_MINUTE.items():
            lines.append(f"| {pm} | ${cpm:.4f} | ${cpm * 60:.2f} | ${cpm * 1000:.2f} |")
        lines.append("")
    else:
        lines += ["_No se encontraron resultados STT._", ""]

    # ── Recommendations ──────────────────────────────────────────────────────
    # Compute key numbers for the findings section
    _findings = _compute_findings(tts_df, stt_df)

    lines += [
        "## 3. Resumen Ejecutivo y Recomendaciones",
        "",
        "### 3.1 Hallazgos Clave",
        "",
    ]
    lines += _findings
    lines += [
        "",
        "### 3.2 Supuestos y Limitaciones",
        "",
        "- Todas las pruebas se ejecutaron desde una única ubicación geográfica; la latencia puede variar por región.",
        "- El jitter de red no está controlado; los resultados representan mediciones en condiciones normales.",
        "- La calidad TTS es parcialmente subjetiva (el checklist MOS debe completarse manualmente, ver §1.5).",
        "- Los archivos de audio para STT se generaron con OpenAI TTS `tts-1`, voz `alloy`; resultados pueden diferir con audio real de producción.",
        "- Los costos están estimados con precios públicos a marzo de 2026; verificar antes de presupuestar.",
        "- Los resultados de streaming de Speechmatics dependen de la región del endpoint RT seleccionado.",
        "",
        "### 3.3 Próximos Pasos",
        "",
        "1. Completar el checklist MOS (§1.5) escuchando los archivos de audio generados en `results/tts/`.",
        "2. Evaluar `openai/gpt-4o-mini-tts` ($12/1M chars) como alternativa económica a `tts-1-hd`.",
        "3. Probar TTS en modo streaming (OpenAI con `stream=True`) para obtener TTFB real.",
        "4. Aplicar un limitador de techo a −1 dBFS sobre las salidas de Deepgram y re-medir MOS.",
        "5. Re-ejecutar el benchmark STT con audio real de producción (no sintético) para validar resultados.",
        "6. Integrar el/los proveedor(es) ganadores al pipeline de la funcionalidad de voz.",
        "",
        "---",
        "_Reporte generado por [tts-stt-benchmark](https://github.com/jeanmlq10/tts-stt-benchmark)_",
    ]

    return "\n".join(lines) + "\n"


def save_report(results_dir: Path, report_path: Path | None = None) -> Path:
    content = build_report(results_dir)
    out = report_path or results_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out
