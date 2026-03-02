"""
Unit tests for the report builder (no live results needed — uses synthetic data).
"""

import json
import tempfile
from pathlib import Path

import pytest
from tts_stt_benchmark.reporting.report_builder import build_report, save_report
from tts_stt_benchmark.models import TTSResult, TTSLatency, TTSQuality, STTResult, STTLatency, STTQuality
from tts_stt_benchmark.storage import save_results_json


def _make_tts_results() -> list[TTSResult]:
    results = []
    for rep in range(1, 4):
        results.append(TTSResult(
            provider="openai",
            model="tts-1-hd",
            language="en",
            text_id="en_short_01",
            text_chars=60,
            repetition=rep,
            latency=TTSLatency(
                time_to_first_byte_s=0.10 + rep * 0.01,
                total_synthesis_s=0.40 + rep * 0.05,
                audio_duration_s=2.1,
            ),
            quality=TTSQuality(rms_dbfs=-18.0, clipping_detected=False),
            audio_path=f"/tmp/openai_rep{rep}.wav",
        ))
    return results


def _make_stt_results() -> list[STTResult]:
    results = []
    for rep in range(1, 4):
        results.append(STTResult(
            provider="openai_whisper_standard",
            model="whisper-1",
            language="en",
            audio_id="en_clean_01",
            audio_duration_s=3.8,
            repetition=rep,
            latency=STTLatency(
                time_to_first_transcript_s=0.8 + rep * 0.1,
                total_transcription_s=1.0 + rep * 0.1,
                audio_duration_s=3.8,
            ),
            quality=STTQuality(wer=0.05, cer=0.02, hypothesis="good morning", reference="good morning"),
            transcript="good morning",
            mode="batch",
        ))
    return results


@pytest.fixture
def results_dir_with_data(tmp_path: Path) -> Path:
    tts_dir = tmp_path / "tts" / "20260302T120000"
    tts_dir.mkdir(parents=True)
    save_results_json(_make_tts_results(), tts_dir / "results.json")

    stt_dir = tmp_path / "stt" / "20260302T120001"
    stt_dir.mkdir(parents=True)
    save_results_json(_make_stt_results(), stt_dir / "results.json")
    return tmp_path


class TestBuildReport:
    def test_returns_string(self, results_dir_with_data):
        report = build_report(results_dir_with_data)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_required_sections(self, results_dir_with_data):
        report = build_report(results_dir_with_data)
        assert "## 1. TTS Results" in report
        assert "## 2. STT Results" in report
        assert "## 3. Executive Summary" in report
        assert "MOS" in report

    def test_contains_provider_names(self, results_dir_with_data):
        report = build_report(results_dir_with_data)
        assert "openai" in report
        assert "openai_whisper_standard" in report

    def test_empty_results_dir(self, tmp_path: Path):
        """Report should still build with no data."""
        report = build_report(tmp_path)
        assert "TTS / STT Benchmark Report" in report
        assert "_No TTS results found._" in report
        assert "_No STT results found._" in report

    def test_save_report_creates_file(self, results_dir_with_data):
        out_path = results_dir_with_data / "test_report.md"
        saved = save_report(results_dir_with_data, out_path)
        assert saved == out_path
        assert out_path.exists()
        content = out_path.read_text()
        assert "TTS / STT Benchmark Report" in content
