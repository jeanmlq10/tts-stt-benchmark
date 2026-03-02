"""
Unit tests for storage utilities (JSON/CSV serialisation round-trip).
"""

import json
import tempfile
from pathlib import Path

import pytest
from tts_stt_benchmark.models import (
    TTSResult, TTSLatency, TTSQuality,
    STTResult, STTLatency, STTQuality,
    result_to_dict,
)
from tts_stt_benchmark.storage import (
    save_results_json, load_results_json,
    save_results_csv, append_result_json, load_jsonl,
)


def _make_tts_result(text_id: str = "test_01", rep: int = 1) -> TTSResult:
    return TTSResult(
        provider="openai",
        model="tts-1-hd",
        language="en",
        text_id=text_id,
        text_chars=42,
        repetition=rep,
        latency=TTSLatency(
            time_to_first_byte_s=0.12,
            total_synthesis_s=0.45,
            audio_duration_s=2.3,
        ),
        quality=TTSQuality(rms_dbfs=-18.5, clipping_detected=False),
        audio_path="/tmp/test.wav",
    )


def _make_stt_result(audio_id: str = "audio_01", rep: int = 1) -> STTResult:
    return STTResult(
        provider="openai_whisper_standard",
        model="whisper-1",
        language="es",
        audio_id=audio_id,
        audio_duration_s=4.1,
        repetition=rep,
        latency=STTLatency(
            time_to_first_transcript_s=1.2,
            total_transcription_s=1.5,
            audio_duration_s=4.1,
        ),
        quality=STTQuality(wer=0.08, cer=0.04, hypothesis="hola mundo", reference="hola mundo"),
        transcript="hola mundo",
        mode="batch",
    )


class TestJsonRoundTrip:
    def test_tts_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results = [_make_tts_result("t1", 1), _make_tts_result("t2", 2)]
            save_results_json(results, path)
            loaded = load_results_json(path)
            assert len(loaded) == 2
            assert loaded[0]["text_id"] == "t1"
            assert loaded[1]["repetition"] == 2
            assert loaded[0]["latency"]["time_to_first_byte_s"] == pytest.approx(0.12)

    def test_stt_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stt.json"
            results = [_make_stt_result()]
            save_results_json(results, path)
            loaded = load_results_json(path)
            assert loaded[0]["quality"]["wer"] == pytest.approx(0.08)


class TestCsvExport:
    def test_csv_has_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            results = [_make_tts_result()]
            save_results_csv(results, path)
            import csv
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert len(rows) == 1
            assert "latency_time_to_first_byte_s" in rows[0]
            assert "quality_rms_dbfs" in rows[0]


class TestJsonlAppend:
    def test_append_multiple_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            for rep in range(1, 4):
                append_result_json(_make_tts_result(rep=rep), path)
            rows = load_jsonl(path)
            assert len(rows) == 3
            assert rows[2]["repetition"] == 3


class TestResultToDict:
    def test_nested_structure(self):
        r = _make_tts_result()
        d = result_to_dict(r)
        assert isinstance(d["latency"], dict)
        assert isinstance(d["quality"], dict)
        assert d["provider"] == "openai"
