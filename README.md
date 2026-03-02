# TTS / STT Benchmark Harness

Automated benchmark scripts to compare **Text-to-Speech** and **Speech-to-Text** providers on latency and quality. Produces a consolidated Markdown report with recommendations per use case (real-time conversation vs. offline generation, ES / EN).

---

## Providers evaluated

| Type | Provider | Model(s) |
|------|----------|----------|
| TTS | OpenAI | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| TTS | Deepgram | `aura-2` |
| TTS | Google Cloud | `gemini-2.5-flash-preview-tts`, `neural2` |
| STT | OpenAI Whisper | `whisper-1` (standard + mini) |
| STT | Speechmatics | `default`, `enhanced` (batch + streaming) |

---

## Metrics

### TTS
| Metric | Description |
|--------|-------------|
| TTFB | Time to First Byte / first audio chunk (streaming) |
| Total synthesis time | Wall-clock from request to last byte |
| Audio duration | Length of synthesised audio |
| Clipping | Samples at ±full scale |
| Abrupt cut | Hard waveform cut at the end |
| RMS loudness (dBFS) | Overall loudness |
| MOS checklist | Subjective naturalness / intelligibility (manual) |

### STT
| Metric | Description |
|--------|-------------|
| TTFT | Time to First Transcript / partial result |
| Total transcription time | Wall-clock to final transcript |
| WER | Word Error Rate vs. ground-truth |
| CER | Character Error Rate vs. ground-truth |

Latency results are aggregated over N repetitions and reported as **p50 / p90 / mean**.

---

## Project structure

```
tts-stt-benchmark/
├── dataset/
│   ├── tts/
│   │   ├── es/texts.json        # ES TTS test texts
│   │   └── en/texts.json        # EN TTS test texts
│   └── stt/
│       ├── es/manifest.json     # ES audio files + ground-truth
│       └── en/manifest.json     # EN audio files + ground-truth
├── src/tts_stt_benchmark/
│   ├── adapters/
│   │   ├── tts/                 # OpenAI, Deepgram, Google adapters
│   │   └── stt/                 # Whisper, Speechmatics adapters
│   ├── cli/                     # run_tts, run_stt, run_benchmark, generate_report
│   ├── metrics/                 # wer_cer.py, audio_checks.py, stats.py
│   ├── reporting/               # report_builder.py
│   ├── config.py
│   ├── models.py
│   ├── runner.py
│   └── storage.py
├── tests/
│   └── unit/                    # pytest unit tests
├── results/                     # generated at runtime (gitignored)
├── pyproject.toml
└── .env.example
```

---

## Quick start

### 1. Prerequisites

- Python ≥ 3.11
- A virtual environment (venv / conda / uv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required environment variables:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI TTS + Whisper |
| `DEEPGRAM_API_KEY` | Deepgram Aura |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud TTS |
| `SPEECHMATICS_API_KEY` | Speechmatics |

### 3. Add STT audio files *(optional but required for WER/CER)*

Place 16 kHz mono WAV files in `dataset/stt/<lang>/` matching the filenames in `manifest.json`.  
Update `reference` strings in the manifests if needed.

---

## Running the benchmark

### Full benchmark (TTS + STT + report)

```bash
run_benchmark --repetitions 5
```

### TTS only

```bash
# All providers, ES + EN, streaming, 5 repetitions
run_tts --provider openai --provider deepgram --provider google \
        --lang es --lang en --streaming --repetitions 5

# Single provider, specific model
run_tts --provider openai --model tts-1 --lang en --repetitions 3
```

### STT only

```bash
# All providers, batch mode
run_stt --provider speechmatics --provider openai_whisper_standard \
        --provider openai_whisper_mini \
        --lang es --lang en --mode batch --repetitions 5

# Speechmatics streaming
run_stt --provider speechmatics --lang en --mode streaming --repetitions 3
```

### Generate report from existing results

```bash
generate_report --results_dir results/
# Output: results/report.md
```

---

## CLI reference

### `run_tts`

```
Usage: run_tts [OPTIONS]

Options:
  -p, --provider [openai|deepgram|google]  TTS provider(s) [default: all]
  -m, --model TEXT                         Override model name
  -l, --lang TEXT                          Language(s) [default: es, en]
  --text_file PATH                         Path(s) to texts JSON file
  --streaming / --no-streaming             Use streaming [default: streaming]
  -n, --repetitions INTEGER                Repetitions per test case
  -o, --output_dir PATH                    Output directory
```

### `run_stt`

```
Usage: run_stt [OPTIONS]

Options:
  -p, --provider [speechmatics|openai_whisper_standard|openai_whisper_mini]
  -l, --lang TEXT                          Language(s) [default: es, en]
  --audio_dir PATH                         Base dir with audio files
  --mode [batch|streaming]                 [default: batch]
  -n, --repetitions INTEGER
  -o, --output_dir PATH
```

### `run_benchmark`

```
Usage: run_benchmark [OPTIONS]

Options:
  --tts / --no-tts
  --stt / --no-sst
  --tts_providers ...
  --stt_providers ...
  --streaming / --no-streaming
  --stt_mode [batch|streaming]
  -n, --repetitions INTEGER
  -o, --output_dir PATH
```

### `generate_report`

```
Usage: generate_report [OPTIONS]

Options:
  -r, --results_dir PATH    [default: results/]
  -o, --output PATH         Output .md file [default: results/report.md]
```

---

## Running tests

```bash
pytest tests/unit/ -v
# With coverage
pytest tests/unit/ --cov=tts_stt_benchmark --cov-report=term-missing
```

---

## Output format

Results are written to `results/tts/<run_id>/` and `results/stt/<run_id>/`:

```
results/
└── tts/
│   └── 20260302T120000/
│       ├── results.json       # All results (structured)
│       ├── results.csv        # Flat CSV for spreadsheet analysis
│       ├── results.jsonl      # Incremental (written during run)
│       └── openai/tts-1-hd/en/en_short_01_rep01.wav  # Audio files
└── stt/
    └── 20260302T120000/
        ├── results.json
        ├── results.csv
        └── results.jsonl
```

### Result schema (TTS)

```json
{
  "provider": "openai",
  "model": "tts-1-hd",
  "language": "en",
  "text_id": "en_short_01",
  "text_chars": 60,
  "repetition": 1,
  "timestamp": "2026-03-02T12:00:00",
  "latency": {
    "time_to_first_byte_s": 0.12,
    "time_to_first_chunk_s": 0.12,
    "total_synthesis_s": 0.45,
    "audio_duration_s": 2.3
  },
  "quality": {
    "clipping_detected": false,
    "silence_at_start_s": 0.0,
    "silence_at_end_s": 0.02,
    "rms_dbfs": -18.5,
    "has_abrupt_cut": false,
    "mos_score": null
  },
  "audio_path": "results/tts/.../en_short_01_rep01.wav",
  "error": null,
  "streaming": true
}
```

---

## Cost estimates

Costs are estimated based on public pricing as of March 2026. **Verify before budgeting.**

### TTS (per 1M characters)

| Provider / Model | USD |
|---|---|
| OpenAI tts-1 | $15.00 |
| OpenAI tts-1-hd | $30.00 |
| OpenAI gpt-4o-mini-tts | $12.00 |
| Deepgram Aura-2 | $15.00 |
| Google Gemini-TTS | ~$12.00 |

### STT (per minute of audio)

| Provider / Model | USD/min | USD/hour |
|---|---|---|
| OpenAI Whisper-1 | $0.006 | $0.36 |
| Speechmatics standard | $0.025 | $1.50 |
| Speechmatics enhanced | $0.040 | $2.40 |
| Speechmatics real-time | $0.030 | $1.80 |

---

## Assumptions & limitations

- Latency is measured as wall-clock time from the client; it includes network round-trip and cannot isolate pure model inference time.
- Tests run from a single location. Results will vary by region.
- TTS MOS scores require manual evaluation (the checklist is in section 1.5 of the report).
- Google TTS does not support true streaming; TTFB ≈ total synthesis time.
- Speechmatics streaming uses the EU2 WebSocket endpoint by default.

---

## Reproducibility checklist

- [ ] Clone repo on a clean machine
- [ ] `cp .env.example .env` and fill in API keys
- [ ] `pip install -e ".[dev]"`
- [ ] Place STT audio WAV files in `dataset/stt/<lang>/`
- [ ] Run `run_benchmark --repetitions 5`
- [ ] Inspect `results/report.md`
- [ ] Complete MOS checklist in section 1.5 of the report
