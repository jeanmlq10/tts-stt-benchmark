# TTS / STT Benchmark Harness

Automated benchmark harness to compare **Text-to-Speech (TTS)** and **Speech-to-Text (STT)** providers on latency and audio/transcription quality. Designed for engineering teams that need objective data to decide which provider to use in production for Spanish and English voice applications.

Produces a consolidated Markdown report with p50/p90 latency tables, audio quality checks, WER/CER scores, cost projections, and per-use-case recommendations.

---

## What we are testing and why

Voice APIs differ significantly in latency, audio naturalness, and accuracy depending on the use case. This harness was built to answer three concrete questions:

1. **Which TTS provider produces the most natural audio in Spanish and English?**  
   We synthesise 7 controlled texts per language covering different difficulty levels — short greetings, medium informational, long financial/news, acronyms, and mixed numbers/symbols — and measure both objective audio checks and subjective MOS.

2. **Which STT provider transcribes most accurately, especially under real-world conditions?**  
   We evaluate batch and streaming transcription on clean and light-noise audio, computing WER and CER against verified human ground-truth references.

3. **What is the real latency cost at production scale?**  
   We measure TTFB (time to first audio byte / first partial transcript) and total synthesis/transcription time over multiple repetitions, then aggregate to p50/p90. Streaming vs. batch comparisons are included for providers that support it.

---

## Current status — March 2026

| Component | State |
|-----------|-------|
| Project scaffolding | ✅ Complete |
| TTS adapters (OpenAI, Deepgram) | ✅ Tested live |
| TTS adapter (Google Cloud) | ⏸ Skipped — credentials not yet configured |
| STT adapters (Whisper, Speechmatics) | ✅ Implemented, pending audio files |
| Metrics (WER/CER, audio checks, p50/p90) | ✅ Complete |
| Report generator | ✅ Complete |
| Unit tests | ✅ 35/35 passing |
| STT audio dataset (WAV files) | 🔲 Not yet generated |
| Full benchmark run | 🔲 Pending STT audio dataset |

### Preliminary TTS observations (1 repetition, no streaming, EN only)

| Provider | Texts | Errors | Latency range |
|----------|-------|--------|---------------|
| OpenAI `tts-1-hd` | 7/7 | 0 | 4.0 s – 8.3 s |
| Deepgram `aura-2` | 7/7 | 0 | 3.5 s – 29.1 s |

> **Note (Deepgram):** The 29 s latency on the long text (~530 chars) is notably high and may indicate throttling or a cold-start penalty. This will be confirmed with multiple repetitions in the full benchmark run.

> **Note (latency):** These numbers are non-streaming batch requests, which represent the worst-case latency scenario. Streaming TTFB values are expected to be significantly lower (typically 200–800 ms for short texts).

---

---

## Test dataset

### TTS texts (7 per language, ES + EN)

| ID | Category | Chars | Purpose |
|----|----------|-------|---------|
| `*_short_01` | Short greeting | ~60 | Typical IVR / assistant opener |
| `*_short_02` | Short + codes | ~70 | Flight info with codes and times |
| `*_medium_01` | Medium informational | ~220–235 | AI/tech news paragraph |
| `*_medium_02` | Medium + special | ~200–210 | OTP code, phone numbers, emails |
| `*_long_01` | Long financial/news | ~520–530 | Economic bulletin with percentages and figures |
| `*_acronyms_01` | Acronyms | ~115–130 | CEO, CTO, R&D, RFP, EU, Q4 — abbreviation stress test |
| `*_numbers_01` | Numbers + symbols | ~105–115 | Prices, decimals, currency, VAT |

These categories are designed to stress-test provider weaknesses: acronyms and mixed numbers are known pain points for TTS naturalness, while long texts reveal latency scaling behavior.

### STT audio files (4 per language, ES + EN)

| ID | Condition | Ground-truth type |
|----|-----------|-------------------|
| `*_clean_01` | Clean speech | Short greeting |
| `*_clean_02` | Clean speech | Medium informational |
| `*_noise_01` | Light background noise (−30 dBFS) | Numbers / prices |
| `*_medium_01` | Clean speech | Flight codes / times |

Audio files must be **16 kHz mono PCM WAV**. The `reference` field in each manifest contains the verified human transcription used as ground-truth for WER/CER computation.

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

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI TTS + Whisper | ✅ |
| `DEEPGRAM_API_KEY` | Deepgram Aura | ✅ |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud TTS | Optional — Google TTS is automatically skipped if not set |
| `SPEECHMATICS_API_KEY` | Speechmatics | ✅ |

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
