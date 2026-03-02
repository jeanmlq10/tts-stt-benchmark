"""
Result storage: save/load JSON and CSV files.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Sequence

from tts_stt_benchmark.models import TTSResult, STTResult, result_to_dict


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into a single-level dict for CSV export."""
    out: dict = {}
    for key, val in d.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            out.update(_flatten(val, f"{full_key}_"))
        else:
            out[full_key] = val
    return out


def save_results_json(results: Sequence[TTSResult | STTResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result_to_dict(r) for r in results]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def load_results_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_results_csv(results: Sequence[TTSResult | STTResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten(result_to_dict(r)) for r in results]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_result_json(result: TTSResult | STTResult, path: Path) -> None:
    """Append a single result to a JSON-lines (.jsonl) file for incremental writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
