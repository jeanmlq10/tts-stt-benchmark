"""
Statistical aggregation utilities: percentile stats from a list of float values.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

import numpy as np

from tts_stt_benchmark.models import PercentileStats


def compute_stats(values: Sequence[float]) -> PercentileStats:
    """Compute p50, p90, mean, min, max for a list of values."""
    if not values:
        nan = float("nan")
        return PercentileStats(p50=nan, p90=nan, mean=nan, min=nan, max=nan, n=0)

    arr = np.array(values, dtype=float)
    return PercentileStats(
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=len(arr),
    )


def compute_stats_dict(values: Sequence[float | None]) -> PercentileStats:
    """Like compute_stats but filters out None/NaN values."""
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return compute_stats(clean)
