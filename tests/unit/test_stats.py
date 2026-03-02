"""
Unit tests for percentile stats utilities.
"""

import math
import pytest
from tts_stt_benchmark.metrics.stats import compute_stats, compute_stats_dict


class TestComputeStats:
    def test_basic_values(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.n == 5
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.mean == pytest.approx(3.0)
        assert stats.p50 == pytest.approx(3.0)
        assert stats.p90 == pytest.approx(4.6, abs=0.1)

    def test_single_value(self):
        stats = compute_stats([42.0])
        assert stats.p50 == pytest.approx(42.0)
        assert stats.p90 == pytest.approx(42.0)
        assert stats.n == 1

    def test_empty_list(self):
        stats = compute_stats([])
        assert stats.n == 0
        assert math.isnan(stats.p50)
        assert math.isnan(stats.mean)

    def test_all_same(self):
        stats = compute_stats([5.0] * 10)
        assert stats.p50 == pytest.approx(5.0)
        assert stats.p90 == pytest.approx(5.0)


class TestComputeStatsDictFiltersNone:
    def test_filters_none(self):
        stats = compute_stats_dict([1.0, None, 3.0, None, 5.0])
        assert stats.n == 3
        assert stats.mean == pytest.approx(3.0)

    def test_all_none(self):
        stats = compute_stats_dict([None, None])
        assert stats.n == 0

    def test_mixed_nan(self):
        stats = compute_stats_dict([float("nan"), 2.0, 4.0])
        assert stats.n == 2
