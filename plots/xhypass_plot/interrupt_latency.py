"""Shared parsing and statistics for interrupt-latency paper figures."""

from __future__ import annotations

import re
from pathlib import Path


REGION_RE = re.compile(
    r"^t0_region>\s*(.*?)(?=^t1_region>)", re.MULTILINE | re.DOTALL
)
BUCKET_RE = re.compile(r"^\[(\d+)\]:(\d+)\s*$", re.MULTILINE)


def parse_last_t0_histogram(path: Path) -> dict[int, int]:
    """Return the last complete t0 histogram as ``latency_ns: count``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    regions = REGION_RE.findall(text)
    if not regions:
        raise ValueError(f"no complete t0_region/t1_region pair in {path}")

    histogram: dict[int, int] = {}
    for bucket, count in BUCKET_RE.findall(regions[-1]):
        latency_ns = int(bucket) * 10
        histogram[latency_ns] = histogram.get(latency_ns, 0) + int(count)
    if not histogram:
        raise ValueError(f"empty final t0_region in {path}")
    return histogram


def load_run_maxima(series_dir: Path) -> tuple[list[float], list[Path]]:
    """Return each published run's final-t0 maximum in microseconds."""
    paths = sorted(series_dir.glob("rtos_run*.log"))
    if not paths:
        raise FileNotFoundError(
            f"no published rtos_run*.log files in {series_dir}"
        )
    maxima_us = [max(parse_last_t0_histogram(path)) / 1000 for path in paths]
    return maxima_us, paths


def weighted_quantile(histogram: dict[int, int], quantile: float) -> int:
    """Return the first histogram bucket reaching ``quantile``."""
    target = quantile * sum(histogram.values())
    cumulative = 0
    for latency_ns, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return latency_ns
    raise AssertionError("non-empty histogram did not reach quantile")
