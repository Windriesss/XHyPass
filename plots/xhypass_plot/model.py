from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunRecord:
    platform: str
    environment: str
    experiment: str
    batch: str
    run: int
    duration_seconds: int
    path: Path
    latency_us: np.ndarray
    counts: np.ndarray
    overflow: int
    interval_us: int = 0

    @property
    def samples(self) -> int:
        return int(self.counts.sum())

    def mean(self) -> float:
        return float(np.average(self.latency_us, weights=self.counts))

    def quantile(self, q: float) -> float:
        cumulative = np.cumsum(self.counts)
        target = q * cumulative[-1]
        index = min(int(np.searchsorted(cumulative, target, side="left")), len(cumulative) - 1)
        return float(self.latency_us[index])

    def maximum(self) -> float:
        return float(self.latency_us[-1])

    def minimum(self) -> float:
        return float(self.latency_us[0])


def combine(records: list[RunRecord]) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, int] = {}
    for record in records:
        for latency, count in zip(record.latency_us, record.counts, strict=True):
            merged[int(latency)] = merged.get(int(latency), 0) + int(count)
    values = np.array(sorted(merged), dtype=float)
    counts = np.array([merged[int(value)] for value in values], dtype=np.int64)
    return values, counts


def weighted_quantile(values: np.ndarray, counts: np.ndarray, q: float) -> float:
    cumulative = np.cumsum(counts)
    index = min(
        int(np.searchsorted(cumulative, q * cumulative[-1], side="left")),
        len(values) - 1,
    )
    return float(values[index])
