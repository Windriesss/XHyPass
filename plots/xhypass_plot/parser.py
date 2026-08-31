from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .model import RunRecord


HISTOGRAM_ROW = re.compile(r"^(\d+)\s+(\d+)$")
OVERFLOW_ROW = re.compile(r"^# Histogram Overflows:\s*(\d+)")
STRESS_METRIC = re.compile(
    r"^stress-ng: info:\s+\[\d+\]\s+vm\s+(\d+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$"
)
EXCLUDED_DATA_DIRECTORIES = frozenset({"backup_fboot"})


def is_excluded_data_path(path: Path, data_root: Path) -> bool:
    """Return whether a discovered result lives below an excluded directory."""
    try:
        parts = path.relative_to(data_root).parts
    except ValueError:
        parts = path.parts
    excluded = {name.casefold() for name in EXCLUDED_DATA_DIRECTORIES}
    return any(part.casefold() in excluded for part in parts)


def parse_histogram(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    values: list[int] = []
    counts: list[int] = []
    overflow = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := HISTOGRAM_ROW.match(line.strip()):
            values.append(int(match.group(1)))
            counts.append(int(match.group(2)))
        elif match := OVERFLOW_ROW.match(line.strip()):
            overflow = int(match.group(1))
    if not values:
        raise ValueError(f"No histogram rows found: {path}")
    return np.asarray(values, dtype=float), np.asarray(counts, dtype=np.int64), overflow


def scan_runs(platform: str, data_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for metadata_path in data_root.rglob("metadata.json"):
        if is_excluded_data_path(metadata_path, data_root):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            continue
        config = metadata.get("configuration", {})
        environment = config.get("environment_name")
        experiment = config.get("experiment_name")
        duration = config.get("experiment", {}).get("duration_seconds")
        interval_us = int(config.get("experiment", {}).get("interval_us", 0))
        hist_path = metadata_path.parent / "hist.txt"
        if not environment or not experiment or duration is None or not hist_path.exists():
            continue
        values, counts, overflow = parse_histogram(hist_path)
        batch = str(metadata.get("condition") or metadata_path.parent.parent.name)
        directory_run = re.search(r"run_(\d+)$", metadata_path.parent.name)
        run_index = (
            int(directory_run.group(1))
            if directory_run
            else int(metadata.get("run_index", 0))
        )
        records.append(
            RunRecord(
                platform=platform,
                environment=environment,
                experiment=experiment,
                batch=batch,
                run=run_index,
                duration_seconds=int(duration),
                path=metadata_path.parent,
                latency_us=values,
                counts=counts,
                overflow=overflow,
                interval_us=interval_us,
            )
        )
    return records


def parse_stress_log(record: RunRecord) -> dict | None:
    path = record.path / "stress-ng.log"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STRESS_METRIC.match(line.strip()):
            return {
                "platform": record.platform,
                "environment": record.environment,
                "experiment": record.experiment,
                "batch": record.batch,
                "run": record.run,
                "interval_us": record.interval_us,
                "bogo_ops": int(match.group(1)),
                "real_seconds": float(match.group(2)),
                "user_seconds": float(match.group(3)),
                "system_seconds": float(match.group(4)),
                "bogo_ops_per_real_second": float(match.group(5)),
                "bogo_ops_per_cpu_second": float(match.group(6)),
            }
    return None
