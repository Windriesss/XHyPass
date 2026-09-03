#!/usr/bin/env python3
"""Compute caller-observed transition statistics without pooling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "condition",
    "iteration",
    "direction",
    "attempt",
    "rc",
    "counter_cycles",
    "duration_ns",
    "request_elapsed_ns",
}
DIRECTIONS = ("DYN-to-RTO", "RTO-to-DYN")
PERCENTILES = {
    "p50_us": 0.5,
    "p99_us": 0.99,
    "p999_us": 0.999,
}


def discover_attempt_files(data_root: Path) -> list[Path]:
    files = sorted(data_root.glob("**/transition_attempts.csv"))
    if not files:
        files = sorted(data_root.glob("**/attempts.csv"))
    return [path for path in files if "analysis" not in path.parts]


def read_attempts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")
    versions = set(frame["schema_version"].dropna().astype(int))
    if versions != {SCHEMA_VERSION}:
        raise ValueError(f"{path}: unsupported schema versions: {sorted(versions)}")
    if not set(frame["direction"]).issubset(DIRECTIONS):
        raise ValueError(f"{path}: invalid direction value")
    for column in (
        "iteration",
        "attempt",
        "rc",
        "counter_cycles",
        "duration_ns",
        "request_elapsed_ns",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["duration_ns"] < 0).any() or (frame["request_elapsed_ns"] < 0).any():
        raise ValueError(f"{path}: negative duration")
    frame["source_file"] = str(path)
    return frame


def _percentile(values: pd.Series, quantile: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=float), quantile)) / 1000.0


def summarize_run(platform: str, frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    identity_columns = ["run_id", "condition", "direction"]
    for (run_id, condition, direction), group in frame.groupby(
        identity_columns, sort=True
    ):
        request_groups = group.groupby("iteration", sort=False)
        terminal = request_groups.tail(1)
        successful_terminal = terminal[terminal["rc"] == 0]
        record = {
            "platform": platform,
            "run_id": str(run_id),
            "condition": str(condition),
            "direction": str(direction),
            "requests": int(len(terminal)),
            "successful_requests": int(len(successful_terminal)),
            "failed_requests": int((terminal["rc"] != 0).sum()),
            "hvc_attempts": int(len(group)),
            "retries": int((group["attempt"] > 0).sum()),
            "busy_returns": int((group["rc"] == -16).sum()),
            "again_returns": int((group["rc"] == -11).sum()),
            "other_errors": int(
                ((group["rc"] != 0) & ~group["rc"].isin((-16, -11))).sum()
            ),
            "first_try_success_rate": float(
                ((terminal["rc"] == 0) & (terminal["attempt"] == 0)).mean()
            ),
        }
        if successful_terminal.empty:
            for label in (*PERCENTILES, "max_us"):
                record[label] = float("nan")
        else:
            for label, quantile in PERCENTILES.items():
                record[label] = _percentile(
                    successful_terminal["request_elapsed_ns"], quantile
                )
            record["max_us"] = (
                float(successful_terminal["request_elapsed_ns"].max()) / 1000.0
            )
        if successful_terminal.empty:
            record["completion_p99_us"] = float("nan")
            record["completion_max_us"] = float("nan")
        else:
            record["completion_p99_us"] = _percentile(
                successful_terminal["request_elapsed_ns"], 0.99
            )
            record["completion_max_us"] = (
                float(successful_terminal["request_elapsed_ns"].max()) / 1000.0
            )
        records.append(record)
    return records


def cross_run_summary(run_statistics: pd.DataFrame) -> list[dict]:
    metrics = (
        "first_try_success_rate",
        "p50_us",
        "p99_us",
        "p999_us",
        "max_us",
        "completion_p99_us",
        "completion_max_us",
    )
    records: list[dict] = []
    for (platform, condition, direction), group in run_statistics.groupby(
        ["platform", "condition", "direction"], sort=True
    ):
        record = {
            "platform": platform,
            "condition": condition,
            "direction": direction,
            "independent_runs": int(group["run_id"].nunique()),
            "requests": int(group["requests"].sum()),
            "hvc_attempts": int(group["hvc_attempts"].sum()),
            "failed_requests": int(group["failed_requests"].sum()),
            "retries": int(group["retries"].sum()),
            "busy_returns": int(group["busy_returns"].sum()),
            "again_returns": int(group["again_returns"].sum()),
            "other_errors": int(group["other_errors"].sum()),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_median"] = (
                float(np.median(values)) if values.size else None
            )
            record[f"{metric}_min"] = float(np.min(values)) if values.size else None
            record[f"{metric}_max"] = float(np.max(values)) if values.size else None
        records.append(record)
    return records


def analyze(platform: str, data_root: Path, output_root: Path) -> tuple[Path, Path]:
    files = discover_attempt_files(data_root)
    if not files:
        raise FileNotFoundError(f"no transition attempt CSVs below {data_root}")
    attempts = pd.concat([read_attempts(path) for path in files], ignore_index=True)
    duplicated = attempts.duplicated(
        ["run_id", "condition", "iteration", "direction", "attempt"]
    )
    if duplicated.any():
        raise ValueError("duplicate transition-attempt identity")
    run_statistics = pd.DataFrame(summarize_run(platform, attempts))
    if run_statistics.empty:
        raise ValueError("no transition groups found")
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "run_statistics.csv"
    summary_path = output_root / "summary.json"
    run_statistics.to_csv(run_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "latency_definition": (
            "caller-observed complete request latency from request_elapsed_ns"
        ),
        "source_files": [str(path) for path in files],
        "groups": cross_run_summary(run_statistics),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or args.data_root / "analysis"
    run_path, summary_path = analyze(args.platform, args.data_root, output_root)
    print(f"Run-level statistics: {run_path}")
    print(f"Cross-run summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
