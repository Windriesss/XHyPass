#!/usr/bin/env python3
"""Single source of truth for the formal RK3588 NN mixed-workload data.

Both the comprehensive 4x3 figure and the CSV/LaTeX paper table import this
module.  Keep run discovery, file matching, pooled inference statistics, and
the two-stream cyclictest aggregation here so their statistical definitions
cannot drift independently.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from xhypass_plot.parser import is_excluded_data_path, parse_histogram


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "RK3588" / "NN"
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

FORMAL_PROFILE = "dual-tflite-formal-v1"
BARE_FORMAL_PROFILE = "dual-tflite-formal-v2-cgroup"
FORMAL_DURATION_SECONDS = 600

WORKLOADS = ("light", "medium", "heavy")
WORKLOAD_TITLES = {
    "light": "Light",
    "medium": "Medium",
    "heavy": "Heavy",
}
ENVIRONMENT_ORDER = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
    "XHyPass",
)
CONFIGURATION_LABELS = {
    "bare": "Bare-metal",
    "jailhouse": "Jailhouse",
    "xen_credit2": "Credit2",
    "xen_credit2_WFX": "Credit2 + native WFx",
    "xen_null": "Null",
    "xen_null_WFX": "Null + native WFx",
    "XHyPass": "XHyPass",
}

_WARNED_RESULT_ISSUES: set[str] = set()


def warn_result_issue(message: str) -> None:
    if message not in _WARNED_RESULT_ISSUES:
        print(f"[WARN] {message}")
        _WARNED_RESULT_ISSUES.add(message)


def percentile(values: np.ndarray, q: float) -> float:
    if not len(values):
        return float("nan")
    return float(np.percentile(values, q))


def incomplete_run_issues(run: Path) -> list[str]:
    """Return artifacts missing from a complete three-demand formal run."""
    results = run / "results"
    issues: list[str] = []
    for workload in WORKLOADS:
        for filename in (
            f"mnasnet_{workload}.csv",
            f"inception_{workload}.csv",
        ):
            path = results / filename
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(filename)
        histograms = [
            path
            for path in results.glob(f"cyclictest_{workload}_*.txt")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not histograms:
            issues.append(f"cyclictest_{workload}_*.txt")
    return issues


def is_formal_configuration(metadata: dict, config: dict) -> bool:
    experiment = config.get("experiment", {})
    environment = str(config.get("environment_name", ""))
    expected_profile = (
        BARE_FORMAL_PROFILE if environment == "bare" else FORMAL_PROFILE
    )
    try:
        duration_seconds = int(experiment.get("duration_seconds", 0))
    except (TypeError, ValueError):
        return False
    profile_names = [
        str(profile.get("name"))
        for profile in experiment.get("profiles", [])
        if isinstance(profile, dict)
    ]
    return (
        str(experiment.get("profile_name")) == expected_profile
        and duration_seconds == FORMAL_DURATION_SECONDS
        and profile_names == list(WORKLOADS)
        and str(metadata.get("condition", "")).startswith(f"{expected_profile}_")
    )


def discover_latest_condition_runs() -> dict[str, list[Path]]:
    """Select the latest complete formal condition for every environment."""
    candidates: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    for metadata_path in DATA_ROOT.rglob("metadata.json"):
        if is_excluded_data_path(metadata_path, DATA_ROOT):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = metadata.get("configuration", {})
        if metadata.get("status") != "completed":
            continue
        if config.get("experiment_name") != "NN":
            continue
        if not is_formal_configuration(metadata, config):
            continue
        results = metadata_path.parent / "results"
        if not results.is_dir():
            continue
        incomplete = incomplete_run_issues(metadata_path.parent)
        if incomplete:
            warn_result_issue(
                "Incomplete three-level NN run excluded: "
                f"{metadata_path.parent}; missing: {', '.join(incomplete)}"
            )
            continue
        environment = str(config.get("environment_name"))
        condition = str(metadata.get("condition"))
        updated = str(metadata.get("updated_at", ""))
        candidates[(environment, condition)].append((updated, metadata_path.parent))

    selected: dict[str, list[Path]] = {}
    for environment in {key[0] for key in candidates}:
        conditions = [key for key in candidates if key[0] == environment]
        latest_key = max(
            conditions,
            key=lambda key: max(updated for updated, _ in candidates[key]),
        )
        selected[environment] = [
            path
            for _, path in sorted(candidates[latest_key], key=lambda item: item[0])
        ]
    return selected


def read_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ("timestamp", "response_ms", "inference_ms"):
                columns[key].append(float(row[key]))
    return {
        key: np.asarray(values, dtype=float)
        for key, values in columns.items()
    }


def combine_csv(runs: list[Path], filename: str) -> dict[str, np.ndarray]:
    chunks: list[dict[str, np.ndarray]] = []
    for run in runs:
        path = run / "results" / filename
        if not path.is_file():
            warn_result_issue(f"Missing NN result, skipped: {path}")
            continue
        try:
            chunks.append(read_csv(path))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            warn_result_issue(f"Could not parse NN result, skipped: {path}: {exc}")
    return {
        key: (
            np.concatenate(
                [chunk.get(key, np.array([], dtype=float)) for chunk in chunks]
            )
            if chunks
            else np.array([], dtype=float)
        )
        for key in ("timestamp", "response_ms", "inference_ms")
    }


def combine_histograms(
    runs: list[Path], filename: str
) -> tuple[np.ndarray, np.ndarray, int]:
    merged: dict[int, int] = defaultdict(int)
    overflows = 0
    for run in runs:
        path = run / "results" / filename
        if not path.is_file():
            warn_result_issue(f"Missing cyclictest result, skipped: {path}")
            continue
        try:
            values, counts, overflow = parse_histogram(path)
        except (OSError, ValueError) as exc:
            warn_result_issue(
                f"Could not parse cyclictest result, skipped: {path}: {exc}"
            )
            continue
        for value, count in zip(values, counts, strict=True):
            merged[int(value)] += int(count)
        overflows += overflow
    values = np.asarray(sorted(merged), dtype=float)
    counts = np.asarray(
        [merged[int(value)] for value in values], dtype=np.int64
    )
    return values, counts, overflows


def available_cyclic_series(
    runs: list[Path], workload: str
) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    legacy = re.compile(rf"cyclictest_{re.escape(workload)}_cpu(\d+)\.txt$")
    xen = re.compile(
        rf"cyclictest_{re.escape(workload)}_vcpu(\d+)_(vm[12])\.txt$"
    )
    for run in runs:
        results = run / "results"
        if not results.is_dir():
            continue
        for path in results.glob(f"cyclictest_{workload}_*.txt"):
            if legacy.fullmatch(path.name):
                found[path.name] = "Cyclictest"
            elif match := xen.fullmatch(path.name):
                found[path.name] = "dom0" if match.group(2) == "vm1" else "dom1"
    return sorted(found.items(), key=lambda item: item[1])


def combine_cyclic_series(
    runs: list[Path], workload: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Merge both cyclictest streams across all selected runs."""
    merged: dict[int, int] = defaultdict(int)
    overflows = 0
    series = available_cyclic_series(runs, workload)
    for filename, _ in series:
        values, counts, stream_overflows = combine_histograms(runs, filename)
        for value, count in zip(values, counts, strict=True):
            merged[int(value)] += int(count)
        overflows += stream_overflows
    values = np.asarray(sorted(merged), dtype=float)
    counts = np.asarray(
        [merged[int(value)] for value in values], dtype=np.int64
    )
    return values, counts, overflows, len(series)


def calculate_one(
    environment: str, workload: str, runs: list[Path]
) -> dict:
    """Calculate the canonical paper statistics for one configuration."""
    mnas_response = combine_csv(runs, f"mnasnet_{workload}.csv")["response_ms"]
    inception_response = combine_csv(
        runs, f"inception_{workload}.csv"
    )["response_ms"]
    values, counts, overflows, series_count = combine_cyclic_series(
        runs, workload
    )
    populated = values[counts > 0] if len(values) else values
    cyclic_max = float(np.max(populated)) if len(populated) else float("nan")
    return {
        "environment": environment,
        "platform": CONFIGURATION_LABELS.get(environment, environment),
        "workload": workload,
        "runs": len(runs),
        "cyclic_filename": f"Merged {series_count} cyclictest streams",
        "cyclic_label": "Merged",
        "cyclic_series_count": series_count,
        "cyclic_samples": int(np.sum(counts)),
        "cyclic_max_us": cyclic_max,
        "cyclic_overflows": overflows,
        "mnas_requests": len(mnas_response),
        "mnas_avg_ms": (
            float(np.mean(mnas_response)) if len(mnas_response) else float("nan")
        ),
        "mnas_p95_ms": percentile(mnas_response, 95),
        "inception_requests": len(inception_response),
        "inception_avg_ms": (
            float(np.mean(inception_response))
            if len(inception_response)
            else float("nan")
        ),
        "inception_p95_ms": percentile(inception_response, 95),
    }


def build_statistics(runs_by_env: dict[str, list[Path]]) -> list[dict]:
    """Return all 3 demand x 7 configuration canonical paper rows."""
    rows: list[dict] = []
    for workload in WORKLOADS:
        for environment in ENVIRONMENT_ORDER:
            runs = runs_by_env.get(environment)
            if runs:
                rows.append(calculate_one(environment, workload, runs))
    return rows
