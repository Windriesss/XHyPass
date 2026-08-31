#!/usr/bin/env python3
"""
Extract representative statistics from the latest complete NN experiment.

Statistics:
    - Cyclictest maximum latency (us)
    - MnasNet response mean (ms)
    - MnasNet response P95 (ms)
    - Inception response mean (ms)
    - Inception response P95 (ms)

The script follows the same data-selection logic as the plotting script:
1. Scan data/RK3588/NN/**/metadata.json
    2. Keep completed NN experiments only
    3. For each environment, select the latest condition
    4. Merge all runs belonging to that condition
    5. Select the representative cyclictest series using the same rule
       as the plotting script

Expected environments:
    bare
    jailhouse
    xen_credit2
    xen_credit2_WFX
    xen_null
    xen_null_WFX
    XHyPass
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nn_mixed_workload_stats_core as mixed_stats
from xhypass_plot.parser import is_excluded_data_path, parse_histogram


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PLATFORM = "RK3588"
PLATFORM = "RK3588"

DATA_ROOT = PROJECT_ROOT / "data" / DATA_PLATFORM / "NN"

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
OUTPUT_PREFIX = f"{PLATFORM.lower()}_"

FORMAL_PROFILE = "dual-tflite-formal-v1"
BARE_FORMAL_PROFILE = "dual-tflite-formal-v2-cgroup"
FORMAL_DURATION_SECONDS = 600

WORKLOADS = (
    "light",
    "medium",
    "heavy",
)

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

LABELS = {
    "bare": "Bare-metal",
    "jailhouse": "Jailhouse",
    "xen_credit2": "Xen(Credit2)",
    "xen_credit2_WFX": "Xen(Credit2 + native WFx)",
    "xen_null": "Xen(Null)",
    "xen_null_WFX": "Xen(Null + native WFx)",
    "XHyPass": "XHyPass",
}

# Keep related native/WFX environments visually paired, while assigning the
# two Xen scheduler families different hues.  XHyPass remains the warm accent
# color used to emphasize the proposed platform.
COLORS = {
    "bare": "#3B82C4",
    "jailhouse": "#7561A8",
    "xen_credit2": "#D64F70",
    "xen_credit2_WFX": "#F09AB1",
    "xen_null": "#2A9D8F",
    "xen_null_WFX": "#86CFC5",
    "XHyPass": "#F28E2B",
}


_WARNED_RESULT_ISSUES: set[str] = set()


# ============================================================================
# Utility
# ============================================================================

def _warn_result_issue(message: str) -> None:
    if message not in _WARNED_RESULT_ISSUES:
        print(f"[WARN] {message}")
        _WARNED_RESULT_ISSUES.add(message)


def _percentile(values: np.ndarray, q: float) -> float:
    if not len(values):
        return float("nan")

    return float(np.percentile(values, q))


def _fmt(value: float, digits: int = 2) -> str:
    if np.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


def _incomplete_run_issues(run: Path) -> list[str]:
    """Return missing artifacts that make a three-level NN run unusable.

    Some interrupted Xen attempts reach the completed directory with only the
    heavy files left behind.  Mixing those partial attempts into all workload
    rows inflates ``Runs`` and gives each demand level a different effective
    sample set.  A run is therefore admitted only when both model CSVs and at
    least one cyclictest histogram exist for light, medium, and heavy.
    """
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


def _is_formal_configuration(metadata: dict, config: dict) -> bool:
    """Accept only the complete 600-second three-demand formal campaign."""
    experiment = config.get("experiment", {})
    environment = str(config.get("environment_name", ""))
    expected_profile = (
        BARE_FORMAL_PROFILE
        if environment == "bare"
        else FORMAL_PROFILE
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
        and str(metadata.get("condition", "")).startswith(
            f"{expected_profile}_"
        )
    )


# ============================================================================
# Discover experiment runs
# ============================================================================

def _discover_latest_condition_runs() -> dict[str, list[Path]]:
    """
    Discover the latest completed NN condition for each environment.

    This follows the same rule as the plotting script.

    candidates:
        (environment, condition) ->
            [
                (updated_at, run_path),
                ...
            ]

    For each environment, the condition whose latest run has the greatest
    updated_at is selected. All runs belonging to that condition are returned.
    """

    candidates: dict[
        tuple[str, str],
        list[tuple[str, Path]]
    ] = defaultdict(list)

    for metadata_path in DATA_ROOT.rglob("metadata.json"):

        if is_excluded_data_path(metadata_path, DATA_ROOT):
            continue

        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        config = metadata.get("configuration", {})

        if metadata.get("status") != "completed":
            continue

        if config.get("experiment_name") != "NN":
            continue

        if not _is_formal_configuration(metadata, config):
            continue

        results = metadata_path.parent / "results"

        if not results.is_dir():
            continue

        incomplete = _incomplete_run_issues(metadata_path.parent)
        if incomplete:
            _warn_result_issue(
                "Incomplete three-level NN run excluded: "
                f"{metadata_path.parent}; missing: {', '.join(incomplete)}"
            )
            continue

        environment = str(
            config.get("environment_name")
        )

        condition = str(
            metadata.get("condition")
        )

        updated = str(
            metadata.get("updated_at", "")
        )

        candidates[
            (environment, condition)
        ].append(
            (updated, metadata_path.parent)
        )

    selected: dict[str, list[Path]] = {}

    environments = {
        key[0]
        for key in candidates
    }

    for environment in environments:

        conditions = [
            key
            for key in candidates
            if key[0] == environment
        ]

        latest_key = max(
            conditions,
            key=lambda key: max(
                updated
                for updated, _ in candidates[key]
            ),
        )

        selected[environment] = [
            path
            for _, path in sorted(
                candidates[latest_key],
                key=lambda item: item[0],
            )
        ]

    return selected


# ============================================================================
# HTTP inference CSV
# ============================================================================

def _read_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as stream:

        for row in csv.DictReader(stream):

            for key in (
                "timestamp",
                "response_ms",
                "inference_ms",
            ):
                columns[key].append(
                    float(row[key])
                )

    return {
        key: np.asarray(
            values,
            dtype=float,
        )
        for key, values in columns.items()
    }


def _combine_csv(
    runs: list[Path],
    filename: str,
) -> dict[str, np.ndarray]:

    chunks = []

    for run in runs:

        path = run / "results" / filename

        if not path.is_file():

            _warn_result_issue(
                f"Missing NN result, skipped: {path}"
            )

            continue

        try:
            chunks.append(
                _read_csv(path)
            )

        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            _warn_result_issue(
                f"Could not parse NN result, "
                f"skipped: {path}: {exc}"
            )

    return {
        key: (
            np.concatenate(
                [
                    chunk.get(
                        key,
                        np.array([], dtype=float),
                    )
                    for chunk in chunks
                ]
            )
            if chunks
            else np.array([], dtype=float)
        )
        for key in (
            "timestamp",
            "response_ms",
            "inference_ms",
        )
    }


# ============================================================================
# Cyclictest histogram
# ============================================================================

def _combine_histograms(
    runs: list[Path],
    filename: str,
) -> tuple[np.ndarray, np.ndarray, int]:

    merged: dict[int, int] = defaultdict(int)

    overflows = 0

    for run in runs:

        path = run / "results" / filename

        if not path.is_file():

            _warn_result_issue(
                f"Missing cyclictest result, skipped: {path}"
            )

            continue

        try:
            values, counts, overflow = parse_histogram(path)

        except (OSError, ValueError) as exc:

            _warn_result_issue(
                f"Could not parse cyclictest result, "
                f"skipped: {path}: {exc}"
            )

            continue

        for value, count in zip(
            values,
            counts,
            strict=True,
        ):
            merged[int(value)] += int(count)

        overflows += overflow

    values = np.asarray(
        sorted(merged),
        dtype=float,
    )

    counts = np.asarray(
        [
            merged[int(value)]
            for value in values
        ],
        dtype=np.int64,
    )

    return values, counts, overflows


# ============================================================================
# Discover cyclictest files
# ============================================================================

def _completion_rate(
    completion_seconds: np.ndarray,
    bin_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return request completions per second in bins relative to the first one.

    Kept as a small public helper for older analysis callers.  The timestamps
    are completion timestamps, so the first sample defines elapsed time zero.
    """

    samples = np.asarray(completion_seconds, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")

    elapsed = samples - float(samples.min())
    bin_count = max(1, int(np.floor(float(elapsed.max()) / bin_seconds)) + 1)
    edges = np.arange(bin_count + 1, dtype=float) * bin_seconds
    counts, _ = np.histogram(elapsed, bins=edges)
    centers = edges[:-1] + bin_seconds / 2.0
    return centers, counts.astype(float) / bin_seconds


def _available_cpus(runs: list[Path], workload: str) -> list[int]:
    """Discover legacy non-Xen cyclictest CPU result files."""

    pattern = re.compile(
        rf"cyclictest_{re.escape(workload)}_cpu(\d+)\.txt$"
    )
    cpus: set[int] = set()
    for run in runs:
        results = run / "results"
        if not results.is_dir():
            continue
        for path in results.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                cpus.add(int(match.group(1)))
    return sorted(cpus)


def _available_cyclic_series(
    runs: list[Path],
    workload: str,
) -> list[tuple[str, str]]:

    """
    Discover both:

        legacy:
            cyclictest_light_cpu6.txt

        Xen per-domain:
            cyclictest_light_vcpu6_vm1.txt
            cyclictest_light_vcpu6_vm2.txt

    vm1 -> dom0
    vm2 -> dom1
    """

    found: dict[str, str] = {}

    legacy = re.compile(
        rf"cyclictest_{re.escape(workload)}_cpu(\d+)\.txt$"
    )

    xen = re.compile(
        rf"cyclictest_{re.escape(workload)}_vcpu(\d+)_(vm[12])\.txt$"
    )

    for run in runs:

        results = run / "results"

        if not results.is_dir():
            continue

        for path in results.glob(
            f"cyclictest_{workload}_*.txt"
        ):

            if match := legacy.fullmatch(path.name):

                found[path.name] = "Cyclictest"

            elif match := xen.fullmatch(path.name):

                domain = (
                    "dom0"
                    if match.group(2) == "vm1"
                    else "dom1"
                )

                found[path.name] = domain

    return sorted(
        found.items(),
        key=lambda item: item[1],
    )


def _representative_cyclic_series(
    runs: list[Path],
    workload: str,
) -> list[tuple[str, str]]:

    """
    Use exactly the same representative-series rule as the plot.

    Priority:
        1. cpu6
        2. dom0 vCPU6
        3. any dom0 vCPU
        4. first available series
    """

    series = _available_cyclic_series(
        runs,
        workload,
    )

    if not series:
        return []

    preferences = (
        re.compile(
            rf"cyclictest_{re.escape(workload)}_cpu6\.txt$"
        ),
        re.compile(
            rf"cyclictest_{re.escape(workload)}_vcpu6_vm1\.txt$"
        ),
        re.compile(
            rf"cyclictest_{re.escape(workload)}_vcpu\d+_vm1\.txt$"
        ),
    )

    for pattern in preferences:

        for filename, label in series:

            if pattern.fullmatch(filename):

                return [
                    (filename, label)
                ]

    return [
        series[0]
    ]


def _combine_cyclic_series(
    runs: list[Path],
    workload: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Merge every cyclictest stream for a workload into one histogram."""
    merged: dict[int, int] = defaultdict(int)
    overflows = 0
    series = _available_cyclic_series(runs, workload)
    for filename, _ in series:
        values, counts, stream_overflows = _combine_histograms(runs, filename)
        for value, count in zip(values, counts, strict=True):
            merged[int(value)] += int(count)
        overflows += stream_overflows
    values = np.asarray(sorted(merged), dtype=float)
    counts = np.asarray(
        [merged[int(value)] for value in values],
        dtype=np.int64,
    )
    return values, counts, overflows, len(series)


# ============================================================================
# Statistics
# ============================================================================

def _calculate_one(
    environment: str,
    workload: str,
    runs: list[Path],
) -> dict:

    # ----------------------------------------------------------------------
    # MnasNet
    # ----------------------------------------------------------------------

    mnas = _combine_csv(
        runs,
        f"mnasnet_{workload}.csv",
    )

    mnas_response = mnas["response_ms"]

    if len(mnas_response):

        mnas_avg = float(
            np.mean(mnas_response)
        )

        mnas_p95 = _percentile(
            mnas_response,
            95,
        )

        mnas_count = len(mnas_response)

    else:

        mnas_avg = float("nan")
        mnas_p95 = float("nan")
        mnas_count = 0

    # ----------------------------------------------------------------------
    # Inception
    # ----------------------------------------------------------------------

    inception = _combine_csv(
        runs,
        f"inception_{workload}.csv",
    )

    inception_response = inception[
        "response_ms"
    ]

    if len(inception_response):

        inception_avg = float(
            np.mean(inception_response)
        )

        inception_p95 = _percentile(
            inception_response,
            95,
        )

        inception_count = len(
            inception_response
        )

    else:

        inception_avg = float("nan")
        inception_p95 = float("nan")
        inception_count = 0

    # ----------------------------------------------------------------------
    # Cyclictest
    # ----------------------------------------------------------------------

    cyclic_max = float("nan")
    cyclic_samples = 0
    cyclic_overflows = 0
    cyclic_filename = "N/A"
    cyclic_label = "Merged"
    cyclic_series_count = 0

    values, counts, cyclic_overflows, cyclic_series_count = (
        _combine_cyclic_series(runs, workload)
    )

    if cyclic_series_count:
        cyclic_filename = f"Merged {cyclic_series_count} cyclictest streams"

        if len(values) and counts.sum():

            cyclic_max = float(
                values[-1]
            )

            cyclic_samples = int(
                counts.sum()
            )

    return {
        "environment": environment,
        "platform": LABELS.get(
            environment,
            environment,
        ),
        "workload": workload,
        "runs": len(runs),

        "cyclic_filename": cyclic_filename,
        "cyclic_label": cyclic_label,
        "cyclic_series_count": cyclic_series_count,
        "cyclic_samples": cyclic_samples,
        "cyclic_max_us": cyclic_max,
        "cyclic_overflows": cyclic_overflows,

        "mnas_requests": mnas_count,
        "mnas_avg_ms": mnas_avg,
        "mnas_p95_ms": mnas_p95,

        "inception_requests": inception_count,
        "inception_avg_ms": inception_avg,
        "inception_p95_ms": inception_p95,
    }


def _build_statistics(
    runs_by_env: dict[str, list[Path]],
) -> list[dict]:

    rows = []

    for workload in WORKLOADS:

        for environment in ENVIRONMENT_ORDER:

            runs = runs_by_env.get(
                environment
            )

            if not runs:
                continue

            rows.append(
                _calculate_one(
                    environment,
                    workload,
                    runs,
                )
            )

    return rows


# ============================================================================
# Output
# ============================================================================

def _print_selected_runs(
    runs_by_env: dict[str, list[Path]],
) -> None:

    print()
    print("=" * 100)
    print("Selected latest NN conditions")
    print("=" * 100)

    for environment in ENVIRONMENT_ORDER:

        runs = runs_by_env.get(
            environment
        )

        platform = LABELS.get(
            environment,
            environment,
        )

        if not runs:

            print(
                f"{platform:<20} : NOT FOUND"
            )

            continue

        print(
            f"{platform:<20} : {len(runs)} run(s)"
        )

        for run in runs:
            print(
                f"{'':22}{run}"
            )


def _print_statistics(
    rows: list[dict],
) -> None:

    print()
    print("=" * 114)
    print(
        "Representative statistics"
    )
    print("=" * 114)

    print(
        f"{'Demand':<8} "
        f"{'Platform':<18} "
        f"{'Cyclic Max':>11} "
        f"{'Mnas Avg':>12} "
        f"{'Mnas P95':>12} "
        f"{'Inc Avg':>12} "
        f"{'Inc P95':>12} "
        f"{'Runs':>6}"
    )

    print(
        f"{'':<8} "
        f"{'':<18} "
        f"{'(us)':>11} "
        f"{'(ms)':>12} "
        f"{'(ms)':>12} "
        f"{'(ms)':>12} "
        f"{'(ms)':>12} "
        f"{'':>6}"
    )

    print("-" * 114)

    last_workload = None

    for row in rows:

        workload = row["workload"]

        if (
            last_workload is not None
            and workload != last_workload
        ):
            print("-" * 114)

        print(
            f"{WORKLOAD_TITLES[workload]:<8} "
            f"{row['platform']:<18} "
            f"{_fmt(row['cyclic_max_us'], 0):>11} "
            f"{_fmt(row['mnas_avg_ms']):>12} "
            f"{_fmt(row['mnas_p95_ms']):>12} "
            f"{_fmt(row['inception_avg_ms']):>12} "
            f"{_fmt(row['inception_p95_ms']):>12} "
            f"{row['runs']:>6}"
        )

        last_workload = workload


def _print_cyclic_sources(
    rows: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(
        "Representative cyclictest sources"
    )
    print("=" * 100)

    for row in rows:

        print(
            f"{WORKLOAD_TITLES[row['workload']]:<8} "
            f"{row['platform']:<18} "
            f"streams={row['cyclic_series_count']:<3} "
            f"samples={row['cyclic_samples']:<10} "
            f"overflow={row['cyclic_overflows']}"
        )


def _print_latex(
    rows: list[dict],
) -> None:

    print()
    print("=" * 100)
    print("LaTeX table rows")
    print("=" * 100)
    print()

    rows_by_workload: dict[
        str,
        list[dict]
    ] = defaultdict(list)

    for row in rows:
        rows_by_workload[
            row["workload"]
        ].append(row)

    for workload_index, workload in enumerate(
        WORKLOADS
    ):

        workload_rows = rows_by_workload.get(
            workload,
            [],
        )

        if workload_index > 0:
            print(r"\midrule")

        for index, row in enumerate(
            workload_rows
        ):

            cyclic = _fmt(
                row["cyclic_max_us"],
                0,
            )

            mnas_avg = _fmt(
                row["mnas_avg_ms"],
                2,
            )

            mnas_p95 = _fmt(
                row["mnas_p95_ms"],
                2,
            )

            inc_avg = _fmt(
                row["inception_avg_ms"],
                2,
            )

            inc_p95 = _fmt(
                row["inception_p95_ms"],
                2,
            )

            if index == 0:

                prefix = (
                    rf"\multirow{{7}}{{*}}"
                    rf"{{{WORKLOAD_TITLES[workload]}}}"
                )

                print(
                    f"{prefix}\n"
                    f"& {row['platform']:<16} "
                    f"& {cyclic:>4} "
                    f"& {mnas_avg:>7} "
                    f"& {mnas_p95:>7} "
                    f"& {inc_avg:>7} "
                    f"& {inc_p95:>7} \\\\"
                )

            else:

                print(
                    f"& {row['platform']:<16} "
                    f"& {cyclic:>4} "
                    f"& {mnas_avg:>7} "
                    f"& {mnas_p95:>7} "
                    f"& {inc_avg:>7} "
                    f"& {inc_p95:>7} \\\\"
                )


def _write_csv(
    rows: list[dict],
) -> Path:

    output = OUTPUT_ROOT / f"{OUTPUT_PREFIX}nn_representative_stats.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output.resolve()


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=float)
    samples = np.sort(samples[np.isfinite(samples)])
    if not len(samples):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    probability = np.arange(1, len(samples) + 1, dtype=float) / len(samples)
    return samples, probability


def _weighted_ccdf(
    values: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(values) or not int(np.sum(counts)):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    tail_counts = np.cumsum(counts[::-1], dtype=float)[::-1]
    return values, tail_counts / float(np.sum(counts))


THROUGHPUT_BIN_SECONDS = 2.0


def _mean_completion_rate(
    runs: list[Path],
    filename: str,
    bin_seconds: float = THROUGHPUT_BIN_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    rates: list[np.ndarray] = []
    for run in runs:
        path = run / "results" / filename
        if not path.is_file():
            continue
        try:
            timestamps = _read_csv(path)["timestamp"]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        _, rate = _completion_rate(timestamps, bin_seconds=bin_seconds)
        if len(rate):
            rates.append(rate)
    if not rates:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    length = max(len(rate) for rate in rates)
    matrix = np.full((len(rates), length), np.nan, dtype=float)
    for index, rate in enumerate(rates):
        matrix[index, :len(rate)] = rate
    elapsed = (
        np.arange(length, dtype=float) * bin_seconds + bin_seconds / 2.0
    )
    return elapsed, np.nanmean(matrix, axis=0)


def _plot_4x3(runs_by_env: dict[str, list[Path]]) -> list[Path]:
    """Plot formal NN results without exposing board-specific CPU numbers."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        }
    )
    figure, axes = plt.subplots(
        4,
        len(WORKLOADS),
        figsize=(15.0, 9.5),
        constrained_layout=False,
    )
    line_styles = {
        "xen_credit2_WFX": (0, (5, 2)),
        "xen_null_WFX": (0, (5, 2)),
    }
    legend_labels = {
        **LABELS,
        "xen_credit2": "Credit2",
        "xen_credit2_WFX": "Credit2 + native WFx",
        "xen_null": "Null",
        "xen_null_WFX": "Null + native WFx",
    }

    for column, workload in enumerate(WORKLOADS):
        for environment in ENVIRONMENT_ORDER:
            runs = runs_by_env.get(environment, [])
            if not runs:
                continue
            color = COLORS[environment]
            style = line_styles.get(environment, "-")
            label = legend_labels[environment]
            if environment == "XHyPass":
                line_width = 1.75
                zorder = 5
            elif environment in {"xen_credit2", "xen_credit2_WFX"}:
                line_width = 1.3
                zorder = 3
            else:
                line_width = 1.05
                zorder = 2

            elapsed, rate = _mean_completion_rate(
                runs, f"mnasnet_{workload}.csv"
            )
            if len(elapsed):
                axes[0, column].plot(
                    elapsed, rate, color=color, linestyle=style,
                    linewidth=line_width, label=label, zorder=zorder,
                )

            values, counts, _, series_count = _combine_cyclic_series(
                runs, workload
            )
            if series_count:
                x_values, probability = _weighted_ccdf(values, counts)
                if len(x_values):
                    axes[1, column].plot(
                        x_values, probability, color=color, linestyle=style,
                        linewidth=line_width, label=label, zorder=zorder,
                    )

            for row_index, model in ((2, "mnasnet"), (3, "inception")):
                response = _combine_csv(
                    runs, f"{model}_{workload}.csv"
                )["response_ms"]
                x_values, probability = _cdf(response)
                if len(x_values):
                    axes[row_index, column].plot(
                        x_values, probability, color=color, linestyle=style,
                        linewidth=line_width, label=label, zorder=zorder,
                    )

    row_labels = (
        "MnasNet throughput (req/s)",
        "Cyclictest latency CCDF",
        "MnasNet response-time CDF",
        "Inception response-time CDF",
    )
    for row_index, row_label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(row_label, fontsize=9.5, labelpad=4)
        for column in range(len(WORKLOADS)):
            axis = axes[row_index, column]
            axis.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.45,
                alpha=0.18,
            )
            axis.grid(False, which="minor")
            axis.tick_params(axis="both", labelsize=8.5, pad=2)
            axis.tick_params(axis="y", labelleft=True)
            if column > 0:
                axis.set_ylabel("")
            if row_index == 0:
                axis.set_title(
                    WORKLOAD_TITLES[WORKLOADS[column]],
                    fontsize=11.5,
                    pad=5,
                )
                axis.set_xlim(0.0, 600.0)
                axis.set_ylim(0.0, 60.0)
                axis.set_yticks(np.arange(0.0, 61.0, 10.0))
            if row_index == 1:
                axis.set_yscale("log")
                axis.set_xlim(0.0, 72.0)
                axis.set_ylim(1e-7, 1.0)
                axis.set_yticks(
                    10.0 ** np.arange(0, -8, -1, dtype=float)
                )
            elif row_index == 2:
                axis.set_xlim(0.0, 2000.0)
                axis.set_ylim(0.0, 1.01)
                axis.set_yticks(
                    np.arange(0.0, 1.01, 0.2)
                )
            elif row_index == 3:
                axis.set_xlim(480.0, 1300.0)
                axis.set_ylim(0.0, 1.01)
                axis.set_yticks(
                    np.arange(0.0, 1.01, 0.2)
                )

    center_column = len(WORKLOADS) // 2
    axes[0, center_column].set_xlabel(
        "Elapsed time (s)", fontsize=9.5, labelpad=2
    )
    axes[1, center_column].set_xlabel(
        r"Latency ($\mu$s)", fontsize=9.5, labelpad=2
    )
    axes[2, center_column].set_xlabel(
        "Response time (ms)", fontsize=9.5, labelpad=2
    )
    axes[3, center_column].set_xlabel(
        "Response time (ms)", fontsize=9.5, labelpad=2
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=7,
            frameon=False,
            fontsize=8.7,
            columnspacing=1.2,
            handlelength=2.5,
            handletextpad=0.5,
        )

    figure.subplots_adjust(
        left=0.066,
        right=0.992,
        top=0.955,
        bottom=0.085,
        hspace=0.16,
        wspace=0.14,
    )

    outputs = [
        OUTPUT_ROOT / f"{OUTPUT_PREFIX}nn_comprehensive_4x3.png",
        OUTPUT_ROOT / f"{OUTPUT_PREFIX}nn_comprehensive_4x3.pdf",
    ]
    figure.savefig(outputs[0], dpi=240, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    plt.close(figure)
    return [path.resolve() for path in outputs]


# ============================================================================
# Canonical mixed-workload statistics API
# ============================================================================
#
# Keep the historical private names for callers such as
# plot_nn_scatter_2x3.py, but route every formal-run selection and numerical
# aggregation through one shared implementation.  The 4x3 figure, scatter
# helpers, terminal table, CSV, and LaTeX table therefore cannot silently
# diverge in run selection or statistical definitions.

_percentile = mixed_stats.percentile
_discover_latest_condition_runs = mixed_stats.discover_latest_condition_runs
_read_csv = mixed_stats.read_csv
_combine_csv = mixed_stats.combine_csv
_combine_histograms = mixed_stats.combine_histograms
_available_cyclic_series = mixed_stats.available_cyclic_series
_combine_cyclic_series = mixed_stats.combine_cyclic_series
_calculate_one = mixed_stats.calculate_one
_build_statistics = mixed_stats.build_statistics


# ============================================================================
# Validation
# ============================================================================

def _validate_environments(
    runs_by_env: dict[str, list[Path]],
) -> None:

    missing = [
        environment
        for environment in ENVIRONMENT_ORDER
        if environment not in runs_by_env
    ]

    unknown = [
        environment
        for environment in runs_by_env
        if environment
        not in ENVIRONMENT_ORDER
    ]

    if missing:

        print()
        print(
            "[WARN] Missing expected environments:"
        )

        for environment in missing:
            print(
                f"    {environment}"
            )

    if unknown:

        print()
        print(
            "[INFO] Additional environments found "
            "but not included in the paper table:"
        )

        for environment in unknown:
            print(
                f"    {environment}"
            )


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    print(
        f"Data root: {DATA_ROOT.resolve()}"
    )

    if not DATA_ROOT.is_dir():

        raise RuntimeError(
            f"NN data directory does not exist: "
            f"{DATA_ROOT.resolve()}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for old_output in (
        OUTPUT_ROOT / f"{OUTPUT_PREFIX}nn_comprehensive_4x3.png",
        OUTPUT_ROOT / f"{OUTPUT_PREFIX}nn_comprehensive_4x3.pdf",
    ):
        old_output.unlink(missing_ok=True)

    runs_by_env = (
        _discover_latest_condition_runs()
    )

    if not runs_by_env:

        raise RuntimeError(
            f"No completed NN runs found under "
            f"{DATA_ROOT.resolve()}"
        )

    _validate_environments(
        runs_by_env
    )

    _print_selected_runs(
        runs_by_env
    )

    rows = _build_statistics(
        runs_by_env
    )

    if not rows:

        raise RuntimeError(
            "No valid statistics could be calculated."
        )

    _print_statistics(
        rows
    )

    _print_cyclic_sources(
        rows
    )

    _print_latex(
        rows
    )

    figure_outputs = _plot_4x3(runs_by_env)

    print()
    print("=" * 100)
    print("Generated files")
    print("=" * 100)
    for output in figure_outputs:
        print(f"- {output}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
