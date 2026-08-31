from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import RunRecord, combine, weighted_quantile
from .parser import parse_stress_log, scan_runs


@dataclass
class AnalysisConfig:
    data_sources: dict[str, Path]
    output_root: Path
    environments: tuple[str, ...]
    experiments: tuple[str, ...]
    duration_seconds: int
    intervals_us: tuple[int, ...] = ()
    expected_runs_per_condition: int | None = None


QUANTILES = (("p50", .5), ("p95", .95), ("p99", .99), ("p99.9", .999), ("p99.99", .9999))
COLORS = {"bare": "#4472C4", "jailhouse": "#ED7D31", "xen_credit2": "#70AD47", "xen_credit2_WFX": "#A5A5A5", "xen_null": "#7030A0", "xen_null_WFX": "#5B9BD5", "XHyPass": "#C00000"}
LABELS = {"bare": "Bare-metal", "jailhouse": "Jailhouse", "xen_credit2": "Xen(Credit2)", "xen_credit2_WFX": "Xen(Credit2 + native WFx)", "xen_null": "Xen(Null)", "xen_null_WFX": "Xen(Null + native WFx)", "XHyPass": "XHyPass", "cyclictest": "Unloaded", "cyclictest-stress": "Loaded"}


def analyze(config: AnalysisConfig) -> Path:
    records = [record for platform, root in config.data_sources.items() for record in scan_runs(platform, root)]
    records = [r for r in records if r.environment in config.environments and r.experiment in config.experiments and r.duration_seconds == config.duration_seconds and (not config.intervals_us or r.interval_us in config.intervals_us)]
    records = _latest_batches(records)
    records = _select_expected_runs(records, config.expected_runs_per_condition)
    if not records:
        raise RuntimeError("No matching completed experiment records found")

    coverage_rows = _coverage(config, records)
    incomplete = [row for row in coverage_rows if row["status"] == "incomplete"]
    if incomplete:
        details = "; ".join(
            f"{row['platform']}/{row['environment']}/{row['experiment']}/"
            f"i{row['interval_us']}: {row['completed_runs']}/"
            f"{row['expected_runs']}"
            for row in incomplete
        )
        raise RuntimeError(
            f"Refusing to plot incomplete "
            f"{config.expected_runs_per_condition}-run conditions: " + details
        )

    platforms = "-".join(sorted({r.platform for r in records}))
    output = config.output_root
    output_prefix = f"{platforms.lower()}_cyclictest_"
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for old_output in output.glob(f"{output_prefix}{pattern}"):
            old_output.unlink()

    groups = _groups(records)
    aggregate_rows = [_aggregate_row(key, group) for key, group in groups.items()]

    _plot_cdf(output, groups, output_prefix)
    _plot_latency_boxplots(output, groups, config, output_prefix)
    _plot_run_maximum(output, groups, config, output_prefix)
    _plot_tail(output, aggregate_rows, output_prefix)
    return output


def _latest_batches(records: list[RunRecord]) -> list[RunRecord]:
    latest: dict[tuple[str, str, str, int, int], str] = {}
    for record in records:
        key = (record.platform, record.environment, record.experiment, record.duration_seconds, record.interval_us)
        latest[key] = max(latest.get(key, ""), record.batch)
    return [r for r in records if r.batch == latest[(r.platform, r.environment, r.experiment, r.duration_seconds, r.interval_us)]]


def _select_expected_runs(
    records: list[RunRecord], expected: int | None
) -> list[RunRecord]:
    """Select run_1..run_N once per condition for deterministic plotting."""
    if expected is None:
        return records
    grouped: dict[tuple[str, str, str, int, int], dict[int, RunRecord]] = {}
    for record in sorted(records, key=lambda item: str(item.path)):
        key = (
            record.platform,
            record.environment,
            record.experiment,
            record.duration_seconds,
            record.interval_us,
        )
        runs = grouped.setdefault(key, {})
        if record.run in runs:
            raise RuntimeError(
                f"Duplicate completed run {record.run} for condition {key}"
            )
        runs[record.run] = record

    selected: list[RunRecord] = []
    for _, runs in sorted(grouped.items()):
        selected.extend(
            runs[index] for index in range(1, expected + 1) if index in runs
        )
    return selected


def _groups(records: list[RunRecord]) -> dict[tuple[str, str, str, int], list[RunRecord]]:
    groups: dict[tuple[str, str, str, int], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.platform, record.environment, record.experiment, record.interval_us), []).append(record)
    return dict(sorted(groups.items()))


def _run_row(record: RunRecord) -> dict:
    row = {"platform": record.platform, "environment": record.environment, "experiment": record.experiment, "interval_us": record.interval_us, "batch": record.batch, "run": record.run, "duration_seconds": record.duration_seconds, "samples": record.samples, "overflow": record.overflow, "min_us": record.minimum(), "mean_us": record.mean(), "max_us": record.maximum(), "path": str(record.path.resolve())}
    row.update({f"{name}_us": record.quantile(q) for name, q in QUANTILES})
    return row


def _aggregate_row(key: tuple[str, str, str, int], records: list[RunRecord]) -> dict:
    values, counts = combine(records)
    row = {"platform": key[0], "environment": key[1], "experiment": key[2], "interval_us": key[3], "runs": len(records), "samples": int(counts.sum()), "overflow": sum(r.overflow for r in records), "min_us": float(values[0]), "mean_us": float(np.average(values, weights=counts)), "max_us": float(values[-1]), "run_max_mean_us": float(np.mean([r.maximum() for r in records])), "run_max_std_us": float(np.std([r.maximum() for r in records], ddof=1)) if len(records) > 1 else 0.0}
    row.update({f"{name}_us": weighted_quantile(values, counts, q) for name, q in QUANTILES})
    return row


def _stress_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["platform"], row["environment"], row["experiment"], row["interval_us"]), []).append(row)
    result = []
    for (platform, environment, experiment, interval_us), group in sorted(grouped.items()):
        throughput = np.asarray([row["bogo_ops_per_real_second"] for row in group], dtype=float)
        result.append({"platform": platform, "environment": environment, "experiment": experiment, "interval_us": interval_us, "runs": len(group), "bogo_ops_per_real_second_mean": float(throughput.mean()), "bogo_ops_per_real_second_std": float(throughput.std(ddof=1)) if len(group) > 1 else 0.0, "coefficient_of_variation_percent": float(throughput.std(ddof=1) / throughput.mean() * 100) if len(group) > 1 else 0.0})
    return result


def _coverage(config: AnalysisConfig, records: list[RunRecord]) -> list[dict]:
    counts: dict[tuple[str, str, str, int], int] = {}
    for record in records:
        key = (record.platform, record.environment, record.experiment, record.interval_us)
        counts[key] = counts.get(key, 0) + 1
    intervals = config.intervals_us or tuple(sorted({r.interval_us for r in records}))
    rows = []
    for platform in config.data_sources:
        for experiment in config.experiments:
            for interval_us in intervals:
                for environment in config.environments:
                    actual = counts.get((platform, environment, experiment, interval_us), 0)
                    expected = config.expected_runs_per_condition
                    rows.append({
                        "platform": platform,
                        "environment": environment,
                        "experiment": experiment,
                        "interval_us": interval_us,
                        "completed_runs": actual,
                        "expected_runs": expected if expected is not None else "",
                        "missing_runs": max(0, expected - actual) if expected is not None else "",
                        "status": "complete" if expected is None or actual >= expected else "incomplete",
                    })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _label_environment(environment: str) -> str:
    return LABELS.get(environment, environment)


def _save(fig, output: Path, name: str, prefix: str = "") -> None:
    fig.tight_layout()
    fig.savefig(output / f"{prefix}{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{prefix}{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _platform_label_from_groups(groups) -> str:
    platforms = sorted({key[0] for key in groups})
    return " / ".join(platforms)


def _plot_cdf(output: Path, groups, output_prefix: str = "") -> None:
    combinations = sorted({(key[2], key[3]) for key in groups})
    for experiment, interval_us in combinations:
        selected = [(key, value) for key, value in groups.items() if key[2] == experiment and key[3] == interval_us]
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        for key, records in selected:
            values, counts = combine(records)
            cdf = np.cumsum(counts) / counts.sum()
            color = COLORS.get(key[1], "#666666")
            label = _label_environment(key[1])
            axes[0].step(values, cdf, where="post", label=label, color=color, linewidth=2)
            exceedance = 1.0 - cdf
            mask = exceedance > 0
            axes[1].step(values[mask], exceedance[mask], where="post", label=label, color=color, linewidth=2)
        axes[0].set(xscale="log", xlabel="Cyclictest latency (us)", ylabel="Cumulative probability")
        axes[0].set_ylim(0, 1.005)
        axes[0].grid(True, which="both", alpha=.25)
        axes[0].legend(fontsize=8)
        axes[1].set(xscale="log", yscale="log", xlabel="Cyclictest latency (us)", ylabel="Probability of a higher latency")
        axes[1].grid(True, which="both", alpha=.25)
        _save(fig, output, f"latency_cdf_{experiment}_i{interval_us}", output_prefix)


def _weighted_box_stats(records: list[RunRecord], label: str) -> dict:
    """Build Tukey boxplot statistics directly from weighted histograms."""
    values, counts = combine(records)
    q1 = weighted_quantile(values, counts, .25)
    median = weighted_quantile(values, counts, .5)
    q3 = weighted_quantile(values, counts, .75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    fliers = values[(values < lower_fence) | (values > upper_fence)]
    return {
        "label": label,
        "whislo": float(inside[0]),
        "q1": q1,
        "med": median,
        "q3": q3,
        "whishi": float(inside[-1]),
        # Plot one marker per outlying latency bucket, as in the reference.
        "fliers": fliers.tolist(),
        "mean": float(np.average(values, weights=counts)),
        "max": float(values[-1]),
    }


def _plot_latency_boxplots(
    output: Path,
    groups,
    config: AnalysisConfig,
    output_prefix: str = "",
) -> None:
    intervals = config.intervals_us or tuple(sorted({key[3] for key in groups}))
    for experiment in config.experiments:
        fig, axes = plt.subplots(1, len(intervals), figsize=(14.5, 5.5), squeeze=False)
        for column, interval_us in enumerate(intervals):
            ax = axes[0][column]
            stats, colors = [], []
            for environment in config.environments:
                matches = [
                    group
                    for key, group in groups.items()
                    if key[1:] == (environment, experiment, interval_us)
                ]
                if not matches:
                    continue
                group = matches[0]
                stats.append(
                    _weighted_box_stats(
                        group,
                        f"{_label_environment(environment)}\n(n={len(group)})",
                    )
                )
                colors.append(COLORS.get(environment, "#888888"))
            artists = ax.bxp(
                stats,
                showfliers=True,
                showmeans=False,
                patch_artist=True,
                boxprops={"edgecolor": "black", "linewidth": 1.2},
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"color": "black", "linewidth": 1.2},
                capprops={"color": "black", "linewidth": 1.2},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 4.5,
                    "linestyle": "none",
                },
            )
            for box, color in zip(artists["boxes"], colors, strict=True):
                box.set_facecolor(color)
                box.set_alpha(.75)
            for position, item in enumerate(stats, start=1):
                ax.annotate(
                    f"Max: {item['max']:.0f}",
                    (position, item["max"]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_ylabel("Cyclictest latency (us)")
            ax.tick_params(axis="x", labelrotation=22, labelsize=8)
            ax.grid(True, axis="y", linestyle="--", color="#999999", alpha=.65)
            ax.set_axisbelow(True)
            upper = max(item["max"] for item in stats)
            ax.set_ylim(0, max(upper * 1.16, 10))
        _save(fig, output, f"latency_boxplot_{experiment}", output_prefix)


def _plot_run_maximum(
    output: Path,
    groups,
    config: AnalysisConfig,
    output_prefix: str = "",
) -> None:
    intervals = config.intervals_us or tuple(sorted({key[3] for key in groups}))
    fig, axes = plt.subplots(len(config.experiments), len(intervals), figsize=(14.5, 8.2), squeeze=False)
    for row, experiment in enumerate(config.experiments):
        for column, interval_us in enumerate(intervals):
            ax = axes[row][column]
            data, labels, colors = [], [], []
            for environment in config.environments:
                matches = [group for key, group in groups.items() if key[1:] == (environment, experiment, interval_us)]
                if not matches:
                    continue
                group = matches[0]
                data.append([record.maximum() for record in group])
                labels.append(f"{_label_environment(environment)}\n(n={len(group)})")
                colors.append(COLORS.get(environment, "#666666"))
            if data:
                boxes = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
                for box, color in zip(boxes["boxes"], colors, strict=True):
                    box.set_facecolor(color)
                    box.set_alpha(.65)
            ax.set_yscale("log")
            ax.set_ylabel("Per-run maximum latency (us)")
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            ax.grid(True, axis="y", which="both", alpha=.25)
    _save(fig, output, "run_maximum_stability", output_prefix)


def _plot_tail(
    output: Path,
    rows: list[dict],
    output_prefix: str = "",
) -> None:
    experiments = sorted({row["experiment"] for row in rows}, key=lambda item: (item.endswith("stress"), item))
    intervals = sorted({row["interval_us"] for row in rows})
    metrics = (("p99.9_us", "P99.9"), ("p99.99_us", "P99.99"), ("max_us", "Max"))
    fig, axes = plt.subplots(
        len(experiments),
        len(intervals),
        figsize=(14.5, 8.2),
        squeeze=False,
        sharey="row",
    )
    for row_index, experiment in enumerate(experiments):
        experiment_rows = [item for item in rows if item["experiment"] == experiment]
        row_upper = max(float(item["max_us"]) for item in experiment_rows) * 1.20
        for column, interval_us in enumerate(intervals):
            ax = axes[row_index][column]
            selected = [item for item in experiment_rows if item["interval_us"] == interval_us]
            selected.sort(key=lambda item: list(COLORS).index(item["environment"]) if item["environment"] in COLORS else 999)
            x = np.arange(len(selected))
            width = .24
            for metric_index, (metric, label) in enumerate(metrics):
                values = [float(item[metric]) for item in selected]
                bars = ax.bar(
                    x + (metric_index - 1) * width,
                    values,
                    width,
                    label=label,
                    alpha=(.65, .8, 1)[metric_index],
                    edgecolor="black",
                    linewidth=.35,
                )
                ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=6)
            ax.set_xticks(x, [_label_environment(item["environment"]) for item in selected], rotation=25, ha="right", fontsize=8)
            ax.set_ylim(0, max(row_upper, 1))
            ax.set_ylabel("Latency (us)")
            ax.grid(True, axis="y", linestyle="--", alpha=.35)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
    _save(fig, output, "tail_quantiles", output_prefix)


def _markdown(records, rows, coverage_rows, stress_rows, stress_summary_rows) -> str:
    incomplete = [row for row in coverage_rows if row["status"] == "incomplete"]
    lines = ["# Cyclictest analysis", "", f"- Completed runs analyzed: {len(records)}", f"- Duration per run: {records[0].duration_seconds} s", f"- Histogram overflows: {sum(r.overflow for r in records)}", f"- Incomplete conditions: {len(incomplete)}", "", "## Coverage", ""]
    if incomplete:
        for row in incomplete:
            lines.append(f"- {row['environment']} / {row['experiment']} / interval={row['interval_us']} us: {row['completed_runs']}/{row['expected_runs']} runs")
    else:
        lines.append("- All configured conditions are complete.")
    lines += ["", "## Aggregate results", "", "| Platform | Environment | Load | Interval (us) | Runs | Samples | Mean (us) | P99.9 (us) | P99.99 (us) | Max (us) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['platform']} | {LABELS.get(row['environment'], row['environment'])} | {LABELS.get(row['experiment'], row['experiment'])} | {row['interval_us']} | {row['runs']} | {row['samples']} | {row['mean_us']:.3f} | {row['p99.9_us']:.0f} | {row['p99.99_us']:.0f} | {row['max_us']:.0f} |")
    lines += ["", "## Initial observations", ""]
    lookup = {(row['platform'], row['environment'], row['experiment'], row['interval_us']): row for row in rows}
    for platform in sorted({row['platform'] for row in rows}):
        for environment in sorted({row['environment'] for row in rows}):
            for interval_us in sorted({row['interval_us'] for row in rows}):
                base = lookup.get((platform, environment, "cyclictest", interval_us))
                stress = lookup.get((platform, environment, "cyclictest-stress", interval_us))
                if base and stress:
                    ratio = stress['max_us'] / base['max_us']
                    lines.append(f"- {platform} {LABELS.get(environment, environment)}, interval={interval_us} us: stress changes aggregate max from {base['max_us']:.0f} to {stress['max_us']:.0f} us ({ratio:.2f}x); stress P99.99 is {stress['p99.99_us']:.0f} us.")
    if stress_rows:
        lines += ["", "## Stress-ng load stability", "", "| Platform | Environment | Interval (us) | Runs | Bogo ops/s mean | Std. dev. | CV |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in stress_summary_rows:
            lines.append(f"| {row['platform']} | {LABELS.get(row['environment'], row['environment'])} | {row['interval_us']} | {row['runs']} | {row['bogo_ops_per_real_second_mean']:.2f} | {row['bogo_ops_per_real_second_std']:.2f} | {row['coefficient_of_variation_percent']:.2f}% |")
        lines += ["", f"Parsed stress-ng metrics for {len(stress_rows)} runs; see `stress_ng.csv` for per-run values."]
    lines += ["", "These results describe the collected runs; formal claims should use additional repetitions/platforms and a predefined statistical protocol.", ""]
    return "\n".join(lines)
