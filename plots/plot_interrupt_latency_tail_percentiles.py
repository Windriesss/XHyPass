#!/usr/bin/env python3
"""Plot run-level interrupt-latency tail statistics unloaded and loaded.

Every percentile is computed independently from the final complete ``t0``
histogram of each published run.  Raw samples from different runs are never
merged.  The plotted marker is the median of five run-level values and the
vertical error bar is their min-max range.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from xhypass_plot.interrupt_latency import (
    parse_last_t0_histogram,
    weighted_quantile,
)
from xhypass_plot.style import apply_paper_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
EXPECTED_RUNS = 5

PLATFORMS = {
    "RK3588": "RK3588",
    "E2000Q": "E2000Q",
}

CONFIGURATIONS = (
    ("bare", "Bare-metal", "#3B82C4", "o"),
    ("xen_credit2", "Credit2", "#D64F70", "s"),
    ("xen_null_WFX", "Null + native WFx", "#86CFC5", "D"),
    ("XHyPass", "XHyPass", "#F28E2B", "^"),
)

CONDITIONS = (
    ("idle", "Unloaded"),
    ("stress", "Loaded"),
)

METRICS = (
    ("P99", 0.99),
    ("P99.9", 0.999),
    ("P99.99", 0.9999),
    ("Max", None),
)


def configure_style() -> None:
    apply_paper_style()


def run_metrics(path: Path) -> dict[str, float]:
    """Return tail metrics from one run's final complete t0 histogram."""
    histogram = parse_last_t0_histogram(path)
    metrics: dict[str, float] = {}
    for label, quantile in METRICS:
        latency_ns = (
            max(histogram)
            if quantile is None
            else weighted_quantile(histogram, quantile)
        )
        metrics[label] = latency_ns / 1000.0
    return metrics


def load_run_level_statistics(
    project_root: Path = PROJECT_ROOT,
) -> dict[tuple[str, str, str, str], list[float]]:
    """Load exactly five independent runs for every plotted series."""
    statistics: dict[tuple[str, str, str, str], list[float]] = {}
    for platform, data_name in PLATFORMS.items():
        for condition, _ in CONDITIONS:
            for environment, _, _, _ in CONFIGURATIONS:
                series_dir = (
                    project_root
                    / "data"
                    / data_name
                    / "int-latency"
                    / environment
                    / condition
                )
                paths = sorted(series_dir.glob("rtos_run*.log"))
                if len(paths) != EXPECTED_RUNS:
                    raise RuntimeError(
                        f"Expected {EXPECTED_RUNS} published {condition} runs "
                        f"for {platform}/{environment}, found {len(paths)}: "
                        f"{paths}"
                    )
                rows = [run_metrics(path) for path in paths]
                for metric, _ in METRICS:
                    statistics[(platform, condition, environment, metric)] = [
                        row[metric] for row in rows
                    ]
    return statistics


def median_value(
    statistics: dict[tuple[str, str, str, str], list[float]],
    platform: str,
    condition: str,
    environment: str,
    metric: str,
) -> float:
    return float(
        np.median(statistics[(platform, condition, environment, metric)])
    )


def print_summary(
    statistics: dict[tuple[str, str, str, str], list[float]],
) -> None:
    for condition, condition_label in CONDITIONS:
        print(f"\n{condition_label}")
        print(
            "Platform | Configuration | Percentile | run values | median | min | max"
        )
        print("-" * 118)
        for platform in PLATFORMS:
            for environment, configuration, _, _ in CONFIGURATIONS:
                for metric, _ in METRICS:
                    values = statistics[
                        (platform, condition, environment, metric)
                    ]
                    rendered = (
                        "[" + ", ".join(f"{value:.2f}" for value in values) + "]"
                    )
                    print(
                        f"{platform} | {configuration} | {metric} | "
                        f"{rendered} | {np.median(values):.2f} | "
                        f"{min(values):.2f} | {max(values):.2f}"
                    )

        print("\nXHyPass relative reduction vs Null + native WFx")
        print(
            "Platform | Percentile | XHyPass median | "
            "Null + native WFx median | reduction"
        )
        print("-" * 94)
        for platform in PLATFORMS:
            for metric, _ in METRICS:
                xhypass = median_value(
                    statistics, platform, condition, "XHyPass", metric
                )
                baseline = median_value(
                    statistics, platform, condition, "xen_null_WFX", metric
                )
                reduction = (baseline - xhypass) / baseline * 100.0
                print(
                    f"{platform} | {metric} | {xhypass:.2f} us | "
                    f"{baseline:.2f} us | {reduction:.2f}%"
                )

        print("\nCredit2 vs Null + native WFx difference")
        print(
            "Platform | Percentile | Credit2 median | "
            "Null + native WFx median | Credit2 - Null + native WFx"
        )
        print("-" * 112)
        for platform in PLATFORMS:
            for metric, _ in METRICS:
                credit2 = median_value(
                    statistics, platform, condition, "xen_credit2", metric
                )
                null_wfx = median_value(
                    statistics, platform, condition, "xen_null_WFX", metric
                )
                difference = credit2 - null_wfx
                relative = difference / null_wfx * 100.0
                print(
                    f"{platform} | {metric} | {credit2:.2f} us | "
                    f"{null_wfx:.2f} us | {difference:+.2f} us "
                    f"({relative:+.2f}%)"
                )

        print("\nTail growth ratios computed from run-level medians")
        print("Platform | Configuration | P99.99 / P99 | Max / P99.99")
        print("-" * 78)
        for platform in PLATFORMS:
            for environment, configuration, _, _ in CONFIGURATIONS:
                p99 = median_value(
                    statistics, platform, condition, environment, "P99"
                )
                p9999 = median_value(
                    statistics, platform, condition, environment, "P99.99"
                )
                maximum = median_value(
                    statistics, platform, condition, environment, "Max"
                )
                print(
                    f"{platform} | {configuration} | {p9999 / p99:.3f} | "
                    f"{maximum / p9999:.3f}"
                )


def plot_tail_percentiles(
    statistics: dict[tuple[str, str, str, str], list[float]],
    output: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.25),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.99,
        bottom=0.11,
        top=0.86,
        hspace=0.28,
        wspace=0.10,
    )
    centers = np.arange(len(METRICS), dtype=float)
    offsets = np.linspace(-0.09, 0.09, len(CONFIGURATIONS))

    # Keep the matrix consistent with interrupt_latency_runmax.pdf:
    # platform is encoded by row and load condition by column.
    for row, platform in enumerate(PLATFORMS):
        for column, (condition, condition_label) in enumerate(CONDITIONS):
            axis = axes[row, column]
            for offset, (environment, _, color, marker) in zip(
                offsets, CONFIGURATIONS, strict=True
            ):
                x_values = centers + offset
                medians: list[float] = []
                lower: list[float] = []
                upper: list[float] = []
                for metric, _ in METRICS:
                    values = np.asarray(
                        statistics[(platform, condition, environment, metric)],
                        dtype=float,
                    )
                    median = float(np.median(values))
                    medians.append(median)
                    lower.append(median - float(values.min()))
                    upper.append(float(values.max()) - median)

                axis.errorbar(
                    x_values,
                    medians,
                    yerr=np.asarray([lower, upper]),
                    fmt=marker + "-",
                    color=color,
                    ecolor=color,
                    linewidth=1.25,
                    elinewidth=1.0,
                    capsize=2.6,
                    capthick=0.9,
                    markersize=5.0,
                    markerfacecolor=color,
                    markeredgecolor="#303030",
                    markeredgewidth=0.6,
                    zorder=3,
                )
            if row == 0:
                axis.set_title(condition_label, pad=5)
            axis.set_xlim(-0.38, len(METRICS) - 0.62)
            axis.set_ylim(0, 22)
            axis.set_xticks(centers, [label for label, _ in METRICS])
            axis.set_yticks((0, 5, 10, 15, 20))
            axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            axis.xaxis.grid(False)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[row, 0].set_ylabel(r"Interrupt latency ($\mu$s)", labelpad=5)
        box = axes[row, 0].get_position()
        figure.text(
            0.018,
            box.y0 + box.height / 2,
            platform,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="#303030",
            markeredgewidth=0.6,
            linewidth=1.25,
            markersize=5.0,
            label=configuration,
        )
        for _, configuration, color, marker in CONFIGURATIONS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        columnspacing=1.1,
        handlelength=1.7,
        handletextpad=0.45,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    print(f"\nSaved {output.resolve()}")


def main() -> int:
    statistics = load_run_level_statistics()
    print_summary(statistics)
    plot_tail_percentiles(
        statistics,
        OUTPUT_ROOT / "interrupt_latency_tail_percentiles.pdf",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
