#!/usr/bin/env python3
"""Plot run-level cyclictest maxima for RK3588 and E2000Q.

Each marker is the median of the independent run-level maxima.  Its vertical
error bar spans the minimum to maximum run-level maximum.  The two platforms
share one y-axis scale so that their results remain directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from xhypass_plot.parser import scan_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# These are code-level experiment parameters by design; no CLI arguments are
# needed for routine use.
DURATION_SECONDS = 600
INTERVAL_US = 1_000
EXPECTED_RUNS = 5
ALLOW_INCOMPLETE = True
MINIMUM_RUNS = 4
Y_MAX_US = 75
Y_TICKS_US = (0, 15, 30, 45, 60, 75)

PLATFORMS = {
    "RK3588": "RK3588",
    "E2000Q": "E2000Q",
}

ENVIRONMENTS = (
    ("bare", "Bare metal"),
    ("jailhouse", "Jailhouse"),
    ("xen_credit2", "Credit2"),
    ("xen_credit2_WFX", "Credit2 + native WFx"),
    ("xen_null", "Null"),
    ("xen_null_WFX", "Null + native WFx"),
    ("XHyPass", "XHyPass"),
)

CONDITIONS = (
    ("cyclictest", "Unloaded"),
    ("cyclictest-stress", "Loaded"),
)

COLORS = {
    "bare": "#3B82C4",
    "jailhouse": "#7561A8",
    "xen_credit2": "#D64F70",
    "xen_credit2_WFX": "#F09AB1",
    "xen_null": "#2A9D8F",
    "xen_null_WFX": "#86CFC5",
    "XHyPass": "#F28E2B",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_platform_maxima(
    platform: str,
    data_root: Path,
) -> dict[tuple[str, str], list[float]]:
    """Load one maximum per completed formal run without merging samples."""
    records = [
        record
        for record in scan_runs(platform, data_root)
        if record.duration_seconds == DURATION_SECONDS
        and record.interval_us == INTERVAL_US
        and record.environment in {key for key, _ in ENVIRONMENTS}
        and record.experiment in {key for key, _ in CONDITIONS}
    ]
    maxima: dict[tuple[str, str], list[float]] = {}
    for environment, _ in ENVIRONMENTS:
        for experiment, condition_label in CONDITIONS:
            matching = [
                record
                for record in records
                if record.environment == environment
                and record.experiment == experiment
            ]
            by_run = {}
            for record in matching:
                if record.run in by_run:
                    raise RuntimeError(
                        f"Duplicate run {record.run} for "
                        f"{platform}/{environment}/{experiment}: "
                        f"{by_run[record.run].path}, {record.path}"
                    )
                by_run[record.run] = record
            values = [
                by_run[run].maximum()
                for run in sorted(by_run)
            ]
            if len(values) != EXPECTED_RUNS:
                message = (
                    f"{platform}/{environment}/{condition_label}: expected "
                    f"{EXPECTED_RUNS} runs, found {len(values)}"
                )
                if not ALLOW_INCOMPLETE or len(values) < MINIMUM_RUNS:
                    raise RuntimeError(message)
                print(f"[WARN] {message}; plotting available runs with n label.")
            maxima[(environment, experiment)] = values
    return maxima


def axis_labels(
    maxima: dict[tuple[str, str], list[float]],
    experiment: str,
) -> list[str]:
    labels = []
    for environment, label in ENVIRONMENTS:
        rendered = {
            "Bare metal": "Bare\nmetal",
            "Credit2 + native WFx": "Credit2\n+ native WFx",
            "Null + native WFx": "Null\n+ native WFx",
        }.get(label, label)
        count = len(maxima[(environment, experiment)])
        if count != EXPECTED_RUNS:
            rendered += f"\n(n={count})"
        labels.append(rendered)
    return labels


def plot_matrix(output: Path) -> None:
    configure_style()
    maxima_by_platform = {
        platform: load_platform_maxima(
            platform,
            PROJECT_ROOT / "data" / data_name,
        )
        for platform, data_name in PLATFORMS.items()
    }

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
        bottom=0.135,
        top=0.91,
        hspace=0.34,
        wspace=0.16,
    )
    x_positions = np.arange(len(ENVIRONMENTS), dtype=float)

    for row, (platform, maxima) in enumerate(maxima_by_platform.items()):
        for column, (experiment, condition_label) in enumerate(CONDITIONS):
            axis = axes[row, column]
            for x_value, (environment, _) in zip(
                x_positions, ENVIRONMENTS, strict=True
            ):
                values = np.asarray(
                    maxima[(environment, experiment)], dtype=float
                )
                median = float(np.median(values))
                minimum = float(values.min())
                maximum = float(values.max())
                color = COLORS[environment]
                axis.errorbar(
                    x_value,
                    median,
                    yerr=np.asarray(
                        [[median - minimum], [maximum - median]]
                    ),
                    fmt="o",
                    markersize=4.7,
                    markerfacecolor=color,
                    markeredgecolor="#303030",
                    markeredgewidth=0.7,
                    ecolor=color,
                    elinewidth=1.15,
                    capsize=3.0,
                    capthick=1.0,
                    zorder=3,
                )
                axis.annotate(
                    f"{maximum:.0f}",
                    (x_value, maximum),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color="#303030",
                    zorder=4,
                )

            axis.set_xlim(-0.55, len(ENVIRONMENTS) - 0.45)
            axis.set_xticks(
                x_positions,
                axis_labels(maxima, experiment),
            )
            axis.set_ylim(0, Y_MAX_US)
            axis.set_yticks(Y_TICKS_US)
            axis.tick_params(
                axis="x", length=0, labelbottom=True, labelsize=7.0, pad=4
            )
            axis.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            axis.xaxis.grid(False)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row == 0:
                axis.set_title(condition_label, pad=5)

        axes[row, 0].set_ylabel(
            r"Cyclictest latency ($\mu$s)",
            labelpad=5,
        )
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

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)

    print(f"Saved {output.resolve()}")
    for platform, maxima in maxima_by_platform.items():
        print(f"[{platform}]")
        for environment, label in ENVIRONMENTS:
            unloaded = maxima[(environment, "cyclictest")]
            loaded = maxima[(environment, "cyclictest-stress")]
            print(
                f"{label:22s} | unloaded={unloaded} "
                f"median={np.median(unloaded):.1f} us | "
                f"loaded={loaded} median={np.median(loaded):.1f} us"
            )


def main() -> int:
    plot_matrix(OUTPUT_ROOT / "cyclictest_runmax.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
