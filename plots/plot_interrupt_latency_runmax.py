#!/usr/bin/env python3
"""Plot median and min-max range of five run-level latency maxima."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xhypass_plot.interrupt_latency import load_run_maxima
from xhypass_plot.style import apply_paper_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
EXPECTED_RUNS = 5

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
    ("idle", "Unloaded"),
    ("stress", "Loaded"),
)

# Keep the same environment palette as the NN figures: related Xen variants
# share a hue, while their native-WFx versions use a lighter shade.
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
    apply_paper_style()


def load_platform_maxima(
    data_root: Path,
) -> dict[tuple[str, str], list[float]]:
    maxima: dict[tuple[str, str], list[float]] = {}
    for environment, _ in ENVIRONMENTS:
        for condition, *_ in CONDITIONS:
            values, paths = load_run_maxima(
                data_root / environment / condition
            )
            if len(values) != EXPECTED_RUNS:
                raise RuntimeError(
                    f"Expected {EXPECTED_RUNS} published runs for "
                    f"{environment}/{condition}, found {len(values)}: {paths}"
                )
            maxima[(environment, condition)] = values
    return maxima


def plot_matrix(output: Path) -> None:
    configure_style()
    maxima_by_platform = {
        display_name: load_platform_maxima(
            PROJECT_ROOT / "data" / data_name / "int-latency"
        )
        for display_name, data_name in PLATFORMS.items()
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
    x_labels = [
        "Bare\nmetal",
        "Jailhouse",
        "Credit2",
        "Credit2\n+ native WFx",
        "Null",
        "Null\n+ native WFx",
        "XHyPass",
    ]

    for row, (platform, maxima) in enumerate(maxima_by_platform.items()):
        for column, (condition, condition_label) in enumerate(CONDITIONS):
            axis = axes[row, column]
            for x_value, (environment, _) in zip(
                x_positions, ENVIRONMENTS, strict=True
            ):
                values = np.asarray(maxima[(environment, condition)], dtype=float)
                median = float(np.median(values))
                maximum = float(values.max())
                color = COLORS[environment]
                axis.errorbar(
                    x_value,
                    median,
                    yerr=np.asarray(
                        [[median - float(values.min())],
                         [float(values.max()) - median]]
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
                    f"{maximum:.2f}",
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
            axis.set_xticks(x_positions, x_labels)
            axis.set_ylim(0, 22)
            axis.set_yticks((0, 5, 10, 15, 20))
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
            r"Interrupt latency ($\mu$s)",
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
            idle = maxima[(environment, "idle")]
            stress = maxima[(environment, "stress")]
            print(
                f"{label:18s} | "
                f"without={idle} median={np.median(idle):.2f} us | "
                f"with={stress} median={np.median(stress):.2f} us"
            )


def main() -> int:
    plot_matrix(OUTPUT_ROOT / "interrupt_latency_runmax.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
