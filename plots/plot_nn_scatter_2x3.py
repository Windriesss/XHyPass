#!/usr/bin/env python3
"""Plot formal RK3588 NN metrics as readable 2x3 scatter panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyze_nn import (
    COLORS,
    ENVIRONMENT_ORDER,
    LABELS,
    OUTPUT_ROOT,
    PLATFORM,
    WORKLOADS,
    WORKLOAD_TITLES,
    _available_cyclic_series,
    _build_statistics,
    _combine_histograms,
    _discover_latest_condition_runs,
)
from xhypass_plot.style import PAPER_RC


OUTPUT_PNG = OUTPUT_ROOT / f"{PLATFORM.lower()}_nn_metrics_scatter_2x3.png"
OUTPUT_PDF = OUTPUT_ROOT / f"{PLATFORM.lower()}_nn_metrics_scatter_2x3.pdf"

# Compact paper layout. These constants keep the visual-only change easy to
# tune or revert without touching data selection and metric calculations.
FIGURE_SIZE = (14.5, 5.8)
X_LIMIT_US = (0, 72.0)
ROW_SCALES = {
    "mnas_avg_ms": (0.0, 510.0, np.arange(0.0, 501.0, 100.0)),
    "inception_avg_ms": (500.0, 1210.0, np.arange(500.0, 1201.0, 100.0)),
}

MARKERS = {
    "bare": "o",
    "jailhouse": "s",
    "xen_credit2": "D",
    "xen_credit2_WFX": "D",
    "xen_null": "^",
    "xen_null_WFX": "^",
    "XHyPass": "*",
}

MARKER_AREA = 62
HOLLOW_MARKER_AREA_SCALE = 1.12
XHYPASS_MARKER_AREA = 110
MARKER_EDGE_COLOR = "black"
MARKER_EDGE_WIDTH = 0.75
HOLLOW_MARKER_EDGE_WIDTH = 1.3
LEGEND_MARKER_SIZE = 6.5
XHYPASS_LEGEND_MARKER_SIZE = 8.5
HOLLOW_ENVIRONMENTS = frozenset(
    {"xen_credit2_WFX", "xen_null_WFX"}
)
GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.30
SPINE_LINEWIDTH = 0.7
SPINE_COLOR = "0.55"
PLOT_RC = dict(PAPER_RC)

LEGEND_LABELS = {
    **LABELS,
    "xen_credit2": "Credit2",
    "xen_credit2_WFX": "Credit2 + native WFx",
    "xen_null": "Null",
    "xen_null_WFX": "Null + native WFx",
}


def marker_area(environment: str) -> float:
    if environment == "XHyPass":
        return float(XHYPASS_MARKER_AREA)
    if environment in HOLLOW_ENVIRONMENTS:
        return float(MARKER_AREA * HOLLOW_MARKER_AREA_SCALE)
    return float(MARKER_AREA)


def marker_face_color(environment: str) -> str:
    return "none" if environment in HOLLOW_ENVIRONMENTS else COLORS[environment]


def marker_edge_color(environment: str) -> str:
    if environment in HOLLOW_ENVIRONMENTS:
        return COLORS[environment]
    return MARKER_EDGE_COLOR


def marker_edge_width(environment: str) -> float:
    if environment in HOLLOW_ENVIRONMENTS:
        return HOLLOW_MARKER_EDGE_WIDTH
    return MARKER_EDGE_WIDTH


def marker_legend_size(environment: str) -> float:
    if environment == "XHyPass":
        return XHYPASS_LEGEND_MARKER_SIZE
    if environment in HOLLOW_ENVIRONMENTS:
        return float(LEGEND_MARKER_SIZE * np.sqrt(HOLLOW_MARKER_AREA_SCALE))
    return LEGEND_MARKER_SIZE


def marker_zorder(environment: str) -> int:
    return 5 if environment == "XHyPass" else 3


def _paired_cyclic_filename(
    environment: str,
    workload: str,
    model: str,
    runs: list[Path],
) -> str:
    """Select the cyclictest stream from the model's own execution domain."""
    filenames = [
        filename
        for filename, _ in _available_cyclic_series(runs, workload)
    ]
    if environment == "bare":
        # Bare metal has no VM boundary, so use a stable arbitrary pairing.
        suffix = "cpu7.txt" if model == "mnas" else "cpu6.txt"
    elif environment == "jailhouse":
        # MnasNet is in the non-root cell; Inception is in the root cell.
        suffix = "cpu3.txt" if model == "mnas" else "cpu6.txt"
    else:
        # Xen and XHyPass: vm2 hosts MnasNet; vm1 hosts Inception.
        suffix = "_vm2.txt" if model == "mnas" else "_vm1.txt"
    matches = [filename for filename in filenames if filename.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {environment}/{workload}/{model} cyclictest stream "
            f"ending in {suffix!r}, found: {filenames}"
        )
    return matches[0]


def _paired_cyclic_max(
    environment: str,
    workload: str,
    model: str,
    runs: list[Path],
) -> float:
    filename = _paired_cyclic_filename(environment, workload, model, runs)
    values, _, _ = _combine_histograms(runs, filename)
    return float(values[-1]) if len(values) else float("nan")


def _plot(
    rows: list[dict],
    runs_by_env: dict[str, list[Path]],
) -> list[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            2,
            3,
            figsize=FIGURE_SIZE,
            sharex="col",
            sharey="row",
            constrained_layout=False,
        )
        model_rows = (
            ("mnas", "mnas_avg_ms", "MnasNet mean response time (ms)"),
            ("inception", "inception_avg_ms", "Inception mean response time (ms)"),
        )
        for column, workload in enumerate(WORKLOADS):
            workload_rows = [row for row in rows if row["workload"] == workload]
            for row_index, (model, metric, ylabel) in enumerate(model_rows):
                axis = axes[row_index, column]
                if row_index == 0:
                    axis.set_title(WORKLOAD_TITLES[workload], pad=4)
                for row in workload_rows:
                    environment = row["environment"]
                    x_value = _paired_cyclic_max(
                        environment,
                        workload,
                        model,
                        runs_by_env[environment],
                    )
                    y_value = float(row[metric])
                    if not np.isfinite(x_value) or not np.isfinite(y_value):
                        continue
                    axis.scatter(
                        x_value,
                        y_value,
                        s=marker_area(environment),
                        marker=MARKERS[environment],
                        facecolors=marker_face_color(environment),
                        edgecolors=marker_edge_color(environment),
                        linewidth=marker_edge_width(environment),
                        zorder=marker_zorder(environment),
                    )
                if column == 0:
                    axis.set_ylabel(ylabel)
                lower, upper, ticks = ROW_SCALES[metric]
                axis.set_ylim(lower, upper)
                axis.set_yticks(ticks)
                axis.tick_params(axis="y", labelleft=True)
                axis.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=GRID_LINEWIDTH,
                    alpha=GRID_ALPHA,
                )
                axis.grid(False, which="minor")
                axis.set_xlim(*X_LIMIT_US)
                for spine in axis.spines.values():
                    spine.set_linewidth(SPINE_LINEWIDTH)
                    spine.set_color(SPINE_COLOR)

        figure.supxlabel(
            r"Paired-domain cyclictest maximum latency ($\mu$s)",
            x=0.53,
            y=0.085,
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker=MARKERS[environment],
                color="none",
                markerfacecolor=marker_face_color(environment),
                markeredgecolor=marker_edge_color(environment),
                markeredgewidth=marker_edge_width(environment),
                markersize=marker_legend_size(environment),
                label=LEGEND_LABELS[environment],
            )
            for environment in ENVIRONMENT_ORDER
            if any(row["environment"] == environment for row in rows)
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=7,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        figure.subplots_adjust(
            left=0.066,
            right=0.995,
            top=0.94,
            bottom=0.19,
            wspace=0.12,
            hspace=0.08,
        )
        figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight")
        figure.savefig(OUTPUT_PDF, bbox_inches="tight")
        plt.close(figure)
    return [OUTPUT_PNG.resolve(), OUTPUT_PDF.resolve()]


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs_by_env = _discover_latest_condition_runs()
    if not runs_by_env:
        raise RuntimeError("No complete formal NN results were found.")
    rows = _build_statistics(runs_by_env)
    if not rows:
        raise RuntimeError("No valid NN statistics could be calculated.")

    outputs = _plot(rows, runs_by_env)
    print("Generated files:")
    for output in outputs:
        print(f"- {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
