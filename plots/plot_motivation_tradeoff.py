#!/usr/bin/env python3
"""Plot the RK3588 Heavy/MnasNet motivation trade-off figure.

The point construction intentionally reuses the same formal-run discovery,
statistics, and paired-domain cyclictest helper as plot_nn_scatter_2x3.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from analyze_nn import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    _build_statistics,
    _discover_latest_condition_runs,
)
from plot_nn_scatter_2x3 import (
    GRID_ALPHA,
    GRID_LINEWIDTH,
    LEGEND_LABELS,
    MARKERS,
    PLOT_RC,
    SPINE_COLOR,
    SPINE_LINEWIDTH,
    _paired_cyclic_max,
    marker_area,
    marker_edge_color,
    marker_edge_width,
    marker_face_color,
    marker_legend_size,
    marker_zorder,
)


WORKLOAD = "heavy"
MODEL = "mnas"
METRIC = "mnas_avg_ms"
FIGURE_SIZE = (4.2, 3.2)

ENVIRONMENTS = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "XHyPass",
)

OUTPUT_PNG = OUTPUT_ROOT / "motivation_tradeoff.png"
OUTPUT_PDF = OUTPUT_ROOT / "motivation_tradeoff.pdf"


def _collect_points() -> list[dict[str, float | str | int]]:
    runs_by_env = _discover_latest_condition_runs()
    if not runs_by_env:
        raise RuntimeError("No complete formal RK3588 NN results were found.")

    rows = _build_statistics(runs_by_env)
    heavy_rows = {
        str(row["environment"]): row
        for row in rows
        if row["workload"] == WORKLOAD
    }

    points: list[dict[str, float | str | int]] = []
    for environment in ENVIRONMENTS:
        if environment not in heavy_rows or environment not in runs_by_env:
            raise RuntimeError(
                f"Missing formal Heavy result for {environment}."
            )
        row = heavy_rows[environment]
        x_value = _paired_cyclic_max(
            environment,
            WORKLOAD,
            MODEL,
            runs_by_env[environment],
        )
        y_value = float(row[METRIC])
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            raise RuntimeError(
                f"Non-finite Motivation point for {environment}: "
                f"x={x_value}, y={y_value}"
            )
        points.append(
            {
                "environment": environment,
                "x": x_value,
                "y": y_value,
                "runs": int(row["runs"]),
            }
        )
    return points


def _axis_limits(points: list[dict[str, float | str | int]]) -> tuple[
    tuple[float, float], tuple[float, float]
]:
    x_values = np.asarray([float(point["x"]) for point in points])
    y_values = np.asarray([float(point["y"]) for point in points])

    # Keep the latency origin visible and round both upper bounds outward.
    x_upper = float(
        max(15.0, np.ceil((float(x_values.max()) + 5.0) / 5.0) * 5.0)
    )
    y_lower = float(
        np.floor((float(y_values.min()) - 10.0) / 5.0) * 5.0
    )
    y_upper = float(
        np.ceil((float(y_values.max()) + 10.0) / 5.0) * 5.0
    )
    return (0.0, x_upper), (y_lower, y_upper)


def _plot(points: list[dict[str, float | str | int]]) -> tuple[
    Path, Path, tuple[float, float], tuple[float, float]
]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    x_limits, y_limits = _axis_limits(points)

    plt.style.use("seaborn-v0_8-whitegrid")
    with plt.rc_context(PLOT_RC):
        figure, axis = plt.subplots(
            figsize=FIGURE_SIZE,
            constrained_layout=True,
        )

        for point in points:
            environment = str(point["environment"])
            axis.scatter(
                float(point["x"]),
                float(point["y"]),
                s=marker_area(environment),
                marker=MARKERS[environment],
                facecolors=marker_face_color(environment),
                edgecolors=marker_edge_color(environment),
                linewidth=marker_edge_width(environment),
                zorder=marker_zorder(environment),
            )

        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.xaxis.set_major_locator(MultipleLocator(15.0))
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
        )
        axis.set_xlabel(r"Cyclictest maximum latency ($\mu$s)")
        axis.set_ylabel("MnasNet mean response time (ms)")
        axis.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
        axis.grid(False, which="minor")
        for spine in axis.spines.values():
            spine.set_linewidth(SPINE_LINEWIDTH)
            spine.set_color(SPINE_COLOR)

        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=MARKERS[environment],
                markerfacecolor=marker_face_color(environment),
                markeredgecolor=marker_edge_color(environment),
                markeredgewidth=marker_edge_width(environment),
                markersize=marker_legend_size(environment),
                label=LEGEND_LABELS[environment],
            )
            for environment in ENVIRONMENTS
        ]
        axis.legend(
            handles=handles,
            loc="upper right",
            ncol=2,
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            columnspacing=0.8,
            handletextpad=0.35,
        )

        figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
        figure.savefig(OUTPUT_PDF, bbox_inches="tight")
        plt.close(figure)

    return OUTPUT_PNG.resolve(), OUTPUT_PDF.resolve(), x_limits, y_limits


def main() -> int:
    points = _collect_points()
    output_png, output_pdf, x_limits, y_limits = _plot(points)

    print("Motivation trade-off points (Heavy / MnasNet domain):")
    for point in points:
        environment = str(point["environment"])
        print(
            f"- {LEGEND_LABELS[environment]:<12} "
            f"x={float(point['x']):.0f} us, "
            f"y={float(point['y']):.2f} ms, "
            f"runs={int(point['runs'])}"
        )
    print(f"Figure size: {FIGURE_SIZE}")
    print(f"xlim: {x_limits}")
    print(f"ylim: {y_limits}")
    print(f"PNG: {output_png}")
    print(f"PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
