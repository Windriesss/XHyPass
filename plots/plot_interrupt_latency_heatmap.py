"""Plot RK3588 and E2000Q interrupt-latency distributions as paper heatmaps.

The RTOS prints cumulative histograms repeatedly during a run.  Only the last
complete ``t0_region`` block in each published ``rtos_run*.log`` is used, so a
sample is never counted more than once.  Histogram bucket indices are in units
of 10 ns.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOARD_PLATFORMS = {
    "RK3588": "RK3588",
    "E2000Q": "E2000Q",
}
ENVIRONMENTS = (
    ("bare", "Bare-metal"),
    ("jailhouse", "Jailhouse"),
    ("xen_credit2", "Xen(Credit2)"),
    ("xen_credit2_WFX", "Xen(Credit2 + native WFx)"),
    ("xen_null", "Xen(Null)"),
    ("xen_null_WFX", "Xen(Null + native WFx)"),
    ("XHyPass", "XHyPass"),
)
CONDITIONS = (("idle", "Unloaded"), ("stress", "Loaded"))

REGION_RE = re.compile(
    r"^t0_region>\s*(.*?)(?=^t1_region>)", re.MULTILINE | re.DOTALL
)
BUCKET_RE = re.compile(r"^\[(\d+)\]:(\d+)\s*$", re.MULTILINE)


def parse_last_t0_histogram(path: Path) -> dict[int, int]:
    """Return the last complete t0 histogram as ``latency_ns: count``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    regions = REGION_RE.findall(text)
    if not regions:
        raise ValueError(f"no complete t0_region/t1_region pair in {path}")

    histogram: dict[int, int] = {}
    for bucket, count in BUCKET_RE.findall(regions[-1]):
        latency_ns = int(bucket) * 10
        histogram[latency_ns] = histogram.get(latency_ns, 0) + int(count)
    if not histogram:
        raise ValueError(f"empty final t0_region in {path}")
    return histogram


def load_series(series_dir: Path) -> tuple[dict[int, int], list[Path]]:
    """Aggregate final histograms from published runs in one series."""
    paths = sorted(series_dir.glob("rtos_run*.log"))
    if not paths:
        raise FileNotFoundError(f"no published rtos_run*.log files in {series_dir}")

    combined: defaultdict[int, int] = defaultdict(int)
    for path in paths:
        for latency_ns, count in parse_last_t0_histogram(path).items():
            combined[latency_ns] += count
    return dict(combined), paths


def load_run_maxima(series_dir: Path) -> tuple[list[float], list[Path]]:
    """Return the final t0 maximum of every published run, in microseconds."""
    paths = sorted(series_dir.glob("rtos_run*.log"))
    if not paths:
        raise FileNotFoundError(f"no published rtos_run*.log files in {series_dir}")
    maxima_us = [max(parse_last_t0_histogram(path)) / 1000 for path in paths]
    return maxima_us, paths


def rebin(
    histogram: dict[int, int], bin_edges_ns: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Normalize counts into shared bins; return values and clipped samples."""
    latencies = np.fromiter(histogram.keys(), dtype=float)
    counts = np.fromiter(histogram.values(), dtype=np.int64)
    total = int(counts.sum())
    in_range = latencies < bin_edges_ns[-1]
    rebinned, _ = np.histogram(
        latencies[in_range], bins=bin_edges_ns, weights=counts[in_range]
    )
    clipped = int(counts[~in_range].sum())
    normalized = rebinned.astype(float) / total
    return normalized, total, clipped


def weighted_quantile(histogram: dict[int, int], quantile: float) -> int:
    target = quantile * sum(histogram.values())
    cumulative = 0
    for latency_ns, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return latency_ns
    raise AssertionError("non-empty histogram did not reach quantile")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def plot_heatmap(
    data_root: Path,
    output: Path,
    *,
    bin_width_ns: int,
    x_max_us: float,
    board_label: str | None = None,
) -> None:
    configure_style()
    x_max_ns = int(round(x_max_us * 1000))
    bin_edges_ns = np.arange(0, x_max_ns + bin_width_ns, bin_width_ns)

    series: dict[
        tuple[str, str], tuple[np.ndarray, int, int] | None
    ] = {}
    summaries: list[str] = []
    # Load and report one condition at a time so all environments can be
    # compared together in the console summary.
    for condition_key, condition_label in CONDITIONS:
        if summaries:
            summaries.append("")
        summaries.append(f"[{condition_label}]")
        for platform_key, platform_label in ENVIRONMENTS:
            try:
                histogram, paths = load_series(
                    data_root / platform_key / condition_key
                )
            except FileNotFoundError:
                series[(platform_key, condition_key)] = None
                summaries.append(
                    f"{platform_label:18s} | {condition_label:12s} | "
                    "no valid published run"
                )
                continue
            values, total, clipped = rebin(histogram, bin_edges_ns)
            series[(platform_key, condition_key)] = values, total, clipped
            summaries.append(
                f"{platform_label:18s} | {condition_label:12s} | "
                f"runs={len(paths)} samples={total} "
                f"p50={weighted_quantile(histogram, 0.50) / 1000:.2f} us "
                f"p99={weighted_quantile(histogram, 0.99) / 1000:.2f} us "
                f"max={max(histogram) / 1000:.2f} us clipped={clipped}"
            )

    colors = (
        "#061826",
        "#164866",
        "#4a568b",
        "#8f4b83",
        "#d34b67",
        "#f67b49",
        "#ffd166",
    )
    cmap = LinearSegmentedColormap.from_list("paper_heat", colors)
    cmap.set_bad("white")
    norm = LogNorm(vmin=1e-6, vmax=1.0, clip=True)

    fig, axes = plt.subplots(
        len(ENVIRONMENTS),
        len(CONDITIONS),
        figsize=(3.55, 3.55),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.31,
        right=0.84,
        bottom=0.15,
        top=0.89,
        hspace=0.42,
        wspace=0.10,
    )
    mesh = None
    x_us = bin_edges_ns / 1000
    for row, (platform_key, platform_label) in enumerate(ENVIRONMENTS):
        for col, (condition_key, condition_label) in enumerate(CONDITIONS):
            ax = axes[row, col]
            entry = series[(platform_key, condition_key)]
            if entry is None:
                ax.set_facecolor("#eeeeee")
                ax.patch.set_hatch("////")
                ax.patch.set_edgecolor("#c8c8c8")
                ax.text(
                    0.5,
                    0.5,
                    "No valid run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#555555",
                    fontsize=6.2,
                )
            else:
                values, _, _ = entry
                masked = np.ma.masked_equal(values[np.newaxis, :], 0.0)
                mesh = ax.pcolormesh(
                    x_us,
                    np.array([0.0, 1.0]),
                    masked,
                    cmap=cmap,
                    norm=norm,
                    shading="flat",
                    rasterized=True,
                )
            ax.set_xlim(0, x_max_us)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False)
            if row != len(ENVIRONMENTS) - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Use a dedicated, left-aligned label column instead of axes y-labels.
    # This keeps names of different lengths on one clean vertical baseline.
    for row, (platform_key, platform_label) in enumerate(ENVIRONMENTS):
        panel_box = axes[row, 0].get_position()
        fig.text(
            0.035,
            panel_box.y0 + panel_box.height / 2,
            platform_label,
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold" if platform_key == "XHyPass" else "normal",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Interrupt latency ($\mu$s)", labelpad=2.0)
        ax.set_xticks(np.linspace(0, x_max_us, 5))

    if mesh is None:
        raise AssertionError("no heatmap was drawn")
    cbar_ax = fig.add_axes((0.875, 0.15, 0.028, 0.74))
    colorbar = fig.colorbar(mesh, cax=cbar_ax)
    colorbar.set_ticks((1e-6, 1e-4, 1e-2, 1.0))
    colorbar.set_ticklabels((r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", "1"))
    colorbar.ax.tick_params(length=2.0, pad=1.5)
    colorbar.set_label("Normalized frequency", labelpad=2.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)

    print("\n".join(summaries))
    print(f"Saved {output.resolve()}")


def plot_run_max_boxplot(
    data_root: Path,
    output: Path,
    *,
    board_label: str,
) -> None:
    """Plot the distribution of per-run final-t0 maxima for every environment."""
    configure_style()
    labels = [label for _, label in ENVIRONMENTS]
    maxima: dict[tuple[str, str], list[float]] = {}
    global_max = 0.0
    for condition_key, _ in CONDITIONS:
        for environment_key, _ in ENVIRONMENTS:
            try:
                values, _ = load_run_maxima(
                    data_root / environment_key / condition_key
                )
            except FileNotFoundError:
                values = []
            maxima[(environment_key, condition_key)] = values
            if values:
                global_max = max(global_max, max(values))

    if global_max <= 0:
        raise AssertionError("no per-run maxima were found")

    colors = ("#4c78a8", "#d4506c")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.30, top=0.82, wspace=0.08)
    for ax, (condition_key, condition_label), color in zip(
        axes, CONDITIONS, colors
    ):
        positions: list[int] = []
        datasets: list[list[float]] = []
        for position, (environment_key, _) in enumerate(ENVIRONMENTS, start=1):
            values = maxima[(environment_key, condition_key)]
            if values:
                positions.append(position)
                datasets.append(values)

        if datasets:
            boxes = ax.boxplot(
                datasets,
                positions=positions,
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": "#333333", "linewidth": 0.8},
                capprops={"color": "#333333", "linewidth": 0.8},
                boxprops={"edgecolor": "#333333", "linewidth": 0.8},
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

        for position, (environment_key, _) in enumerate(ENVIRONMENTS, start=1):
            values = maxima[(environment_key, condition_key)]
            if not values:
                ax.text(
                    position,
                    global_max * 0.5,
                    "No data",
                    rotation=90,
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=6.0,
                )
                continue
            offsets = np.linspace(-0.09, 0.09, len(values))
            ax.scatter(
                position + offsets,
                values,
                s=13,
                facecolors="white",
                edgecolors="#222222",
                linewidths=0.65,
                zorder=3,
            )
            maximum = max(values)
            ax.annotate(
                f"{maximum:.2f}",
                (position, maximum),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.8,
                fontweight="bold",
            )

        ax.set_xlim(0.45, len(ENVIRONMENTS) + 0.55)
        ax.set_ylim(0, global_max * 1.18)
        ax.set_xticks(range(1, len(ENVIRONMENTS) + 1))
        ax.set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Per-run maximum interrupt latency ($\mu$s)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    print(f"Saved {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="root containing PLATFORM/int-latency/environment/condition data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=tuple(BOARD_PLATFORMS),
        default=list(BOARD_PLATFORMS),
        help="platforms to draw; both are generated by default",
    )
    parser.add_argument("--bin-width-ns", type=int, default=100)
    parser.add_argument("--x-max-us", type=float, default=20.0)
    args = parser.parse_args()
    if args.bin_width_ns <= 0 or args.x_max_us <= 0:
        parser.error("bin width and x-axis maximum must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    for board in arguments.platforms:
        data_platform = BOARD_PLATFORMS[board]
        print(f"\n{'=' * 78}\n{board}\n{'=' * 78}")
        plot_heatmap(
            arguments.data_root / data_platform / "int-latency",
            arguments.output_dir
            / f"{board.lower()}_interrupt_latency_heatmap.pdf",
            bin_width_ns=arguments.bin_width_ns,
            x_max_us=arguments.x_max_us,
            board_label=board,
        )
        plot_run_max_boxplot(
            arguments.data_root / data_platform / "int-latency",
            arguments.output_dir
            / f"{board.lower()}_interrupt_latency_max_boxplot.pdf",
            board_label=board,
        )
