#!/usr/bin/env python3
"""Reproduce all figures released with the XHyPass artifact."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
PLOTS_ROOT = REPOSITORY_ROOT / "plots"
SCRIPTS = (
    "plot_motivation_tradeoff.py",
    "plot_interrupt_latency_runmax.py",
    "plot_interrupt_latency_tail_percentiles.py",
    "plot_cyclictest_runmax.py",
    "plot_nn_scatter_2x3.py",
    "analyze_nn.py",
)
EXPECTED_PDFS = (
    "motivation_tradeoff.pdf",
    "interrupt_latency_runmax.pdf",
    "interrupt_latency_tail_percentiles.pdf",
    "cyclictest_runmax.pdf",
    "rk3588_nn_metrics_scatter_2x3.pdf",
    "rk3588_nn_comprehensive_4x3.pdf",
)


def main() -> int:
    for script in SCRIPTS:
        print(f"[reproduce] {script}", flush=True)
        subprocess.run(
            [sys.executable, "-B", str(PLOTS_ROOT / script)],
            cwd=REPOSITORY_ROOT,
            check=True,
        )

    missing = [
        name for name in EXPECTED_PDFS
        if not (PLOTS_ROOT / "output" / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing expected figures: {', '.join(missing)}")

    print("Reproduced figures:")
    for name in EXPECTED_PDFS:
        print(f"- plots/output/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
