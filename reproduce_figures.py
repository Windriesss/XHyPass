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
EXPECTED_OUTPUTS = (
    "plots/output/motivation_tradeoff.pdf",
    "plots/output/interrupt_latency_runmax.pdf",
    "plots/output/interrupt_latency_tail_percentiles.pdf",
    "plots/output/cyclictest_runmax.pdf",
    "plots/output/rk3588_nn_metrics_scatter_2x3.pdf",
    "plots/output/rk3588_nn_comprehensive_4x3.pdf",
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
        path for path in EXPECTED_OUTPUTS
        if not (REPOSITORY_ROOT / path).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing expected figures: {', '.join(missing)}")

    subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPOSITORY_ROOT / "tools" / "check_pdf_fonts.py"),
            *(str(REPOSITORY_ROOT / path) for path in EXPECTED_OUTPUTS),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    print("Reproduced figures:")
    for path in EXPECTED_OUTPUTS:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
