#!/usr/bin/env python3
"""Run one 600 s Bare-metal cyclictest-stress round at a 10 ms interval."""

from pathlib import Path

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


ENVIRONMENT = "bare"
EXPERIMENT = "cyclictest-stress"
INTERVAL_US = 10_000
DURATION_SECONDS = 600
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PLATFORM


def build_config() -> dict:
    original = (
        settings.ENVIRONMENT,
        settings.EXPERIMENT,
        settings.INTERVAL_US,
        settings.DURATION_SECONDS,
    )
    try:
        settings.ENVIRONMENT = ENVIRONMENT
        settings.EXPERIMENT = EXPERIMENT
        settings.INTERVAL_US = INTERVAL_US
        settings.DURATION_SECONDS = DURATION_SECONDS
        return settings.build_run_config()
    finally:
        (
            settings.ENVIRONMENT,
            settings.EXPERIMENT,
            settings.INTERVAL_US,
            settings.DURATION_SECONDS,
        ) = original


def main() -> int:
    output = ExperimentRunner(build_config(), DATA_ROOT, dry_run=False).run(
        RUNS, REBOOT_POLICY
    )[0]
    print(f"Completed: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
