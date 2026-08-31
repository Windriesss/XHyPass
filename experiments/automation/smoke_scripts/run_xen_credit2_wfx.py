#!/usr/bin/env python3
"""Editable entry point for xen_credit2_WFX experiments."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


# Edit this block; no command-line arguments are needed.
EXPERIMENT = "cyclictest"  # cyclictest / cyclictest-stress
RUNS = 1
REBOOT_POLICY = "each-run"
DURATION_SECONDS = 10       # use 600 for formal data
INTERVAL_US = 1_000
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False


def build_run_config() -> dict:
    original = (
        settings.ENVIRONMENT,
        settings.EXPERIMENT,
        settings.DURATION_SECONDS,
        settings.INTERVAL_US,
    )
    try:
        settings.ENVIRONMENT = "xen_credit2_WFX"
        settings.EXPERIMENT = EXPERIMENT
        settings.DURATION_SECONDS = DURATION_SECONDS
        settings.INTERVAL_US = INTERVAL_US
        return settings.build_run_config()
    finally:
        (
            settings.ENVIRONMENT,
            settings.EXPERIMENT,
            settings.DURATION_SECONDS,
            settings.INTERVAL_US,
        ) = original


def main() -> int:
    outputs = ExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    if not DRY_RUN:
        print("Completed:")
        for output in outputs:
            print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
