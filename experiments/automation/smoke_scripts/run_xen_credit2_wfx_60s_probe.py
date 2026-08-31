#!/usr/bin/env python3
"""Run one 60 s xen_credit2_WFX cyclictest probe."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


ENVIRONMENT = "xen_credit2_WFX"
EXPERIMENT = "cyclictest"
DURATION_SECONDS = 60
INTERVAL_US = 1_000
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM


def build_run_config() -> dict:
    original = (
        settings.ENVIRONMENT,
        settings.EXPERIMENT,
        settings.DURATION_SECONDS,
        settings.INTERVAL_US,
    )
    try:
        settings.ENVIRONMENT = ENVIRONMENT
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
    output = ExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=False
    ).run(RUNS, REBOOT_POLICY)[0]
    print(f"Completed: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
