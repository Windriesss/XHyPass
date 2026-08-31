#!/usr/bin/env python3
"""Editable E2000Q bare cyclictest entry point."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from xhypass_lab.config import resolved_run_config
from xhypass_lab.platforms import ENVIRONMENTS, load_platform_config
from xhypass_lab.runner import ExperimentRunner


# Edit parameters here; no command-line arguments are needed.
PLATFORM = "E2000Q"
ENVIRONMENT = "bare"
EXPERIMENT = "cyclictest"
SERIAL_PORT = "COM12"
SERIAL_BAUDRATE = 115_200
RUNS = 1
REBOOT_POLICY = "each-run"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False

CPU = 3
THREADS = 1
PRIORITY = 99
INTERVAL_US = 1_000
DURATION_SECONDS = 10
HISTOGRAM_LIMIT_US = 10_000
GRACE_SECONDS = 60
STRESS_CPUS = "3"
STRESS_VM_WORKERS = 1
STRESS_VM_BYTES = "256M"


def build_run_config(
    experiment: str | None = None,
    environment: str | None = None,
    duration_seconds: int | None = None,
    interval_us: int | None = None,
) -> dict:
    experiment = experiment or EXPERIMENT
    environment = environment or ENVIRONMENT
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown E2000Q environment: {environment}")
    config = load_platform_config(PLATFORM)
    config["serial"]["port"] = SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    overrides = {
        "cpu": CPU,
        "threads": THREADS,
        "priority": PRIORITY,
        "interval_us": INTERVAL_US if interval_us is None else interval_us,
        "duration_seconds": (
            DURATION_SECONDS if duration_seconds is None else duration_seconds
        ),
        "histogram_limit_us": HISTOGRAM_LIMIT_US,
        "grace_seconds": GRACE_SECONDS,
    }
    if experiment == "cyclictest-stress":
        overrides.update(
            {
                "stress_cpus": STRESS_CPUS,
                "stress_vm_workers": STRESS_VM_WORKERS,
                "stress_vm_bytes": STRESS_VM_BYTES,
            }
        )
    return resolved_run_config(
        config,
        environment,
        experiment,
        overrides,
    )


def main() -> int:
    outputs = ExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    for output in outputs:
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
