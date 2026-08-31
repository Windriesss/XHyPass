#!/usr/bin/env python3
"""Smoke-test four E2000Q Xen environments over COM12."""

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts.run_e2000q_experiment import (
    DATA_ROOT,
    DRY_RUN,
    REBOOT_POLICY,
    build_run_config,
)
from xhypass_lab.runner import ExperimentRunner


# All values are editable here; no command-line arguments are needed.
ENVIRONMENTS = (
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
)
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
RUNS = 1


def main() -> int:
    for environment in ENVIRONMENTS:
        for experiment in EXPERIMENTS:
            print("=" * 78)
            print(f"E2000Q Xen smoke: {environment} / {experiment}")
            print("=" * 78)
            outputs = ExperimentRunner(
                build_run_config(experiment, environment),
                DATA_ROOT,
                dry_run=DRY_RUN,
            ).run(RUNS, REBOOT_POLICY)
            for output in outputs:
                print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
