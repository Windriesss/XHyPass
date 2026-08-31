#!/usr/bin/env python3
"""Run the two E2000Q bare experiments once for 10 seconds each."""

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


# Edit parameters in run_e2000q_experiment.py. This smoke entry point runs both.
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
RUNS = 1


def main() -> int:
    for experiment in EXPERIMENTS:
        print("=" * 78)
        print(f"E2000Q bare smoke: {experiment}")
        print("=" * 78)
        outputs = ExperimentRunner(
            build_run_config(experiment), DATA_ROOT, dry_run=DRY_RUN
        ).run(RUNS, REBOOT_POLICY)
        for output in outputs:
            print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
