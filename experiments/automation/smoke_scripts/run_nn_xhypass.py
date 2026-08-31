#!/usr/bin/env python3
"""XHyPass dual-domain NN experiment with per-domain RTO modules."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_nn_xen_credit2 as base
from xhypass_lab.nn_runner import NNExperimentRunner


# Edit these values directly when changing the smoke/formal campaign.
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False
DURATION_SECONDS = 30
CYCLICTEST_INTERVAL_US = 1_000
DOM0_WORKLOAD_CPUS = "0-5"
DOM1_WORKLOAD_CPUS = "0-5"
DOM0_CYCLICTEST_CPU = 6
DOM1_CYCLICTEST_CPU = 6
RTO_MODULE_CPU = 6
DOM1_START_ATTEMPTS = 3
DOM1_CONSOLE_READY_TIMEOUT_SECONDS = 60


def build_run_config() -> dict:
    # XHyPass inherits Xen-credit2 boot, dom1, vCPU pinning and NN settings;
    # NNExperimentRunner adds the module lifecycle on both domains.
    run = base.build_run_config("XHyPass")
    run["environment"]["dom1_start_attempts"] = DOM1_START_ATTEMPTS
    run["environment"]["dom1_console_ready_timeout_seconds"] = (
        DOM1_CONSOLE_READY_TIMEOUT_SECONDS
    )
    run["experiment"].update(
        {
            "duration_seconds": DURATION_SECONDS,
            "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
            "dom0_workload_cpus": DOM0_WORKLOAD_CPUS,
            "dom1_workload_cpus": DOM1_WORKLOAD_CPUS,
            "dom0_cyclictest_cpu": DOM0_CYCLICTEST_CPU,
            "dom1_cyclictest_cpu": DOM1_CYCLICTEST_CPU,
            "xhypass_module_cpu": RTO_MODULE_CPU,
        }
    )
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN XHyPass result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
