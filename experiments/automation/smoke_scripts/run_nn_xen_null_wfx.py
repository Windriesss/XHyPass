#!/usr/bin/env python3
"""Xen-null native-WFX dual-domain NN smoke experiment."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_nn_xen_null as base
from xhypass_lab.nn_runner import NNExperimentRunner


# Edit these values directly when changing the smoke/formal campaign.
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False


def build_run_config() -> dict:
    # xen_null_WFX inherits the null-scheduler topology and only replaces
    # the local boot-artifact source with Xen_null_nativeWFX.
    return base.build_run_config("xen_null_WFX")


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN Xen-null native-WFX result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
