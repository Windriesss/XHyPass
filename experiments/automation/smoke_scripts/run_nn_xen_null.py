#!/usr/bin/env python3
"""Xen-null dual-domain NN smoke experiment."""

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
DOM0_CYCLICTEST_CPU = 3
DOM1_CYCLICTEST_CPU = 3
DOM0_WORKLOAD_CPUS = "0-2"
DOM1_WORKLOAD_CPUS = "0-2"


def build_run_config(environment_name: str = "xen_null") -> dict:
    # xen_null inherits the common Xen workflow but supplies its own boot
    # artifacts, four-vCPU topology, null-scheduler config and pCPU pinning.
    run = base.build_run_config(environment_name)
    run["experiment"].update(
        {
            "dom0_cyclictest_cpu": DOM0_CYCLICTEST_CPU,
            "dom1_cyclictest_cpu": DOM1_CYCLICTEST_CPU,
            "dom0_workload_cpus": DOM0_WORKLOAD_CPUS,
            "dom1_workload_cpus": DOM1_WORKLOAD_CPUS,
        }
    )
    # Null-scheduler guests must be pinned immediately after `xl create`.
    # Keeping this in the COM10 initialization sequence avoids letting dom1
    # boot for several seconds with its temporary scheduler placement.
    env = run["environment"]
    dom1_pin = str(env["dom1_pin_command"])
    commands = [
        str(command)
        for command in env["dom0_init_commands"]
        if str(command) != dom1_pin
    ]
    create_indexes = [
        index
        for index, command in enumerate(commands)
        if command.strip().startswith("xl create ")
    ]
    if len(create_indexes) != 1:
        raise ValueError(
            "xen_null NN requires exactly one xl create command before dom1 pinning"
        )
    commands.insert(create_indexes[0] + 1, dom1_pin)
    env["dom0_init_commands"] = commands
    env["dom1_pin_during_init"] = True
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN Xen-null result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
