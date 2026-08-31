#!/usr/bin/env python3
"""RK3588 Xen credit2/null/WFX dom0less interrupt-latency experiments."""

import copy
from pathlib import Path

from xhypass_lab.config import resolved_run_config
from xhypass_lab.int_latency_runner import IntLatencyRunner
from xhypass_lab.platforms import load_platform_config


# Edit experiment parameters here; no command-line arguments are required.
PLATFORM = "RK3588"
ENVIRONMENTS = (
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
)
CONDITIONS = ("idle", "stress")
RUNS_PER_CONDITION = 1
DURATION_SECONDS = 600
GRACE_SECONDS = 180
CONTROL_SERIAL_PORT = "COM10"
RTOS_SERIAL_PORT = "COM14"
SERIAL_BAUDRATE = 115_200
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "experiments"
DATA_ROOT = REPOSITORY_ROOT / "data" / PLATFORM
DRY_RUN = False


SOURCE_ROOT = EXPERIMENTS_ROOT / "boot-artifacts" / "RK3588-int-latency" / "xen"
TARGET_ROOT = EXPERIMENTS_ROOT / "deploy" / "RK3588" / "dom0less"

RTOS_BINARIES = {
    "idle": "qsemos-rt.bin_native",
    "stress": "qsemos-rt.bin_native_stress",
}

XEN_DTBS = {
    "xen_credit2": "tl3588-evm-xen-dom0less.dtb.credit2",
    "xen_credit2_WFX": "tl3588-evm-xen-dom0less.dtb.credit2.WFX",
    "xen_null": "tl3588-evm-xen-dom0less.dtb.null",
    "xen_null_WFX": "tl3588-evm-xen-dom0less.dtb.null.WFX",
}

PIN_COMMANDS_BY_ENVIRONMENT = {
    "xen_credit2": (
        "xl vcpu-pin 0 3 7",
        "xl vcpu-pin 1 0 3",
    ),
    "xen_credit2_WFX": (
        "xl vcpu-pin 0 3 7",
        "xl vcpu-pin 1 0 3",
    ),
}

XEN_NULL_DYNAMIC_PIN_SWAP = {
    "moving_domain_id": 1,
    "moving_vcpu_id": 0,
    "target_pcpu": 3,
}


def build_run_config(environment_name: str, condition: str) -> dict:
    if environment_name not in XEN_DTBS:
        raise ValueError(f"Unknown Xen int-latency environment: {environment_name}")
    if condition not in RTOS_BINARIES:
        raise ValueError(f"Unknown int-latency condition: {condition}")

    config = load_platform_config(PLATFORM)
    config["serial"]["port"] = CONTROL_SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    run = resolved_run_config(
        config,
        environment_name,
        "int-latency",
        {
            "condition": condition,
            "duration_seconds": DURATION_SECONDS,
            "timeout_seconds": DURATION_SECONDS + GRACE_SECONDS,
        },
    )
    environment = run["environment"]
    environment["boot_command"] = "run boot_xen_dom0less"
    environment["int_latency_boot_files"] = [
        {
            "source": str(SOURCE_ROOT / RTOS_BINARIES[condition]),
            "destination": str(TARGET_ROOT / "qsemos-rt.bin"),
        },
        {
            "source": str(SOURCE_ROOT / XEN_DTBS[environment_name]),
            "destination": str(TARGET_ROOT / "tl3588-evm-xen-dom0less.dtb"),
        },
    ]
    if environment_name in {"xen_null", "xen_null_WFX"}:
        environment.pop("int_latency_pin_commands", None)
        environment["int_latency_dynamic_pin_swap"] = dict(
            XEN_NULL_DYNAMIC_PIN_SWAP
        )
    else:
        environment["int_latency_pin_commands"] = list(
            PIN_COMMANDS_BY_ENVIRONMENT[environment_name]
        )
    run["control_serial"] = copy.deepcopy(config["serial"])
    run["rtos_serial"] = copy.deepcopy(config["serial"])
    run["rtos_serial"]["port"] = RTOS_SERIAL_PORT
    return run


def main() -> int:
    configs = [
        build_run_config(environment, condition)
        for environment in ENVIRONMENTS
        for condition in CONDITIONS
        for _ in range(RUNS_PER_CONDITION)
    ]
    outputs = IntLatencyRunner(
        configs, DATA_ROOT, dry_run=DRY_RUN
    ).run()
    for output in outputs:
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
