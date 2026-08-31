#!/usr/bin/env python3
"""RK3588 bare interrupt-latency experiment over COM10 + COM14."""

import copy
from pathlib import Path

from xhypass_lab.config import resolved_run_config
from xhypass_lab.int_latency_runner import IntLatencyRunner
from xhypass_lab.platforms import load_platform_config


# Edit experiment parameters here; no command-line arguments are required.
PLATFORM = "RK3588"
ENVIRONMENT = "bare"
CONDITIONS = ("idle", "stress")
RUNS_PER_CONDITION = 1
DURATION_SECONDS = 600
GRACE_SECONDS = 180
CONTROL_SERIAL_PORT = "COM10"
RTOS_SERIAL_PORT = "COM14"
SERIAL_BAUDRATE = 115_200
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PLATFORM
DRY_RUN = False


BOOT_COMMANDS = {
    "idle": "run boot_rtos_idle",
    "stress": "run boot_rtos_stress",
}


def build_run_config(condition: str) -> dict:
    if condition not in BOOT_COMMANDS:
        raise ValueError(f"Unknown int-latency condition: {condition}")
    config = load_platform_config(PLATFORM)
    config["serial"]["port"] = CONTROL_SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    run = resolved_run_config(
        config,
        ENVIRONMENT,
        "int-latency",
        {
            "condition": condition,
            "duration_seconds": DURATION_SECONDS,
            "timeout_seconds": DURATION_SECONDS + GRACE_SECONDS,
        },
    )
    run["environment"]["boot_command"] = BOOT_COMMANDS[condition]
    run["control_serial"] = copy.deepcopy(config["serial"])
    run["rtos_serial"] = copy.deepcopy(config["serial"])
    run["rtos_serial"]["port"] = RTOS_SERIAL_PORT
    return run


def main() -> int:
    configs = [
        build_run_config(condition)
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
