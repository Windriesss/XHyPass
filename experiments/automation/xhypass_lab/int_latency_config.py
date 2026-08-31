from __future__ import annotations

import copy
from typing import Any

from .config import ConfigError, resolved_run_config
from .platforms import load_platform_config


CONDITIONS = ("idle", "stress")


def build_int_latency_run_config(
    platform: str,
    environment: str,
    condition: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dual-UART int-latency run from declarative platform config."""
    if condition not in CONDITIONS:
        raise ConfigError(
            f"Unknown int-latency condition: {condition}; choose from {CONDITIONS}"
        )

    config = load_platform_config(platform)
    serial_profile = config.get("int_latency")
    if not isinstance(serial_profile, dict):
        raise ConfigError(f"Platform {platform} has no int-latency serial profile")

    experiment_overrides = {"condition": condition}
    if overrides:
        experiment_overrides.update(copy.deepcopy(overrides))
    run = resolved_run_config(
        config,
        environment,
        "int-latency",
        experiment_overrides,
    )

    environment_config = run["environment"]
    # Readiness must be declared by this exact environment.  Inherited
    # `int_latency_ready` from bare must not make a partially configured
    # Jailhouse/Xen environment runnable.
    declared_environment = config["environments"][environment]
    if not declared_environment.get("int_latency_ready", False):
        raise ConfigError(
            f"Platform {platform} environment {environment} is not ready for "
            "int-latency yet"
        )

    # Like readiness, condition-specific boot commands are environment-local.
    # Jailhouse extends bare for normal Linux settings but must keep boot_oee,
    # not inherit bare's boot_rtos_idle / boot_rtos_stress commands.
    boot_commands = declared_environment.get("int_latency_boot_commands")
    if isinstance(boot_commands, dict):
        try:
            environment_config["boot_command"] = str(boot_commands[condition])
        except KeyError as exc:
            raise ConfigError(
                f"Platform {platform} environment {environment} has no "
                f"int-latency boot command for {condition}"
            ) from exc

    int_latency_boot_command = environment_config.get("int_latency_boot_command")
    if int_latency_boot_command:
        environment_config["boot_command"] = str(int_latency_boot_command)

    if environment_config.get("environment_type") == "xen":
        pin_commands = environment_config.get("int_latency_pin_commands")
        if (
            not isinstance(pin_commands, list)
            or not pin_commands
            or not all(isinstance(command, str) and command.strip() for command in pin_commands)
        ):
            raise ConfigError(
                f"Platform {platform} environment {environment} must define "
                "non-empty int_latency_pin_commands"
            )

    boot_files_by_condition = declared_environment.get(
        "int_latency_boot_files_by_condition"
    )
    if isinstance(boot_files_by_condition, dict):
        try:
            selected_files = copy.deepcopy(boot_files_by_condition[condition])
        except KeyError as exc:
            raise ConfigError(
                f"Platform {platform} environment {environment} has no "
                f"int-latency boot files for {condition}"
            ) from exc
        if not isinstance(selected_files, list) or not selected_files:
            raise ConfigError(
                f"Platform {platform} environment {environment} has invalid "
                f"int-latency boot files for {condition}"
            )
        for item in selected_files:
            if not isinstance(item, dict) or not item.get("source") or not item.get(
                "destination"
            ):
                raise ConfigError(
                    f"Platform {platform} environment {environment} has an "
                    "invalid int-latency boot file mapping"
                )
        environment_config["int_latency_boot_files"] = selected_files

    base_serial = copy.deepcopy(config["serial"])
    control_serial = copy.deepcopy(base_serial)
    control_serial.update(copy.deepcopy(serial_profile.get("control_serial", {})))
    rtos_serial = copy.deepcopy(base_serial)
    rtos_serial.update(copy.deepcopy(serial_profile.get("rtos_serial", {})))
    run["control_serial"] = control_serial
    run["rtos_serial"] = rtos_serial
    return run
