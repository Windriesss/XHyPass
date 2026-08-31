from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_environment(
    environments: dict[str, Any], name: str, chain: tuple[str, ...] = ()
) -> dict[str, Any]:
    if name not in environments:
        raise ConfigError(f"Unknown environment: {name}")
    if name in chain:
        raise ConfigError(f"Environment inheritance cycle: {' -> '.join((*chain, name))}")
    current = copy.deepcopy(environments[name])
    parent = current.pop("extends", None)
    if not parent:
        return current
    return _deep_merge(
        _resolve_environment(environments, str(parent), (*chain, name)), current
    )


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    for key in ("serial", "environments", "experiments"):
        if key not in config:
            raise ConfigError(f"Missing top-level config key: {key}")
    return config


def resolved_run_config(
    config: dict[str, Any], environment: str, experiment: str, overrides: dict[str, Any]
) -> dict[str, Any]:
    if environment not in config["environments"]:
        raise ConfigError(f"Unknown environment: {environment}")
    if experiment not in config["experiments"]:
        raise ConfigError(f"Unknown experiment: {experiment}")

    environment_config = _resolve_environment(config["environments"], environment)
    experiment_config = copy.deepcopy(config["experiments"][experiment])
    experiment_config.update(
        copy.deepcopy(environment_config.get("experiment_overrides", {}))
    )
    result = {
        "platform_name": str(config.get("platform", {}).get("name", "RK3588")),
        "serial": copy.deepcopy(config["serial"]),
        "environment_name": environment,
        "environment": environment_config,
        "experiment_name": experiment,
        "experiment": experiment_config,
        "timeouts": copy.deepcopy(config.get("timeouts", {})),
        "reboot": copy.deepcopy(config.get("reboot", {})),
    }
    for key, value in overrides.items():
        if value is not None:
            result["experiment"][key] = value
    boot_command = str(result["environment"].get("boot_command", ""))
    if not boot_command or boot_command.startswith("TODO_"):
        raise ConfigError(
            f"Platform '{result['platform_name']}' environment '{environment}' "
            "has no usable U-Boot command yet"
        )
    return result
