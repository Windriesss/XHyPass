from __future__ import annotations

from pathlib import Path

from .config import ConfigError, load_config


PLATFORMS = ("RK3588", "E2000Q")
ENVIRONMENTS = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
    "XHyPass",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def platform_config_path(platform: str, root: Path | None = None) -> Path:
    if platform not in PLATFORMS:
        raise ConfigError(f"Unknown platform: {platform}; choose from {PLATFORMS}")
    return (root or project_root()) / "config" / platform / "lab.json"


def platform_data_root(platform: str, root: Path | None = None) -> Path:
    if platform not in PLATFORMS:
        raise ConfigError(f"Unknown platform: {platform}; choose from {PLATFORMS}")
    return (root or project_root()) / "data" / platform


def load_platform_config(
    platform: str,
    root: Path | None = None,
    *,
    require_ready: bool = True,
) -> dict:
    config = load_config(platform_config_path(platform, root))
    metadata = config.get("platform", {})
    configured_name = metadata.get("name")
    if configured_name != platform:
        raise ConfigError(
            f"Platform config mismatch: requested {platform}, got {configured_name!r}"
        )
    missing = set(ENVIRONMENTS) - set(config["environments"])
    if missing:
        raise ConfigError(
            f"Platform {platform} is missing environments: {sorted(missing)}"
        )
    if require_ready and not metadata.get("ready", False):
        raise ConfigError(
            f"Platform {platform} is scaffolded but not ready; fill in "
            f"{platform_config_path(platform, root)} and set platform.ready=true"
        )
    return config
