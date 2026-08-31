#!/usr/bin/env python3
"""Replace workstation-specific paths in copied experiment metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Any


DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def normalize_path(value: str) -> str:
    if not DRIVE_PATH.match(value):
        return value
    parts = list(PureWindowsPath(value).parts)
    lowered = [part.casefold() for part in parts]
    if "xhypassscripts" in lowered:
        index = lowered.index("xhypassscripts") + 1
        tail = parts[index:]
        if tail and tail[0].casefold() == "data":
            if len(tail) > 1 and tail[1].casefold() == "tl3588":
                tail[1] = "RK3588"
            elif len(tail) > 1 and tail[1].casefold() not in {"rk3588", "e2000q"}:
                tail.insert(1, "RK3588")
        elif tail and tail[0].casefold() == "experiment_material":
            tail = ["experiments", "workloads", *tail[1:]]
        elif tail and tail[0].casefold() == "boot_env":
            tail = ["experiments", "boot-artifacts", *tail[1:]]
        return Path(*tail).as_posix()
    if "sftp" in lowered:
        index = lowered.index("sftp") + 1
        tail = parts[index:]
        platform = "E2000Q" if any("e2000q" in p.casefold() for p in tail) else "RK3588"
        if tail:
            tail = tail[1:]
        return Path("experiments", "deploy", platform, *tail).as_posix()
    return Path(parts[-1]).as_posix()


def rewrite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, str):
        return normalize_path(value)
    return value


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    paths = sorted((repository_root / "data").rglob("*.json"))
    paths.extend(
        sorted(
            (
                repository_root
                / "experiments"
                / "automation"
                / "config"
            ).rglob("*.json")
        )
    )
    for path in paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(rewrite(content), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Sanitized {len(paths)} JSON files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
