#!/usr/bin/env python3
"""Validate empirical transition-correctness records."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


REQUIRED_NOTIFICATION_FIELDS = (
    "produced",
    "consumed",
    "lost",
    "duplicates",
    "unexpected",
)


def validate_record(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("final_mode") != "DYN":
        errors.append("final_mode must be DYN")
    if payload.get("passed") is not True:
        errors.append("passed must be true")
    if int(payload.get("watchdog_timeouts", -1)) != 0:
        errors.append("watchdog_timeouts must be zero")
    notifications = payload.get("notifications")
    if not isinstance(notifications, dict) or not notifications:
        errors.append("notifications must be a non-empty object")
    else:
        for kind, result in notifications.items():
            if result.get("skipped") is True:
                continue
            for field in REQUIRED_NOTIFICATION_FIELDS:
                if field not in result:
                    errors.append(f"{kind}: missing {field}")
            for field in ("lost", "duplicates", "unexpected"):
                if int(result.get(field, -1)) != 0:
                    errors.append(f"{kind}: {field} must be zero")
            for field in ("reordered", "wrong_cpu"):
                if int(result.get(field, 0)) != 0:
                    errors.append(f"{kind}: {field} must be zero")
            produced = int(result.get("produced", -1))
            consumed = int(result.get("consumed", -2))
            if produced <= 0:
                errors.append(f"{kind}: produced must be positive")
            if produced != consumed:
                errors.append(f"{kind}: produced != consumed")
            if int(result.get("timeouts", 0)) != 0:
                errors.append(f"{kind}: timeouts must be zero")
            if kind == "device-SPI":
                spi_timer = payload.get("spi_timer", {})
                if spi_timer.get("affinity_verified") is not True:
                    errors.append("device-SPI: affinity must be verified")
                if int(spi_timer.get("gic_hwirq", -1)) != 321:
                    errors.append("device-SPI: gic_hwirq must be 321")
    cases = payload.get("state_cases")
    if not isinstance(cases, list) or not cases:
        errors.append("state_cases must be a non-empty list")
    else:
        for index, case in enumerate(cases):
            if not case.get("passed", False):
                errors.append(f"state_cases[{index}] did not pass")
            if case.get("final_mode") not in ("DYN", "RTO"):
                errors.append(f"state_cases[{index}] has invalid final_mode")
    return errors


def expand_records(values: list[str]) -> list[Path]:
    """Expand files, directories, and shell-independent glob patterns."""
    records: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            records.extend(sorted(path.rglob("correctness.json")))
            continue
        matches = [Path(match) for match in sorted(glob.glob(value))]
        records.extend(matches or [path])
    unique = list(dict.fromkeys(records))
    if not unique:
        raise FileNotFoundError("no correctness records found")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"correctness record not found: {missing[0]}")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()
    failures = 0
    records = expand_records(args.records)
    for path in records:
        errors = validate_record(path)
        if errors:
            failures += 1
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[PASS] {path}")
    print(f"Validated {len(records)} record(s): {len(records) - failures} passed, "
          f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
