#!/usr/bin/env python3
"""Run the fixed 600-second bare/Jailhouse cyclictest campaign.

No command-line parameters are required. Execute this file directly with the
project's Python environment.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


# Fixed campaign definition. Each case reboots into its target environment and
# executes five independent 600-second runs.
CAMPAIGN = (
    ("bare", "cyclictest"),
    ("bare", "cyclictest-stress"),
    ("jailhouse", "cyclictest"),
    ("jailhouse", "cyclictest-stress"),
)

DURATION_SECONDS = 600
RUNS_PER_CASE = 5
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PLATFORM


def build_case_config(environment: str, experiment: str) -> dict:
    """Build one case from the shared, hardware-validated settings."""
    original_environment = settings.ENVIRONMENT
    original_experiment = settings.EXPERIMENT
    original_duration = settings.DURATION_SECONDS
    try:
        settings.ENVIRONMENT = environment
        settings.EXPERIMENT = experiment
        settings.DURATION_SECONDS = DURATION_SECONDS
        return settings.build_run_config()
    finally:
        settings.ENVIRONMENT = original_environment
        settings.EXPERIMENT = original_experiment
        settings.DURATION_SECONDS = original_duration


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = DATA_ROOT / "campaigns" / campaign_id / "campaign.json"
    summary = {
        "campaign_id": campaign_id,
        "status": "running",
        "duration_seconds": DURATION_SECONDS,
        "runs_per_case": RUNS_PER_CASE,
        "reboot_policy": REBOOT_POLICY,
        "started_at": datetime.now().astimezone().isoformat(),
        "cases": [],
    }
    write_summary(summary_path, summary)

    print("=" * 72)
    print("RK3588 fixed 600-second experiment campaign")
    print("Cases: bare/jailhouse x cyclictest/cyclictest-stress")
    print(f"Campaign summary: {summary_path.resolve()}")
    print("=" * 72)

    for index, (environment, experiment) in enumerate(CAMPAIGN, start=1):
        case = {
            "index": index,
            "environment": environment,
            "experiment": experiment,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
        }
        summary["cases"].append(case)
        write_summary(summary_path, summary)

        print("\n" + "=" * 72)
        print(
            f"CASE {index}/{len(CAMPAIGN)}: {environment} / {experiment} / "
            f"{DURATION_SECONDS}s"
        )
        print("=" * 72)

        try:
            config = build_case_config(environment, experiment)
            outputs = ExperimentRunner(config, DATA_ROOT, dry_run=False).run(
                RUNS_PER_CASE, REBOOT_POLICY
            )
        except Exception as exc:
            case["status"] = "failed"
            case["finished_at"] = datetime.now().astimezone().isoformat()
            case["error"] = str(exc)
            summary["status"] = "failed"
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            write_summary(summary_path, summary)
            print(f"\nCAMPAIGN STOPPED: case {index} failed: {exc}")
            print(f"Summary: {summary_path.resolve()}")
            return 1

        case["status"] = "completed"
        case["finished_at"] = datetime.now().astimezone().isoformat()
        case["output_directories"] = [str(path.resolve()) for path in outputs]
        write_summary(summary_path, summary)

    summary["status"] = "completed"
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_summary(summary_path, summary)
    print("\n" + "=" * 72)
    print("CAMPAIGN COMPLETED")
    print(f"Summary: {summary_path.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
