#!/usr/bin/env python3
"""Smoke-test four Xen environments with two aligned cyclictest workloads."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


ENVIRONMENTS = (
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
)
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
CAMPAIGN = tuple(
    (environment, experiment)
    for environment in ENVIRONMENTS
    for experiment in EXPERIMENTS
)

DURATION_SECONDS = 10
INTERVAL_US = 1_000
RUNS_PER_CASE = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM


def build_case_config(environment: str, experiment: str) -> dict:
    original = (
        settings.ENVIRONMENT,
        settings.EXPERIMENT,
        settings.DURATION_SECONDS,
        settings.INTERVAL_US,
    )
    try:
        settings.ENVIRONMENT = environment
        settings.EXPERIMENT = experiment
        settings.DURATION_SECONDS = DURATION_SECONDS
        settings.INTERVAL_US = INTERVAL_US
        return settings.build_run_config()
    finally:
        (
            settings.ENVIRONMENT,
            settings.EXPERIMENT,
            settings.DURATION_SECONDS,
            settings.INTERVAL_US,
        ) = original


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = (
        DATA_ROOT / "campaigns" / f"xen-smoke-{campaign_id}" / "campaign.json"
    )
    summary = {
        "campaign_id": campaign_id,
        "campaign_type": "xen-variants-smoke",
        "status": "running",
        "duration_seconds": DURATION_SECONDS,
        "interval_us": INTERVAL_US,
        "runs_per_case": RUNS_PER_CASE,
        "reboot_policy": REBOOT_POLICY,
        "started_at": datetime.now().astimezone().isoformat(),
        "cases": [],
    }
    write_summary(summary_path, summary)

    print("=" * 72)
    print("RK3588 Xen variants smoke campaign")
    print(f"Cases: {len(CAMPAIGN)}, duration: {DURATION_SECONDS}s, interval: {INTERVAL_US}us")
    print(f"Campaign summary: {summary_path.resolve()}")
    print("=" * 72)

    failures = 0
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
            f"{DURATION_SECONDS}s / interval={INTERVAL_US}us"
        )
        print("=" * 72)

        try:
            config = build_case_config(environment, experiment)
            outputs = ExperimentRunner(config, DATA_ROOT, dry_run=False).run(
                RUNS_PER_CASE, REBOOT_POLICY
            )
        except Exception as exc:
            failures += 1
            case["status"] = "failed"
            case["error"] = str(exc)
            print(f"[CASE FAILED] {environment} / {experiment}: {exc}")
        else:
            case["status"] = "completed"
            case["output_directories"] = [
                str(output.resolve()) for output in outputs
            ]
        finally:
            case["finished_at"] = datetime.now().astimezone().isoformat()
            write_summary(summary_path, summary)

    summary["completed_cases"] = len(CAMPAIGN) - failures
    summary["failed_cases"] = failures
    summary["status"] = "completed" if failures == 0 else "completed_with_failures"
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_summary(summary_path, summary)

    print("\n" + "=" * 72)
    print(
        f"SMOKE CAMPAIGN FINISHED: {summary['completed_cases']} passed, "
        f"{summary['failed_cases']} failed"
    )
    print(f"Summary: {summary_path.resolve()}")
    print("=" * 72)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
