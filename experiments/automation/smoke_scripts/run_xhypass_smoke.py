#!/usr/bin/env python3
"""Run 10-second cyclictest smoke cases for the XHyPass environment."""

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


ENVIRONMENT = "XHyPass"
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
DURATION_SECONDS = 10
INTERVAL_US = 1_000
RUNS_PER_CASE = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM


def build_case_config(experiment: str) -> dict:
    original = (
        settings.ENVIRONMENT,
        settings.EXPERIMENT,
        settings.DURATION_SECONDS,
        settings.INTERVAL_US,
    )
    try:
        settings.ENVIRONMENT = ENVIRONMENT
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
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = DATA_ROOT / "campaigns" / f"xhypass-smoke-{campaign_id}" / "campaign.json"
    summary = {
        "campaign_id": campaign_id,
        "campaign_type": "xhypass-smoke",
        "status": "running",
        "duration_seconds": DURATION_SECONDS,
        "interval_us": INTERVAL_US,
        "started_at": datetime.now().astimezone().isoformat(),
        "cases": [],
    }
    write_summary(summary_path, summary)
    failures = 0
    for index, experiment in enumerate(EXPERIMENTS, start=1):
        case = {
            "index": index,
            "environment": ENVIRONMENT,
            "experiment": experiment,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
        }
        summary["cases"].append(case)
        write_summary(summary_path, summary)
        print(f"\n{'=' * 72}\nCASE {index}/2: {ENVIRONMENT} / {experiment} / 10s\n{'=' * 72}")
        try:
            outputs = ExperimentRunner(
                build_case_config(experiment), DATA_ROOT, dry_run=False
            ).run(RUNS_PER_CASE, REBOOT_POLICY)
        except Exception as exc:
            failures += 1
            case.update(status="failed", error=str(exc))
            print(f"[CASE FAILED] {experiment}: {exc}")
        else:
            case.update(
                status="completed",
                output_directories=[str(path.resolve()) for path in outputs],
            )
        finally:
            case["finished_at"] = datetime.now().astimezone().isoformat()
            write_summary(summary_path, summary)
    summary.update(
        completed_cases=len(EXPERIMENTS) - failures,
        failed_cases=failures,
        status="completed" if not failures else "completed_with_failures",
        finished_at=datetime.now().astimezone().isoformat(),
    )
    write_summary(summary_path, summary)
    print(f"\nXHyPass smoke: {len(EXPERIMENTS) - failures} passed, {failures} failed")
    print(f"Summary: {summary_path.resolve()}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
