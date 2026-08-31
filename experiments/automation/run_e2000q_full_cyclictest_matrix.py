#!/usr/bin/env python3
"""Run the formal 7 x 2 x 5 E2000Q cyclictest matrix.

Successful histograms are counted at each condition root. Re-running this
script skips existing successes and fills only the missing rounds.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smoke_scripts.run_e2000q_experiment import DATA_ROOT, DRY_RUN, build_run_config
from xhypass_lab.runner import ExperimentRunner, condition_name, safe_name


# Edit parameters here; no command-line arguments are needed.
ENVIRONMENTS = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
    "XHyPass",
)
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
DURATION_SECONDS = 600
INTERVAL_US = 1_000
RUNS_PER_CONDITION = 5
REBOOT_POLICY = "each-run"
CONTINUE_ON_FAILURE = True
PAUSE_ON_INFRASTRUCTURE_FAILURE = True
SUMMARY_PATH = DATA_ROOT / "campaigns" / "full-cyclictest-matrix" / "campaign.json"


def _completed_count(run: dict) -> int:
    series = (
        DATA_ROOT
        / safe_name(run["experiment_name"])
        / safe_name(run["environment_name"])
        / condition_name(run)
    )
    return sum(1 for path in series.glob("hist_run*.txt") if path.is_file())


def _write_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _is_infrastructure_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "boot failed after",
            "no reboot marker appeared",
            "unable to reach u-boot",
            "missed the u-boot stop window",
            "could not open port",
            "serialexception",
        )
    )


def main() -> int:
    conditions = [
        (environment, experiment)
        for environment in ENVIRONMENTS
        for experiment in EXPERIMENTS
    ]
    summary = {
        "platform": "E2000Q",
        "status": "running",
        "environments": list(ENVIRONMENTS),
        "experiments": list(EXPERIMENTS),
        "duration_seconds": DURATION_SECONDS,
        "interval_us": INTERVAL_US,
        "runs_per_condition": RUNS_PER_CONDITION,
        "planned_successes": len(conditions) * RUNS_PER_CONDITION,
        "updated_at": datetime.now().astimezone().isoformat(),
        "conditions": [],
    }
    if not DRY_RUN:
        _write_summary(summary)

    failures = 0
    for index, (environment, experiment) in enumerate(conditions, start=1):
        run = build_run_config(
            experiment,
            environment,
            duration_seconds=DURATION_SECONDS,
            interval_us=INTERVAL_US,
        )
        completed_before = _completed_count(run)
        condition_record = {
            "index": index,
            "environment": environment,
            "experiment": experiment,
            "completed_before": completed_before,
            "attempts": [],
        }
        summary["conditions"].append(condition_record)
        print("=" * 78)
        print(
            f"CONDITION {index}/{len(conditions)}: {environment} / {experiment}; "
            f"{completed_before}/{RUNS_PER_CONDITION} successes already present"
        )
        print("=" * 78)

        for target_round in range(completed_before + 1, RUNS_PER_CONDITION + 1):
            attempt = {
                "target_round": target_round,
                "started_at": datetime.now().astimezone().isoformat(),
                "status": "running",
            }
            condition_record["attempts"].append(attempt)
            if not DRY_RUN:
                summary["updated_at"] = datetime.now().astimezone().isoformat()
                _write_summary(summary)
            try:
                ExperimentRunner(run, DATA_ROOT, dry_run=DRY_RUN).run(
                    1, REBOOT_POLICY
                )
            except (Exception, KeyboardInterrupt) as exc:
                failures += 1
                attempt["status"] = "failed"
                attempt["error"] = str(exc)
                print(
                    f"[MATRIX/FAILED] {environment} / {experiment} / "
                    f"target round {target_round}: {exc}"
                )
                if (
                    PAUSE_ON_INFRASTRUCTURE_FAILURE
                    and _is_infrastructure_failure(exc)
                ):
                    summary["status"] = "paused_infrastructure_failure"
                    summary["paused_at"] = datetime.now().astimezone().isoformat()
                    summary["pause_reason"] = str(exc)
                    summary["failed_attempts_this_run"] = failures
                    if not DRY_RUN:
                        _write_summary(summary)
                    print(
                        "[MATRIX/PAUSED] Infrastructure failure detected; "
                        "remaining conditions were not attempted. Re-run the "
                        "same script after the board/network recovers."
                    )
                    return 2
                if not CONTINUE_ON_FAILURE or isinstance(exc, KeyboardInterrupt):
                    if not DRY_RUN:
                        summary["status"] = "interrupted"
                        _write_summary(summary)
                    raise
            else:
                attempt["status"] = "completed"
            finally:
                attempt["finished_at"] = datetime.now().astimezone().isoformat()
                if not DRY_RUN:
                    summary["updated_at"] = datetime.now().astimezone().isoformat()
                    _write_summary(summary)

        condition_record["completed_after"] = _completed_count(run)

    completed = sum(
        min(
            _completed_count(
                build_run_config(
                    experiment,
                    environment,
                    duration_seconds=DURATION_SECONDS,
                    interval_us=INTERVAL_US,
                )
            ),
            RUNS_PER_CONDITION,
        )
        for environment, experiment in conditions
    )
    summary["completed_successes"] = completed
    summary["failed_attempts_this_run"] = failures
    summary["status"] = (
        "completed" if completed == summary["planned_successes"] else "incomplete"
    )
    summary["updated_at"] = datetime.now().astimezone().isoformat()
    if not DRY_RUN:
        _write_summary(summary)
    print(
        f"[MATRIX/FINISHED] {completed}/{summary['planned_successes']} completed; "
        f"{failures} failed attempts this run"
    )
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
