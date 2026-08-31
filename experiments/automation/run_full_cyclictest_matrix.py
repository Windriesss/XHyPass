#!/usr/bin/env python3
"""Run the complete 600-second cyclictest experiment matrix.

All settings are edited directly in this file; no command-line arguments are
required. Progress is persisted after every round so a long campaign always
has an inspectable summary, including failures.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smoke_scripts import run_experiment as settings
from xhypass_lab.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Fixed formal-experiment settings. Edit here when a new campaign is needed.
# ---------------------------------------------------------------------------
ENVIRONMENTS = (
    "bare",
    "jailhouse",
    # "xen_credit2",
    # "xen_credit2_WFX",
    # "xen_null",
    # "xen_null_WFX",
    # "XHyPass",
)
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
INTERVALS_US = (1_000, 10_000)
DURATION_SECONDS = 600
RUNS_PER_CONDITION = 5
REBOOT_POLICY = "each-run"
CONTINUE_ON_FAILURE = True
RESUME_INCOMPLETE_CAMPAIGN = True
MAX_CONSECUTIVE_INFRASTRUCTURE_FAILURES = 2
PLATFORM = "RK3588"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PLATFORM


class CampaignPaused(RuntimeError):
    """Raised after repeated control-plane failures to preserve pending work."""


def _is_infrastructure_failure(exc: Exception) -> bool:
    """Recognize failures that normally mean the board/control path is wedged."""
    message = str(exc).lower()
    prompt_timeout = "timed out" in message and (
        "login" in message or "shell prompt" in message
    )
    connection_failure = any(
        marker in message
        for marker in (
            "ssh connection",
            "ssh unavailable",
            "could not connect",
            "connection timed out",
        )
    )
    return prompt_timeout or connection_failure


def build_case_config(environment: str, experiment: str, interval_us: int) -> dict:
    """Build one case while restoring the shared editable settings afterward."""
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
        settings.INTERVAL_US = interval_us
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


def _campaign_matches(summary: dict) -> bool:
    return (
        summary.get("campaign_type") == "full-cyclictest-matrix"
        and summary.get("platform", "RK3588") == PLATFORM
        and summary.get("environments") == list(ENVIRONMENTS)
        and summary.get("experiments") == list(EXPERIMENTS)
        and summary.get("intervals_us") == list(INTERVALS_US)
        and summary.get("duration_seconds") == DURATION_SECONDS
        and summary.get("runs_per_condition") == RUNS_PER_CONDITION
        and summary.get("reboot_policy") == REBOOT_POLICY
    )


def find_incomplete_campaign() -> tuple[Path, dict] | None:
    """Return the newest compatible campaign that may still need successes."""
    if not RESUME_INCOMPLETE_CAMPAIGN:
        return None
    campaign_root = DATA_ROOT / "campaigns"
    candidates = sorted(
        campaign_root.glob("full-cyclictest-matrix-*/campaign.json"), reverse=True
    )
    for path in candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") in {
            "running",
            "interrupted",
            "completed_with_failures",
        } and _campaign_matches(summary):
            return path, summary
    return None


def _condition_key(environment: str, experiment: str, interval_us: int) -> tuple:
    return environment, experiment, int(interval_us)


def main() -> int:
    if RUNS_PER_CONDITION < 1:
        raise ValueError("RUNS_PER_CONDITION must be at least 1")

    conditions = tuple(
        (environment, experiment, interval_us)
        for experiment in EXPERIMENTS
        for interval_us in INTERVALS_US
        for environment in ENVIRONMENTS
    )
    total_rounds = len(conditions) * RUNS_PER_CONDITION
    resumed = find_incomplete_campaign()
    if resumed:
        summary_path, summary = resumed
        now = datetime.now().astimezone().isoformat()
        for condition in summary.get("conditions", []):
            condition.pop("failed_rounds", None)
            for record in condition.get("rounds", []):
                if record.get("status") == "running":
                    record["status"] = "interrupted"
                    record["finished_at"] = now
        summary["status"] = "running"
        summary.pop("failed_rounds", None)
        summary.pop("finished_at", None)
        summary.setdefault("resume_events", []).append({"resumed_at": now})
        campaign_id = str(summary["campaign_id"])
        print(f"[MATRIX/RESUME] Resuming campaign {campaign_id}")
    else:
        campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_path = (
            DATA_ROOT
            / "campaigns"
            / f"full-cyclictest-matrix-{campaign_id}"
            / "campaign.json"
        )
        summary = {
            "campaign_id": campaign_id,
            "campaign_type": "full-cyclictest-matrix",
            "platform": PLATFORM,
            "status": "running",
            "environments": list(ENVIRONMENTS),
            "experiments": list(EXPERIMENTS),
            "intervals_us": list(INTERVALS_US),
            "duration_seconds": DURATION_SECONDS,
            "runs_per_condition": RUNS_PER_CONDITION,
            "reboot_policy": REBOOT_POLICY,
            "condition_count": len(conditions),
            "total_rounds": total_rounds,
            "started_at": datetime.now().astimezone().isoformat(),
            "conditions": [],
        }
    write_summary(summary_path, summary)

    print("=" * 78)
    print(f"{PLATFORM} full cyclictest matrix")
    print(
        f"{len(ENVIRONMENTS)} environments x {len(EXPERIMENTS)} experiments x "
        f"{len(INTERVALS_US)} intervals x {RUNS_PER_CONDITION} rounds"
    )
    print(f"Total: {len(conditions)} conditions / {total_rounds} rounds")
    print(f"Each round: {DURATION_SECONDS}s; reboot policy: {REBOOT_POLICY}")
    print(f"Campaign summary: {summary_path.resolve()}")
    print("=" * 78)

    all_existing_rounds = [
        record
        for condition in summary.get("conditions", [])
        for record in condition.get("rounds", [])
    ]
    completed_rounds = sum(
        record.get("status") == "completed" for record in all_existing_rounds
    )
    failed_attempts = sum(
        record.get("status") == "failed" for record in all_existing_rounds
    )
    interrupted = False
    paused = False
    consecutive_infrastructure_failures = 0
    try:
        for condition_index, (environment, experiment, interval_us) in enumerate(
            conditions, start=1
        ):
            key = _condition_key(environment, experiment, interval_us)
            condition = next(
                (
                    item
                    for item in summary["conditions"]
                    if _condition_key(
                        item["environment"], item["experiment"], item["interval_us"]
                    ) == key
                ),
                None,
            )
            if condition is None:
                condition = {
                    "index": condition_index,
                    "environment": environment,
                    "experiment": experiment,
                    "interval_us": interval_us,
                    "duration_seconds": DURATION_SECONDS,
                    "status": "running",
                    "started_at": datetime.now().astimezone().isoformat(),
                    "rounds": [],
                }
                summary["conditions"].append(condition)
            else:
                condition["status"] = "running"
            write_summary(summary_path, summary)

            print("\n" + "=" * 78)
            print(
                f"CONDITION {condition_index}/{len(conditions)}: {environment} / "
                f"{experiment} / interval={interval_us}us / {DURATION_SECONDS}s"
            )
            print("=" * 78)

            for round_index in range(1, RUNS_PER_CONDITION + 1):
                successful = next(
                    (
                        record
                        for record in condition["rounds"]
                        if int(record["round"]) == round_index
                        and record.get("status") == "completed"
                    ),
                    None,
                )
                if successful:
                    print(
                        f"[MATRIX/RESUME] Skip condition {condition_index}, "
                        f"round {round_index}: successful result already exists"
                    )
                    continue
                attempt = 1 + sum(
                    int(record["round"]) == round_index
                    for record in condition["rounds"]
                )
                round_record = {
                    "round": round_index,
                    "attempt": attempt,
                    "status": "running",
                    "started_at": datetime.now().astimezone().isoformat(),
                }
                condition["rounds"].append(round_record)
                write_summary(summary_path, summary)
                print(
                    f"\n[MATRIX] condition {condition_index}/{len(conditions)}, "
                    f"round {round_index}/{RUNS_PER_CONDITION}, "
                    f"successful results {completed_rounds}/{total_rounds}, "
                    f"attempt {attempt}"
                )

                pause_after_round = False
                try:
                    outputs = ExperimentRunner(
                        build_case_config(environment, experiment, interval_us),
                        DATA_ROOT,
                        dry_run=False,
                    ).run(1, REBOOT_POLICY)
                except Exception as exc:
                    failed_attempts += 1
                    round_record.update(status="failed", error=str(exc))
                    print(
                        f"[MATRIX FAILED] {environment} / {experiment} / "
                        f"interval={interval_us} / round={round_index}: {exc}"
                    )
                    if _is_infrastructure_failure(exc):
                        consecutive_infrastructure_failures += 1
                        pause_after_round = (
                            consecutive_infrastructure_failures
                            >= MAX_CONSECUTIVE_INFRASTRUCTURE_FAILURES
                        )
                    else:
                        consecutive_infrastructure_failures = 0
                    if not CONTINUE_ON_FAILURE:
                        raise
                else:
                    consecutive_infrastructure_failures = 0
                    completed_rounds += 1
                    round_record.update(
                        status="completed",
                        output_directories=[
                            str(output.resolve()) for output in outputs
                        ],
                    )
                finally:
                    round_record["finished_at"] = (
                        datetime.now().astimezone().isoformat()
                    )
                    summary["completed_rounds"] = completed_rounds
                    summary["failed_attempts"] = failed_attempts
                    summary["pending_rounds"] = total_rounds - completed_rounds
                    write_summary(summary_path, summary)

                if pause_after_round:
                    raise CampaignPaused(
                        f"{consecutive_infrastructure_failures} consecutive "
                        "board/SSH/login failures; restore or power-cycle the "
                        "board, then run this script again to resume"
                    )

            terminal_records = [
                record
                for record in condition["rounds"]
                if record.get("status") in {"completed", "failed"}
            ]
            condition["completed_rounds"] = sum(
                record["status"] == "completed" for record in terminal_records
            )
            condition["failed_attempts"] = sum(
                record["status"] == "failed" for record in terminal_records
            )
            condition.pop("failed_rounds", None)
            condition["pending_rounds"] = max(
                0, RUNS_PER_CONDITION - condition["completed_rounds"]
            )
            condition["status"] = (
                "completed"
                if condition["pending_rounds"] == 0
                else "incomplete"
            )
            condition["finished_at"] = datetime.now().astimezone().isoformat()
            write_summary(summary_path, summary)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[MATRIX] Interrupted by user; progress has been saved.")
    except CampaignPaused as exc:
        paused = True
        print(f"\n[MATRIX/PAUSED] {exc}")
    finally:
        summary["completed_rounds"] = completed_rounds
        summary["failed_attempts"] = failed_attempts
        summary["pending_rounds"] = total_rounds - completed_rounds
        summary["status"] = (
            "interrupted"
            if interrupted or paused
            else "completed"
            if completed_rounds >= total_rounds
            else "completed_with_failures"
        )
        summary["finished_at"] = datetime.now().astimezone().isoformat()
        write_summary(summary_path, summary)

    print("\n" + "=" * 78)
    print(
        f"MATRIX FINISHED: {completed_rounds} completed, "
        f"{total_rounds - completed_rounds} pending, "
        f"{failed_attempts} failed attempts, {total_rounds} planned successes"
    )
    print(f"Summary: {summary_path.resolve()}")
    print("=" * 78)
    if interrupted or paused:
        return 130
    return 0 if completed_rounds >= total_rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())
