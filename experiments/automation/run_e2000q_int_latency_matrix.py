#!/usr/bin/env python3
"""Resumable E2000Q int-latency matrix for all seven environments."""

import json
import re
from datetime import datetime
from pathlib import Path

from xhypass_lab.int_latency_config import build_int_latency_run_config
from xhypass_lab.int_latency_runner import IntLatencyRunner, PostRunRecoveryError


# This is the target total, not the number appended on every invocation.
# Keep 1 for the current smoke pass. Change it to 5 for the formal campaign;
# conditions with one valid result will then receive only four additional runs.
TARGET_RUNS_PER_CONDITION = 5
ENVIRONMENTS = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
    "XHyPass",
)
CONDITIONS = ("idle", "stress")
DURATION_SECONDS = 600
GRACE_SECONDS = 180
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "E2000Q"
CONTINUE_ON_FAILURE = True
DRY_RUN = False
SUMMARY_PATH = (
    DATA_ROOT / "campaigns" / "int-latency-matrix" / "campaign.json"
)


def build_run_config(environment: str, condition: str) -> dict:
    return build_int_latency_run_config(
        "E2000Q",
        environment,
        condition,
        {
            "duration_seconds": DURATION_SECONDS,
            "timeout_seconds": DURATION_SECONDS + GRACE_SECONDS,
        },
    )


def result_series(config: dict, data_root: Path = DATA_ROOT) -> Path:
    experiment = config["experiment"]
    return (
        data_root
        / "int-latency"
        / config["environment_name"]
        / experiment["condition"]
    )


def successful_results(config: dict, data_root: Path = DATA_ROOT) -> list[Path]:
    series = result_series(config, data_root)
    completion = re.compile(
        str(config["experiment"]["completion_pattern"]), re.I | re.M
    )
    valid: list[tuple[int, Path]] = []
    for path in series.glob("rtos_run*.log"):
        name_match = re.fullmatch(r"rtos_run(\d+)\.log", path.name)
        if not name_match:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if completion.search(text):
            valid.append((int(name_match.group(1)), path))
    return [path for _, path in sorted(valid)]


def _write_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    if TARGET_RUNS_PER_CONDITION <= 0:
        raise ValueError("TARGET_RUNS_PER_CONDITION must be positive")

    started_at = datetime.now().astimezone().isoformat()
    summary = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "target_runs_per_condition": TARGET_RUNS_PER_CONDITION,
        "environments": list(ENVIRONMENTS),
        "conditions": list(CONDITIONS),
        "completed": [],
        "failures": [],
    }
    if not DRY_RUN:
        _write_summary(summary)

    observed = {
        (environment, condition): len(
            successful_results(build_run_config(environment, condition))
        )
        for environment in ENVIRONMENTS
        for condition in CONDITIONS
    }

    for environment in ENVIRONMENTS:
        for condition in CONDITIONS:
            config = build_run_config(environment, condition)
            completed = observed[(environment, condition)]
            print("=" * 78)
            print(
                f"[MATRIX] {environment} / {condition}: "
                f"{completed}/{TARGET_RUNS_PER_CONDITION} valid runs"
            )
            while completed < TARGET_RUNS_PER_CONDITION:
                target_round = completed + 1
                try:
                    IntLatencyRunner(
                        [config], DATA_ROOT, dry_run=DRY_RUN
                    ).run()
                    if DRY_RUN:
                        completed += 1
                    else:
                        refreshed = len(successful_results(config))
                        if refreshed <= completed:
                            raise RuntimeError(
                                "Runner returned without publishing a valid "
                                "RTOS completion log"
                            )
                        completed = refreshed
                    observed[(environment, condition)] = completed
                except (Exception, KeyboardInterrupt) as exc:
                    failure = {
                        "environment": environment,
                        "condition": condition,
                        "target_round": target_round,
                        "error": str(exc),
                        "at": datetime.now().astimezone().isoformat(),
                    }
                    summary["failures"].append(failure)
                    summary["updated_at"] = failure["at"]
                    if not DRY_RUN:
                        _write_summary(summary)
                    print(
                        f"[MATRIX/FAILED] {environment} / {condition} / "
                        f"target round {target_round}: {exc}"
                    )
                    if isinstance(exc, PostRunRecoveryError):
                        observed[(environment, condition)] = len(
                            successful_results(config)
                        )
                        summary["status"] = "paused_recovery_failure"
                        summary["completed"] = [
                            {
                                "environment": env,
                                "condition": cond,
                                "valid_runs": observed[(env, cond)],
                            }
                            for env in ENVIRONMENTS
                            for cond in CONDITIONS
                        ]
                        if not DRY_RUN:
                            _write_summary(summary)
                        print(
                            "[MATRIX/PAUSED] The experiment result is valid, "
                            "but the board did not return to U-Boot. Recover "
                            "the board and re-run; this result will be skipped."
                        )
                        return 2
                    if isinstance(exc, KeyboardInterrupt) or not CONTINUE_ON_FAILURE:
                        summary["status"] = "interrupted"
                        if not DRY_RUN:
                            _write_summary(summary)
                        return 130 if isinstance(exc, KeyboardInterrupt) else 1
                    break

            summary["completed"] = [
                {
                    "environment": env,
                    "condition": cond,
                    "valid_runs": observed[(env, cond)],
                }
                for env in ENVIRONMENTS
                for cond in CONDITIONS
            ]
            summary["updated_at"] = datetime.now().astimezone().isoformat()
            if not DRY_RUN:
                _write_summary(summary)

    total_completed = sum(item["valid_runs"] for item in summary["completed"])
    total_planned = (
        len(ENVIRONMENTS) * len(CONDITIONS) * TARGET_RUNS_PER_CONDITION
    )
    summary["status"] = (
        "completed" if total_completed >= total_planned else "incomplete"
    )
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    summary["updated_at"] = summary["finished_at"]
    if not DRY_RUN:
        _write_summary(summary)
    print("=" * 78)
    print(
        f"[MATRIX/FINISHED] {total_completed}/{total_planned} valid results; "
        f"{len(summary['failures'])} failed attempts this run"
    )
    print(f"[SUMMARY] {SUMMARY_PATH.resolve()}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
