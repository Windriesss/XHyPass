#!/usr/bin/env python3
"""Formal seven-environment NN campaign; edit constants below and run directly."""

from __future__ import annotations

import json
import tarfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from smoke_scripts import (
    run_nn_bare,
    run_nn_jailhouse,
    run_nn_xen_credit2,
    run_nn_xen_credit2_wfx,
    run_nn_xen_null,
    run_nn_xen_null_wfx,
    run_nn_xhypass,
)
from xhypass_lab.nn_runner import NNExperimentRunner
from xhypass_lab.runner import condition_name, safe_name


# ---------------------------------------------------------------------------
# Formal experiment settings: edit here; no command-line arguments are used.
# One run executes light, medium, and heavy sequentially. The scheduler gives
# every environment its round-N attempt before starting the round-N+1 pass;
# failures are recorded but never stop later environments or priority passes.
# The default campaign contains 7 x 5 x 3 x 600 s = 17.5 workload hours.
# ---------------------------------------------------------------------------
ENVIRONMENTS = (
    "bare",
    "jailhouse",
    "xen_credit2",
    "xen_credit2_WFX",
    "xen_null",
    "xen_null_WFX",
    "XHyPass",
)
RUNS_PER_ENVIRONMENT = 5
DURATION_SECONDS = 600
CYCLICTEST_INTERVAL_US = 1_000
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / PLATFORM
DRY_RUN = False
CONTINUE_AFTER_FAILURE = True
GRACE_SECONDS = 600

LOAD_PROFILES = (
    {
        "name": "light",
        "inception_qps": 0.2,
        "peak_qps": 5,
        "poisson_qps": 3,
        "constant_qps": 3,
    },
    {
        "name": "medium",
        "inception_qps": 0.2,
        "peak_qps": 12,
        "poisson_qps": 6,
        "constant_qps": 6,
    },
    {
        "name": "heavy",
        "inception_qps": 0.2,
        "peak_qps": 20,
        "poisson_qps": 9,
        "constant_qps": 9,
    },
)

BUILDERS: dict[str, Callable[[], dict]] = {
    "bare": run_nn_bare.build_run_config,
    "jailhouse": run_nn_jailhouse.build_run_config,
    "xen_credit2": run_nn_xen_credit2.build_run_config,
    "xen_credit2_WFX": run_nn_xen_credit2_wfx.build_run_config,
    "xen_null": run_nn_xen_null.build_run_config,
    "xen_null_WFX": run_nn_xen_null_wfx.build_run_config,
    "XHyPass": run_nn_xhypass.build_run_config,
}


def build_formal_config(environment: str) -> dict:
    run = BUILDERS[environment]()
    run["experiment"].update(
        {
            "profile_name": (
                "dual-tflite-formal-v2-cgroup"
                if environment == "bare"
                else "dual-tflite-formal-v1"
            ),
            "duration_seconds": DURATION_SECONDS,
            "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
            "profiles": [dict(profile) for profile in LOAD_PROFILES],
            "grace_seconds": GRACE_SECONDS,
        }
    )
    return run


def _series_directory(run: dict) -> Path:
    return (
        DATA_ROOT
        / safe_name(run["experiment_name"])
        / safe_name(run["environment_name"])
        / condition_name(run)
    )


def _valid_result_archives(run: dict) -> list[Path]:
    series = _series_directory(run)
    valid = []
    for path in sorted(series.glob("nn_results_run*.tar.gz")):
        try:
            with tarfile.open(path, "r:gz") as archive:
                if not any(
                    member.isfile() and member.name.startswith("results/")
                    for member in archive.getmembers()
                ):
                    raise ValueError("archive contains no result files")
        except (OSError, tarfile.TarError, ValueError) as exc:
            print(f"[WARN] Invalid published NN result, not counted: {path}: {exc}")
            continue
        valid.append(path)
    return valid


def _write_campaign(state: dict) -> Path:
    path = DATA_ROOT / "campaigns" / "full-nn-matrix" / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _snapshot(configs: dict[str, dict], failures: list[dict]) -> dict:
    completed = {
        environment: len(_valid_result_archives(run))
        for environment, run in configs.items()
    }
    return {
        "campaign": "full-nn-matrix",
        "platform": PLATFORM,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {
            "environments": list(ENVIRONMENTS),
            "runs_per_environment": RUNS_PER_ENVIRONMENT,
            "duration_seconds_per_level": DURATION_SECONDS,
            "levels": [profile["name"] for profile in LOAD_PROFILES],
            "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
            "reboot_policy": REBOOT_POLICY,
            "scheduling": "round-robin-continuous",
            "dry_run": DRY_RUN,
        },
        "completed_successes": completed,
        "planned_successes": len(ENVIRONMENTS) * RUNS_PER_ENVIRONMENT,
        "failures_this_invocation": failures,
    }


def main() -> int:
    unknown = set(ENVIRONMENTS) - set(BUILDERS)
    if unknown:
        raise ValueError(f"No NN configuration builder for: {sorted(unknown)}")
    configs = {environment: build_formal_config(environment) for environment in ENVIRONMENTS}
    failures: list[dict] = []
    summary_path = _write_campaign(_snapshot(configs, failures))

    print("=" * 78)
    print(f"{PLATFORM} FULL NN MATRIX")
    print(
        f"7 environments, {RUNS_PER_ENVIRONMENT} successful rounds each, "
        f"light/medium/heavy x {DURATION_SECONDS}s"
    )
    print(f"Progress: {summary_path.resolve()}")
    print("=" * 78)

    try:
        for target_round in range(1, RUNS_PER_ENVIRONMENT + 1):
            print("=" * 78)
            print(
                f"[ROUND {target_round}/{RUNS_PER_ENVIRONMENT}] "
                "Trying every environment before the next priority round"
            )
            for environment in ENVIRONMENTS:
                run = configs[environment]
                completed = len(_valid_result_archives(run))
                if completed >= target_round:
                    print(
                        f"[ROUND/SKIP] {environment}: {completed} successful "
                        f"run(s), target {target_round} already satisfied"
                    )
                    continue
                print(
                    f"[ROUND/RUN] {environment}: {completed} successful "
                    f"run(s); targeting round {target_round}"
                )
                try:
                    NNExperimentRunner(run, DATA_ROOT, dry_run=DRY_RUN).run(
                        1, REBOOT_POLICY
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    failure = {
                        "environment": environment,
                        "target_round": target_round,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"[FAILED] {environment}: {failure['error']}")
                    traceback.print_exc()
                    _write_campaign(_snapshot(configs, failures))
                    if not CONTINUE_AFTER_FAILURE:
                        return 1
                else:
                    print(
                        f"[OK] {environment}: "
                        f"{len(_valid_result_archives(run))}/"
                        f"{RUNS_PER_ENVIRONMENT} successful rounds"
                    )
                    _write_campaign(_snapshot(configs, failures))

            missing = [
                environment
                for environment in ENVIRONMENTS
                if len(_valid_result_archives(configs[environment]))
                < target_round
            ]
            if missing:
                print(
                    f"[ROUND/CONTINUE] Round {target_round} still has missing "
                    f"successes for: {', '.join(missing)}. Failures were "
                    "recorded; continuing with the next priority round."
                )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Progress saved; rerun this script to continue.")
        _write_campaign(_snapshot(configs, failures))
        return 130

    state = _snapshot(configs, failures)
    summary_path = _write_campaign(state)
    completed_total = sum(state["completed_successes"].values())
    planned_total = int(state["planned_successes"])
    print("=" * 78)
    print(
        f"NN MATRIX FINISHED: {completed_total}/{planned_total} "
        f"successful rounds, {len(failures)} failed attempts this invocation"
    )
    if completed_total < planned_total:
        print(
            "The full priority pass finished with missing successes. Rerun "
            "this script to fill only the remaining results."
        )
    print(f"Summary: {summary_path.resolve()}")
    print("=" * 78)
    return 0 if completed_total >= planned_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
