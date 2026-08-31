#!/usr/bin/env python3
"""Finish RK3588 int-latency first, then finish the NN campaign."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import run_full_nn_matrix
import run_tl3588_int_latency_matrix


# Edit these constants directly; no command-line arguments are required.
NN_RUNS_PER_ENVIRONMENT = 5
INT_LATENCY_RUNS_PER_CONDITION = 5
DRY_RUN = False

ROOT = Path(__file__).resolve().parent
STATUS_PATH = (
    ROOT
    / "data"
    / "RK3588"
    / "campaigns"
    / "nn-then-int-latency"
    / "status.json"
)


def _write_status(stage: str, status: str, return_code: int | None = None) -> None:
    if DRY_RUN:
        return
    payload = {
        "platform": "RK3588",
        "workflow": "int-latency then NN",
        "stage": stage,
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "nn_runs_per_environment": NN_RUNS_PER_ENVIRONMENT,
        "int_latency_runs_per_condition": INT_LATENCY_RUNS_PER_CONDITION,
    }
    if return_code is not None:
        payload["return_code"] = return_code
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _configure_campaigns() -> None:
    run_full_nn_matrix.RUNS_PER_ENVIRONMENT = NN_RUNS_PER_ENVIRONMENT
    run_full_nn_matrix.DRY_RUN = DRY_RUN
    run_tl3588_int_latency_matrix.TARGET_RUNS_PER_CONDITION = (
        INT_LATENCY_RUNS_PER_CONDITION
    )
    run_tl3588_int_latency_matrix.DRY_RUN = DRY_RUN


def main() -> int:
    _configure_campaigns()
    print("=" * 78)
    print("RK3588 COMPLETE CAMPAIGN: INT-LATENCY -> NN")
    print(
        f"NN target: {NN_RUNS_PER_ENVIRONMENT} successful rounds per environment"
    )
    print(
        "int-latency target: "
        f"{INT_LATENCY_RUNS_PER_CONDITION} valid runs per environment/condition"
    )
    print("Existing valid results will be skipped by each resumable matrix.")
    print("=" * 78)

    _write_status("int-latency", "running")
    try:
        int_latency_code = int(run_tl3588_int_latency_matrix.main())
    except KeyboardInterrupt:
        int_latency_code = 130
    if int_latency_code != 0:
        _write_status("int-latency", "incomplete", int_latency_code)
        print(
            "[WORKFLOW/STOPPED] int-latency is not complete. Recover the "
            "board if necessary and rerun this same script; successful "
            "int-latency results will be skipped. NN was not started."
        )
        return int_latency_code

    _write_status("int-latency", "completed", 0)
    print("[WORKFLOW] int-latency complete; starting RK3588 NN matrix...")
    _write_status("NN", "running")
    try:
        nn_code = int(run_full_nn_matrix.main())
    except KeyboardInterrupt:
        nn_code = 130
    if nn_code != 0:
        _write_status("NN", "incomplete", nn_code)
        print(
            "[WORKFLOW/STOPPED] NN is not complete. Recover the board if "
            "necessary and rerun this same script; completed int-latency "
            "and successful NN results will be skipped."
        )
        return nn_code

    _write_status("all", "completed", 0)
    print("=" * 78)
    print("RK3588 INT-LATENCY AND NN CAMPAIGNS ARE COMPLETE")
    print(f"Workflow status: {STATUS_PATH.resolve()}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
