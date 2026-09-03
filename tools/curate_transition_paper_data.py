#!/usr/bin/env python3
"""Validate and curate the formal XHyPass transition datasets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CORRECTNESS_SOURCES = {
    "sgi": "SGI",
    "timer": "timer-PPI",
    "event": "event-channel",
    "spi": "device-SPI",
}
ZERO_FIELDS = (
    "lost",
    "duplicates",
    "unexpected",
    "reordered",
    "wrong_cpu",
    "timeouts",
)
LATENCY_COLUMNS = {
    "schema_version",
    "run_id",
    "condition",
    "iteration",
    "direction",
    "attempt",
    "rc",
    "counter_cycles",
    "duration_ns",
    "request_elapsed_ns",
}
DIRECTIONS = ("DYN-to-RTO", "RTO-to-DYN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-campaign", type=Path, required=True)
    parser.add_argument("--spi-campaign", type=Path, required=True)
    parser.add_argument("--latency-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_manifest(campaign: Path, *, allow_failed: bool = False) -> dict:
    path = campaign / "campaign_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status != "completed" and not (allow_failed and status == "failed"):
        raise ValueError(f"{campaign.name}: campaign is not completed")
    if payload.get("platform") != "RK3588":
        raise ValueError(f"{campaign.name}: platform is not RK3588")
    return payload


def validate_correctness(path: Path, source: str, expected_run: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema")
    if payload.get("run_id") != expected_run:
        raise ValueError(f"{path}: unexpected run_id")
    if payload.get("transition_pairs") != 10000:
        raise ValueError(f"{path}: transition_pairs must be 10000")
    if payload.get("passed") is not True or payload.get("final_mode") != "DYN":
        raise ValueError(f"{path}: correctness run did not pass")
    if payload.get("watchdog_timeouts") != 0 or payload.get("setup_errors") != 0:
        raise ValueError(f"{path}: watchdog or setup error")

    counters = payload.get("transition_counters", {})
    for name in ("enter_successes", "exit_successes"):
        if counters.get(name) != 10000:
            raise ValueError(f"{path}: {name} must be 10000")
    if counters.get("other_errors") != 0:
        raise ValueError(f"{path}: other_errors must be zero")
    successful_transitions = counters["enter_successes"] + counters["exit_successes"]
    total_attempts = counters.get("enter_attempts", 0) + counters.get("exit_attempts", 0)
    transient_returns = counters.get("busy_returns", 0) + counters.get("again_returns", 0)
    if total_attempts != successful_transitions + transient_returns:
        raise ValueError(f"{path}: transition attempt accounting mismatch")

    name = CORRECTNESS_SOURCES[source]
    result = payload.get("notifications", {}).get(name)
    if not isinstance(result, dict) or result.get("skipped") is True:
        raise ValueError(f"{path}: {name} was not exercised")
    if result.get("produced", 0) <= 0:
        raise ValueError(f"{path}: no {name} notifications")
    if result.get("produced") != result.get("consumed"):
        raise ValueError(f"{path}: produced/consumed mismatch")
    if result.get("handler_entries") != result.get("consumed"):
        raise ValueError(f"{path}: handler/consumed mismatch")
    for field in ZERO_FIELDS:
        if result.get(field, 0) != 0:
            raise ValueError(f"{path}: {field} must be zero")

    if source == "spi":
        timer = payload.get("spi_timer", {})
        if timer.get("gic_hwirq") != 321:
            raise ValueError(f"{path}: SPI hwirq must be 321")
        if timer.get("affinity_cpu") != 6 or timer.get("affinity_verified") is not True:
            raise ValueError(f"{path}: SPI affinity was not verified on CPU6")

    return {
        "produced": int(result["produced"]),
        "consumed": int(result["consumed"]),
        "transition_pairs": int(payload["transition_pairs"]),
        "busy_returns": int(counters.get("busy_returns", 0)),
        "again_returns": int(counters.get("again_returns", 0)),
        "enter_retries": int(counters.get("enter_retries", 0)),
        "exit_retries": int(counters.get("exit_retries", 0)),
    }


def curate_correctness(
    campaign: Path, source: str, output: Path, expected_runs: int
) -> dict:
    paths = sorted((campaign / "correctness" / source).glob("run_*/correctness-*.json"))
    if len(paths) != expected_runs:
        raise ValueError(
            f"{campaign.name}/{source}: expected {expected_runs} records, got {len(paths)}"
        )
    total_produced = 0
    total_pairs = 0
    busy_returns = again_returns = enter_retries = exit_retries = 0
    for number, path in enumerate(paths, 1):
        run_id = f"run-{number:03d}"
        stats = validate_correctness(path, source, run_id)
        total_produced += stats["produced"]
        total_pairs += stats["transition_pairs"]
        busy_returns += stats["busy_returns"]
        again_returns += stats["again_returns"]
        enter_retries += stats["enter_retries"]
        exit_retries += stats["exit_retries"]
        destination = output / "correctness" / source / f"run_{number:03d}" / "correctness.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return {
        "source": source,
        "notification": CORRECTNESS_SOURCES[source],
        "independent_runs": expected_runs,
        "transition_pairs": total_pairs,
        "produced": total_produced,
        "consumed": total_produced,
        "failed_runs": 0,
        "busy_returns": busy_returns,
        "again_returns": again_returns,
        "enter_retries": enter_retries,
        "exit_retries": exit_retries,
    }


def validate_latency(path: Path, expected_run: str, expected_pairs: int) -> dict:
    terminal: dict[tuple[int, str], tuple[int, int]] = {}
    identities: set[tuple[int, str, int]] = set()
    rows = retries = busy = again = other_errors = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not LATENCY_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing latency columns")
        for row in reader:
            rows += 1
            if int(row["schema_version"]) != 1 or row["run_id"] != expected_run:
                raise ValueError(f"{path}: schema or run_id mismatch")
            if row["condition"] != "idle" or row["direction"] not in DIRECTIONS:
                raise ValueError(f"{path}: condition or direction mismatch")
            iteration = int(row["iteration"])
            attempt = int(row["attempt"])
            rc = int(row["rc"])
            identity = (iteration, row["direction"], attempt)
            if identity in identities:
                raise ValueError(f"{path}: duplicate attempt identity")
            identities.add(identity)
            if iteration < 0 or iteration >= expected_pairs:
                raise ValueError(f"{path}: iteration outside expected range")
            if int(row["duration_ns"]) < 0 or int(row["request_elapsed_ns"]) < 0:
                raise ValueError(f"{path}: negative latency")
            terminal[(iteration, row["direction"])] = (attempt, rc)
            retries += int(attempt > 0)
            busy += int(rc == -16)
            again += int(rc == -11)
            other_errors += int(rc not in (0, -16, -11))

    expected_requests = expected_pairs * len(DIRECTIONS)
    if len(terminal) != expected_requests:
        raise ValueError(f"{path}: expected {expected_requests} requests, got {len(terminal)}")
    if any(rc != 0 for _, rc in terminal.values()):
        raise ValueError(f"{path}: non-successful terminal request")
    return {
        "rows": rows,
        "requests": expected_requests,
        "retries": retries,
        "busy_returns": busy,
        "again_returns": again,
        "other_errors": other_errors,
    }


def curate_latency(campaign: Path, output: Path, runs: int, pairs: int) -> dict:
    paths = sorted((campaign / "latency").glob("run_*/latency-*.csv"))
    if len(paths) != runs:
        raise ValueError(f"{campaign.name}: expected {runs} latency CSVs, got {len(paths)}")
    totals = {key: 0 for key in (
        "rows", "requests", "retries", "busy_returns", "again_returns", "other_errors"
    )}
    for number, path in enumerate(paths, 1):
        stats = validate_latency(path, f"run-{number:03d}", pairs)
        for key, value in stats.items():
            totals[key] += value
        destination = output / "latency" / "idle" / f"run_{number:03d}" / "transition_attempts.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return {"independent_runs": runs, "pairs_per_run": pairs, **totals}


def write_metadata(output: Path, manifest: dict, validation: dict) -> None:
    (output / "paper_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = """# RK3588 transition paper data

This directory contains the formal raw data used for the XHyPass transition
evaluation. The four correctness sources contain 100 independent runs with
10,000 DYN/RTO transition pairs per run. The latency dataset contains 30
independent idle runs with 100,000 pairs per run.

## Layout

- `correctness/{sgi,timer,event,spi}/run_NNN/correctness.json`
- `latency/idle/run_NNN/transition_attempts.csv`
- `latency/analysis/`: reproducible run-level and cross-run statistics
- `paper_data_manifest.json`: parameters and source campaign identifiers
- `validation_summary.json`: aggregate validation results

Validate correctness records from the repository root:

```bash
python experiments/transition/validate_correctness.py \\
  data/paper/RK3588/transition/correctness/*/run_*/correctness.json
```

Regenerate latency statistics:

```bash
python experiments/transition/analyze_transition.py \\
  --platform RK3588 \\
  --data-root data/paper/RK3588/transition/latency
```

The complete diagnostic campaigns and console logs remain under
`data/RK3588/transition/campaigns/`. This directory copies only the selected
paper-result CSV and JSON records.

The SGI, timer-PPI, and event-channel records come from a campaign whose
overall status is `failed` only because its separate SPI run 100 was not
started after a serial-control timeout. All 100 selected runs for each of the
three sources are present and independently validated. The formal SPI records
come exclusively from the completed SPI-only campaign.

Latency statistics use the caller-observed complete request latency stored in
`request_elapsed_ns`, including any retry time. `duration_ns` records an
individual ioctl attempt and is retained only as raw diagnostic data.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    campaigns = {
        "correctness": args.correctness_campaign.resolve(),
        "spi": args.spi_campaign.resolve(),
        "latency": args.latency_campaign.resolve(),
    }
    manifests = {
        "correctness": load_manifest(campaigns["correctness"], allow_failed=True),
        "spi": load_manifest(campaigns["spi"]),
        "latency": load_manifest(campaigns["latency"]),
    }
    if manifests["correctness"].get("runs") != 100 or \
       manifests["correctness"].get("correctness_iterations") != 10000:
        raise ValueError("SGI/timer/event campaign parameters are not formal")
    if manifests["spi"].get("runs") != 100 or \
       manifests["spi"].get("correctness_iterations") != 10000:
        raise ValueError("SPI campaign parameters are not formal")
    if manifests["latency"].get("runs") != 30 or \
       manifests["latency"].get("latency_iterations") != 100000:
        raise ValueError("latency campaign parameters are not formal")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transition-paper-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        correctness = []
        for source in ("sgi", "timer", "event"):
            correctness.append(
                curate_correctness(campaigns["correctness"], source, staged, 100)
            )
        correctness.append(curate_correctness(campaigns["spi"], "spi", staged, 100))
        latency = curate_latency(campaigns["latency"], staged, 30, 100000)

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": "RK3588",
            "scope": "Xen-dom0-Linux",
            "source_campaigns": {
                "sgi_timer_event": campaigns["correctness"].name,
                "spi": campaigns["spi"].name,
                "latency": campaigns["latency"].name,
            },
            "source_campaign_status": {
                name: payload["status"] for name, payload in manifests.items()
            },
            "correctness": {
                "runs_per_source": 100,
                "transition_pairs_per_run": 10000,
                "rto_cpu": 6,
                "producer_cpu": 0,
                "timer_delay_us": 50,
                "spi_delay_us": 50,
                "event_timeout_ms": 1000,
                "max_retries": manifests["correctness"].get("max_retries", 1000),
                "retry_delay_us": manifests["correctness"].get("retry_delay_us", 10),
                "sources": list(CORRECTNESS_SOURCES),
            },
            "latency": {
                "condition": "idle",
                "independent_runs": 30,
                "transition_pairs_per_run": 100000,
                "rto_cpu": 6,
                "max_retries": manifests["latency"].get("max_retries", 1000),
                "retry_delay_us": manifests["latency"].get("retry_delay_us", 10),
                "dwell_us": manifests["latency"].get("dwell_us", 0),
                "latency_field": "request_elapsed_ns",
                "latency_definition": "caller-observed complete request latency",
            },
        }
        validation = {"correctness": correctness, "latency": latency}
        write_metadata(staged, manifest, validation)
        staged.replace(output)
    print(f"Curated paper data: {output}")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
