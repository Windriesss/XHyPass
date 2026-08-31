#!/usr/bin/env python3
"""Bare-metal dual-TFLite NN experiment; edit settings here and run directly."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from xhypass_lab.config import load_config, resolved_run_config
from xhypass_lab.nn_runner import NNExperimentRunner


# Serial and run control.
SERIAL_PORT = "COM10"
SERIAL_BAUDRATE = 115_200
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False

# Each run executes light, medium, and heavy sequentially. Therefore a 600 s
# duration means about 3 x 600 s of workload time, plus boot/transfer overhead.
DURATION_SECONDS = 30
SEED = 12_345
WORKERS = 8
INCEPTION_THREADS = 6
MNASNET_THREADS = 6
WORKLOAD_CPUS = "0-5"
CGROUP_WEIGHT = 100
CGROUP_SHARES = 1_024
CYCLICTEST_INTERVAL_US = 1_000
CYCLICTEST_PRIORITY = 99
HISTOGRAM_LIMIT_US = 10_000
PERIOD_SECONDS = 60
GRACE_SECONDS = 300
CLEANUP_REMOTE = True

# Parameters map directly to loadgen.cpp:
# constant-qps, qps (Poisson base), peak-qps, and period-sec.
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


def build_run_config() -> dict:
    config = load_config(PROJECT_ROOT / "config/RK3588/lab.json")
    config["serial"]["port"] = SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    run = resolved_run_config(config, "bare", "cyclictest", {})
    run["experiment_name"] = "NN"
    run["experiment"] = {
        "profile_name": "dual-tflite-v2-cgroup",
        "remote_root": "~/NN",
        "local_run_script": str(
            PROJECT_ROOT.parent
            / "workloads"
            / "NN"
            / "bare"
            / "run_all.sh"
        ),
        "duration_seconds": DURATION_SECONDS,
        "seed": SEED,
        "workers": WORKERS,
        "inception_threads": INCEPTION_THREADS,
        "mnasnet_threads": MNASNET_THREADS,
        "workload_cpus": WORKLOAD_CPUS,
        "cgroup_weight": CGROUP_WEIGHT,
        "cgroup_shares": CGROUP_SHARES,
        "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
        "cyclictest_priority": CYCLICTEST_PRIORITY,
        "histogram_limit_us": HISTOGRAM_LIMIT_US,
        "period_seconds": PERIOD_SECONDS,
        "profiles": [dict(profile) for profile in LOAD_PROFILES],
        "grace_seconds": GRACE_SECONDS,
        "cleanup_remote": CLEANUP_REMOTE,
        "transfer_chunk_bytes": 262_144,
    }
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
