#!/usr/bin/env python3
"""Jailhouse dual-cell NN experiment; edit settings here and run directly."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from xhypass_lab.config import load_config, resolved_run_config
from xhypass_lab.nn_runner import NNExperimentRunner


SERIAL_PORT = "COM10"
SERIAL_BAUDRATE = 115_200
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False

# Three sequential levels run on both cells concurrently. A 600 s duration is
# therefore about 1800 s of workload time per round, not 3600 s.
DURATION_SECONDS = 30
SEED = 12_345
WORKERS = 8
ROOTCELL_MODEL_THREADS = 3
NONROOT_MODEL_THREADS = 3
ROOTCELL_CYCLICTEST_CPU = 6
NONROOT_CYCLICTEST_CPU = 3
CYCLICTEST_INTERVAL_US = 1_000
CYCLICTEST_PRIORITY = 99
HISTOGRAM_LIMIT_US = 10_000
PERIOD_SECONDS = 60
START_DELAY_SECONDS = 2
GRACE_SECONDS = 300
CLEANUP_REMOTE = True

# Root/non-root private link used for copying the ramfs-side experiment files.
ROOTCELL_INTERNAL_INTERFACE = "enP1p0s3"
ROOTCELL_INTERNAL_IP = "202.197.10.50"
NONROOT_INTERNAL_INTERFACE = "eth0"
NONROOT_INTERNAL_IP = "202.197.10.51"
ROOTCELL_INTERNAL_PASSWORD = ""
SCP_RETRIES = 5
SCP_TIMEOUT_SECONDS = 600

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
    project_root = PROJECT_ROOT
    config = load_config(project_root / "config" / "RK3588" / "lab.json")
    config["serial"]["port"] = SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    run = resolved_run_config(config, "jailhouse", "cyclictest", {})
    run["experiment_name"] = "NN"
    run["experiment"] = {
        "profile_name": "dual-tflite-v1",
        "duration_seconds": DURATION_SECONDS,
        "seed": SEED,
        "workers": WORKERS,
        # Kept for the common loadgen assignment builder.
        "inception_threads": ROOTCELL_MODEL_THREADS,
        "mnasnet_threads": NONROOT_MODEL_THREADS,
        "rootcell_model_threads": ROOTCELL_MODEL_THREADS,
        "nonroot_model_threads": NONROOT_MODEL_THREADS,
        "rootcell_cyclictest_cpu": ROOTCELL_CYCLICTEST_CPU,
        "nonroot_cyclictest_cpu": NONROOT_CYCLICTEST_CPU,
        "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
        "cyclictest_priority": CYCLICTEST_PRIORITY,
        "histogram_limit_us": HISTOGRAM_LIMIT_US,
        "period_seconds": PERIOD_SECONDS,
        "start_delay_seconds": START_DELAY_SECONDS,
        "profiles": [dict(profile) for profile in LOAD_PROFILES],
        "local_rootcell_script": str(
            project_root.parent / "workloads" / "NN" / "jailhouse" / "run_vm1.sh"
        ),
        "local_nonroot_script": str(
            project_root.parent / "workloads" / "NN" / "jailhouse" / "run_vm2.sh"
        ),
        "jailhouse_linux_command": (
            "cd ~ && jailhouse cell linux ./tl3588-linux-demo.cell ./Image "
            "-d /usr/share/jailhouse/cells/dts/inmate-tl3588.dtb "
            "-i new_phytium_rootfs.cpio.gz "
            "-c \"isolcpus=3 rcu_nocb=3\""
        ),
        "rootcell_internal_interface": ROOTCELL_INTERNAL_INTERFACE,
        "rootcell_internal_ip": ROOTCELL_INTERNAL_IP,
        "nonroot_internal_interface": NONROOT_INTERNAL_INTERFACE,
        "nonroot_internal_ip": NONROOT_INTERNAL_IP,
        "rootcell_internal_password": ROOTCELL_INTERNAL_PASSWORD,
        "scp_retries": SCP_RETRIES,
        "scp_timeout_seconds": SCP_TIMEOUT_SECONDS,
        "grace_seconds": GRACE_SECONDS,
        "cleanup_remote": CLEANUP_REMOTE,
        "transfer_chunk_bytes": 262_144,
    }
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN Jailhouse result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
