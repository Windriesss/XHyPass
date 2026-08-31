#!/usr/bin/env python3
"""Xen-credit2 dual-domain NN experiment; edit settings here and run directly."""

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

# light/medium/heavy run sequentially, while dom0 and dom1 run concurrently.
PROFILE_NAME = "dual-tflite-heavy-only"
DURATION_SECONDS = 30
SEED = 12_345
WORKERS = 8
DOM0_MODEL_THREADS = 6
DOM1_MODEL_THREADS = 3
DOM0_CYCLICTEST_CPU = 6
DOM1_CYCLICTEST_CPU = 6
DOM0_WORKLOAD_CPUS = "0-5"
DOM1_WORKLOAD_CPUS = "0-5"
CYCLICTEST_INTERVAL_US = 1_000
CYCLICTEST_PRIORITY = 99
HISTOGRAM_LIMIT_US = 10_000
PERIOD_SECONDS = 60
START_DELAY_SECONDS = 2
DOM0_LAUNCH_DELAY_SECONDS = 0
DOM1_LAUNCH_DELAY_SECONDS = 0
WARMUP_SECONDS = 0
WARMUP_QPS = 1
DOMAIN_SYNC_TIMEOUT_SECONDS = 180
GRACE_SECONDS = 300
CLEANUP_REMOTE = True

# Private dom0/dom1 bridge used only when dom1 is missing NN dependencies.
DOM0_BRIDGE_INTERFACE = "xenbr0"
DOM0_BRIDGE_IP = "202.197.67.50"
DOM1_INTERFACE = "enX0"
DOM1_IP = "202.197.67.51"
DOM1_NETMASK = "255.255.255.0"
DOM1_NETWORK_SETTLE_SECONDS = 5
# During the workload, probe dom1 SSH periodically.  Two consecutive failures
# cause enX0 to be repaired through the COM10 dom1 console.
DOM1_NETWORK_MONITOR_INTERVAL_SECONDS = 10
DOM1_NETWORK_MONITOR_FAILURES = 2
DOM1_NETWORK_PROBE_TIMEOUT_SECONDS = 2
XEN_BOOT_HEALTH_ATTEMPTS = 3
XEN_CONTROL_READY_TIMEOUT_SECONDS = 20
DOM1_START_ATTEMPTS = 3
DOM1_CONSOLE_READY_TIMEOUT_SECONDS = 60

LOAD_PROFILES = (
    {
        "name": "heavy",
        "inception_qps": 0.2,
        "peak_qps": 20,
        "poisson_qps": 9,
        "constant_qps": 9,
    },
)


def build_run_config(environment_name: str = "xen_credit2") -> dict:
    project_root = PROJECT_ROOT
    config = load_config(project_root / "config" / "RK3588" / "lab.json")
    config["serial"]["port"] = SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    run = resolved_run_config(config, environment_name, "cyclictest", {})
    run["experiment_name"] = "NN"
    run["environment"]["xen_boot_health_attempts"] = XEN_BOOT_HEALTH_ATTEMPTS
    run["environment"]["xen_control_ready_timeout_seconds"] = (
        XEN_CONTROL_READY_TIMEOUT_SECONDS
    )
    run["environment"]["dom1_start_attempts"] = DOM1_START_ATTEMPTS
    run["environment"]["dom1_console_ready_timeout_seconds"] = (
        DOM1_CONSOLE_READY_TIMEOUT_SECONDS
    )
    run["experiment"] = {
        "profile_name": PROFILE_NAME,
        "duration_seconds": DURATION_SECONDS,
        "seed": SEED,
        "workers": WORKERS,
        # Common assignment names retained for loadgen parameter generation.
        "inception_threads": DOM0_MODEL_THREADS,
        "mnasnet_threads": DOM1_MODEL_THREADS,
        "dom0_model_threads": DOM0_MODEL_THREADS,
        "dom1_model_threads": DOM1_MODEL_THREADS,
        "dom0_cyclictest_cpu": DOM0_CYCLICTEST_CPU,
        "dom1_cyclictest_cpu": DOM1_CYCLICTEST_CPU,
        "dom0_workload_cpus": DOM0_WORKLOAD_CPUS,
        "dom1_workload_cpus": DOM1_WORKLOAD_CPUS,
        "cyclictest_interval_us": CYCLICTEST_INTERVAL_US,
        "cyclictest_priority": CYCLICTEST_PRIORITY,
        "histogram_limit_us": HISTOGRAM_LIMIT_US,
        "period_seconds": PERIOD_SECONDS,
        "start_delay_seconds": START_DELAY_SECONDS,
        "dom0_launch_delay_seconds": DOM0_LAUNCH_DELAY_SECONDS,
        "dom1_launch_delay_seconds": DOM1_LAUNCH_DELAY_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "warmup_qps": WARMUP_QPS,
        "domain_sync_timeout_seconds": DOMAIN_SYNC_TIMEOUT_SECONDS,
        "profiles": [dict(profile) for profile in LOAD_PROFILES],
        "remote_xen_dir": "~/NN/xen_credit2",
        "dom0_bridge_interface": DOM0_BRIDGE_INTERFACE,
        "dom0_bridge_ip": DOM0_BRIDGE_IP,
        "dom1_interface": DOM1_INTERFACE,
        "dom1_ip": DOM1_IP,
        "dom1_netmask": DOM1_NETMASK,
        "dom1_network_settle_seconds": DOM1_NETWORK_SETTLE_SECONDS,
        "dom1_network_monitor_interval_seconds": (
            DOM1_NETWORK_MONITOR_INTERVAL_SECONDS
        ),
        "dom1_network_monitor_failures": DOM1_NETWORK_MONITOR_FAILURES,
        "dom1_network_probe_timeout_seconds": (
            DOM1_NETWORK_PROBE_TIMEOUT_SECONDS
        ),
        "local_dom0_script": str(
            project_root.parent
            / "workloads"
            / "NN"
            / "xen_credit2"
            / "run_vm1.sh"
        ),
        "local_dom1_script": str(
            project_root.parent
            / "workloads"
            / "NN"
            / "xen_credit2"
            / "run_vm2.sh"
        ),
        "grace_seconds": GRACE_SECONDS,
        "cleanup_remote": CLEANUP_REMOTE,
        "transfer_chunk_bytes": 262_144,
    }
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN Xen-credit2 result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
