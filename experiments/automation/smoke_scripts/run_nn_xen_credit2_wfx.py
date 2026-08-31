#!/usr/bin/env python3
"""Run one 30-second heavy NN experiment on xen_credit2_WFX."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_nn_xen_credit2 as base
from xhypass_lab.nn_runner import NNExperimentRunner


# Edit these values directly when changing the smoke/formal campaign.
RUNS = 1
REBOOT_POLICY = "each-run"
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM
DRY_RUN = False
DURATION_SECONDS = 30
LOAD_PROFILES = (
    {
        "name": "heavy",
        "inception_qps": 0.2,
        "peak_qps": 20,
        "poisson_qps": 9,
        "constant_qps": 9,
    },
)


def build_run_config() -> dict:
    # xen_credit2_WFX inherits the complete Xen-credit2 workflow and only
    # replaces local_boot_files.source_dir with Xen_credit2_nativeWFX.
    run = base.build_run_config("xen_credit2_WFX")
    run["experiment"]["profile_name"] = "dual-tflite-heavy-only"
    run["experiment"]["duration_seconds"] = DURATION_SECONDS
    run["experiment"]["profiles"] = [dict(profile) for profile in LOAD_PROFILES]
    return run


def main() -> int:
    outputs = NNExperimentRunner(
        build_run_config(), DATA_ROOT, dry_run=DRY_RUN
    ).run(RUNS, REBOOT_POLICY)
    print("NN Xen-credit2 native-WFX result directories:")
    for output in outputs:
        print(f"- {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
