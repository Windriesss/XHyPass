#!/usr/bin/env python3
"""Run 7 E2000Q environments x 2 experiments as 10-second smoke tests."""

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

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
DURATION_SECONDS = 10
INTERVAL_US = 1_000
RUNS_PER_CONDITION = 1
REBOOT_POLICY = "each-run"
CONTINUE_ON_FAILURE = True
SKIP_COMPLETED = True


def _completed_count(run: dict) -> int:
    series = (
        DATA_ROOT
        / safe_name(run["experiment_name"])
        / safe_name(run["environment_name"])
        / condition_name(run)
    )
    return sum(1 for path in series.glob("hist_run*.txt") if path.is_file())


def main() -> int:
    failures = 0
    total = len(ENVIRONMENTS) * len(EXPERIMENTS)
    index = 0
    for environment in ENVIRONMENTS:
        for experiment in EXPERIMENTS:
            index += 1
            print("=" * 78)
            print(
                f"SMOKE {index}/{total}: {environment} / {experiment} / "
                f"{DURATION_SECONDS}s"
            )
            print("=" * 78)
            run = build_run_config(
                experiment,
                environment,
                duration_seconds=DURATION_SECONDS,
                interval_us=INTERVAL_US,
            )
            if SKIP_COMPLETED and _completed_count(run) >= RUNS_PER_CONDITION:
                print("[SMOKE/SKIP] A successful result already exists")
                continue
            try:
                ExperimentRunner(run, DATA_ROOT, dry_run=DRY_RUN).run(
                    RUNS_PER_CONDITION, REBOOT_POLICY
                )
            except (Exception, KeyboardInterrupt) as exc:
                failures += 1
                print(f"[SMOKE/FAILED] {environment} / {experiment}: {exc}")
                if not CONTINUE_ON_FAILURE or isinstance(exc, KeyboardInterrupt):
                    raise
    print(f"[SMOKE/FINISHED] {total - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
