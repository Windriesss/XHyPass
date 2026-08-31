#!/usr/bin/env python3
"""Edit data sources below, then run without command-line arguments."""

from pathlib import Path

from xhypass_plot.parser import scan_runs
from xhypass_plot.report import AnalysisConfig, analyze


EXPERIMENTS = ("cyclictest", "cyclictest-stress")
ENVIRONMENTS = ("bare", "jailhouse", "xen_credit2", "xen_credit2_WFX", "xen_null", "xen_null_WFX", "XHyPass")
DURATION_SECONDS = 600
EXPECTED_RUNS_PER_CONDITION = 5
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

# Keep each board independent: their CPU topology and formal interval matrix
# differ, and results from different boards must never be merged into one box.
PLATFORM_PROFILES = {
    "RK3588": {
        "data_root": Path(__file__).resolve().parents[1] / "data" / "RK3588",
        "intervals_us": (1_000, 10_000),
    },
    "E2000Q": {
        "data_root": Path(__file__).resolve().parents[1] / "data" / "E2000Q",
        "intervals_us": (1_000,),
    },
}


def build_platform_config(platform: str) -> AnalysisConfig:
    profile = PLATFORM_PROFILES[platform]
    return AnalysisConfig(
        data_sources={platform: profile["data_root"]},
        output_root=OUTPUT_ROOT,
        environments=ENVIRONMENTS,
        experiments=EXPERIMENTS,
        duration_seconds=DURATION_SECONDS,
        intervals_us=profile["intervals_us"],
        expected_runs_per_condition=EXPECTED_RUNS_PER_CONDITION,
    )


def has_matching_results(platform: str, config: AnalysisConfig) -> bool:
    root = config.data_sources[platform]
    return any(
        record.environment in config.environments
        and record.experiment in config.experiments
        and record.duration_seconds == config.duration_seconds
        and record.interval_us in config.intervals_us
        for record in scan_runs(platform, root)
    )


def main() -> int:
    generated = []
    for platform in PLATFORM_PROFILES:
        config = build_platform_config(platform)
        if not has_matching_results(platform, config):
            print(
                f"[SKIP] {platform}: no completed {DURATION_SECONDS}s formal "
                "results match the configured intervals"
            )
            continue
        try:
            result = analyze(config)
        except RuntimeError as error:
            print(f"[SKIP] {platform}: {error}")
            continue
        generated.append(result)
        print(f"[{platform}] Analysis written to: {result}")
    if not generated:
        print("No platform has matching completed formal results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
