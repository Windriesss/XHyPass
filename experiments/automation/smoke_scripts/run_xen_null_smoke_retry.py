#!/usr/bin/env python3
"""Retry the four xen_null smoke cases after framework fixes."""

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from smoke_scripts import run_xen_variants_smoke as smoke


ENVIRONMENTS = ("xen_null", "xen_null_WFX")
EXPERIMENTS = ("cyclictest", "cyclictest-stress")
CAMPAIGN = tuple(
    (environment, experiment)
    for environment in ENVIRONMENTS
    for experiment in EXPERIMENTS
)


def main() -> int:
    original_campaign = smoke.CAMPAIGN
    try:
        smoke.CAMPAIGN = CAMPAIGN
        return smoke.main()
    finally:
        smoke.CAMPAIGN = original_campaign


if __name__ == "__main__":
    raise SystemExit(main())
