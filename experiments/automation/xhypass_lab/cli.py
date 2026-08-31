from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config, resolved_run_config
from .runner import ExperimentRunner


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RK3588/XHyPass experiment controller")
    project_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "RK3588" / "lab.json",
    )
    sub = p.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="boot an environment and execute experiments")
    run.add_argument("environment", help="bare, jailhouse, xen, or xhypass")
    run.add_argument("experiment", help="experiment plugin, currently cyclictest")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--duration", type=int, dest="duration_seconds")
    run.add_argument("--cpu", type=int)
    run.add_argument("--interval-us", type=int)
    run.add_argument("--priority", type=int)
    run.add_argument("--histogram-limit-us", type=int)
    run.add_argument("--reboot", choices=("once", "each-run"), default="each-run")
    run.add_argument("--data-root", type=Path, default=project_root / "data")
    run.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.runs < 1:
        parser().error("--runs must be at least 1")
    try:
        cfg = load_config(args.config)
        overrides = {
            "duration_seconds": args.duration_seconds,
            "cpu": args.cpu,
            "interval_us": args.interval_us,
            "priority": args.priority,
            "histogram_limit_us": args.histogram_limit_us,
        }
        resolved = resolved_run_config(cfg, args.environment, args.experiment, overrides)
        outputs = ExperimentRunner(resolved, args.data_root, args.dry_run).run(args.runs, args.reboot)
    except (ConfigError, OSError, TimeoutError, RuntimeError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("Completed:")
        for path in outputs:
            print(path.resolve())
    return 0
