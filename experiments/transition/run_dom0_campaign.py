#!/usr/bin/env python3
"""Run the XHyPass transition experiment entirely inside Xen dom0 Linux."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_ROOT = REPO_ROOT / "experiments" / "automation"
if str(AUTOMATION_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMATION_ROOT))

from xhypass_lab.config import load_config, resolved_run_config  # noqa: E402
from dom0_campaign_serial import main as diagnostic_main  # noqa: E402
GUEST_DIR = REPO_ROOT / "guest" / "linux" / "interrupt_passthrough"


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def command_text(parts: list[str | int]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=AUTOMATION_ROOT / "config" / "RK3588" / "lab.json",
    )
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=10000)
    parser.add_argument("--correctness-iterations", type=positive_int, default=10000)
    parser.add_argument("--rto-cpu", type=int, default=6)
    parser.add_argument("--producer-cpu", type=int, default=0)
    parser.add_argument("--event-sgi", type=int, default=7)
    parser.add_argument("--mode", choices=("latency", "correctness", "both"), default="both")
    parser.add_argument("--remote-dir", default="/root/xhypass-transition")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "RK3588" / "transition",
    )
    parser.add_argument("--command-timeout", type=positive_int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_artifacts(mode: str) -> dict[str, Path]:
    artifacts = {"interrupt_passthrough.ko": GUEST_DIR / "interrupt_passthrough.ko"}
    if mode in ("latency", "both"):
        artifacts["xhypass_transition_bench"] = GUEST_DIR / "xhypass_transition_bench"
    if mode in ("correctness", "both"):
        artifacts["xhypass_correctness"] = GUEST_DIR / "xhypass_correctness"
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("build guest artifacts first: " + ", ".join(missing))
    return artifacts


def main() -> int:
    """Use the COM10-first, staged diagnostic campaign implementation."""
    return diagnostic_main()


def legacy_main() -> int:
    args = parse_args()
    if args.rto_cpu < 0 or args.producer_cpu < 0:
        raise SystemExit("CPU numbers must be non-negative")
    if args.rto_cpu == args.producer_cpu:
        raise SystemExit("producer CPU must differ from the RTO CPU")
    if args.event_sgi != 7:
        raise SystemExit("the supplied Linux dom0 patch reserves SGI 7")

    artifacts = require_artifacts(args.mode)
    config = load_config(args.config.resolve())
    run_config = resolved_run_config(config, "XHyPass", "cyclictest", {})
    ssh_settings = run_config["environment"]["ssh"]
    platform = run_config["platform_name"]
    remote_root = PurePosixPath(args.remote_dir)
    campaign_log = args.output_root / "campaign.log"

    manifest = {
        "schema_version": 1,
        "scope": "Xen-dom0-Linux",
        "platform": platform,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs,
        "latency_iterations": args.iterations,
        "correctness_iterations": args.correctness_iterations,
        "rto_cpu": args.rto_cpu,
        "producer_cpu": args.producer_cpu,
        "event_sgi": args.event_sgi,
        "mode": args.mode,
    }
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        return 0

    from xhypass_lab.ssh_session import SSHSession

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    session = SSHSession(ssh_settings)
    session.connect()
    module_loaded = False
    try:
        session.run(
            command_text(["mkdir", "-p", str(remote_root)]),
            timeout=30,
            log_path=campaign_log,
        )
        for name, local in artifacts.items():
            session.put(local, str(remote_root / name))
        session.run(
            command_text(["chmod", "0700", *[str(remote_root / name) for name in artifacts]]),
            timeout=30,
            log_path=campaign_log,
        )
        preflight = (
            "set -eu; test \"$(uname -m)\" = aarch64; "
            "test -d /proc/xen; "
            f"test -d /sys/devices/system/cpu/cpu{args.rto_cpu}; "
            f"test -d /sys/devices/system/cpu/cpu{args.producer_cpu}; "
            f"xl vcpu-pin 0 {args.rto_cpu} {args.rto_cpu}; "
            "xl vcpu-list 0; uname -a; cat /proc/cmdline"
        )
        session.run(preflight, timeout=60, log_path=campaign_log)
        session.run(
            f"if grep -q '^interrupt_passthrough ' /proc/modules; then "
            f"taskset -c {args.rto_cpu} rmmod interrupt_passthrough; fi; "
            f"taskset -c {args.rto_cpu} insmod "
            f"{shlex.quote(str(remote_root / 'interrupt_passthrough.ko'))} "
            f"auto_enter=0 rto_cpu={args.rto_cpu} event_sgi={args.event_sgi}",
            timeout=60,
            log_path=campaign_log,
        )
        module_loaded = True

        for run_number in range(1, args.runs + 1):
            run_id = f"run-{run_number:03d}"
            if args.mode in ("latency", "both"):
                local_dir = args.output_root / "idle" / f"run_{run_number:03d}"
                remote_csv = remote_root / f"{run_id}-transition_attempts.csv"
                command = command_text(
                    [
                        "taskset", "-c", args.rto_cpu,
                        str(remote_root / "xhypass_transition_bench"),
                        "--iterations", args.iterations,
                        "--run-id", run_id,
                        "--condition", "idle",
                        "--output", str(remote_csv),
                    ]
                )
                session.run(command, timeout=args.command_timeout, log_path=campaign_log)
                session.get(str(remote_csv), local_dir / "transition_attempts.csv")

            if args.mode in ("correctness", "both"):
                local_dir = args.output_root / "correctness" / f"run_{run_number:03d}"
                remote_json = remote_root / f"{run_id}-correctness.json"
                command = command_text(
                    [
                        "taskset", "-c", args.rto_cpu,
                        str(remote_root / "xhypass_correctness"),
                        "--iterations", args.correctness_iterations,
                        "--producer-cpu", args.producer_cpu,
                        "--platform", platform,
                        "--run-id", run_id,
                        "--condition", "notification-stress",
                        "--output", str(remote_json),
                    ]
                )
                session.run(command, timeout=args.command_timeout, log_path=campaign_log)
                session.get(str(remote_json), local_dir / "correctness.json")

        session.run(
            "dmesg | tail -n 400",
            timeout=30,
            log_path=args.output_root / "dom0_dmesg.log",
        )
    finally:
        if module_loaded:
            session.run(
                f"taskset -c {args.rto_cpu} rmmod interrupt_passthrough",
                timeout=60,
                log_path=campaign_log,
                check=False,
            )
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
