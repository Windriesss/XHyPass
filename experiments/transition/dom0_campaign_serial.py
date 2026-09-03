#!/usr/bin/env python3
"""COM10-controlled, failure-diagnosable XHyPass dom0 campaign."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_ROOT = REPO_ROOT / "experiments" / "automation"
if str(AUTOMATION_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMATION_ROOT))

from xhypass_lab.config import load_config, resolved_run_config  # noqa: E402


GUEST_DIR = REPO_ROOT / "guest" / "linux" / "interrupt_passthrough"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "RK3588" / "transition"
FATAL_PATTERN = (
    r"Kernel panic|Unable to handle kernel|Internal error:|watchdog: BUG:|"
    r"detected stalls|SError Interrupt|Xen BUG at"
)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def shell_join(parts: list[str | int]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=AUTOMATION_ROOT / "config" / "RK3588" / "lab.json",
    )
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=1000)
    parser.add_argument("--correctness-iterations", type=positive_int, default=100)
    parser.add_argument("--correctness-sources", default="sgi,timer,event")
    parser.add_argument("--timer-delay-us", type=positive_int, default=50)
    parser.add_argument("--spi-delay-us", type=positive_int, default=50)
    parser.add_argument("--event-timeout-ms", type=positive_int, default=1000)
    parser.add_argument("--max-retries", type=nonnegative_int, default=1000)
    parser.add_argument("--retry-delay-us", type=nonnegative_int, default=10)
    parser.add_argument("--dwell-us", type=nonnegative_int, default=0)
    parser.add_argument("--spi-timer-path", default="/timer@feae0000")
    parser.add_argument("--spi-hwirq", type=positive_int, default=321)
    parser.add_argument("--rto-cpu", type=int, default=6)
    parser.add_argument("--producer-cpu", type=int, default=0)
    parser.add_argument("--event-sgi", type=int, default=7)
    parser.add_argument("--mode", choices=("latency", "correctness", "both"), default="both")
    parser.add_argument("--control", choices=("serial", "ssh"), default="serial")
    parser.add_argument("--remote-dir", default="/root/xhypass-transition")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--command-timeout", type=positive_int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_sources(text: str) -> list[str]:
    sources = [item.strip() for item in text.split(",") if item.strip()]
    if not sources or any(item not in {"sgi", "timer", "event", "spi"} for item in sources):
        raise SystemExit("--correctness-sources must contain sgi,timer,event,spi")
    if len(sources) != len(set(sources)):
        raise SystemExit("--correctness-sources contains a duplicate")
    return sources


def require_artifacts(mode: str) -> dict[str, Path]:
    artifacts = {
        "interrupt_passthrough.ko": GUEST_DIR / "interrupt_passthrough.ko",
        "xhypass_transition_bench": GUEST_DIR / "xhypass_transition_bench",
    }
    if mode in ("correctness", "both"):
        artifacts["xhypass_correctness"] = GUEST_DIR / "xhypass_correctness"
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("build guest artifacts first: " + ", ".join(missing))
    return artifacts


def record_event(path: Path, stage: str, status: str, **details: object) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "status": status, **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def ensure_ssh(session) -> None:
    if not session.is_active():
        session.reconnect()


def serial_login(serial, environment: dict) -> None:
    prompts = environment.get("shell_prompts", [r"root@[^\r\n]*#\s*$", r"#\s*$"])
    serial.sendline()
    matched, _ = serial.expect(
        [*prompts, environment.get("login_prompt", r"login:\s*$")], 30, clear=True
    )
    if matched < len(prompts):
        return
    serial.sendline(environment.get("username", "root"))
    matched, _ = serial.expect([*prompts, r"Password:\s*$"], 20, clear=True)
    if matched == len(prompts):
        serial.sendline(environment.get("password", ""))
        serial.expect(prompts, 20, clear=True)


def run_serial(serial, command: str, timeout: float, stage: str) -> bytes:
    marker = "XHYPASS_" + uuid.uuid4().hex.upper()
    wrapped = (
        f"echo {marker}_BEGIN; ( {command} ); "
        f"__xhypass_rc=$?; echo {marker}_END:$__xhypass_rc"
    )
    print(f"\n[SERIAL CMD/{stage}] {command}")
    serial.sendline(wrapped)
    matched, output = serial.expect(
        [rf"{marker}_END:[0-9]+", FATAL_PATTERN], timeout, clear=True
    )
    if matched == 1:
        serial.drain(10)
        raise RuntimeError(f"fatal console signature during {stage}")
    result = re.search(rb"%s_END:([0-9]+)" % marker.encode("ascii"), output)
    if not result:
        raise RuntimeError(f"missing completion marker during {stage}")
    rc = int(result.group(1))
    if rc:
        raise RuntimeError(f"serial command failed during {stage}: rc={rc}")
    return output


def snapshot(remote_root: PurePosixPath, name: str) -> tuple[str, list[str]]:
    suffixes = ("dmesg", "xen-dmesg", "interrupts", "modules", "vcpus")
    paths = [remote_root / f"{name}-{suffix}.log" for suffix in suffixes]
    command = (
        f"dmesg > {shlex.quote(str(paths[0]))}; "
        f"xl dmesg > {shlex.quote(str(paths[1]))} 2>&1; "
        f"cat /proc/interrupts > {shlex.quote(str(paths[2]))}; "
        f"cat /proc/modules > {shlex.quote(str(paths[3]))}; "
        f"xl vcpu-list 0 > {shlex.quote(str(paths[4]))} 2>&1"
    )
    return command, [str(path) for path in paths]


def fetch(session, remote_files: list[str], local_dir: Path) -> None:
    ensure_ssh(session)
    for remote in remote_files:
        session.get(remote, local_dir / PurePosixPath(remote).name)


def main() -> int:
    args = parse_args()
    sources = parse_sources(args.correctness_sources)
    if args.rto_cpu < 0 or args.producer_cpu < 0:
        raise SystemExit("CPU numbers must be non-negative")
    if args.rto_cpu == args.producer_cpu:
        raise SystemExit("producer CPU must differ from the RTO CPU")
    if args.event_sgi != 7:
        raise SystemExit("the supplied Linux dom0 patch reserves SGI 7")

    artifacts = require_artifacts(args.mode)
    config = load_config(args.config.resolve())
    run_config = resolved_run_config(config, "XHyPass", "cyclictest", {})
    environment = run_config["environment"]
    platform = run_config["platform_name"]
    remote_root = PurePosixPath(args.remote_dir)
    campaign_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root or DEFAULT_DATA_ROOT / "campaigns" / campaign_id
    ssh_log = output_root / "ssh.log"
    serial_log = output_root / "com10.log"
    event_log = output_root / "events.jsonl"
    manifest = {
        "schema_version": 2, "scope": "Xen-dom0-Linux",
        "platform": platform, "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs, "latency_iterations": args.iterations,
        "correctness_iterations": args.correctness_iterations,
        "correctness_sources": sources, "rto_cpu": args.rto_cpu,
        "producer_cpu": args.producer_cpu, "event_sgi": args.event_sgi,
        "timer_delay_us": args.timer_delay_us,
        "spi_delay_us": args.spi_delay_us,
        "event_timeout_ms": args.event_timeout_ms,
        "max_retries": args.max_retries,
        "retry_delay_us": args.retry_delay_us,
        "dwell_us": args.dwell_us,
        "spi_timer_path": args.spi_timer_path, "spi_hwirq": args.spi_hwirq,
        "mode": args.mode, "control": args.control,
        "command_timeout_seconds": args.command_timeout,
        "output_root": str(output_root),
        "status": "planned" if args.dry_run else "running",
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    from xhypass_lab.serial_session import SerialSession
    from xhypass_lab.ssh_session import SSHSession

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    session = SSHSession(environment["ssh"])
    serial_context = None
    module_loaded = False
    failed = False
    session.connect()
    try:
        session.run(shell_join(["mkdir", "-p", str(remote_root)]), timeout=30,
                    log_path=ssh_log)
        for name, local in artifacts.items():
            session.put(local, str(remote_root / name))
        session.run(
            shell_join(["chmod", "0700", *[str(remote_root / name) for name in artifacts]]),
            timeout=30, log_path=ssh_log,
        )

        if args.control == "serial":
            serial_context = SerialSession(config["serial"], serial_log)
            control = serial_context.__enter__()
            serial_login(control, environment)

            def run_control(command: str, timeout: float, stage: str) -> bytes:
                return run_serial(control, command, timeout, stage)
        else:
            def run_control(command: str, timeout: float, stage: str) -> bytes:
                ensure_ssh(session)
                _, out, err = session.run(command, timeout=timeout, log_path=ssh_log)
                return out + err

        preflight = (
            "set -eu; test \"$(uname -m)\" = aarch64; test -d /proc/xen; "
            f"test -d /sys/devices/system/cpu/cpu{args.rto_cpu}; "
            f"test -d /sys/devices/system/cpu/cpu{args.producer_cpu}; "
            f"xl vcpu-pin 0 {args.rto_cpu} {args.rto_cpu}; dmesg -n 8; "
            "xl vcpu-list 0; uname -a; cat /proc/cmdline"
        )
        run_control(preflight, 60, "preflight")
        record_event(event_log, "preflight", "completed")

        load_module = (
            "if grep -q '^interrupt_passthrough ' /proc/modules; then "
            f"taskset -c {args.rto_cpu} rmmod interrupt_passthrough; fi; "
            f"taskset -c {args.rto_cpu} insmod "
            f"{shlex.quote(str(remote_root / 'interrupt_passthrough.ko'))} "
            f"auto_enter=0 rto_cpu={args.rto_cpu} event_sgi={args.event_sgi} "
            f"spi_timer_path={shlex.quote(args.spi_timer_path)} "
            f"spi_expected_hwirq={args.spi_hwirq}"
        )
        run_control(load_module, 60, "module-load")
        module_loaded = True

        probe_csv = remote_root / "console-log-probe.csv"
        probe_command = shell_join([
            "taskset", "-c", args.rto_cpu,
            str(remote_root / "xhypass_transition_bench"),
            "--iterations", 2, "--run-id", "console-probe",
            "--condition", "console-probe", "--output", str(probe_csv),
        ])
        probe_output = run_control(probe_command, 30, "console-log-probe")
        if re.search(rb"\(XEN\)\s+(?:enter|exit)_rto\s+rc=", probe_output):
            raise RuntimeError(
                "deployed Xen logs every RTO transition; rebuild/deploy the "
                "repository Xen before measuring latency"
            )

        for run_number in range(1, args.runs + 1):
            run_id = f"run-{run_number:03d}"
            stages: list[tuple[str, Path, str, str]] = []
            if args.mode in ("latency", "both"):
                remote_csv = remote_root / f"latency-{run_id}.csv"
                command = shell_join([
                    "taskset", "-c", args.rto_cpu,
                    str(remote_root / "xhypass_transition_bench"),
                    "--iterations", args.iterations, "--run-id", run_id,
                    "--condition", "idle", "--output", str(remote_csv),
                ])
                stages.append((f"latency-{run_id}", output_root / "latency" /
                               f"run_{run_number:03d}", command, str(remote_csv)))
            if args.mode in ("correctness", "both"):
                for source in sources:
                    remote_json = remote_root / f"correctness-{source}-{run_id}.json"
                    command = shell_join([
                        "taskset", "-c", args.rto_cpu,
                        str(remote_root / "xhypass_correctness"),
                        "--iterations", args.correctness_iterations,
                        "--producer-cpu", args.producer_cpu, "--source", source,
                        "--timer-delay-us", args.timer_delay_us,
                        "--spi-delay-us", args.spi_delay_us,
                        "--event-timeout-ms", args.event_timeout_ms,
                        "--max-retries", args.max_retries,
                        "--retry-delay-us", args.retry_delay_us,
                        "--dwell-us", args.dwell_us,
                        "--platform", platform, "--run-id", run_id,
                        "--condition", source + "-stress", "--output", str(remote_json),
                    ])
                    stages.append((f"correctness-{source}-{run_id}",
                                   output_root / "correctness" / source /
                                   f"run_{run_number:03d}", command, str(remote_json)))

            for stage, local_dir, command, result_file in stages:
                before, before_files = snapshot(remote_root, stage + "-before")
                run_control(before, 60, stage + "-snapshot-before")
                fetch(session, before_files, local_dir)
                record_event(event_log, stage, "started")
                stage_error: BaseException | None = None
                try:
                    run_control(command, args.command_timeout, stage)
                except BaseException as exc:
                    stage_error = exc
                finally:
                    # A failed correctness command normally still writes its
                    # JSON counters.  Preserve that file and the post-failure
                    # system state before propagating the original error.
                    try:
                        fetch(session, [result_file], local_dir)
                    except Exception as exc:
                        record_event(
                            event_log, stage + "-result-fetch", "failed",
                            error=repr(exc),
                        )
                    try:
                        after, after_files = snapshot(remote_root, stage + "-after")
                        run_control(after, 60, stage + "-snapshot-after")
                        fetch(session, after_files, local_dir)
                    except Exception as exc:
                        record_event(
                            event_log, stage + "-snapshot-after", "failed",
                            error=repr(exc),
                        )
                if stage_error is not None:
                    record_event(event_log, stage, "failed", error=repr(stage_error))
                    raise stage_error
                record_event(event_log, stage, "completed")
    except BaseException as exc:
        failed = True
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        record_event(event_log, "campaign", "failed", error=repr(exc))
        raise
    finally:
        if module_loaded and not failed:
            try:
                run_control(f"taskset -c {args.rto_cpu} rmmod interrupt_passthrough",
                            60, "module-unload")
            except Exception as exc:
                record_event(event_log, "module-unload", "failed", error=repr(exc))
        if serial_context is not None:
            serial_context.__exit__(None, None, None)
        session.close()
        if not failed:
            manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0
