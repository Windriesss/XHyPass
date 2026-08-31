from __future__ import annotations

import base64
import copy
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .serial_session import SerialSession, SerialTimeout
from .ssh_session import SSHSession


SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(value: str) -> str:
    return SAFE_NAME.sub("-", value).strip("-_") or "run"


def condition_name(run_config: dict[str, Any]) -> str:
    """Return a stable, readable directory name for result-affecting settings."""
    exp = run_config["experiment"]
    if run_config["experiment_name"].lower() == "nn":
        parts = [
            safe_name(str(exp.get("profile_name", "dual-tflite-v1"))),
            f"d{int(exp['duration_seconds'])}s",
            f"seed{int(exp['seed'])}",
            f"workers{int(exp['workers'])}",
            f"ct-i{int(exp['cyclictest_interval_us'])}us",
            f"h{int(exp['histogram_limit_us'])}",
        ]
        if "dom0_workload_cpus" in exp or "dom1_workload_cpus" in exp:
            parts.append(
                "wlcpu"
                + safe_name(str(exp.get("dom0_workload_cpus", "all")))
                + "-"
                + safe_name(str(exp.get("dom1_workload_cpus", "all")))
            )
        dom0_launch_delay = int(exp.get("dom0_launch_delay_seconds", 0))
        dom1_launch_delay = int(exp.get("dom1_launch_delay_seconds", 0))
        if dom0_launch_delay or dom1_launch_delay:
            parts.append(
                f"stagger{dom0_launch_delay}-{dom1_launch_delay}s"
            )
        if int(exp.get("warmup_seconds", 0)):
            parts.append(f"warmup{int(exp['warmup_seconds'])}s")
        return "_".join(parts)
    parts = [
        f"cpu{exp['cpu']}",
        f"t{exp['threads']}",
        f"p{exp['priority']}",
        f"i{exp['interval_us']}us",
        f"d{exp['duration_seconds']}s",
        f"h{exp['histogram_limit_us']}",
    ]
    if run_config["experiment_name"] == "cyclictest-stress":
        parts.extend([
            f"stress-cpu{exp['stress_cpus']}",
            f"vm{exp['stress_vm_workers']}",
            safe_name(str(exp["stress_vm_bytes"])),
        ])
    return "_".join(str(part) for part in parts)


class ExperimentRunner:
    def __init__(self, run_config: dict[str, Any], data_root: Path, dry_run: bool = False):
        self.cfg = run_config
        self.data_root = data_root
        self.dry_run = dry_run
        self._jailhouse_initialized = False
        self._xen_initialized = False

    def plan(self, runs: int, reboot_policy: str) -> list[Path]:
        series = (
            self.data_root
            / safe_name(self.cfg["experiment_name"])
            / safe_name(self.cfg["environment_name"])
            / condition_name(self.cfg)
        )
        existing = [
            int(match.group(1))
            for path in series.glob("hist_run*.txt")
            if (match := re.fullmatch(r"hist_run(\d+)\.txt", path.name))
        ] if series.exists() else []
        first = max(existing, default=0) + 1
        completed = series / "_details" / "completed"
        return [completed / f"run_{i:03d}" for i in range(first, first + runs)]

    def run(self, runs: int, reboot_policy: str) -> list[Path]:
        outputs = self.plan(runs, reboot_policy)
        if self.dry_run:
            print("[DRY RUN] 仅显示计划：不会连接串口、不会运行实验、不会创建结果目录。")
            self._print_plan(outputs, reboot_policy)
            return outputs

        # Create the first result directory before opening the serial port so its
        # transcript belongs to that run from the very first received byte.
        transcript = outputs[0] / "serial.log"
        self._create_output_directory(outputs[0], 1)
        print(
            f"[START] 正式运行：{self.cfg['environment_name']} / "
            f"{self.cfg['experiment_name']}，共 {runs} 轮"
        )
        print(
            f"[SERIAL] 正在连接 {self.cfg['serial']['port']} @ "
            f"{self.cfg['serial']['baudrate']} baud"
        )
        with SerialSession(self.cfg["serial"], transcript) as session:
            for index, output in enumerate(outputs, start=1):
                if index > 1:
                    self._create_output_directory(output, index)
                transcript = output / "serial.log"
                if session.transcript_path != transcript:
                    session.switch_transcript(transcript)
                print(f"\n[RUN {index}/{runs}] 结果目录：{output.resolve()}")
                print(f"[RUN {index}/{runs}] 串口日志：{transcript.resolve()}")
                self._write_metadata(output, "running", index, runs, reboot_policy, transcript)
                try:
                    if reboot_policy == "each-run" or index == 1:
                        if self.cfg["environment_name"] == "jailhouse":
                            self._jailhouse_initialized = False
                        if self.cfg["environment"].get("environment_type") == "xen":
                            self._xen_initialized = False
                        self._sync_local_boot_files()
                        print(f"[RUN {index}/{runs}] 重启并启动 {self.cfg['environment_name']} 环境...")
                        self._boot_and_login(session)
                        print(f"[RUN {index}/{runs}] root 登录成功。")
                    print(f"[RUN {index}/{runs}] 开始 {self.cfg['experiment_name']}...")
                    self._run_experiment(session, output)
                except (Exception, KeyboardInterrupt) as exc:
                    # A transcript must be closed before its directory can be
                    # moved on Windows.
                    if session.log:
                        session.log.close()
                        session.log = None
                    failed = self._failed_output_path(output, index)
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(output), str(failed))
                    transcript = failed / "serial.log"
                    session.transcript_path = transcript
                    failure_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    self._write_metadata(failed, failure_status, index, runs, reboot_policy, transcript, str(exc))
                    print(f"\n[FAILED] 第 {index} 轮失败：{exc}")
                    print(f"[FAILED] 已保留结果目录：{failed.resolve()}")
                    raise
                else:
                    self._write_metadata(output, "completed", index, runs, reboot_policy, transcript)
                    published = self._publish_histogram(output)
                    print(f"[RESULT] 主结果：{published.resolve()}")
                    print(f"[RUN {index}/{runs}] 完成，结果已保存到：{output.resolve()}")
        return outputs

    def _create_output_directory(self, output: Path, index: int) -> None:
        """Reserve a fresh detailed-output directory for one attempt."""
        output.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _failed_output_path(output: Path, index: int) -> Path:
        failed_root = output.parent.parent / "failed"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = failed_root / f"attempt_{stamp}_run_{index:03d}"
        suffix = 1
        while candidate.exists():
            candidate = failed_root / f"attempt_{stamp}_run_{index:03d}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _publish_histogram(output: Path) -> Path:
        """Expose the histogram at the condition root for quick comparison."""
        match = re.fullmatch(r"run_(\d+)", output.name)
        if not match:
            raise RuntimeError(f"Unexpected run directory name: {output.name}")
        source = output / "hist.txt"
        if not source.is_file():
            raise RuntimeError(f"Completed run has no histogram: {source}")
        series = output.parent.parent.parent
        published = series / f"hist_run{int(match.group(1))}.txt"
        if published.exists():
            raise FileExistsError(f"Histogram result already exists: {published}")
        shutil.copy2(source, published)
        return published

    def _sync_local_boot_files(self) -> None:
        """Overlay environment-specific host boot files before board reboot."""
        local_boot = self.cfg["environment"].get("local_boot_files")
        if not local_boot:
            return
        source = Path(local_boot["source_dir"])
        target = Path(local_boot["target_dir"])
        if not source.is_dir():
            raise FileNotFoundError(f"Boot-file source directory not found: {source}")
        if not target.is_dir():
            raise FileNotFoundError(f"Boot-file target directory not found: {target}")
        entries = list(source.iterdir())
        if not entries:
            raise RuntimeError(f"Boot-file source directory is empty: {source}")
        print(f"[BOOT FILES] Sync {source} -> {target}")
        for item in entries:
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
            print(f"[BOOT FILES] Copied: {item.name}")

    def _boot_and_login(self, session: SerialSession) -> None:
        env = self.cfg["environment"]
        timeouts = self.cfg["timeouts"]
        login_prompt = env.get("login_prompt", r"tl3588 login:\s*$")
        shell_patterns = env.get("shell_prompts", [r"root@[^\r\n]*[#]\s*$", r"[#]\s*$"])
        uboot_prompt = env.get("uboot_prompt", r"=>\s*$")

        session.buffer.clear()
        session.sendline()
        session.drain(float(timeouts.get("probe", 2)))
        probe = bytes(session.buffer).decode("utf-8", errors="replace")

        if re.search(uboot_prompt, probe, re.I | re.M):
            print("[BOOT] 当前已位于 U-Boot。")
            pass
        else:
            if self.cfg.get("reboot", {}).get("ssh"):
                self._reboot_jailhouse_rootcell(session, env, uboot_prompt)
                session.buffer.clear()
                self._boot_command_until_login(
                    session, env, login_prompt, uboot_prompt
                )
                session.buffer.clear()
                session.sendline(env.get("username", "root"))
                session.expect(shell_patterns, float(timeouts.get("login", 30)))
                session.command(
                    "export PS1='__XHYPASS_PROMPT__# '",
                    r"__XHYPASS_PROMPT__#\s*$",
                    10,
                )
                return
            self._reboot_serial_to_uboot(
                session, env, uboot_prompt, initial_probe=probe
            )

        session.buffer.clear()
        print(f"[U-BOOT] 执行：{env['boot_command']}")
        self._boot_command_until_login(session, env, login_prompt, uboot_prompt)
        print(f"[BOOT] 等待登录提示（最长 {timeouts.get('boot', 180)} 秒）...")
        session.buffer.clear()
        session.sendline(env.get("username", "root"))
        session.expect(shell_patterns, float(timeouts.get("login", 30)))
        session.command("export PS1='__XHYPASS_PROMPT__# '", r"__XHYPASS_PROMPT__#\s*$", 10)

    def _reboot_serial_to_uboot(
        self,
        session: SerialSession,
        env: dict[str, Any],
        uboot_prompt: str,
        *,
        initial_probe: str = "",
    ) -> None:
        """Log in and reboot a serial-only target without requiring SSH config."""
        timeouts = self.cfg["timeouts"]
        login_prompt = env.get("login_prompt", r"tl3588 login:\s*$")
        shell_patterns = env.get(
            "shell_prompts", [r"root@[^\r\n]*[#]\s*$", r"[#]\s*$"]
        )
        probe = initial_probe
        if re.search(login_prompt, probe, re.I | re.M):
            session.sendline(env.get("username", "root"))
            session.expect(
                shell_patterns, float(timeouts.get("login", 30)), clear=True
            )
        elif not any(
            re.search(pattern, probe, re.I | re.M) for pattern in shell_patterns
        ):
            # The target might still be booting, or its prompt may have been
            # emitted before this controller opened the serial port.
            try:
                session.expect([login_prompt, *shell_patterns], 10, clear=True)
                observed = bytes(session.buffer).decode(
                    "utf-8", errors="replace"
                )
                if re.search(login_prompt, observed, re.I | re.M):
                    session.sendline(env.get("username", "root"))
                    session.expect(
                        shell_patterns,
                        float(timeouts.get("login", 30)),
                        clear=True,
                    )
            except SerialTimeout:
                session.sendline()
        if self.cfg.get("reboot", {}).get(
            "prepare_serial_environment", False
        ):
            # Establish a deterministic prompt before probing modules/xl. This
            # must happen before reboot because a loaded passthrough module can
            # otherwise stall Xen shutdown.
            session.command(
                "export PS1='__REBOOT_PROBE__# '",
                r"__REBOOT_PROBE__#\s*$",
                10,
            )
            self._prepare_xen_reboot_serial(session)
        session.send(b"\x03")
        session.sendline()
        session.drain(0.5)
        print("[BOOT] 发送 reboot，等待进入 U-Boot...")
        session.buffer.clear()
        session.sendline(env.get("reboot_command", "reboot"))
        self._reach_uboot(session, uboot_prompt)

    def _reboot_jailhouse_rootcell(
        self, session: SerialSession, env: dict[str, Any], uboot_prompt: str
    ) -> None:
        """Prefer SSH because COM10 may currently be the non-root-cell console."""
        reboot_cfg = self.cfg.get("reboot", {})
        ssh_settings = copy.deepcopy(reboot_cfg["ssh"])
        root_ssh = SSHSession(ssh_settings)
        ssh_log = session.transcript_path.parent / "rootcell_ssh.log"
        try:
            print(
                f"[REBOOT/SSH] Probe root cell {ssh_settings['host']}:"
                f"{ssh_settings.get('port', 22)} before using COM10..."
            )
            root_ssh.connect()
        except ConnectionError as exc:
            print(f"[REBOOT/SSH] Root-cell SSH unavailable: {exc}")
            is_nonroot = self._serial_console_is_nonroot(session, reboot_cfg)
            if is_nonroot and not reboot_cfg.get("allow_serial_fallback", False):
                raise RuntimeError(
                    "COM10 was identified as the Jailhouse non-root cell from "
                    "/proc/cmdline. Serial reboot was blocked because it would not "
                    "reboot the root cell. Restore root-cell SSH access."
                ) from exc
            if is_nonroot:
                print("[REBOOT] WARNING: non-root serial fallback was explicitly allowed.")
            else:
                print("[REBOOT] COM10 is root/normal Linux; using serial reboot.")
                self._prepare_xen_reboot_serial(session)
            session.send(b"\x03")
            session.sendline()
            session.drain(0.5)
            session.buffer.clear()
            session.sendline(env.get("reboot_command", "reboot"))
            self._reach_uboot(session, uboot_prompt)
            return

        try:
            self._prepare_xen_reboot_ssh(root_ssh, ssh_log)
            print("[REBOOT/SSH] Connected; scheduling root-cell reboot.")
            root_ssh.run(
                "nohup sh -c 'sleep 1; reboot' >/dev/null 2>&1 &",
                timeout=10,
                log_path=ssh_log,
                check=True,
                show_output=False,
            )
        finally:
            root_ssh.close()
        session.buffer.clear()
        self._reach_uboot(session, uboot_prompt)

    @staticmethod
    def _xl_list_has_domain(output: bytes, domain: str) -> bool:
        text = output.decode("utf-8", errors="replace")
        return re.search(rf"^\s*{re.escape(domain)}\s+", text, re.I | re.M) is not None

    def _prepare_xen_reboot_ssh(self, ssh: SSHSession, log_path: Path) -> None:
        """Destroy dom1 before rebooting Xen, which otherwise may hang."""
        self._remove_xhypass_module_ssh(ssh, log_path)
        _, output, _ = ssh.run(
            "if command -v xl >/dev/null 2>&1; then echo __XEN_ENV__; xl list; fi",
            timeout=15,
            log_path=log_path,
            check=False,
        )
        if b"__XEN_ENV__" not in output:
            print("[REBOOT/XEN] xl is unavailable; current system is not Xen.")
            return
        print("[REBOOT/XEN] Xen environment detected by xl list.")
        if self._xl_list_has_domain(output, "dom1"):
            print("[REBOOT/XEN] Destroying dom1 before reboot...")
            ssh.run("xl destroy dom1", timeout=30, log_path=log_path, check=True)
        else:
            print("[REBOOT/XEN] dom1 is not running; no domain needs destruction.")

    @staticmethod
    def _remove_xhypass_module_ssh(ssh: SSHSession, log_path: Path) -> None:
        """Unload the XHyPass RTO module before any Xen teardown operation."""
        _, output, _ = ssh.run(
            "if command -v lsmod >/dev/null 2>&1 && "
            "lsmod | awk '{print $1}' | grep -qx interrupt_passthrough; then "
            "echo __XHYPASS_MODULE_LOADED__; fi",
            timeout=15,
            log_path=log_path,
            check=False,
        )
        if b"__XHYPASS_MODULE_LOADED__" not in output:
            return
        print("[REBOOT/XHYPASS] Unloading interrupt_passthrough before Xen teardown...")
        ssh.run(
            "taskset -c 6 rmmod interrupt_passthrough.ko",
            timeout=30,
            log_path=log_path,
            check=True,
        )

    def _prepare_xen_reboot_serial(self, session: SerialSession) -> None:
        prompt = r"__REBOOT_PROBE__#\s*$"
        module_cpu = int(
            self.cfg.get("reboot", {}).get("xhypass_module_cpu", 6)
        )
        self._unload_xhypass_module_serial(
            session,
            prompt,
            module_cpu,
            settle_seconds=2,
            reason="before reboot",
        )
        output = session.command(
            "if command -v xl >/dev/null 2>&1; then echo __XEN_ENV__; xl list; fi",
            prompt,
            15,
        )
        if b"__XEN_ENV__" not in output:
            print("[REBOOT/XEN] xl is unavailable; current system is not Xen.")
            return
        print("[REBOOT/XEN] Xen environment detected by xl list on COM10.")
        if self._xl_list_has_domain(output, "dom1"):
            print("[REBOOT/XEN] Destroying dom1 before serial reboot...")
            session.command("xl destroy dom1", prompt, 30)
        else:
            print("[REBOOT/XEN] dom1 is not running; no domain needs destruction.")

    @staticmethod
    def _module_is_loaded(module_list: bytes, module_name: str) -> bool:
        return re.search(
            rb"^" + re.escape(module_name.encode("utf-8")) + rb"\s+",
            module_list,
            re.M,
        ) is not None

    def _serial_module_is_loaded(
        self, session: SerialSession, prompt: str, module_name: str
    ) -> bool:
        # E2000Q hvc0 can deliver the tabular lsmod output after the shell
        # prompt, allowing it to collide with the next transmitted command.
        # Query /proc/modules directly and wait for a short explicit marker.
        output = session.command(
            f"grep -q '^{module_name} ' /proc/modules;echo XM=$?",
            r"(?:^|[\r\n])XM=\d+",
            15,
        )
        match = re.search(rb"(?:^|[\r\n])XM=(\d+)", output)
        if not match:
            raise RuntimeError("Module-state probe returned no valid XM marker")
        status = int(match.group(1))
        if status not in (0, 1):
            raise RuntimeError(
                f"Module-state probe failed with shell status {status}"
            )
        # Consume any delayed hvc0 output, then obtain a fresh prompt before
        # another command is allowed onto the serial console.
        session.drain(0.5)
        session.command("", prompt, 10)
        return status == 0

    def _unload_xhypass_module_serial(
        self,
        session: SerialSession,
        prompt: str,
        cpu: int,
        *,
        settle_seconds: float,
        reason: str,
    ) -> None:
        if not self._serial_module_is_loaded(
            session, prompt, "interrupt_passthrough"
        ):
            return
        print(
            "[XHYPASS] Unloading interrupt_passthrough "
            f"on CPU{cpu} {reason}..."
        )
        unload = session.command(
            f"taskset -c {cpu} rmmod interrupt_passthrough.ko;echo XR=$?",
            r"(?:^|[\r\n])XR=\d+",
            30,
        )
        match = re.search(rb"(?:^|[\r\n])XR=(\d+)", unload)
        if not match or int(match.group(1)) != 0:
            status = match.group(1).decode() if match else "missing"
            raise RuntimeError(
                f"Could not unload interrupt_passthrough (XR={status})"
            )
        session.drain(settle_seconds)
        session.command("", prompt, 10)
        if self._serial_module_is_loaded(
            session, prompt, "interrupt_passthrough"
        ):
            raise RuntimeError(
                "interrupt_passthrough is still present in lsmod after rmmod"
            )

    def _serial_console_is_nonroot(
        self, session: SerialSession, reboot_cfg: dict[str, Any]
    ) -> bool:
        """Log in on COM10 and identify the Jailhouse inmate from its cmdline."""
        login_prompt = reboot_cfg.get("nonroot_login_prompt", r"[^\s]+ login:\s*$")
        shell_prompts = reboot_cfg.get(
            "shell_prompts", [r"root@[^\r\n]*[#]\s*$", r"[#]\s*$"]
        )
        session.buffer.clear()
        session.sendline()
        matched, _ = session.expect([login_prompt, *shell_prompts], 10)
        if matched == 0:
            session.buffer.clear()
            session.sendline(reboot_cfg.get("nonroot_username", "root"))
            matched, _ = session.expect([r"Password:\s*$", *shell_prompts], 15)
            if matched == 0:
                session.buffer.clear()
                session.sendline(reboot_cfg.get("nonroot_password", ""))
                session.expect(shell_prompts, 15)

        session.command(
            "export PS1='__REBOOT_PROBE__# '", r"__REBOOT_PROBE__#\s*$", 10
        )
        raw = session.command(
            "echo __CMDLINE_BEGIN__; cat /proc/cmdline; echo __CMDLINE_END__",
            r"__REBOOT_PROBE__#\s*$",
            10,
        )
        cmdline = self._extract_cmdline(raw)
        if cmdline is None:
            raise RuntimeError("Could not identify the COM10 Linux environment")
        markers = reboot_cfg.get(
            "nonroot_cmdline_markers", ["isolcpus=3", "rcu_nocb=3"]
        )
        print(f"[REBOOT/SERIAL] /proc/cmdline: {cmdline.strip()}")
        return all(marker in cmdline for marker in markers)

    @staticmethod
    def _has_output_line(raw: bytes, marker: str) -> bool:
        """Ignore command echo and accept a marker only as a complete output line."""
        pattern = rb"(?:^|\r*\n)" + re.escape(marker.encode("utf-8")) + rb"\r*(?:\n|$)"
        return re.search(pattern, raw) is not None

    @staticmethod
    def _extract_cmdline(raw: bytes) -> str | None:
        match = re.search(
            rb"__CMDLINE_BEGIN__\r*\n(.*?)\r*\n__CMDLINE_END__", raw, re.S
        )
        if not match:
            return None
        return match.group(1).decode("utf-8", errors="replace")

    def _boot_command_until_login(
        self,
        session: SerialSession,
        env: dict[str, Any],
        login_prompt: str,
        uboot_prompt: str,
    ) -> None:
        """Run a U-Boot command and retry configured transient failures."""
        max_attempts = int(env.get("boot_max_attempts", 1))
        reset_cycles = int(env.get("boot_reset_cycles", 1))
        retry_delay = float(env.get("boot_retry_delay_seconds", 1.0))
        reset_delay = float(env.get("boot_reset_delay_seconds", 1.0))
        retry_patterns = list(env.get("boot_retry_patterns", []))
        immediate_reset_patterns = set(
            env.get("boot_immediate_reset_patterns", [])
        )
        boot_timeout = float(self.cfg["timeouts"].get("boot", 180))

        for reset_cycle in range(1, reset_cycles + 1):
            for attempt in range(1, max_attempts + 1):
                session.buffer.clear()
                cycle_text = (
                    f", reset cycle {reset_cycle}/{reset_cycles}"
                    if reset_cycles > 1
                    else ""
                )
                print(
                    f"[U-BOOT] Boot attempt {attempt}/{max_attempts}"
                    f"{cycle_text}: {env['boot_command']}"
                )
                session.sendline(env["boot_command"])
                matched, _ = session.expect(
                    [login_prompt, *retry_patterns], boot_timeout
                )
                if matched == 0:
                    return

                error_pattern = retry_patterns[matched - 1]
                print(f"[BOOT RETRY] Detected transient error: {error_pattern}")
                self._recover_uboot_prompt(session, uboot_prompt)
                immediate_reset = error_pattern in immediate_reset_patterns
                if not immediate_reset and attempt < max_attempts:
                    time.sleep(retry_delay)
                    continue

                if reset_cycle >= reset_cycles:
                    raise RuntimeError(
                        "Boot failed after "
                        f"{reset_cycles} reset cycles x {max_attempts} attempts: "
                        f"{error_pattern}"
                    )

                reset_command = str(env.get("boot_reset_command", "reset"))
                if immediate_reset:
                    print(
                        f"[BOOT RESET] `{error_pattern}` requires a fresh "
                        f"board reset; executing U-Boot `{reset_command}`..."
                    )
                else:
                    print(
                        "[BOOT RESET] Boot retries exhausted; executing U-Boot "
                        f"`{reset_command}` before the next cycle..."
                    )
                session.buffer.clear()
                session.sendline(reset_command)
                self._reach_uboot(session, uboot_prompt)
                time.sleep(reset_delay)
                break

    @staticmethod
    def _recover_uboot_prompt(session: SerialSession, uboot_prompt: str) -> None:
        """Wait for U-Boot to return, sending one Ctrl+C only when necessary."""
        try:
            session.expect([uboot_prompt], 3)
        except SerialTimeout:
            print("[BOOT RETRY] Sending one Ctrl+C to recover the U-Boot prompt.")
            session.buffer.clear()
            session.send(b"\x03")
            session.expect([uboot_prompt], 10)

    def _reach_uboot(self, session: SerialSession, uboot_prompt: str) -> None:
        timeout = float(self.cfg["timeouts"].get("uboot", 60))
        deadline = time.monotonic() + timeout
        prompt_re = re.compile(uboot_prompt.encode("utf-8"), re.I | re.M)
        login_re = re.compile(
            self.cfg["environment"].get("login_prompt", r"login:\s*$").encode("utf-8"),
            re.I | re.M,
        )

        # Do not transmit Ctrl+C while Linux is still processing `reboot`. Wait
        # until the console proves that reset has started. Sending control bytes
        # before this point can interrupt the shell or flood its prompt.
        reset_markers = list(
            self.cfg["environment"].get(
                "reset_markers",
                [r"DDR V\d", r"U-Boot SPL", r"U-Boot 20\d\d"],
            )
        )
        # Xen can spend most of the U-Boot timeout between the Linux shutdown
        # message and the first DDR/U-Boot banner.  These messages prove that
        # reboot has started, so it is safe to arm the Ctrl+C stop loop now.
        for marker in (
            r"reboot:\s*Restarting system",
            r"Hardware Dom0 shutdown:\s*rebooting machine",
        ):
            if marker not in reset_markers:
                reset_markers.append(marker)
        print("[REBOOT] 等待开发板确认重启（此阶段不发送 Ctrl+C）...")
        while time.monotonic() < deadline:
            session._read_once()
            if prompt_re.search(session.buffer):
                print("[U-BOOT] 已检测到 U-Boot 提示符。")
                return
            if any(
                re.search(pattern.encode("utf-8"), session.buffer, re.I | re.M)
                for pattern in reset_markers
            ):
                break
            time.sleep(0.02)
        else:
            raise SerialTimeout(
                f"No reboot marker appeared within {timeout}s; Ctrl+C was not sent"
            )

        print("[U-BOOT] 已确认重启，发送 Ctrl+C 截停自动启动...")
        interrupt_ack_re = re.compile(rb"=>\s*<INTERRUPT>\s*$", re.I | re.M)
        while time.monotonic() < deadline:
            # The countdown is effectively zero. Ctrl+C is sent only after reset
            # was observed, and at a moderate rate until main U-Boot accepts it.
            session.send(b"\x03")
            session._read_once()
            if prompt_re.search(session.buffer) or interrupt_ack_re.search(
                session.buffer
            ):
                print("[U-BOOT] 已检测到 U-Boot 提示符。")
                return
            if login_re.search(session.buffer):
                raise SerialTimeout(
                    "Missed the U-Boot stop window: Linux reached the login prompt. "
                    "No further input was sent. Please retry."
                )
            time.sleep(0.10)
        raise SerialTimeout(
            f"Unable to reach U-Boot prompt within {timeout}s while sending Ctrl+C"
        )

    def _run_experiment(self, session: SerialSession, output: Path) -> None:
        name = self.cfg["experiment_name"]
        if name not in {"cyclictest", "cyclictest-stress"}:
            raise NotImplementedError(f"Experiment plugin not implemented yet: {name}")
        if self.cfg["environment_name"] == "jailhouse":
            self._run_jailhouse_cyclictest(session, output)
        elif self.cfg["environment"].get("environment_type") == "xen":
            self._run_xen_cyclictest(session, output)
        else:
            self._run_cyclictest(session, output)

    def _run_xen_cyclictest(self, session: SerialSession, output: Path) -> None:
        env = self.cfg["environment"]
        if env.get("xen_serial_only", False):
            print("[XEN] Serial-only environment; running directly in dom0.")
            prompt = r"__XHYPASS_PROMPT__#\s*$"
            module_cpu = env.get("xhypass_module_cpu")
            if module_cpu is not None:
                module_cpu = int(module_cpu)
                module_file = str(
                    env.get("xhypass_module_file", "interrupt_passthrough.ko")
                )
                settle_seconds = float(
                    env.get("xhypass_module_settle_seconds", 2)
                )
                if not self._serial_module_is_loaded(
                    session, prompt, "interrupt_passthrough"
                ):
                    print(
                        "[XHYPASS] Loading interrupt_passthrough "
                        f"on CPU{module_cpu}..."
                    )
                    # Keep both the command and marker short. E2000Q hvc0 was
                    # observed dropping/repeating characters in the previous
                    # long XY_XHYPASS_INSMOD_RC command line.
                    load = session.command(
                        f"taskset -c {module_cpu} insmod {module_file};echo XI=$?",
                        r"(?:^|[\r\n])XI=\d+",
                        30,
                    )
                    match = re.search(rb"(?:^|[\r\n])XI=(\d+)", load)
                    if not match or int(match.group(1)) != 0:
                        status = match.group(1).decode() if match else "missing"
                        raise RuntimeError(
                            "Could not load interrupt_passthrough "
                            f"(XI={status})"
                        )
                    session.drain(settle_seconds)
                    session.command("", prompt, 10)
                if not self._serial_module_is_loaded(
                    session, prompt, "interrupt_passthrough"
                ):
                    raise RuntimeError(
                        "interrupt_passthrough is absent from lsmod after insmod"
                    )

            pre_output = b""
            for command in env.get("pre_experiment_commands", []):
                print(f"[XHYPASS] Pre-experiment command: {command}")
                pre_output += session.command(
                    command,
                    prompt,
                    float(env.get("pre_experiment_command_timeout", 30)),
                )
            pre_marker = env.get("pre_experiment_success_marker")
            if pre_marker and not self._has_output_line(
                pre_output, str(pre_marker)
            ):
                raise RuntimeError(
                    "XHyPass pre-experiment module load was not confirmed"
                )
            module_delay = float(env.get("post_module_load_delay_seconds", 0))
            if pre_marker and module_delay > 0:
                print(
                    f"[XHYPASS] Module loaded; waiting {module_delay:g}s "
                    "before the experiment..."
                )
                time.sleep(module_delay)

            self._run_cyclictest(session, output)

            if module_cpu is not None:
                self._unload_xhypass_module_serial(
                    session,
                    prompt,
                    module_cpu,
                    settle_seconds=settle_seconds,
                    reason="after the experiment",
                )

            post_output = b""
            for command in env.get("post_experiment_commands", []):
                print(f"[XHYPASS] Post-experiment command: {command}")
                post_output += session.command(
                    command,
                    prompt,
                    float(env.get("post_experiment_command_timeout", 30)),
                )
            post_marker = env.get("post_experiment_success_marker")
            if post_marker and not self._has_output_line(
                post_output, str(post_marker)
            ):
                raise RuntimeError(
                    "XHyPass post-experiment module removal was not confirmed"
                )
            return

        prompt = r"__XHYPASS_PROMPT__#\s*$"
        ssh_log = output / "dom0_ssh.log"

        if not self._xen_initialized:
            print("[XEN] Initializing bridge and dom1 from the serial console...")
            post_delays = env.get("dom0_init_post_delays_seconds", {})
            for command in env["dom0_init_commands"]:
                post_delay = float(post_delays.get(command, 0))
                # xl vcpu-set prints the prompt before asynchronous CPU shutdown
                # messages. For delayed commands, recognize the prompt anywhere
                # in the buffer and drain those messages before continuing.
                command_prompt = r"__XHYPASS_PROMPT__#" if post_delay else prompt
                session.command(
                    command,
                    command_prompt,
                    float(env.get("init_command_timeout", 60)),
                )
                if post_delay:
                    print(
                        f"[XEN] Waiting {post_delay:g}s after `{command}` "
                        "for asynchronous console activity..."
                    )
                    session.drain(post_delay)

        ssh = SSHSession(env["ssh"])
        try:
            print(
                f"[SSH] Connecting to Xen dom0 {env['ssh']['host']}:"
                f"{env['ssh'].get('port', 22)}..."
            )
            ssh.connect()
            print("[SSH] Xen dom0 connection established.")
            if not self._xen_initialized:
                if env.get("dom0_pin_command"):
                    ssh.run(env["dom0_pin_command"], timeout=30, log_path=ssh_log, check=True)
                if env.get("dom1_pin_command"):
                    ssh.run(env["dom1_pin_command"], timeout=30, log_path=ssh_log, check=True)
                for verify_command in env.get("pin_verify_commands", []):
                    ssh.run(
                        verify_command,
                        timeout=30,
                        log_path=ssh_log,
                        check=True,
                    )
                print("[XEN/DOM1] Waiting for the dom1 console login...")
                ssh.wait_for_console_login(
                    env.get("dom1_console_command", "xl console dom1"),
                    username=env.get("dom1_username", "root"),
                    password=env.get("dom1_password", ""),
                    login_prompt=env.get("dom1_login_prompt", r"[A-Za-z0-9_.-]+ login:\s*$"),
                    shell_prompts=env.get("dom1_shell_prompts", [r"root@[^\r\n]*#\s*$", r"#\s*$"]),
                    timeout=float(env.get("dom1_boot_timeout", 180)),
                    log_path=ssh_log,
                )
                settle_seconds = float(
                    env.get("post_dom1_login_delay_seconds", 10)
                )
                if settle_seconds > 0:
                    print(
                        f"[XEN/DOM1] Root login completed; waiting "
                        f"{settle_seconds:g}s for dom1 to stabilize..."
                    )
                    time.sleep(settle_seconds)
                self._xen_initialized = True
        finally:
            ssh.close()
        print("[XEN] dom1 is stable; running the experiment on dom0 via COM10.")
        pre_output = b""
        for command in env.get("pre_experiment_commands", []):
            print(f"[XHYPASS] Pre-experiment command: {command}")
            pre_output += session.command(
                command,
                prompt,
                float(env.get("pre_experiment_command_timeout", 30)),
            )
        marker = env.get("pre_experiment_success_marker")
        if marker and not self._has_output_line(pre_output, str(marker)):
            raise RuntimeError(
                "XHyPass pre-experiment setup did not confirm that "
                "interrupt_passthrough is loaded"
            )
        module_settle_seconds = float(
            env.get("post_module_load_delay_seconds", 0)
        )
        if marker and module_settle_seconds > 0:
            print(
                f"[XHYPASS] interrupt_passthrough is ready; waiting "
                f"{module_settle_seconds:g}s before the experiment..."
            )
            time.sleep(module_settle_seconds)
        self._run_cyclictest(session, output)

    def _run_jailhouse_cyclictest(
        self, session: SerialSession, output: Path
    ) -> None:
        env = self.cfg["environment"]
        prompt = r"__XHYPASS_PROMPT__#\s*$"
        if env.get("jailhouse_rootcell_only", False):
            if not self._jailhouse_initialized:
                self._load_jailhouse_module(session, env, prompt)
                enable_command = env["enable_command"]
                settle_seconds = float(env.get("enable_settle_seconds", 2))
                print(f"[JAILHOUSE] Enabling root cell: {enable_command}")
                enabled = session.command(
                    f"{enable_command}; JH_ENABLE_RC=$?; "
                    f"sleep {settle_seconds:g}; "
                    "printf '\\n__JH_ENABLE_RC__=%s\\n' \"$JH_ENABLE_RC\"",
                    prompt,
                    settle_seconds + 60,
                )
                if not self._has_output_line(enabled, "__JH_ENABLE_RC__=0"):
                    raise RuntimeError("Failed to enable the Jailhouse root cell")
                self._jailhouse_initialized = True
            self._run_cyclictest(session, output)
            return

        if not self._jailhouse_initialized:
            self._load_jailhouse_module(session, env, prompt)
            netmask = env.get("rootcell_netmask")
            ifconfig = f"ifconfig eth0 {env['rootcell_ip']}"
            if netmask:
                ifconfig += f" netmask {netmask}"
            print(f"[JAILHOUSE] Configure root-cell network: {ifconfig}")
            session.command(ifconfig, prompt, 20)

        ssh_log = output / "rootcell_ssh.log"
        ssh = SSHSession(env["ssh"])
        try:
            print(
                f"[SSH] Connecting to root cell {env['ssh']['host']}:"
                f"{env['ssh'].get('port', 22)}..."
            )
            ssh.connect()
            print("[SSH] Root-cell connection established.")
            if not self._jailhouse_initialized:
                ssh.run(
                    env["enable_command"], timeout=60, log_path=ssh_log, check=True
                )

                # COM10 becomes the non-root-cell console as the inmate starts.
                session.buffer.clear()
                ssh.run(
                    env["linux_command"], timeout=60, log_path=ssh_log, check=True
                )
                self._login_nonroot_cell(session, env)
                self._jailhouse_initialized = True
            self._run_cyclictest_ssh(ssh, output, ssh_log)
        finally:
            ssh.close()

    def _load_jailhouse_module(
        self, session: SerialSession, env: dict[str, Any], prompt: str
    ) -> None:
        command = env.get("module_load_command", "insmod jailhouse.ko")
        settle_seconds = float(env.get("module_settle_seconds", 20))
        print(f"[JAILHOUSE] Loading root-cell module: {command}")
        # Loading Jailhouse emits delayed kernel/regulator messages on the serial
        # console.  If the completion marker is printed immediately, those
        # messages can split both the marker and PS1, making a successful insmod
        # look like a serial timeout.  Wait for that burst to settle before
        # emitting the marker used by the controller.  The lsmod guard also
        # makes this safe when retrying in the same boot.
        output = session.command(
            "cd ~; "
            "if command -v lsmod >/dev/null 2>&1 && "
            "lsmod | awk '{print $1}' | grep -qx jailhouse; then "
            "JH_RC=0; "
            f"else {command}; JH_RC=$?; fi; "
            f"sleep {settle_seconds:g}; "
            "printf '\\n__JH_INSMOD_RC__=%s\\n' \"$JH_RC\"",
            prompt,
            settle_seconds + 45,
        )
        if not self._has_output_line(output, "__JH_INSMOD_RC__=0"):
            raise RuntimeError(
                "Failed to load jailhouse.ko immediately after root login"
            )

    def _login_nonroot_cell(self, session: SerialSession, env: dict[str, Any]) -> None:
        login_prompt = env.get(
            "nonroot_login_prompt", env.get("login_prompt", r"[^\s]+ login:\s*$")
        )
        timeout = float(env.get("nonroot_boot_timeout", 180))
        print(f"[NONROOT] Waiting for non-root-cell login prompt ({timeout:g}s)...")
        session.expect([login_prompt], timeout)
        session.buffer.clear()
        session.sendline(env.get("nonroot_username", "root"))
        password = env["nonroot_password"]
        shell_prompt = r"[#]\s*$"
        new_password_prompts = [
            r"New password:\s*$",
            r"Enter new UNIX password:\s*$",
        ]
        matched, _ = session.expect(
            [
                shell_prompt,
                r"Password:\s*$",
                r"Current password:\s*$",
                *new_password_prompts,
            ],
            30,
        )

        if matched == 1:
            print("[NONROOT] Existing password requested; sending configured password.")
            session.buffer.clear()
            session.sendline(password)
            matched, _ = session.expect([shell_prompt, *new_password_prompts], 30)
            if matched == 0:
                self._set_nonroot_prompt(session)
                print("[NONROOT] Login succeeded with the configured password.")
                return
        elif matched == 2:
            print("[NONROOT] Current password requested; sending configured password.")
            session.buffer.clear()
            session.sendline(password)
            session.expect(new_password_prompts, 20)
        elif matched == 0:
            self._set_nonroot_prompt(session)
            print("[NONROOT] Login succeeded; no password change was requested.")
            return

        print("[NONROOT] Forced password change detected; sending new password.")
        session.buffer.clear()
        session.sendline(password)
        matched, _ = session.expect(
            [
                r"Retype password:\s*$",
                r"Retype new password:\s*$",
                r"Retype new UNIX password:\s*$",
                shell_prompt,
            ],
            20,
        )
        if matched == 3:
            self._set_nonroot_prompt(session)
            print("[NONROOT] Password change completed and shell is ready.")
            return
        print("[NONROOT] Confirming the new password.")
        session.buffer.clear()
        session.sendline(password)
        session.expect(
            [
                shell_prompt,
                r"password updated successfully",
                r"passwd: password updated successfully",
            ],
            30,
        )
        # Some passwd implementations print a success line before redrawing the
        # shell prompt; sending a newline makes the prompt observable without
        # changing state.
        session.buffer.clear()
        session.sendline()
        session.expect([shell_prompt], 15)
        self._set_nonroot_prompt(session)
        print("[NONROOT] Login succeeded and the forced password change completed.")

    @staticmethod
    def _set_nonroot_prompt(session: SerialSession) -> None:
        session.command(
            "export PS1='__JH_NONROOT_PROMPT__# '",
            r"__JH_NONROOT_PROMPT__#\s*$",
            10,
        )

    def _run_cyclictest_ssh(
        self, ssh: SSHSession, output: Path, ssh_log: Path
    ) -> None:
        exp = self.cfg["experiment"]
        duration = int(exp["duration_seconds"])
        remote_dir = f"/tmp/xyexp-{int(time.time())}"
        remote_hist = f"{remote_dir}/hist.txt"
        remote_stdout = f"{remote_dir}/cyclictest.log"
        remote_stress = f"{remote_dir}/stress-ng.log"
        stress_enabled = self.cfg["experiment_name"] == "cyclictest-stress"
        stress_check = (
            f" && test -x {exp['stress_binary']}" if stress_enabled else ""
        )
        preflight = (
            f"cd ~ && test -x {exp['binary']}{stress_check} "
            "&& command -v taskset >/dev/null"
        )
        ssh.run(preflight, timeout=15, log_path=ssh_log, check=True)
        stress_launch = ""
        stress_wait = ""
        if stress_enabled:
            stress_launch = (
                f"taskset -c {exp['stress_cpus']} {exp['stress_binary']} "
                f"--vm {int(exp['stress_vm_workers'])} "
                f"--vm-bytes {exp['stress_vm_bytes']} --metrics-brief "
                f"--timeout {duration}s >{remote_stress} 2>&1 & "
                "XY_STRESS_PID=$!; "
            )
            stress_wait = (
                "wait $XY_STRESS_PID; XY_STRESS_RC=$?; "
                "test $XY_STRESS_RC -eq 0 || exit $XY_STRESS_RC; "
            )
        command = (
            f"cd ~ && mkdir -p {remote_dir}; "
            f"{stress_launch}"
            f"taskset -c {int(exp['cpu'])} {exp['binary']} -t{int(exp['threads'])} "
            f"-p {int(exp['priority'])} -m -i {int(exp['interval_us'])} "
            f"-D {duration}s -h {int(exp['histogram_limit_us'])} "
            f"--histfile={remote_hist} -q >{remote_stdout} 2>&1 & "
            "XY_PID=$!; wait $XY_PID; XY_RC=$?; "
            f"{stress_wait}"
            "exit $XY_RC"
        )
        print(
            f"[CYCLICTEST/SSH] Starting on root cell: CPU={exp['cpu']}, "
            f"duration={duration}s"
        )
        ssh.run(
            command,
            timeout=duration + int(exp.get("grace_seconds", 60)),
            log_path=ssh_log,
            check=True,
        )
        ssh.get(remote_hist, output / "hist.txt")
        ssh.get(remote_stdout, output / "cyclictest.log")
        if stress_enabled:
            ssh.get(remote_stress, output / "stress-ng.log")
        if exp.get("cleanup_remote", True):
            ssh.run(
                f"rm -rf {remote_dir}", timeout=15, log_path=ssh_log, check=True
            )
        print("[RESULT] Downloaded root-cell hist.txt and cyclictest.log via SFTP.")

    def _run_cyclictest(self, session: SerialSession, output: Path) -> None:
        exp = self.cfg["experiment"]
        prompt = r"__XHYPASS_PROMPT__#\s*$"
        duration = int(exp["duration_seconds"])
        # Environment initialization may leave the serial shell in another
        # directory (xen_credit2 ends in ~/dom-interrupt). Experiment binaries
        # are stored in root's home and configured with ./ relative paths.
        session.command("cd ~", prompt, 10)
        print(
            f"[CYCLICTEST] CPU={exp['cpu']}，duration={duration}s，"
            f"interval={exp['interval_us']}us，priority={exp['priority']}"
        )
        stress_enabled = self.cfg["experiment_name"] == "cyclictest-stress"
        # E2000Q's hvc0 console can lose characters from long command lines.
        # Keep every probe and its marker deliberately short; otherwise a
        # successful ``__XY_PREFLIGHT__=0`` can arrive as ``__XY_PRE__=0`` and
        # be misreported as a missing executable.
        probes = [
            ("cyclictest", f"test -x {exp['binary']}", "XYCY"),
            ("taskset", "command -v taskset >/dev/null", "XYTS"),
            ("base64", "command -v base64 >/dev/null", "XYB6"),
        ]
        if stress_enabled:
            probes.insert(
                1,
                ("stress-ng", f"test -x {exp['stress_binary']}", "XYST"),
            )
        missing = []
        for name, probe, marker in probes:
            check = session.command(
                f"{probe}; echo {marker}=$?",
                prompt,
                10,
            )
            success_pattern = (
                rb"(?:^|[\r\n])"
                + marker.encode()
                + rb"=0(?:[\r\n]|$)"
            )
            if not re.search(success_pattern, check):
                missing.append(name)
        if missing:
            raise RuntimeError(
                "Board preflight failed; missing or unusable: "
                + ", ".join(missing)
            )
        remote_dir = f"/tmp/xyexp-{int(time.time())}"
        remote_hist = f"{remote_dir}/hist.txt"
        remote_stdout = f"{remote_dir}/cyclictest.log"
        remote_stress = f"{remote_dir}/stress-ng.log"
        stress_launch = ""
        stress_wait = ""
        if stress_enabled:
            stress_launch = (
                f"taskset -c {exp['stress_cpus']} {exp['stress_binary']} "
                f"--vm {int(exp['stress_vm_workers'])} "
                f"--vm-bytes {exp['stress_vm_bytes']} --metrics-brief "
                f"--timeout {duration}s >{remote_stress} 2>&1 & "
                "XY_STRESS_PID=$!; "
            )
            stress_wait = (
                "wait $XY_STRESS_PID; XY_STRESS_RC=$?; "
                "echo __XY_STRESS_DONE__=$XY_STRESS_RC; "
            )
        command = (
            f"mkdir -p {remote_dir}; "
            f"{stress_launch}"
            f"taskset -c {int(exp['cpu'])} {exp['binary']} -t{int(exp['threads'])} "
            f"-p {int(exp['priority'])} -m -i {int(exp['interval_us'])} "
            f"-D {duration}s -h {int(exp['histogram_limit_us'])} "
            f"--histfile={remote_hist} -q >{remote_stdout} 2>&1 & "
            "XY_PID=$!; echo __XY_PID__=$XY_PID"
        )
        session.command(command, prompt, 15)
        print(f"[CYCLICTEST] 任务已启动，等待约 {duration} 秒...")
        wait = (
            "while kill -0 $XY_PID 2>/dev/null; do sleep 1; done; "
            "wait $XY_PID; XY_RC=$?; echo __XY_DONE__=$XY_RC; "
            f"{stress_wait}"
        )
        raw = session.command(wait, prompt, duration + int(exp.get("grace_seconds", 60)))
        done = re.search(rb"__XY_DONE__=(\d+)", raw)
        if not done or int(done.group(1)) != 0:
            raise RuntimeError("cyclictest did not finish successfully")
        if stress_enabled:
            stress_done = re.search(rb"__XY_STRESS_DONE__=(\d+)", raw)
            if not stress_done or int(stress_done.group(1)) != 0:
                raise RuntimeError("stress-ng did not finish successfully")
        self._download_base64(session, remote_hist, output / "hist.txt", prompt)
        self._download_base64(session, remote_stdout, output / "cyclictest.log", prompt)
        if stress_enabled:
            self._download_base64(
                session, remote_stress, output / "stress-ng.log", prompt
            )
        print("[RESULT] 已回收 hist.txt 和 cyclictest.log。")
        if exp.get("cleanup_remote", True):
            session.command(f"rm -rf {remote_dir}", prompt, 10)

    @staticmethod
    def _download_base64(session: SerialSession, remote: str, local: Path, prompt: str) -> None:
        begin, end = "XYB64BEGIN", "XYB64END"
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                raw = session.command(
                    f"echo {begin}; base64 {remote}; echo {end}", prompt, 30
                )
                match = re.search(
                    # Linux UARTs normally use LF or CRLF. Xen's hvc console
                    # emits CRCRLF, so accept any number of CR bytes before LF.
                    rb"XYB64BEGIN\r*\n(.*?)\r*\nXYB64END", raw, re.S
                )
                if not match:
                    raise ValueError("serial transfer markers were incomplete")
                payload = re.sub(rb"\s+", b"", match.group(1))
                decoded = base64.b64decode(payload, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                last_error = exc
                print(
                    f"[TRANSFER/RETRY] {remote}: attempt {attempt}/3 "
                    f"failed ({exc})"
                )
                continue
            local.write_bytes(decoded)
            return
        raise RuntimeError(
            f"Could not transfer remote file after 3 attempts: {remote}"
        ) from last_error

    def _write_metadata(
        self, output: Path, status: str, index: int, runs: int, reboot_policy: str,
        transcript: Path, error: str | None = None,
    ) -> None:
        metadata = {
            "status": status,
            "condition": condition_name(self.cfg),
            "run_index": index,
            "run_count": runs,
            "reboot_policy": reboot_policy,
            "updated_at": datetime.now().astimezone().isoformat(),
            "serial_transcript": str(transcript.resolve()),
            "configuration": self._redact_secrets(self.cfg),
        }
        if error:
            metadata["error"] = error
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def _redact_secrets(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "<redacted>"
                    if "password" in key.lower() or "secret" in key.lower()
                    else cls._redact_secrets(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact_secrets(item) for item in value]
        return value

    def _print_plan(self, outputs: list[Path], reboot_policy: str) -> None:
        print(json.dumps({
            "environment": self.cfg["environment_name"],
            "boot_command": self.cfg["environment"]["boot_command"],
            "experiment": self.cfg["experiment_name"],
            "parameters": self.cfg["experiment"],
            "reboot_policy": reboot_policy,
            "output_directories": [str(path) for path in outputs],
        }, ensure_ascii=False, indent=2))
