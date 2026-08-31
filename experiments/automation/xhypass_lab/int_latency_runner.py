from __future__ import annotations

import copy
import json
import re
import shutil
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import ExperimentRunner, safe_name
from .serial_session import SerialSession, SerialTimeout


class BootTransientError(RuntimeError):
    """A retryable U-Boot/TFTP failure observed on the control console."""


class PostRunRecoveryError(RuntimeError):
    """Experiment data completed, but the board could not return to U-Boot."""


class IntLatencyRunner:
    """Run interrupt-latency experiments using control and RTOS UARTs."""

    def __init__(
        self,
        run_configs: list[dict[str, Any]],
        data_root: Path,
        *,
        dry_run: bool = False,
    ) -> None:
        if not run_configs:
            raise ValueError("At least one int-latency run configuration is required")
        self.run_configs = run_configs
        self.data_root = data_root
        self.dry_run = dry_run

    def _series(self, config: dict[str, Any]) -> Path:
        return (
            self.data_root
            / "int-latency"
            / safe_name(config["environment_name"])
            / safe_name(config["experiment"]["condition"])
        )

    def _next_run_number(self, config: dict[str, Any]) -> int:
        series = self._series(config)
        existing = []
        if series.exists():
            for path in series.glob("rtos_run*.log"):
                match = re.fullmatch(r"rtos_run(\d+)\.log", path.name)
                if match:
                    existing.append(int(match.group(1)))
        return max(existing, default=0) + 1

    def plan(self) -> list[tuple[dict[str, Any], int, Path]]:
        next_numbers: dict[tuple[str, str], int] = {}
        planned = []
        for config in self.run_configs:
            key = (
                config["environment_name"],
                config["experiment"]["condition"],
            )
            number = next_numbers.get(key)
            if number is None:
                number = self._next_run_number(config)
            next_numbers[key] = number + 1
            output = (
                self._series(config)
                / "_details"
                / "completed"
                / f"run_{number:03d}"
            )
            planned.append((config, number, output))
        return planned

    def run(self) -> list[Path]:
        planned = self.plan()
        if self.dry_run:
            print("[DRY RUN] int-latency plan; no serial ports will be opened.")
            for config, number, output in planned:
                print(
                    f"  run {number}: {config['environment_name']} / "
                    f"{config['experiment']['condition']} -> {output.resolve()}"
                )
            return [item[2] for item in planned]

        first_config, _, first_output = planned[0]
        self._create_output_directory(first_output, planned[0][1])
        control_settings = first_config["control_serial"]
        rtos_settings = first_config["rtos_serial"]
        control_log_name = self._serial_log_name(control_settings, "control")
        rtos_log_name = self._serial_log_name(rtos_settings, "rtos")
        print(
            f"[SERIAL/CONTROL] {control_settings['port']} @ "
            f"{control_settings['baudrate']} baud"
        )
        print(
            f"[SERIAL/RTOS] {rtos_settings['port']} @ "
            f"{rtos_settings['baudrate']} baud"
        )

        results: list[Path] = []
        with ExitStack() as stack:
            control = stack.enter_context(
                SerialSession(control_settings, first_output / control_log_name)
            )
            rtos = stack.enter_context(
                SerialSession(rtos_settings, first_output / rtos_log_name)
            )
            uboot_ready = False

            for index, (config, run_number, output) in enumerate(planned):
                if index:
                    self._create_output_directory(output, run_number)
                    control.switch_transcript(output / control_log_name)
                    rtos.switch_transcript(output / rtos_log_name)

                self._write_metadata(output, config, run_number, "running")
                condition = config["experiment"]["condition"]
                print(
                    f"\n[INT-LATENCY] {config['environment_name']} / "
                    f"{condition} / run {run_number}"
                )
                print(f"[RESULT DIR] {output.resolve()}")
                try:
                    self._sync_boot_files(config)
                    if not uboot_ready:
                        self._ensure_uboot(control, config)
                    environment_name = config["environment_name"]
                    if environment_name == "bare":
                        self._run_bare_trial(control, rtos, config)
                    elif environment_name == "jailhouse":
                        self._run_jailhouse_trial(control, rtos, config)
                    elif config["environment"].get("environment_type") == "xen":
                        self._run_xen_dom0less_trial(control, rtos, config)
                    else:
                        raise NotImplementedError(
                            "int-latency environment is not implemented yet: "
                            f"{environment_name}"
                        )
                    uboot_ready = True
                except (Exception, KeyboardInterrupt) as exc:
                    self._close_transcripts(control, rtos)
                    failed = self._failed_output_path(output, run_number)
                    failed.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(output), str(failed))
                    failure_status = (
                        "interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    )
                    error_text = str(exc) or type(exc).__name__
                    self._write_metadata(
                        failed,
                        config,
                        run_number,
                        failure_status,
                        error=error_text,
                    )
                    print(f"[FAILED] int-latency run failed: {error_text}")
                    print(f"[FAILED] Logs retained at: {failed.resolve()}")
                    raise

                self._write_metadata(output, config, run_number, "completed")
                published = self._publish_rtos_log(output, run_number, config)
                print(f"[RESULT] RTOS log: {published.resolve()}")
                results.append(output)

                try:
                    if environment_name == "bare":
                        # The bare RTOS image is expected to reboot itself. Keep
                        # this recovery separate from experiment validity: COM14
                        # data is already complete and published at this point.
                        print(
                            "[BARE] RTOS runtime completed; waiting for "
                            "automatic reboot..."
                        )
                        self._base_runner(config)._reach_uboot(
                            control,
                            config["environment"].get(
                                "uboot_prompt", r"=>\s*$"
                            ),
                        )
                    elif environment_name == "jailhouse":
                        self._finish_jailhouse_trial(control, config)
                    else:
                        self._finish_xen_dom0less_trial(control, config)
                except Exception as exc:
                    self._write_metadata(
                        output,
                        config,
                        run_number,
                        "completed",
                        recovery_error=str(exc),
                    )
                    raise PostRunRecoveryError(
                        f"Experiment result was saved, but board recovery "
                        f"failed: {exc}"
                    ) from exc
                uboot_ready = True

        return results

    def _create_output_directory(self, output: Path, run_number: int) -> None:
        """Preserve an interrupted detailed attempt and reuse its run number."""
        if output.exists():
            failed = self._failed_output_path(output, run_number)
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(failed))
            metadata = failed / "metadata.json"
            if metadata.is_file():
                try:
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                payload.update(
                    {
                        "status": "abandoned",
                        "error": (
                            "Interrupted attempt was found in the completed "
                            "directory during resume"
                        ),
                        "recovered_at": datetime.now().astimezone().isoformat(),
                    }
                )
                metadata.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(
                "[RESUME] Preserved stale int-latency attempt before reusing "
                f"{output.name}: {failed.resolve()}"
            )
        output.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _close_transcripts(control: SerialSession, rtos: SerialSession) -> None:
        for session in (control, rtos):
            if session.log:
                session.log.close()
                session.log = None

    @staticmethod
    def _base_runner(config: dict[str, Any]) -> ExperimentRunner:
        return ExperimentRunner(config, Path("unused"), dry_run=False)

    @staticmethod
    def _sync_boot_files(config: dict[str, Any]) -> None:
        files = config["environment"].get("int_latency_boot_files", [])
        for item in files:
            source = Path(item["source"])
            destination = Path(item["destination"])
            if not source.is_file():
                raise FileNotFoundError(
                    f"int-latency boot artifact is missing: {source}"
                )
            if not destination.parent.is_dir():
                raise FileNotFoundError(
                    "int-latency boot target directory is missing: "
                    f"{destination.parent}"
                )
            print(f"[BOOT FILE] {source} -> {destination}")
            shutil.copy2(source, destination)

    def _ensure_uboot(self, control: SerialSession, config: dict[str, Any]) -> None:
        environment = config["environment"]
        uboot_prompt = environment.get("uboot_prompt", r"=>\s*$")
        control.buffer.clear()
        control.sendline()
        control.drain(float(config["timeouts"].get("probe", 2)))
        if re.search(uboot_prompt.encode(), control.buffer, re.I | re.M):
            print("[U-BOOT] Board is already at the U-Boot prompt.")
            return

        reset_markers = environment.get(
            "reset_markers", [r"DDR V\d", r"U-Boot SPL", r"U-Boot 20\d\d"]
        )
        if any(
            re.search(pattern.encode(), control.buffer, re.I | re.M)
            for pattern in reset_markers
        ):
            print("[U-BOOT] Reset is already in progress; intercepting U-Boot.")
            self._base_runner(config)._reach_uboot(control, uboot_prompt)
            return

        print("[REBOOT] Board is not at U-Boot; using the existing safe reboot flow.")
        base_runner = self._base_runner(config)
        if config.get("reboot", {}).get("ssh"):
            base_runner._reboot_jailhouse_rootcell(
                control, environment, uboot_prompt
            )
        else:
            probe = bytes(control.buffer).decode("utf-8", errors="replace")
            base_runner._reboot_serial_to_uboot(
                control,
                environment,
                uboot_prompt,
                initial_probe=probe,
            )

    def _run_bare_trial(
        self,
        control: SerialSession,
        rtos: SerialSession,
        config: dict[str, Any],
    ) -> None:
        environment = config["environment"]
        experiment = config["experiment"]
        max_attempts = int(environment.get("boot_max_attempts", 3))
        retry_delay = float(environment.get("boot_retry_delay_seconds", 1))
        retry_patterns = [
            re.compile(pattern.encode(), re.I | re.M)
            for pattern in environment.get("boot_retry_patterns", [])
        ]
        uboot_prompt = environment.get("uboot_prompt", r"=>\s*$")

        for attempt in range(1, max_attempts + 1):
            control.buffer.clear()
            self._prepare_rtos_capture(rtos)
            print(
                f"[U-BOOT] Attempt {attempt}/{max_attempts}: "
                f"{environment['boot_command']}"
            )
            started_at = time.monotonic()
            control.sendline(environment["boot_command"])
            try:
                self._monitor_dual_serial(
                    control,
                    rtos,
                    completion_pattern=str(experiment["completion_pattern"]),
                    timeout=float(experiment["timeout_seconds"]),
                    completion_settle_seconds=float(
                        experiment.get("completion_settle_seconds", 5)
                    ),
                    completion_not_before_seconds=(
                        float(experiment["duration_seconds"])
                        * float(experiment.get("completion_not_before_ratio", 0.9))
                    ),
                    started_at=started_at,
                    retry_patterns=retry_patterns,
                    stop_bare_autoboot=True,
                    uboot_prompt=uboot_prompt,
                    reset_markers=list(
                        environment.get(
                            "reset_markers",
                            [r"DDR V\d", r"U-Boot SPL", r"U-Boot 20\d\d"],
                        )
                    ),
                    login_prompt=environment.get(
                        "login_prompt", r"tl3588 login:\s*$"
                    ),
                )
                return
            except BootTransientError as exc:
                print(f"[BOOT RETRY] {exc}")
                self._base_runner(config)._recover_uboot_prompt(
                    control, uboot_prompt
                )
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"RTOS boot failed after {max_attempts} attempts: {exc}"
                    ) from exc
                time.sleep(retry_delay)

    def _run_jailhouse_trial(
        self,
        control: SerialSession,
        rtos: SerialSession,
        config: dict[str, Any],
    ) -> None:
        environment = config["environment"]
        experiment = config["experiment"]
        base = self._base_runner(config)
        prompt = r"__XHYPASS_PROMPT__#\s*$"

        print("[JAILHOUSE/INT] Booting OEE and logging in on COM10...")
        base._boot_and_login(control)
        base._load_jailhouse_module(control, environment, prompt)

        self._checked_serial_command(
            control,
            environment["enable_command"],
            prompt,
            "ENABLE",
            timeout=60,
            settle_seconds=float(environment.get("enable_settle_seconds", 2)),
        )
        self._checked_serial_command(
            control,
            environment["int_latency_create_command"],
            prompt,
            "CREATE",
            timeout=60,
            settle_seconds=2,
        )
        load_command = environment["int_latency_load_commands"][
            experiment["condition"]
        ]
        self._checked_serial_command(
            control,
            load_command,
            prompt,
            "LOAD",
            timeout=60,
            settle_seconds=2,
        )

        # Clear stale RTOS bytes before start, not after it: QSemOS may print on
        # COM14 immediately while the start command is returning on COM10.
        self._prepare_rtos_capture(rtos)
        started_at = time.monotonic()
        with self._background_serial_capture(rtos):
            self._checked_serial_command(
                control,
                environment["int_latency_start_command"],
                prompt,
                "START",
                timeout=60,
                settle_seconds=1,
            )
        self._monitor_dual_serial(
            control,
            rtos,
            completion_pattern=str(experiment["completion_pattern"]),
            timeout=float(experiment["timeout_seconds"]),
            completion_settle_seconds=float(
                experiment.get("completion_settle_seconds", 5)
            ),
            completion_not_before_seconds=(
                float(experiment["duration_seconds"])
                * float(experiment.get("completion_not_before_ratio", 0.9))
            ),
            started_at=started_at,
            retry_patterns=[],
            control_completion_pattern=environment.get(
                "int_latency_reboot_ready_pattern"
            ),
        )

    def _finish_jailhouse_trial(
        self, control: SerialSession, config: dict[str, Any]
    ) -> None:
        environment = config["environment"]
        # Destroying the cell / disabling Jailhouse after the RTOS exits can
        # trigger scheduler warnings (balance_push) while CPUs are returned to
        # Linux.  A full reboot is the authoritative reset between trials, so
        # skip that unstable cleanup path and reboot the root cell directly.
        print("[JAILHOUSE/INT] RTOS completed; rebooting root cell directly...")
        self._send_prompt_independent_reboot(control, environment)
        self._base_runner(config)._reach_uboot(
            control, environment.get("uboot_prompt", r"=>\s*$")
        )

    @staticmethod
    def _recover_rootcell_shell(
        session: SerialSession, environment: dict[str, Any]
    ) -> None:
        """Recover an authenticated root-cell shell after console auto-logout."""
        login_prompt = environment.get("login_prompt", r"tl3588 login:\s*$")
        username = environment.get("username", "root")
        password = environment.get("password", "")
        custom_prompt = r"__XHYPASS_PROMPT__#\s*$"
        shell_prompts = environment.get(
            "shell_prompts", [r"root@[^\r\n]*[#]\s*$", r"[#]\s*$"]
        )
        patterns = [login_prompt, r"Password:\s*$", custom_prompt, *shell_prompts]

        print("[JAILHOUSE/INT] Recovering the root-cell COM10 shell...")
        # Ctrl+C is safe here: the RTOS result is already complete and COM10 is
        # known to be the Linux root cell.  It also escapes a stale Password:
        # prompt left by an earlier command being consumed as a login name.
        session.buffer.clear()
        session.send("\x03")
        session.sendline()
        matched, _ = session.expect(patterns, 15)

        if matched == 1:
            session.buffer.clear()
            session.send("\x03")
            session.sendline()
            matched, _ = session.expect([login_prompt, custom_prompt, *shell_prompts], 15)
            if matched == 0:
                matched = 0
            else:
                matched += 1

        if matched == 0:
            session.buffer.clear()
            session.sendline(username)
            matched, _ = session.expect(
                [r"Password:\s*$", custom_prompt, *shell_prompts], 30
            )
            if matched == 0:
                session.buffer.clear()
                session.sendline(password)
                session.expect([custom_prompt, *shell_prompts], 30)

        # Whether the recovered shell had the default or old custom prompt,
        # redraw a deterministic prompt for the command wrapper below.
        session.buffer.clear()
        session.sendline("export PS1='__XHYPASS_PROMPT__# '")
        session.expect([custom_prompt], 15)

    def _run_xen_dom0less_trial(
        self,
        control: SerialSession,
        rtos: SerialSession,
        config: dict[str, Any],
    ) -> None:
        environment = config["environment"]
        experiment = config["experiment"]
        prompt = r"__XHYPASS_PROMPT__#\s*$"

        # COM14 can begin producing samples as soon as U-Boot launches Xen.
        # Clear it exactly once before boot and never after Xen has started.
        self._prepare_rtos_capture(rtos)

        print("[XEN/INT] Booting Xen dom0less on COM10...")
        started_at = time.monotonic()
        # Actively drain COM14 while COM10 is busy booting, logging in, and
        # pinning vCPUs. Relying on the OS UART buffer here can lose early RTOS
        # samples when Xen boot output takes a while.
        with self._background_serial_capture(rtos):
            self._base_runner(config)._boot_and_login(control)
            dynamic_swap = environment.get("int_latency_dynamic_pin_swap")
            if dynamic_swap:
                self._apply_dynamic_vcpu_swap(control, prompt, dynamic_swap)
            else:
                pin_commands = environment["int_latency_pin_commands"]
                print(
                    f"[XEN/INT] Applying {len(pin_commands)} mandatory vCPU "
                    f"pin commands for {config['environment_name']}..."
                )
                for index, command in enumerate(pin_commands, start=1):
                    self._checked_serial_command(
                        control,
                        command,
                        prompt,
                        f"XEN_PIN_{index}",
                        timeout=30,
                    )
                print("[XEN/INT] Mandatory vCPU pinning completed.")

        self._monitor_dual_serial(
            control,
            rtos,
            completion_pattern=str(experiment["completion_pattern"]),
            timeout=float(experiment["timeout_seconds"]),
            completion_settle_seconds=float(
                experiment.get("completion_settle_seconds", 5)
            ),
            completion_not_before_seconds=(
                float(experiment["duration_seconds"])
                * float(experiment.get("completion_not_before_ratio", 0.9))
            ),
            started_at=started_at,
            retry_patterns=[],
            control_completion_pattern=environment.get(
                "int_latency_reboot_ready_pattern"
            ),
        )

    @staticmethod
    def _xl_vcpu_rows(output: bytes) -> list[tuple[int, int, int]]:
        """Extract (domain, vCPU, pCPU) rows from `xl vcpu-list`."""
        text = output.decode("utf-8", errors="replace")
        rows: list[tuple[int, int, int]] = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                row_domain = int(fields[1])
                row_vcpu = int(fields[2])
                row_pcpu = int(fields[3])
            except ValueError:
                continue
            rows.append((row_domain, row_vcpu, row_pcpu))
        return rows

    @classmethod
    def _xl_vcpu_physical_cpu(
        cls, output: bytes, domain_id: int, vcpu_id: int
    ) -> int:
        """Extract the current pCPU from one row of `xl vcpu-list`."""
        for row_domain, row_vcpu, row_pcpu in cls._xl_vcpu_rows(output):
            if row_domain == domain_id and row_vcpu == vcpu_id:
                return row_pcpu
        raise RuntimeError(
            f"Could not find domain {domain_id} vCPU {vcpu_id} in xl vcpu-list"
        )

    def _apply_dynamic_vcpu_swap(
        self,
        control: SerialSession,
        prompt: str,
        settings: dict[str, Any],
    ) -> None:
        moving_domain = int(settings["moving_domain_id"])
        moving_vcpu = int(settings["moving_vcpu_id"])
        target_pcpu = int(settings["target_pcpu"])

        print("[XEN/INT] Reading the live vCPU placement before dynamic swap...")
        before = self._checked_serial_command(
            control,
            "xl vcpu-list",
            prompt,
            "XEN_VCPU_LIST_BEFORE",
            timeout=30,
        )
        source_pcpu = self._xl_vcpu_physical_cpu(
            before, moving_domain, moving_vcpu
        )
        if source_pcpu == target_pcpu:
            print(
                f"[XEN/INT] domain {moving_domain} vCPU {moving_vcpu} is "
                f"already on pCPU {target_pcpu}; no swap is required."
            )
            return

        target_occupants = [
            (domain_id, vcpu_id)
            for domain_id, vcpu_id, pcpu in self._xl_vcpu_rows(before)
            if pcpu == target_pcpu
            and (domain_id, vcpu_id) != (moving_domain, moving_vcpu)
        ]
        if len(target_occupants) != 1:
            raise RuntimeError(
                f"Dynamic vCPU swap requires exactly one vCPU currently on "
                f"pCPU {target_pcpu}; found {target_occupants}"
            )
        displaced_domain, displaced_vcpu = target_occupants[0]

        commands = (
            f"xl vcpu-pin {moving_domain} {moving_vcpu} {target_pcpu}",
            f"xl vcpu-pin {displaced_domain} {displaced_vcpu} {source_pcpu}",
        )
        print(
            f"[XEN/INT] Swapping pCPU placement: domain {moving_domain} "
            f"vCPU {moving_vcpu} pCPU{source_pcpu}->pCPU{target_pcpu}; "
            f"domain {displaced_domain} vCPU {displaced_vcpu} "
            f"pCPU{target_pcpu}->pCPU{source_pcpu}."
        )
        for index, command in enumerate(commands, start=1):
            self._checked_serial_command(
                control,
                command,
                prompt,
                f"XEN_DYNAMIC_PIN_{index}",
                timeout=30,
            )

        after = self._checked_serial_command(
            control,
            "xl vcpu-list",
            prompt,
            "XEN_VCPU_LIST_AFTER",
            timeout=30,
        )
        actual_moving = self._xl_vcpu_physical_cpu(
            after, moving_domain, moving_vcpu
        )
        actual_displaced = self._xl_vcpu_physical_cpu(
            after, displaced_domain, displaced_vcpu
        )
        if actual_moving != target_pcpu or actual_displaced != source_pcpu:
            raise RuntimeError(
                "Dynamic vCPU swap verification failed: "
                f"domain {moving_domain} vCPU {moving_vcpu}=pCPU"
                f"{actual_moving}, domain {displaced_domain} vCPU "
                f"{displaced_vcpu}=pCPU{actual_displaced}"
            )
        print("[XEN/INT] Dynamic vCPU swap verified successfully.")

    def _finish_xen_dom0less_trial(
        self, control: SerialSession, config: dict[str, Any]
    ) -> None:
        environment = config["environment"]
        # Xen's forced-preemption diagnostics can continuously overwrite the
        # visible Linux prompt.  Waiting for an anchored prompt in that stream
        # prevents reboot from ever being sent.  This sequence works both at a
        # login prompt ("root" logs in) and in an existing root shell ("root"
        # is merely a harmless failed command), then performs the required
        # manual reboot without depending on console-output quiescence.
        print(
            "[XEN/INT] RTOS completed; sending prompt-independent manual "
            "reboot sequence..."
        )
        self._send_prompt_independent_reboot(control, environment)
        self._base_runner(config)._reach_uboot(
            control, environment.get("uboot_prompt", r"=>\s*$")
        )

    @staticmethod
    def _send_prompt_independent_reboot(
        control: SerialSession, environment: dict[str, Any]
    ) -> None:
        """Reboot from either a login prompt or an existing root shell."""
        control.buffer.clear()
        control.send("\x03")
        control.sendline()
        time.sleep(float(environment.get("reboot_console_escape_seconds", 0.5)))
        control.sendline(environment.get("username", "root"))
        # Read the login transition instead of merely sleeping.  Some getty /
        # shell hand-offs consume the first byte sent at their boundary (the
        # observed `reboot -f` became `eboot -f`).  A blank line redraws the
        # shell, while leading spaces protect the actual command if one or more
        # bytes are still swallowed.
        control.drain(float(environment.get("reboot_console_login_seconds", 2)))
        control.sendline()
        control.drain(
            float(environment.get("reboot_console_prompt_settle_seconds", 0.5))
        )
        raw_reboot_command = environment.get(
            "int_latency_reboot_command", "/sbin/reboot -f"
        ).lstrip()
        reboot_command = "    " + raw_reboot_command
        print(f"[INT-LATENCY/REBOOT] COM10: {raw_reboot_command}")
        control.buffer.clear()
        control.sendline(reboot_command)

    @staticmethod
    @contextmanager
    def _background_serial_capture(session: SerialSession):
        stop = threading.Event()
        errors: list[BaseException] = []

        def capture() -> None:
            try:
                while not stop.is_set():
                    session._read_once()
            except BaseException as exc:
                errors.append(exc)
                stop.set()

        worker = threading.Thread(
            target=capture,
            name=f"int-latency-{session.settings.get('port', 'rtos')}",
            daemon=True,
        )
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError("RTOS serial capture thread did not stop")
            if errors:
                raise RuntimeError(
                    f"RTOS serial background capture failed: {errors[0]}"
                ) from errors[0]

    @staticmethod
    def _has_checked_rc(output: bytes, marker: str, rc: int) -> bool:
        """Recognize an RC marker even when async Xen output joins its line.

        Xen's null scheduler writes directly to the same UART while the shell
        prints the command result.  Consequently a successful marker can look
        like ``__INT_XEN_PIN_1_RC__=0(XEN) ...`` instead of occupying a full
        line.  Requiring the marker to begin at a line boundary still rejects
        the shell's command echo; only the trailing line boundary is relaxed
        for known asynchronous console prefixes.
        """
        expected = re.escape(f"{marker}={rc}".encode("utf-8"))
        boundary = rb"(?:^|\r*\n)"
        suffix = rb"(?=\r*(?:\n|$)|\(XEN\)|\[\s*\d)"
        return re.search(boundary + expected + suffix, output) is not None

    @staticmethod
    def _checked_serial_command(
        session: SerialSession,
        command: str,
        prompt: str,
        step: str,
        *,
        timeout: float,
        settle_seconds: float = 0,
    ) -> bytes:
        marker = f"__INT_{step}_RC__"
        print(f"[INT-LATENCY/ROOT] {command}")
        try:
            output = session.command(
                f"{command}; INT_RC=$?; "
                + (f"sleep {settle_seconds:g}; " if settle_seconds else "")
                + f"printf '\\n{marker}=%s\\n' \"$INT_RC\"",
                prompt,
                timeout + settle_seconds,
            )
        except SerialTimeout:
            # Jailhouse can emit a delayed kernel line after the shell prompt.
            # That displaces the anchored prompt even though the command has
            # completed successfully. The explicit complete RC line wins.
            output = bytes(session.buffer)
            if IntLatencyRunner._has_checked_rc(output, marker, 0):
                print(
                    f"[JAILHOUSE/INT] {step} returned RC=0; accepting it "
                    "despite delayed kernel output after the shell prompt."
                )
            else:
                raise
        if not IntLatencyRunner._has_checked_rc(output, marker, 0):
            # A prompt left in the UART/driver queue can arrive immediately
            # after command() clears its in-memory buffer.  In that case the
            # prompt matcher returns before the command echo and RC marker.
            # Keep reading for this command's unique marker instead of
            # reporting a false command failure.
            try:
                session.expect(
                    [
                        rf"(?:^|\r*\n){re.escape(marker)}=-?\d+"
                        rf"(?=\r*(?:\n|$)|\(XEN\)|\[\s*\d)"
                    ],
                    timeout,
                )
                output = bytes(session.buffer)
            except SerialTimeout:
                output = bytes(session.buffer)
        if not IntLatencyRunner._has_checked_rc(output, marker, 0):
            raise RuntimeError(
                f"int-latency step {step} failed: {command}"
            )
        return output

    @staticmethod
    def _prepare_rtos_capture(
        session: SerialSession,
        *,
        quiet_seconds: float = 1.0,
        maximum_drain_seconds: float = 5.0,
    ) -> None:
        """Create a clean UART boundary immediately before launching RTOS."""
        deadline = time.monotonic() + maximum_drain_seconds
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            raw = session._read_once()
            now = time.monotonic()
            if raw:
                quiet_since = now
            elif now - quiet_since >= quiet_seconds:
                break
        if session.port is not None:
            session.port.reset_input_buffer()
        session.buffer.clear()
        # Bytes drained above belong to a previous run. Keep them out of this
        # run's primary COM14 log as well as out of marker matching.
        session.truncate_transcript()

    @staticmethod
    def _completion_marker_is_eligible(
        *, elapsed_seconds: float, completion_not_before_seconds: float
    ) -> bool:
        return elapsed_seconds >= completion_not_before_seconds

    @staticmethod
    def _rtos_run_complete(
        completion_seen: bool,
        *,
        now: float,
        completion_seen_at: float | None,
        completion_settle_seconds: float,
    ) -> bool:
        return (
            completion_seen
            and completion_seen_at is not None
            and now - completion_seen_at >= completion_settle_seconds
        )

    @staticmethod
    def _monitor_dual_serial(
        control: SerialSession,
        rtos: SerialSession,
        *,
        completion_pattern: str,
        timeout: float,
        completion_settle_seconds: float,
        completion_not_before_seconds: float,
        started_at: float,
        retry_patterns: list[re.Pattern[bytes]],
        control_completion_pattern: str | None = None,
        stop_bare_autoboot: bool = False,
        uboot_prompt: str = r"=>\s*$",
        reset_markers: list[str] | None = None,
        login_prompt: str = r"login:\s*$",
    ) -> None:
        completion = re.compile(completion_pattern.encode(), re.I | re.M)
        control_completion = (
            re.compile(control_completion_pattern.encode(), re.I | re.M)
            if control_completion_pattern
            else None
        )
        prompt_re = re.compile(uboot_prompt.encode(), re.I | re.M)
        login_re = re.compile(login_prompt.encode(), re.I | re.M)
        reset_res = [
            re.compile(pattern.encode(), re.I | re.M)
            for pattern in (
                reset_markers
                or [r"DDR V\d", r"U-Boot SPL", r"U-Boot 20\d\d"]
            )
        ]
        deadline = started_at + timeout
        next_progress = time.monotonic() + 10
        completion_seen = False
        completion_seen_at: float | None = None
        control_completion_seen = control_completion is None
        reset_seen = False
        uboot_stopped = False
        while time.monotonic() < deadline:
            control._read_once()
            rtos._read_once()
            now = time.monotonic()
            if (
                not control_completion_seen
                and control_completion is not None
                and control_completion.search(control.buffer)
            ):
                control_completion_seen = True
                print(
                    "[XHYPASS/INT] COM10 reboot-ready marker detected; "
                    "reboot is now permitted after RTOS completion."
                )
            if not completion_seen:
                match = completion.search(rtos.buffer)
                if match:
                    elapsed = now - started_at
                    # Consume this occurrence so an early/stale marker cannot
                    # become eligible merely because host time later advances.
                    del rtos.buffer[: match.end()]
                    if IntLatencyRunner._completion_marker_is_eligible(
                        elapsed_seconds=elapsed,
                        completion_not_before_seconds=(
                            completion_not_before_seconds
                        ),
                    ):
                        completion_seen = True
                        completion_seen_at = now
                        print(
                            f"[RTOS] Fresh completion marker detected at "
                            f"{elapsed:.1f}s; continuing capture for "
                            f"{completion_settle_seconds:.1f}s."
                        )
                    else:
                        print(
                            f"[RTOS/STALE] Ignored completion marker at "
                            f"{elapsed:.1f}s; a fresh marker is required after "
                            f"{completion_not_before_seconds:.1f}s."
                        )

            # Bare QSemOS resets the board almost simultaneously with its
            # completion marker.  We must continue capturing COM14 for five
            # seconds, but cannot postpone stopping U-Boot until afterwards:
            # its zero-second countdown has already launched Linux by then.
            if stop_bare_autoboot and completion_seen and not uboot_stopped:
                if prompt_re.search(control.buffer):
                    uboot_stopped = True
                    print(
                        "[BARE/U-BOOT] Autoboot stopped while completing "
                        "post-marker COM14 capture."
                    )
                else:
                    if not reset_seen and any(
                        pattern.search(control.buffer) for pattern in reset_res
                    ):
                        reset_seen = True
                        print(
                            "[BARE/U-BOOT] Automatic reset detected; "
                            "intercepting U-Boot now while COM14 capture "
                            "continues."
                        )
                    if reset_seen:
                        control.send(b"\x03")
                    if login_re.search(control.buffer):
                        raise SerialTimeout(
                            "Missed the bare RTOS U-Boot stop window during "
                            "post-marker capture"
                        )

            if control_completion_seen and IntLatencyRunner._rtos_run_complete(
                completion_seen,
                now=now,
                completion_seen_at=completion_seen_at,
                completion_settle_seconds=completion_settle_seconds,
            ):
                if stop_bare_autoboot and reset_seen and not uboot_stopped:
                    # The marker-settle interval is complete, but remain in the
                    # dual-UART loop until U-Boot confirms that Ctrl+C landed.
                    continue
                print("[RTOS] Post-marker capture completed.")
                return
            for pattern in retry_patterns:
                if pattern.search(control.buffer):
                    text = pattern.pattern.decode("utf-8", errors="replace")
                    raise BootTransientError(f"transient U-Boot error: {text}")
            if now >= next_progress:
                remaining = max(0, int(deadline - now))
                elapsed = int(now - started_at)
                marker = "seen" if completion_seen else "pending"
                print(
                    f"[RTOS WAIT] elapsed {elapsed}s; marker {marker}; "
                    f"timeout remaining {remaining}s"
                )
                next_progress = now + 10
        tail = bytes(rtos.buffer[-1000:]).decode("utf-8", errors="replace")
        raise SerialTimeout(
            f"Timed out after {timeout:.1f}s waiting for RTOS completion marker "
            f"{completion_pattern!r}. COM14 tail:\n{tail}"
        )

    @staticmethod
    def _serial_log_name(settings: dict[str, Any], role: str) -> str:
        port = safe_name(str(settings.get("port", ""))).lower()
        return f"{port or role}.log"

    def _publish_rtos_log(
        self, output: Path, run_number: int, config: dict[str, Any]
    ) -> Path:
        source = output / self._serial_log_name(config["rtos_serial"], "rtos")
        if not source.is_file():
            raise RuntimeError(f"Completed run has no RTOS serial log: {source}")
        published = self._series_from_output(output) / f"rtos_run{run_number}.log"
        if published.exists():
            raise FileExistsError(f"RTOS result already exists: {published}")
        shutil.copy2(source, published)
        return published

    @staticmethod
    def _series_from_output(output: Path) -> Path:
        # <series>/_details/completed/run_NNN
        return output.parents[2]

    @staticmethod
    def _failed_output_path(output: Path, run_number: int) -> Path:
        failed_root = output.parent.parent / "failed"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = failed_root / f"attempt_{stamp}_run_{run_number:03d}"
        suffix = 1
        while candidate.exists():
            candidate = failed_root / (
                f"attempt_{stamp}_run_{run_number:03d}_{suffix}"
            )
            suffix += 1
        return candidate

    @staticmethod
    def _write_metadata(
        output: Path,
        config: dict[str, Any],
        run_number: int,
        status: str,
        *,
        error: str | None = None,
        recovery_error: str | None = None,
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        path = output / "metadata.json"
        created_at = now
        if path.is_file():
            try:
                created_at = json.loads(path.read_text(encoding="utf-8")).get(
                    "created_at", now
                )
            except (OSError, json.JSONDecodeError):
                pass
        payload = {
            "status": status,
            "created_at": created_at,
            "updated_at": now,
            "platform": config["platform_name"],
            "experiment": "int-latency",
            "environment": config["environment_name"],
            "condition": config["experiment"]["condition"],
            "run": run_number,
            "duration_seconds": int(config["experiment"]["duration_seconds"]),
            "completion_pattern": config["experiment"]["completion_pattern"],
            "completion_not_before_ratio": float(
                config["experiment"].get("completion_not_before_ratio", 0.9)
            ),
            "completion_not_before_seconds": (
                float(config["experiment"]["duration_seconds"])
                * float(
                    config["experiment"].get(
                        "completion_not_before_ratio", 0.9
                    )
                )
            ),
            "completion_settle_seconds": float(
                config["experiment"].get("completion_settle_seconds", 5)
            ),
            "control_serial": copy.deepcopy(config["control_serial"]),
            "rtos_serial": copy.deepcopy(config["rtos_serial"]),
            "boot_files": copy.deepcopy(
                config["environment"].get("int_latency_boot_files", [])
            ),
        }
        payload["control_serial"].pop("show_output", None)
        payload["rtos_serial"].pop("show_output", None)
        if error:
            payload["error"] = error
        if recovery_error:
            payload["recovery_error"] = recovery_error
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
