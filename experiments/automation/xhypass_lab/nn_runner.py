from __future__ import annotations

import base64
import csv
import json
import re
import shlex
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import ExperimentRunner, condition_name, safe_name
from .serial_session import SerialSession, SerialTimeout
from .ssh_session import SSHSession


class NNExperimentRunner(ExperimentRunner):
    """Run the dual-TFLite HTTP workload and collect its complete result set."""

    def plan(self, runs: int, reboot_policy: str) -> list[Path]:
        series = (
            self.data_root
            / safe_name(self.cfg["experiment_name"])
            / safe_name(self.cfg["environment_name"])
            / condition_name(self.cfg)
        )
        existing = [
            int(match.group(1))
            for path in series.glob("nn_results_run*.tar.gz")
            if (match := re.fullmatch(r"nn_results_run(\d+)\.tar\.gz", path.name))
        ] if series.exists() else []
        first = max(existing, default=0) + 1
        completed = series / "_details" / "completed"
        return [completed / f"run_{index:03d}" for index in range(first, first + runs)]

    def _create_output_directory(self, output: Path, index: int) -> None:
        """Quarantine an interrupted NN attempt before reusing its run number.

        A hard stop can leave ``_details/completed/run_NNN`` behind before the
        result archive is published.  Published archives drive successful-run
        numbering, so that stale directory would otherwise block every resume.
        Preserve it under ``failed`` and retry the same successful-run number.
        """
        if output.exists():
            match = re.fullmatch(r"run_(\d+)", output.name)
            run_number = int(match.group(1)) if match else index
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
                "[RESUME] Preserved stale NN attempt before reusing "
                f"{output.name}: {failed.resolve()}"
            )
        super()._create_output_directory(output, index)

    @staticmethod
    def _publish_histogram(output: Path) -> Path:
        match = re.fullmatch(r"run_(\d+)", output.name)
        if not match:
            raise RuntimeError(f"Unexpected NN run directory name: {output.name}")
        source = output / "nn_results.tar.gz"
        if not source.is_file():
            raise RuntimeError(f"Completed NN run has no result archive: {source}")
        published = (
            output.parent.parent.parent
            / f"nn_results_run{int(match.group(1))}.tar.gz"
        )
        if published.exists():
            raise FileExistsError(f"NN result already exists: {published}")
        shutil.copy2(source, published)
        return published

    def _run_experiment(self, session: SerialSession, output: Path) -> None:
        environment = self.cfg["environment_name"]
        if environment == "bare":
            self._run_nn_bare(session, output)
        elif environment == "jailhouse":
            self._run_nn_jailhouse(session, output)
        elif environment in (
            "xen_credit2",
            "xen_credit2_WFX",
            "xen_null",
            "xen_null_WFX",
            "XHyPass",
        ):
            self._run_nn_xen_credit2(session, output)
        else:
            raise NotImplementedError(
                f"The NN runner does not support {environment} yet"
            )

    def _run_nn_bare(self, session: SerialSession, output: Path) -> None:
        exp = self.cfg["experiment"]
        prompt = r"__XHYPASS_PROMPT__#\s*$"
        remote_root = str(exp.get("remote_root", "~/NN"))
        remote_bare = f"{remote_root}/bare"
        duration = int(exp["duration_seconds"])
        total_cases = len(exp["profiles"])
        template = Path(exp["local_run_script"])
        if not template.is_file():
            raise FileNotFoundError(f"NN run script not found: {template}")

        preflight = session.command(
            f"cd {remote_bare} && test -x ../loadgen && "
            "test -x ../benchmark_http_infer && test -x ../cyclictest && "
            "test -f ../inception_v4_299_quant.tflite && "
            "test -f ../mnasnet_1.3_224.tflite && "
            "command -v base64 >/dev/null && command -v tar >/dev/null && "
            "command -v taskset >/dev/null; echo __NN_PREFLIGHT_RC__=$?",
            prompt,
            20,
        )
        if not self._has_output_line(preflight, "__NN_PREFLIGHT_RC__=0"):
            raise RuntimeError(
                "NN preflight failed: remote NN binaries/models or required "
                "base64/tar/taskset command is missing"
            )

        remote_script = f"/tmp/xy_nn_run_all_{int(time.time())}.sh"
        encoded_script = base64.b64encode(template.read_bytes()).decode("ascii")
        uploaded = session.command(
            f"echo {encoded_script} | base64 -d > {remote_script}; "
            f"chmod +x {remote_script}; echo __NN_UPLOAD_RC__=$?",
            prompt,
            30,
        )
        if not self._has_output_line(uploaded, "__NN_UPLOAD_RC__=0"):
            raise RuntimeError("Could not upload the hardened NN run script")
        shutil.copy2(template, output / "run_all.executed.sh")

        assignments = self._environment_assignments(exp)
        command = (
            f"cd {remote_bare} && {assignments} bash {remote_script}; "
            "XY_NN_RC=$?; echo __NN_DONE_RC__=$XY_NN_RC"
        )
        grace = int(exp.get("grace_seconds", 300))
        print(
            f"[NN] Running {total_cases} load levels x {duration}s; "
            f"expected workload time is about {total_cases * duration}s"
        )
        raw = session.command(
            command,
            prompt,
            total_cases * duration + grace,
        )
        done = re.search(rb"__NN_DONE_RC__=(\d+)", raw)
        run_rc = int(done.group(1)) if done else -1

        remote_archive = f"/tmp/xy_nn_results_{int(time.time())}.tar.gz"
        archived = session.command(
            f"cd {remote_bare} && tar -czf {remote_archive} results; "
            "echo __NN_ARCHIVE_RC__=$?",
            prompt,
            60,
        )
        if not self._has_output_line(archived, "__NN_ARCHIVE_RC__=0"):
            raise RuntimeError("Could not archive NN results on the board")

        local_archive = output / "nn_results.tar.gz"
        self._download_remote_chunked(
            session,
            remote_archive,
            local_archive,
            prompt,
            int(exp.get("transfer_chunk_bytes", 262_144)),
        )
        results_dir = output / "results"
        with tarfile.open(local_archive, "r:gz") as archive:
            archive.extractall(output, filter="data")
        summary = self._summarize_results(results_dir, exp)
        (output / "nn_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if exp.get("cleanup_remote", True):
            session.command(
                f"rm -f {remote_script} {remote_archive}", prompt, 15
            )
        if run_rc != 0:
            raise RuntimeError(
                f"NN run_all.sh failed with rc={run_rc}; partial results were collected"
            )
        if summary["http_inference_failures"]:
            raise RuntimeError(
                "NN workload recorded failed HTTP inference requests; results "
                "were collected for diagnosis"
            )
        print(
            f"[RESULT] Collected {summary['csv_files']} CSV files, "
            f"{summary['cyclictest_files']} cyclictest files and server logs"
        )

    def _run_nn_jailhouse(self, session: SerialSession, output: Path) -> None:
        env = self.cfg["environment"]
        exp = self.cfg["experiment"]
        root_prompt = r"__XHYPASS_PROMPT__#\s*$"
        nonroot_prompt = r"__JH_NONROOT_PROMPT__#\s*$"
        ssh_log = output / "rootcell_ssh.log"
        duration = int(exp["duration_seconds"])
        total_timeout = len(exp["profiles"]) * duration + int(
            exp.get("grace_seconds", 300)
        )

        self._load_jailhouse_module(session, env, root_prompt)
        session.command(
            f"ifconfig eth0 {env['rootcell_ip']}", root_prompt, 20
        )
        ssh = SSHSession(env["ssh"])
        try:
            print("[NN/JAILHOUSE] Connecting to the root cell over SSH...")
            ssh.connect()
            root_script = Path(exp["local_rootcell_script"])
            nonroot_script = Path(exp["local_nonroot_script"])
            remote_root_script = f"/tmp/xy_nn_vm1_{int(time.time())}.sh"
            remote_nonroot_script = "/root/NN/jailhouse/run_vm2_automated.sh"
            self._upload_script_ssh(
                ssh, root_script, remote_root_script, ssh_log
            )
            self._upload_script_ssh(
                ssh, nonroot_script, remote_nonroot_script, ssh_log
            )
            shutil.copy2(root_script, output / "run_vm1.executed.sh")
            shutil.copy2(nonroot_script, output / "run_vm2.executed.sh")

            ssh.run(
                env["enable_command"], timeout=60, log_path=ssh_log, check=True
            )
            session.buffer.clear()
            ssh.run(
                exp["jailhouse_linux_command"],
                timeout=60,
                log_path=ssh_log,
                check=True,
            )
            self._login_nonroot_cell(session, env)
            self._jailhouse_initialized = True

            self._prepare_nonroot_files(
                session, ssh, exp, nonroot_prompt, ssh_log
            )
            root_assignments = self._jailhouse_assignments(exp, "rootcell")
            nonroot_assignments = self._jailhouse_assignments(exp, "nonroot")
            root_command = (
                f"cd ~/NN/jailhouse && {root_assignments} "
                f"sh {remote_root_script}"
            )
            nonroot_command = (
                f"cd ~/NN/jailhouse && {nonroot_assignments} "
                f"sh ./run_vm2_automated.sh; "
                "XY_NN_NONROOT_RC=$?; "
                "echo __NN_NONROOT_DONE_RC__=$XY_NN_NONROOT_RC"
            )
            print(
                f"[NN/JAILHOUSE] Starting both cells: 3 x {duration}s per cell"
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                root_future = executor.submit(
                    ssh.run,
                    root_command,
                    timeout=total_timeout,
                    log_path=ssh_log,
                    check=False,
                )
                nonroot_raw = session.command(
                    nonroot_command, nonroot_prompt, total_timeout
                )
                root_rc, _, _ = root_future.result()
            nonroot_done = re.search(
                rb"__NN_NONROOT_DONE_RC__=(\d+)", nonroot_raw
            )
            nonroot_rc = int(nonroot_done.group(1)) if nonroot_done else -1

            self._collect_jailhouse_results(
                session,
                ssh,
                output,
                exp,
                nonroot_prompt,
                ssh_log,
            )
            summary = self._summarize_results(output / "results", exp)
            (output / "nn_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if root_rc != 0 or nonroot_rc != 0:
                raise RuntimeError(
                    "Jailhouse NN scripts failed: "
                    f"rootcell rc={root_rc}, nonroot rc={nonroot_rc}; "
                    "available results were collected"
                )
            if summary["http_inference_failures"]:
                raise RuntimeError(
                    "Jailhouse NN workload recorded failed HTTP inference requests"
                )
            print(
                f"[RESULT] Jailhouse NN: {summary['http_requests']} requests, "
                f"{summary['http_inference_failures']} inference failures"
            )
            if exp.get("cleanup_remote", True):
                ssh.run(
                    f"rm -f {remote_root_script} /tmp/xy_nn_root_results.tar.gz",
                    timeout=20,
                    log_path=ssh_log,
                    check=False,
                )
                session.command(
                    "rm -f /tmp/xy_nn_nonroot_results.tar.gz",
                    nonroot_prompt,
                    15,
                )
        finally:
            ssh.close()

    def _run_nn_xen_credit2(self, session: SerialSession, output: Path) -> None:
        env = self.cfg["environment"]
        exp = self.cfg["experiment"]
        dom0_prompt = r"__XHYPASS_PROMPT__#\s*$"
        dom1_prompt = r"__XEN_DOM1_PROMPT__#\s*$"
        ssh_log = output / "dom0_ssh.log"
        dom1_log = output / "dom1_ssh.log"
        duration = int(exp["duration_seconds"])
        total_timeout = len(exp["profiles"]) * duration + int(
            exp.get("grace_seconds", 300)
        )

        self._ensure_xen_control_plane(session, env, dom0_prompt)
        print("[NN/XEN] Initializing bridge and dom1 on COM10...")
        dom1_pin = str(env.get("dom1_pin_command", "")).strip()
        dom1_pin_during_init = bool(env.get("dom1_pin_during_init", False))
        if dom1_pin_during_init and dom1_pin not in {
            str(command).strip() for command in env["dom0_init_commands"]
        }:
            raise ValueError(
                "dom1_pin_during_init is enabled, but dom1_pin_command is "
                "missing from dom0_init_commands"
            )
        post_delays = env.get("dom0_init_post_delays_seconds", {})
        for command in env["dom0_init_commands"]:
            post_delay = float(post_delays.get(command, 0))
            raw = session.command(
                f"{command}; XY_XEN_INIT_RC=$?; "
                "echo __NN_XEN_INIT_RC__=$XY_XEN_INIT_RC",
                r"__XHYPASS_PROMPT__#" if post_delay else dom0_prompt,
                float(env.get("init_command_timeout", 60)),
            )
            if not self._has_output_line(raw, "__NN_XEN_INIT_RC__=0"):
                raise RuntimeError(
                    f"Xen dom0 initialization command failed: {command}"
                )
            if post_delay:
                session.drain(post_delay)
        bridge_ready = session.command(
            f"ifconfig {exp.get('dom0_bridge_interface', 'xenbr0')} "
            f"{exp.get('dom0_bridge_ip', '202.197.67.50')} netmask "
            "255.255.255.0 up; echo __NN_DOM0_BRIDGE_RC__=$?",
            dom0_prompt,
            20,
        )
        if not self._has_output_line(bridge_ready, "__NN_DOM0_BRIDGE_RC__=0"):
            raise RuntimeError("Could not configure dom0 xenbr0 before SSH")

        ssh = SSHSession(env["ssh"])
        dom1_ssh: SSHSession | None = None
        dom0_rto_loaded = False
        dom1_rto_loaded = False
        is_xhypass = self.cfg["environment_name"] == "XHyPass"
        module_cpu = int(exp.get("xhypass_module_cpu", 6))
        module_file = str(
            env.get("xhypass_module_file", "interrupt_passthrough.ko")
        )
        module_load_args = str(
            env.get("xhypass_module_load_args", "")
        ).strip()
        try:
            print("[NN/XEN] Connecting to dom0 over SSH...")
            ssh.connect()
            ssh.run(
                f"ifconfig {exp.get('dom0_bridge_interface', 'xenbr0')} "
                f"{exp.get('dom0_bridge_ip', '202.197.67.50')}",
                timeout=20,
                log_path=ssh_log,
                check=True,
            )
            if env.get("dom0_pin_command"):
                ssh.run(
                    env["dom0_pin_command"], timeout=30,
                    log_path=ssh_log, check=True,
                )
            if env.get("dom1_pin_command") and not dom1_pin_during_init:
                ssh.run(
                    env["dom1_pin_command"], timeout=30,
                    log_path=ssh_log, check=True,
                )
            for command in env.get("pin_verify_commands", []):
                ssh.run(command, timeout=30, log_path=ssh_log, check=True)

            self._start_xen_dom1_with_retry(ssh, env, exp, ssh_log)
            dom1_settings = dict(env["ssh"])
            dom1_settings.update(
                {
                    "host": exp.get("dom1_ip", "202.197.67.51"),
                    "connect_timeout": env.get("dom1_boot_timeout", 180),
                }
            )
            dom1_ssh = SSHSession(dom1_settings)
            print(
                f"[NN/XEN] Waiting for dom1 SSH at "
                f"{dom1_settings['host']}:{dom1_settings.get('port', 22)}..."
            )
            dom1_ssh.connect()
            settle_seconds = float(env.get("post_dom1_login_delay_seconds", 10))
            if settle_seconds:
                print(
                    f"[NN/XEN] dom1 SSH login complete; waiting "
                    f"{settle_seconds:g}s for stabilization..."
                )
                time.sleep(settle_seconds)
            self._xen_initialized = True

            if is_xhypass:
                module_settle_seconds = float(
                    env.get("post_module_load_delay_seconds", 0)
                )
                self._set_xhypass_nn_module(
                    ssh, ssh_log, "dom0", module_cpu, load=True,
                    settle_seconds=module_settle_seconds,
                    module_file=module_file,
                    load_args=module_load_args,
                )
                dom0_rto_loaded = True
                self._set_xhypass_nn_module(
                    dom1_ssh, dom1_log, "dom1", module_cpu, load=True,
                    settle_seconds=module_settle_seconds,
                    module_file=module_file,
                    load_args=module_load_args,
                )
                dom1_rto_loaded = True

            dom0_script = Path(exp["local_dom0_script"])
            dom1_script = Path(exp["local_dom1_script"])
            remote_dir = str(exp.get("remote_xen_dir", "~/NN/xen_credit2"))
            remote_dom0_script = f"{remote_dir}/run_vm1.sh"
            remote_dom1_script = f"{remote_dir}/run_vm2.sh"
            ssh.run(
                f"mkdir -p {remote_dir}", timeout=20,
                log_path=ssh_log, check=True,
            )
            self._upload_script_ssh(
                ssh, dom0_script, remote_dom0_script, ssh_log
            )
            dom1_ssh.run(
                f"mkdir -p {remote_dir}", timeout=20,
                log_path=dom1_log, check=True,
            )
            self._upload_script_ssh(
                dom1_ssh, dom1_script, remote_dom1_script, dom1_log
            )
            shutil.copy2(dom0_script, output / "run_vm1.executed.sh")
            shutil.copy2(dom1_script, output / "run_vm2.executed.sh")

            dom0_base = self._resolve_xen_nn_base(
                ssh, remote_dir, "inception_v4_299_quant.tflite", ssh_log, "dom0"
            )
            dom1_base = self._resolve_xen_nn_base(
                dom1_ssh, remote_dir, "mnasnet_1.3_224.tflite", dom1_log, "dom1"
            )

            dom0_assignments = (
                self._xen_assignments(exp, "dom0")
                + f" BASE_DIR={shlex.quote(dom0_base)}"
            )
            dom1_assignments = (
                self._xen_assignments(exp, "dom1")
                + f" BASE_DIR={shlex.quote(dom1_base)}"
            )
            sync_token = f"{time.time_ns():x}"
            sync_timeout = int(exp.get("domain_sync_timeout_seconds", 180))
            sync_values = (
                f" SYNC_TOKEN={shlex.quote(sync_token)}"
                f" SYNC_TIMEOUT_SECONDS={sync_timeout}"
            )
            dom0_assignments += sync_values
            dom1_assignments += sync_values
            dom0_command = (
                f"cd {remote_dir} && {dom0_assignments} sh ./run_vm1.sh"
            )
            dom1_command = f"cd {remote_dir} && {dom1_assignments} sh ./run_vm2.sh"
            print(
                f"[NN/XEN] Starting dom0 and dom1: 3 x {duration}s per domain"
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                dom0_future = executor.submit(
                    ssh.run,
                    dom0_command,
                    timeout=total_timeout,
                    log_path=ssh_log,
                    check=False,
                )
                dom1_future = executor.submit(
                    dom1_ssh.run,
                    dom1_command,
                    timeout=total_timeout,
                    log_path=dom1_log,
                    check=False,
                )
                self._monitor_xen_dom1_network(
                    session,
                    ssh,
                    dom1_ssh,
                    env,
                    exp,
                    ssh_log,
                    (dom0_future, dom1_future),
                    sync_token=sync_token,
                    sync_levels=[
                        str(profile["name"]) for profile in exp["profiles"]
                    ],
                    sync_report_path=output / "domain_alignment.json",
                )
                dom0_rc, _, _ = dom0_future.result()
                dom1_rc, _, _ = dom1_future.result()

            # A long-running guest command can finish just as its SSH transport
            # is torn down.  Result collection opens new channels, so never
            # assume the workload connection is reusable here.
            self._refresh_xen_result_connections(
                session, ssh, dom1_ssh, env, exp, ssh_log
            )
            dom0_complete = self._xen_run_complete(
                ssh, remote_dir, "vm1", ssh_log
            )
            dom1_complete = self._xen_run_complete(
                dom1_ssh, remote_dir, "vm2", dom1_log
            )
            if dom0_complete:
                dom0_rc = 0
            if dom1_complete:
                dom1_rc = 0

            self._collect_xen_results(
                ssh, dom1_ssh, output, ssh_log, dom1_log, remote_dir
            )
            self._finalize_domain_alignment(
                output,
                [str(profile["name"]) for profile in exp["profiles"]],
            )
            summary = self._summarize_results(output / "results", exp)
            (output / "nn_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if dom0_rc != 0 or dom1_rc != 0:
                raise RuntimeError(
                    f"Xen NN scripts failed: dom0 rc={dom0_rc}, "
                    f"dom1 rc={dom1_rc}; available results were collected"
                )
            if summary["http_inference_failures"]:
                raise RuntimeError("Xen NN workload recorded failed HTTP requests")
            if is_xhypass:
                try:
                    self._set_xhypass_nn_module(
                        dom1_ssh, dom1_log, "dom1", module_cpu, load=False
                    )
                finally:
                    dom1_rto_loaded = False
                try:
                    self._set_xhypass_nn_module(
                        ssh, ssh_log, "dom0", module_cpu, load=False
                    )
                finally:
                    dom0_rto_loaded = False
            print(
                f"[RESULT] Xen-credit2 NN: {summary['http_requests']} requests, "
                f"{summary['http_inference_failures']} inference failures"
            )
        finally:
            if dom1_rto_loaded and dom1_ssh is not None:
                try:
                    self._set_xhypass_nn_module(
                        dom1_ssh, dom1_log, "dom1", module_cpu, load=False
                    )
                except Exception as exc:
                    print(f"[WARN] Could not clean up XHyPass dom1 module: {exc}")
            if dom0_rto_loaded:
                try:
                    self._set_xhypass_nn_module(
                        ssh, ssh_log, "dom0", module_cpu, load=False
                    )
                except Exception as exc:
                    print(f"[WARN] Could not clean up XHyPass dom0 module: {exc}")
            if dom1_ssh is not None:
                dom1_ssh.close()
            ssh.close()

    @staticmethod
    def _set_xhypass_nn_module(
        ssh: SSHSession,
        log_path: Path,
        domain: str,
        cpu: int,
        *,
        load: bool,
        settle_seconds: float = 0,
        module_file: str = "interrupt_passthrough.ko",
        load_args: str = "",
    ) -> None:
        if load:
            action = f"insmod {module_file}"
            if load_args:
                action += f" {load_args}"
        else:
            action = "rmmod interrupt_passthrough.ko"
        if not ssh.is_active():
            print(f"[NN/XHYPASS] Reconnecting to {domain} before {action}...")
            ssh.reconnect()
        print(
            f"[NN/XHYPASS] {domain}: taskset -c {cpu} "
            f"{action}"
        )
        ssh.run(
            f"cd ~ && taskset -c {cpu} {action}",
            timeout=30,
            log_path=log_path,
            check=True,
        )
        if load and settle_seconds > 0:
            print(
                f"[NN/XHYPASS] {domain}: module loaded; waiting "
                f"{settle_seconds:g}s before continuing..."
            )
            time.sleep(settle_seconds)

    def _ensure_xen_control_plane(
        self, session: SerialSession, env: dict[str, Any], prompt: str
    ) -> None:
        attempts = int(env.get("xen_boot_health_attempts", 3))
        ready_timeout = float(env.get("xen_control_ready_timeout_seconds", 20))
        for attempt in range(1, attempts + 1):
            deadline = time.monotonic() + ready_timeout
            while time.monotonic() < deadline:
                raw = session.command(
                    "if xl list >/dev/null 2>&1; then "
                    "echo __NN_XEN_CONTROL_READY__; fi; "
                    "printf '__NN_XEN_KERNEL__='; uname -r",
                    prompt,
                    15,
                )
                if self._has_output_line(raw, "__NN_XEN_CONTROL_READY__"):
                    print(
                        f"[NN/XEN] Xen control plane healthy "
                        f"(boot attempt {attempt}/{attempts})."
                    )
                    return
                time.sleep(2)
            if attempt >= attempts:
                break

            print(
                f"[NN/XEN] Xen control plane unavailable after boot "
                f"attempt {attempt}/{attempts}; rebooting and retrying..."
            )
            session.buffer.clear()
            session.sendline(env.get("reboot_command", "reboot"))
            self._reach_uboot(session, env.get("uboot_prompt", r"=>\s*$"))
            self._sync_local_boot_files()
            session.buffer.clear()
            self._boot_command_until_login(
                session,
                env,
                env.get("login_prompt", r"tl3588 login:\s*$"),
                env.get("uboot_prompt", r"=>\s*$"),
            )
            session.buffer.clear()
            session.sendline(env.get("username", "root"))
            session.expect(
                env.get(
                    "shell_prompts",
                    [r"root@[^\r\n]*#\s*$", r"#\s*$"],
                ),
                float(self.cfg["timeouts"].get("login", 30)),
            )
            session.command(
                "export PS1='__XHYPASS_PROMPT__# '", prompt, 10
            )
        raise RuntimeError(
            f"Xen control plane failed after {attempts} boot attempts: "
            "xl list is unavailable (privileged Xen interface missing)"
        )

    def _configure_xen_dom1_network(
        self, session: SerialSession, exp: dict[str, Any], prompt: str
    ) -> None:
        interface = str(exp.get("dom1_interface", "enX0"))
        ip = str(exp.get("dom1_ip", "202.197.67.51"))
        netmask = str(exp.get("dom1_netmask", "255.255.255.0"))
        settle = float(exp.get("dom1_network_settle_seconds", 5))
        configured = session.command(
            "systemctl mask --runtime --now NetworkManager.service "
            "NetworkManager-wait-online.service network-manager.service "
            "networking.service systemd-networkd.service "
            "systemd-networkd-wait-online.service >/dev/null 2>&1 || true; "
            f"pkill -f '[u]dhcpc.*{interface}' >/dev/null 2>&1 || true; "
            f"ifconfig {interface} {ip} netmask {netmask} up; "
            f"ip -4 -o addr show dev {interface}; "
            f"ip -4 -o addr show dev {interface} | grep -q ' {ip}/24 '; "
            "echo __NN_DOM1_NET_RC__=$?",
            prompt,
            30,
        )
        if not self._has_output_line(configured, "__NN_DOM1_NET_RC__=0"):
            raise RuntimeError(
                f"Could not configure dom1 {interface} as {ip}/24"
            )
        if settle:
            print(
                f"[NN/XEN] Waiting {settle:g}s to verify that dom1 IPv4 "
                "is not removed by background network services..."
            )
            session.drain(settle)
        verified = session.command(
            f"ip -4 -o addr show dev {interface}; "
            f"ip -4 -o addr show dev {interface} | grep -q ' {ip}/24 '; "
            "echo __NN_DOM1_NET_STABLE_RC__=$?",
            prompt,
            20,
        )
        if not self._has_output_line(
            verified, "__NN_DOM1_NET_STABLE_RC__=0"
        ):
            raise RuntimeError(
                f"dom1 {interface} lost {ip}/24 during network stabilization"
            )
        print(f"[NN/XEN] dom1 network stable: {interface}={ip}/24")

    def _monitor_xen_dom1_network(
        self,
        session: SerialSession,
        ssh: SSHSession,
        dom1_ssh: SSHSession,
        env: dict[str, Any],
        exp: dict[str, Any],
        ssh_log: Path,
        futures: tuple[Any, Any],
        *,
        sync_token: str | None = None,
        sync_levels: list[str] | None = None,
        sync_report_path: Path | None = None,
    ) -> None:
        """Coordinate per-level starts and repair dom1 networking while jobs run."""
        interval = float(
            exp.get("dom1_network_monitor_interval_seconds", 10)
        )
        failure_limit = int(
            exp.get("dom1_network_monitor_failures", 2)
        )
        duration = int(exp["duration_seconds"])
        failures = 0
        levels = list(sync_levels or [])
        sync_dir = f"/tmp/xy_nn_sync_{sync_token}" if sync_token else ""
        next_level = 0
        alignment: dict[str, Any] = {
            "method": "two-domain-ready-go-barrier",
            "token": sync_token,
            "timestamp_note": (
                "Observed start/finish values come from each domain's wall "
                "clock; compare them only when dom0 and dom1 clocks are synced."
            ),
            "levels": [],
        }
        next_network_check = time.monotonic() + interval
        next_sync_poll = time.monotonic()
        while not all(future.done() for future in futures):
            now = time.monotonic()
            if next_level < len(levels) and now >= next_sync_poll:
                level = levels[next_level]
                dom0_ready = self._remote_marker_exists(
                    ssh, f"{sync_dir}/vm1.{level}.ready"
                )
                dom1_ready = self._remote_marker_exists(
                    dom1_ssh, f"{sync_dir}/vm2.{level}.ready"
                )
                if dom0_ready and dom1_ready:
                    go_path = f"{sync_dir}/{level}.go"
                    print(
                        f"[NN/SYNC] Both domains ready for {level}; "
                        "releasing them together..."
                    )
                    with ThreadPoolExecutor(max_workers=2) as release_pool:
                        releases = [
                            release_pool.submit(
                                self._publish_sync_marker, side, go_path
                            )
                            for side in (ssh, dom1_ssh)
                        ]
                        released = [future.result() for future in releases]
                    release_gap_ms = abs(released[0] - released[1]) / 1_000_000
                    alignment["levels"].append(
                        {
                            "name": level,
                            "coordinator_released_at": datetime.now()
                            .astimezone()
                            .isoformat(timespec="milliseconds"),
                            "go_publish_gap_ms": round(release_gap_ms, 3),
                        }
                    )
                    if sync_report_path is not None:
                        sync_report_path.write_text(
                            json.dumps(alignment, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    print(
                        f"[NN/SYNC] {level} released; coordinator publish "
                        f"gap={release_gap_ms:.1f} ms"
                    )
                    next_level += 1
                    # A side cannot reach the next level until the current
                    # DURATION has elapsed.  Avoid thousands of needless SFTP
                    # probes (and sshd units in the small dom1 ramdisk).
                    next_sync_poll = time.monotonic() + max(1, duration - 5)
                    continue
                next_sync_poll = time.monotonic() + 0.5

            wait(futures, timeout=min(0.2, interval))
            failed_future = next(
                (
                    future
                    for future in futures
                    if future.done() and future.exception() is not None
                ),
                None,
            )
            if failed_future is not None:
                # A foreground SSH workload cannot be resumed after its
                # channel exits.  Stop monitoring immediately instead of
                # repeatedly repairing an IP for an already-invalid run.
                failure = failed_future.exception()
                assert failure is not None
                raise failure
            if all(future.done() for future in futures):
                break
            if time.monotonic() < next_network_check:
                continue
            next_network_check = time.monotonic() + interval
            # Reuse the established transport state.  Opening a fresh TCP
            # probe every interval spawns needless per-connection sshd units
            # in the small ramdisk guest.
            if dom1_ssh.is_active():
                failures = 0
                continue
            failures += 1
            print(
                f"[NN/XEN] dom1 SSH probe failed "
                f"({failures}/{failure_limit})."
            )
            if failures < failure_limit:
                continue
            if self._xen_dom1_ip_reachable(ssh, exp, ssh_log):
                print(
                    "[NN/XEN] dom1 IP is reachable from dom0; the workload "
                    "SSH transport alone was lost. No IP repair is needed."
                )
                failures = 0
                continue
            print(
                "[NN/XEN] dom1 IP appears lost; repairing through "
                "the COM10 dom1 console..."
            )
            self._repair_xen_dom1_network_via_serial_console(
                session, env, exp
            )
            failures = 0

        if next_level < len(levels):
            alignment["incomplete_levels"] = levels[next_level:]
            if sync_report_path is not None:
                sync_report_path.write_text(
                    json.dumps(alignment, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    @staticmethod
    def _remote_marker_exists(ssh: SSHSession, remote: str) -> bool:
        if not ssh.is_active():
            return False
        try:
            return ssh.path_exists(remote)
        except Exception:
            return False

    @staticmethod
    def _publish_sync_marker(ssh: SSHSession, remote: str) -> int:
        published_at = time.perf_counter_ns()
        ssh.put_bytes(remote, b"go\n")
        return published_at

    @staticmethod
    def _finalize_domain_alignment(output: Path, levels: list[str]) -> None:
        """Add observed two-domain start/finish deltas to the sync report."""
        report_path = output / "domain_alignment.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {
                "method": "two-domain-ready-go-barrier",
                "levels": [],
            }
        by_name = {
            str(item.get("name")): item
            for item in report.get("levels", [])
            if isinstance(item, dict)
        }
        results = output / "results"

        def timestamp(kind: str, level: str, role: str) -> float | None:
            path = results / f"sync_{kind}_{level}_{role}.txt"
            try:
                return float(path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                return None

        for level in levels:
            item = by_name.setdefault(level, {"name": level})
            start_vm1 = timestamp("start", level, "vm1")
            start_vm2 = timestamp("start", level, "vm2")
            finish_vm1 = timestamp("finish", level, "vm1")
            finish_vm2 = timestamp("finish", level, "vm2")
            if start_vm1 is not None and start_vm2 is not None:
                item["observed_start_gap_ms"] = round(
                    abs(start_vm1 - start_vm2) * 1000, 3
                )
            if finish_vm1 is not None and finish_vm2 is not None:
                item["observed_finish_gap_ms"] = round(
                    abs(finish_vm1 - finish_vm2) * 1000, 3
                )
            if start_vm1 is not None and finish_vm1 is not None:
                item["vm1_elapsed_seconds"] = round(
                    finish_vm1 - start_vm1, 6
                )
            if start_vm2 is not None and finish_vm2 is not None:
                item["vm2_elapsed_seconds"] = round(
                    finish_vm2 - start_vm2, 6
                )
        report["levels"] = [by_name[level] for level in levels]
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        report_path.write_text(rendered, encoding="utf-8")
        (results / "domain_alignment.json").write_text(
            rendered, encoding="utf-8"
        )
        # _collect_xen_results creates the transport archive before alignment
        # can be finalized. Repack it so the published nn_results_runN archive
        # also carries the synchronization evidence.
        with tarfile.open(output / "nn_results.tar.gz", "w:gz") as archive:
            archive.add(results, arcname="results")

    @staticmethod
    def _xen_dom1_ip_reachable(
        ssh: SSHSession,
        exp: dict[str, Any],
        log_path: Path,
    ) -> bool:
        ip = str(exp.get("dom1_ip", "202.197.67.51"))
        timeout = max(
            1, int(float(exp.get("dom1_network_probe_timeout_seconds", 2)))
        )
        rc, _, _ = ssh.run(
            f"ping -c 1 -W {timeout} {ip} >/dev/null 2>&1",
            timeout=timeout + 3,
            log_path=log_path,
            check=False,
            show_output=False,
        )
        return rc == 0

    @staticmethod
    def _configure_xen_dom1_via_console(
        ssh: SSHSession,
        env: dict[str, Any],
        exp: dict[str, Any],
        log_path: Path,
        *,
        initial: bool,
    ) -> None:
        """Configure dom1 networking through `xl console dom1` over dom0 SSH."""
        interface = str(exp.get("dom1_interface", "enX0"))
        ip = str(exp.get("dom1_ip", "202.197.67.51"))
        netmask = str(exp.get("dom1_netmask", "255.255.255.0"))
        service_setup = (
            "systemctl mask --runtime --now NetworkManager.service "
            "NetworkManager-wait-online.service network-manager.service "
            "networking.service systemd-networkd.service "
            "systemd-networkd-wait-online.service >/dev/null 2>&1 || true; "
            f"pkill -f '[u]dhcpc.*{interface}' >/dev/null 2>&1 || true; "
            if initial else ""
        )
        marker = "__NN_DOM1_XLCONSOLE_NET_RC__"
        command = (
            f"{service_setup}ifconfig {interface} {ip} netmask {netmask} up; "
            f"ip -4 -o addr show dev {interface} | grep -q ' {ip}/24 '; "
            f"echo {marker}=$?"
        )
        print(
            f"[NN/XEN] Configuring dom1 {interface}={ip}/24 through "
            "dom0 `xl console dom1`..."
        )
        console_timeout = float(
            env.get("dom1_console_ready_timeout_seconds", 60)
            if initial
            else env.get("dom1_boot_timeout", 180)
        )
        ssh.run_xen_console_command(
            env.get("dom1_console_command", "xl console dom1"),
            command,
            username=env.get("dom1_username", "root"),
            password=env.get("dom1_password", ""),
            login_prompt=env.get(
                "dom1_login_prompt", r"[A-Za-z0-9_.-]+ login:\s*$"
            ),
            guest_shell_prompts=env.get(
                "dom1_shell_prompts", [r"root@[^\r\n]*#\s*$", r"#\s*$"]
            ),
            completion_pattern=rf"{marker}=0(?:\r?\n|$)",
            timeout=console_timeout,
            log_path=log_path,
        )
        print(f"[NN/XEN] dom1 network ready via xl console: {interface}={ip}/24")

    def _start_xen_dom1_with_retry(
        self,
        ssh: SSHSession,
        env: dict[str, Any],
        exp: dict[str, Any],
        log_path: Path,
    ) -> None:
        """Recreate dom1 when its initial console never reaches login/shell."""
        attempts = max(1, int(env.get("dom1_start_attempts", 3)))
        create_commands = [
            str(command)
            for command in env.get("dom0_init_commands", [])
            if str(command).strip().startswith("xl create ")
        ]
        if len(create_commands) != 1:
            raise RuntimeError(
                "Expected exactly one `xl create` command for dom1 restart; "
                f"found {create_commands}"
            )
        create_command = create_commands[0]

        for attempt in range(1, attempts + 1):
            try:
                print(
                    f"[NN/XEN] dom1 console readiness attempt "
                    f"{attempt}/{attempts}..."
                )
                self._configure_xen_dom1_via_console(
                    ssh, env, exp, log_path, initial=True
                )
                return
            except TimeoutError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"dom1 did not reach login/shell after {attempts} "
                        "start attempts"
                    ) from exc
                print(
                    "[NN/XEN] dom1 console did not reach login/shell; "
                    "destroying and recreating dom1..."
                )
                if not ssh.is_active():
                    ssh.reconnect()
                ssh.run(
                    "xl destroy dom1",
                    timeout=30,
                    log_path=log_path,
                    check=False,
                )
                time.sleep(float(env.get("dom1_destroy_settle_seconds", 2)))
                ssh.run(
                    create_command,
                    timeout=float(env.get("init_command_timeout", 60)),
                    log_path=log_path,
                    check=True,
                )
                if env.get("dom1_pin_command"):
                    ssh.run(
                        env["dom1_pin_command"],
                        timeout=30,
                        log_path=log_path,
                        check=True,
                    )
                for command in env.get("pin_verify_commands", []):
                    if "dom1" in str(command):
                        ssh.run(
                            command,
                            timeout=30,
                            log_path=log_path,
                            check=True,
                        )
                time.sleep(float(env.get("dom1_recreate_settle_seconds", 2)))

    def _repair_xen_dom1_network(
        self,
        session: SerialSession,
        exp: dict[str, Any],
        prompt: str,
    ) -> None:
        """Recover dom1 IPv4 through COM10 without restarting network services."""
        interface = str(exp.get("dom1_interface", "enX0"))
        ip = str(exp.get("dom1_ip", "202.197.67.51"))
        netmask = str(exp.get("dom1_netmask", "255.255.255.0"))

        # Kernel/systemd messages can arrive after the shell prompt.  Request a
        # fresh prompt before sending the repair command instead of assuming
        # the console is currently at a clean prompt boundary.
        last_error: Exception | None = None
        for attempt in range(1, 4):
            session.buffer.clear()
            session.sendline()
            try:
                session.expect([prompt], 10)
                break
            except SerialTimeout as exc:
                last_error = exc
                print(
                    f"[NN/XEN] Waiting for a clean dom1 console prompt "
                    f"({attempt}/3)..."
                )
        else:
            raise RuntimeError(
                "Could not recover the dom1 COM10 shell prompt before IP repair"
            ) from last_error

        repaired = session.command(
            f"ifconfig {interface} {ip} netmask {netmask} up; "
            f"ip -4 -o addr show dev {interface} | grep -q ' {ip}/24 '; "
            "echo __NN_DOM1_REPAIR_RC__=$?",
            prompt,
            30,
        )
        if not self._has_output_line(repaired, "__NN_DOM1_REPAIR_RC__=0"):
            raise RuntimeError(
                f"Could not repair dom1 {interface} as {ip}/24 through COM10"
            )
        print(f"[NN/XEN] dom1 IP repaired through COM10: {interface}={ip}/24")

    def _repair_xen_dom1_network_via_serial_console(
        self,
        session: SerialSession,
        env: dict[str, Any],
        exp: dict[str, Any],
    ) -> None:
        """Enter dom1 from the COM10 dom0 shell, repair IPv4, then detach."""
        dom0_prompt = r"__XHYPASS_PROMPT__#\s*$"
        dom1_prompt = r"__XEN_DOM1_PROMPT__#\s*$"
        login_prompt = env.get(
            "dom1_login_prompt", r"[A-Za-z0-9_.-]+ login:\s*$"
        )
        shell_prompts = env.get(
            "dom1_shell_prompts", [r"root@[^\r\n]*#\s*$", r"#\s*$"]
        )
        console_command = env.get("dom1_console_command", "xl console dom1")
        timeout = float(env.get("dom1_console_ready_timeout_seconds", 60))
        primary_error: Exception | None = None

        print(f"[NN/XEN] COM10: entering dom1 with `{console_command}`...")
        session.buffer.clear()
        session.sendline(console_command)
        # xl console does not redraw an already-running guest's prompt until
        # it receives input.  Wake the dom1 console before waiting; otherwise
        # a healthy guest at a silent shell is mistaken for a dead guest.
        time.sleep(.5)
        session.sendline()
        try:
            matched, _ = session.expect(
                [dom1_prompt, login_prompt, *shell_prompts], timeout
            )
            if matched == 1:
                session.buffer.clear()
                session.sendline(env.get("dom1_username", "root"))
                matched, _ = session.expect(
                    [dom1_prompt, *shell_prompts],
                    float(self.cfg["timeouts"].get("login", 30)),
                )
            if matched != 0:
                session.command(
                    "export PS1='__XEN_DOM1_PROMPT__# '", dom1_prompt, 10
                )
            self._repair_xen_dom1_network(session, exp, dom1_prompt)
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            print("[NN/XEN] COM10: detaching from dom1 console...")
            session.buffer.clear()
            session.send(b"\x1d")
            try:
                session.expect([dom0_prompt], 15)
            except Exception as detach_error:
                if primary_error is None:
                    raise RuntimeError(
                        "dom1 IP was repaired, but COM10 could not return "
                        "to the dom0 shell"
                    ) from detach_error
                print(
                    "[WARN] COM10 could not confirm the dom0 prompt after "
                    f"dom1 repair failure: {detach_error}"
                )

    def _refresh_xen_result_connections(
        self,
        session: SerialSession,
        ssh: SSHSession,
        dom1_ssh: SSHSession,
        env: dict[str, Any],
        exp: dict[str, Any],
        ssh_log: Path,
    ) -> None:
        """Ensure result collection uses live dom0 and dom1 transports."""
        print("[NN/XEN] Checking SSH connections before result collection...")
        if not ssh.is_active():
            print("[NN/XEN] dom0 SSH is stale; reconnecting...")
            ssh.reconnect()

        # Do not touch a healthy guest network.  A stale Paramiko transport may
        # only need reconnecting; COM10 repair is reserved for an endpoint that
        # is actually unreachable.
        if dom1_ssh.is_active():
            print("[NN/XEN] dom1 SSH and IP are healthy; no repair needed.")
        elif self._xen_dom1_ip_reachable(ssh, exp, ssh_log):
            print("[NN/XEN] dom1 IP is reachable; reconnecting stale SSH...")
            dom1_ssh.reconnect()
        else:
            print(
                "[NN/XEN] dom1 SSH endpoint is unreachable; "
                "repairing through the COM10 dom1 console..."
            )
            self._repair_xen_dom1_network_via_serial_console(
                session, env, exp
            )
            dom1_ssh.reconnect()

    @staticmethod
    def _xen_run_complete(
        ssh: SSHSession,
        remote_dir: str,
        side: str,
        log_path: Path,
    ) -> bool:
        marker = f"{remote_dir}/results/run_complete_{side}"
        rc, _, _ = ssh.run(
            f"test -f {marker}",
            timeout=20,
            log_path=log_path,
            check=False,
            show_output=False,
        )
        if rc == 0:
            print(f"[NN/XEN] {side} completion marker verified.")
            return True
        print(f"[NN/XEN] {side} completion marker is missing.")
        return False

    @staticmethod
    def _resolve_xen_nn_base(
        ssh: SSHSession,
        remote_dir: str,
        model: str,
        log_path: Path,
        domain: str,
    ) -> str:
        command = (
            f"cd {remote_dir}; "
            "for base in .. .; do "
            "for executable in loadgen benchmark_http_infer cyclictest; do "
            "if [ -f \"$base/$executable\" ]; then "
            "chmod +x \"$base/$executable\"; fi; done; done; "
            "XY_NN_BASE=; "
            "for base in .. .; do "
            "if test -x \"$base/loadgen\" && "
            "test -x \"$base/benchmark_http_infer\" && "
            "test -x \"$base/cyclictest\" && "
            f"test -f \"$base/{model}\"; then "
            "XY_NN_BASE=$base; break; fi; done; "
            "if [ -n \"$XY_NN_BASE\" ] && command -v taskset >/dev/null && "
            "command -v base64 >/dev/null && command -v tar >/dev/null; then "
            "echo __NN_BASE__=$XY_NN_BASE; exit 0; fi; "
            "echo __NN_PREFLIGHT_DIAGNOSTIC__; "
            "for base in .. .; do "
            "echo BASE=$base; ls -l \"$base/loadgen\" "
            "\"$base/benchmark_http_infer\" \"$base/cyclictest\" "
            f"\"$base/{model}\" 2>&1; done; "
            "for tool in taskset base64 tar; do "
            "command -v \"$tool\" || echo MISSING_COMMAND=$tool; done; exit 1"
        )
        rc, output, _ = ssh.run(
            command, timeout=30, log_path=log_path, check=False
        )
        match = re.search(rb"(?:^|\n)__NN_BASE__=(\.\.?)(?:\r?\n|$)", output)
        if rc != 0 or not match:
            raise RuntimeError(
                f"{domain} NN preflight failed; see {log_path.name} for "
                "the exact missing file or permission"
            )
        base = match.group(1).decode("ascii")
        print(f"[NN/XEN] {domain} NN dependency base: {remote_dir}/{base}")
        return base

    @staticmethod
    def _enter_xen_dom1_console(
        session: SerialSession, env: dict[str, Any], prompt: str
    ) -> None:
        print("[NN/XEN] Entering dom1 with xl console dom1 on COM10...")
        session.buffer.clear()
        session.sendline(env.get("dom1_console_command", "xl console dom1"))
        session.expect(
            [env.get("dom1_login_prompt", r"[A-Za-z0-9_.-]+ login:\s*$")],
            float(env.get("dom1_boot_timeout", 180)),
        )
        session.buffer.clear()
        session.sendline(env.get("dom1_username", "root"))
        shell_prompts = env.get(
            "dom1_shell_prompts", [r"root@[^\r\n]*#\s*$", r"#\s*$"]
        )
        matched, _ = session.expect([*shell_prompts, r"Password:\s*$"], 30)
        if matched == len(shell_prompts):
            session.buffer.clear()
            session.sendline(env.get("dom1_password", ""))
            session.expect(shell_prompts, 30)
        session.command("export PS1='__XEN_DOM1_PROMPT__# '", prompt, 10)
        print("[NN/XEN] dom1 root login succeeded on COM10.")

    def _collect_xen_results(
        self,
        ssh: SSHSession,
        dom1_ssh: SSHSession,
        output: Path,
        ssh_log: Path,
        dom1_log: Path,
        remote_dir: str,
    ) -> None:
        dom0_archive = "/tmp/xy_nn_xen_dom0_results.tar.gz"
        dom1_archive = "/tmp/xy_nn_xen_dom1_results.tar.gz"
        ssh.run(
            f"tar -czf {dom0_archive} -C {remote_dir} results",
            timeout=60, log_path=ssh_log, check=True,
        )
        dom0_local = output / "dom0_results.tar.gz"
        ssh.get(dom0_archive, dom0_local)
        dom1_ssh.run(
            f"tar -czf {dom1_archive} -C {remote_dir} results",
            timeout=60, log_path=dom1_log, check=True,
        )
        dom1_local = output / "dom1_results.tar.gz"
        dom1_ssh.get(dom1_archive, dom1_local)
        dom0_dir = output / "dom0"
        dom1_dir = output / "dom1"
        dom0_dir.mkdir()
        dom1_dir.mkdir()
        with tarfile.open(dom0_local, "r:gz") as archive:
            archive.extractall(dom0_dir, filter="data")
        with tarfile.open(dom1_local, "r:gz") as archive:
            archive.extractall(dom1_dir, filter="data")
        merged = output / "results"
        merged.mkdir()
        for source in (dom0_dir / "results", dom1_dir / "results"):
            for item in source.iterdir():
                destination = merged / item.name
                if destination.exists():
                    raise RuntimeError(
                        f"dom0/dom1 NN result name collision: {item.name}"
                    )
                shutil.copy2(item, destination)
        with tarfile.open(output / "nn_results.tar.gz", "w:gz") as archive:
            archive.add(merged, arcname="results")
        ssh.run(
            f"rm -f {dom0_archive}", timeout=20,
            log_path=ssh_log, check=False,
        )
        dom1_ssh.run(
            f"rm -f {dom1_archive}", timeout=20,
            log_path=dom1_log, check=False,
        )

    def _xen_assignments(self, exp: dict[str, Any], side: str) -> str:
        assignments = self._environment_assignments(exp)
        if side == "dom0":
            extra = {
                "MODEL_THREADS": exp["dom0_model_threads"],
                "CPU_CYCLIC": exp["dom0_cyclictest_cpu"],
                "WORKLOAD_CPUS": exp.get("dom0_workload_cpus", "0-5"),
                "LAUNCH_DELAY": exp.get("dom0_launch_delay_seconds", 0),
            }
        else:
            extra = {
                "MODEL_THREADS": exp["dom1_model_threads"],
                "CPU_CYCLIC": exp["dom1_cyclictest_cpu"],
                "WORKLOAD_CPUS": exp.get("dom1_workload_cpus", "0-5"),
                "LAUNCH_DELAY": exp.get("dom1_launch_delay_seconds", 0),
            }
        extra["START_DELAY"] = exp.get("start_delay_seconds", 2)
        extra["WARMUP_SECONDS"] = exp.get("warmup_seconds", 0)
        extra["WARMUP_QPS"] = exp.get("warmup_qps", 1)
        extra["RUN_LEVELS"] = " ".join(
            str(profile["name"]) for profile in exp["profiles"]
        )
        return assignments + " " + " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in extra.items()
        )

    @staticmethod
    def _upload_script_ssh(
        ssh: SSHSession, local: Path, remote: str, log_path: Path
    ) -> None:
        if not local.is_file():
            raise FileNotFoundError(f"NN script not found: {local}")
        payload = base64.b64encode(local.read_bytes()).decode("ascii")
        ssh.run(
            f"echo {payload} | base64 -d > {remote}; chmod +x {remote}",
            timeout=30,
            log_path=log_path,
            check=True,
            show_output=False,
        )

    def _prepare_nonroot_files(
        self,
        session: SerialSession,
        ssh: SSHSession,
        exp: dict[str, Any],
        prompt: str,
        ssh_log: Path,
    ) -> None:
        session.command(
            "cd ~ && rm -rf NN && rm -f cyclictest stress-ng",
            prompt,
            30,
        )
        root_ip = exp["rootcell_internal_ip"]
        copies = (
            f"scp -q root@{root_ip}:/root/cyclictest ./",
            f"scp -q root@{root_ip}:/root/stress-ng ./",
            f"scp -q -r root@{root_ip}:/root/NN .",
        )
        retries = int(exp.get("scp_retries", 5))
        for copy_command in copies:
            copied = False
            for attempt in range(1, retries + 1):
                print(
                    f"[NN/JAILHOUSE/SCP] attempt {attempt}/{retries}: "
                    f"{copy_command}"
                )
                ssh.run(
                    f"ifconfig {exp['rootcell_internal_interface']} "
                    f"{exp['rootcell_internal_ip']}",
                    timeout=20,
                    log_path=ssh_log,
                    check=True,
                )
                session.command(
                    f"ifconfig {exp['nonroot_internal_interface']} "
                    f"{exp['nonroot_internal_ip']}",
                    prompt,
                    20,
                )
                command = (
                    "cd ~ && "
                    + copy_command.replace(
                        "scp ",
                        "scp -o StrictHostKeyChecking=no "
                        "-o UserKnownHostsFile=/dev/null "
                        "-o ConnectTimeout=10 ",
                        1,
                    )
                    + "; echo __NN_SCP_RC__=$?"
                )
                raw = self._nonroot_interactive_command(
                    session,
                    command,
                    prompt,
                    str(exp.get("rootcell_internal_password", "")),
                    float(exp.get("scp_timeout_seconds", 600)),
                )
                if self._has_output_line(raw, "__NN_SCP_RC__=0"):
                    copied = True
                    break
                print("[NN/JAILHOUSE/SCP] Copy failed; restoring both IPs and retrying.")
            if not copied:
                raise RuntimeError(
                    f"Could not copy NN dependency after {retries} attempts: "
                    f"{copy_command}"
                )
        chmod = session.command(
            "cd ~ && chmod +x cyclictest stress-ng NN/loadgen "
            "NN/benchmark_http_infer NN/cyclictest "
            "NN/jailhouse/run_vm2_automated.sh; "
            "echo __NN_CHMOD_RC__=$?",
            prompt,
            30,
        )
        if not self._has_output_line(chmod, "__NN_CHMOD_RC__=0"):
            raise RuntimeError("Could not set non-root-cell NN executable permissions")

    @staticmethod
    def _nonroot_interactive_command(
        session: SerialSession,
        command: str,
        prompt: str,
        password: str,
        timeout: float,
    ) -> bytes:
        deadline = time.monotonic() + timeout
        collected = bytearray()
        session.buffer.clear()
        session.sendline(command)
        patterns = [
            r"Are you sure you want to continue connecting.*\?\s*$",
            r"(?:password|Password):\s*$",
            prompt,
        ]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Interactive command timed out: {command}")
            matched, raw = session.expect(patterns, remaining)
            collected.extend(raw)
            if matched == 2:
                return bytes(collected)
            session.buffer.clear()
            session.sendline("yes" if matched == 0 else password)

    def _collect_jailhouse_results(
        self,
        session: SerialSession,
        ssh: SSHSession,
        output: Path,
        exp: dict[str, Any],
        prompt: str,
        ssh_log: Path,
    ) -> None:
        root_archive = "/tmp/xy_nn_root_results.tar.gz"
        nonroot_archive = "/tmp/xy_nn_nonroot_results.tar.gz"
        ssh.run(
            f"tar -czf {root_archive} -C ~/NN/jailhouse results",
            timeout=60,
            log_path=ssh_log,
            check=True,
        )
        root_local = output / "rootcell_results.tar.gz"
        ssh.get(root_archive, root_local)
        archived = session.command(
            f"tar -czf {nonroot_archive} -C ~/NN/jailhouse results; "
            "echo __NN_NONROOT_ARCHIVE_RC__=$?",
            prompt,
            60,
        )
        if not self._has_output_line(
            archived, "__NN_NONROOT_ARCHIVE_RC__=0"
        ):
            raise RuntimeError("Could not archive non-root-cell NN results")
        nonroot_local = output / "nonroot_results.tar.gz"
        self._download_remote_chunked(
            session,
            nonroot_archive,
            nonroot_local,
            prompt,
            int(exp.get("transfer_chunk_bytes", 262_144)),
        )
        root_dir = output / "rootcell"
        nonroot_dir = output / "nonroot"
        root_dir.mkdir()
        nonroot_dir.mkdir()
        with tarfile.open(root_local, "r:gz") as archive:
            archive.extractall(root_dir, filter="data")
        with tarfile.open(nonroot_local, "r:gz") as archive:
            archive.extractall(nonroot_dir, filter="data")
        merged = output / "results"
        merged.mkdir()
        for source in (root_dir / "results", nonroot_dir / "results"):
            for item in source.iterdir():
                destination = merged / item.name
                if destination.exists():
                    raise RuntimeError(
                        f"Root/non-root NN result name collision: {item.name}"
                    )
                shutil.copy2(item, destination)
        combined = output / "nn_results.tar.gz"
        with tarfile.open(combined, "w:gz") as archive:
            archive.add(merged, arcname="results")

    def _jailhouse_assignments(
        self, exp: dict[str, Any], side: str
    ) -> str:
        assignments = self._environment_assignments(exp)
        if side == "rootcell":
            extra = {
                "MODEL_THREADS": exp["rootcell_model_threads"],
                "CPU_CYCLIC": exp["rootcell_cyclictest_cpu"],
                "START_DELAY": exp.get("start_delay_seconds", 2),
            }
        else:
            extra = {
                "MODEL_THREADS": exp["nonroot_model_threads"],
                "CPU_CYCLIC": exp["nonroot_cyclictest_cpu"],
                "START_DELAY": exp.get("start_delay_seconds", 2),
            }
        return assignments + " " + " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in extra.items()
        )

    @staticmethod
    def _environment_assignments(exp: dict[str, Any]) -> str:
        values: dict[str, Any] = {
            "DURATION": exp["duration_seconds"],
            "SEED": exp["seed"],
            "WORKERS": exp["workers"],
            "INCEPTION_THREAD": exp["inception_threads"],
            "MNAS_THREAD": exp["mnasnet_threads"],
            "CT_INTERVAL_US": exp["cyclictest_interval_us"],
            "CT_PRIORITY": exp["cyclictest_priority"],
            "HISTOGRAM_LIMIT_US": exp["histogram_limit_us"],
            "PERIOD_SEC": exp["period_seconds"],
        }
        if "workload_cpus" in exp:
            values["WORKLOAD_CPUS"] = exp["workload_cpus"]
        if "cgroup_weight" in exp:
            values["CGROUP_WEIGHT"] = exp["cgroup_weight"]
        if "cgroup_shares" in exp:
            values["CGROUP_SHARES"] = exp["cgroup_shares"]
        for profile in exp["profiles"]:
            prefix = str(profile["name"]).upper()
            values.update(
                {
                    f"{prefix}_INCEPTION_QPS": profile["inception_qps"],
                    f"{prefix}_PEAK_QPS": profile["peak_qps"],
                    f"{prefix}_POISSON_QPS": profile["poisson_qps"],
                    f"{prefix}_CONSTANT_QPS": profile["constant_qps"],
                }
            )
        return " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in values.items()
        )

    def _download_remote_chunked(
        self,
        session: SerialSession,
        remote: str,
        local: Path,
        prompt: str,
        chunk_bytes: int,
    ) -> None:
        sized = session.command(
            f"echo __NN_SIZE__=$(wc -c < {remote})", prompt, 15
        )
        match = re.search(rb"(?:^|\r*\n)__NN_SIZE__=(\d+)\r*(?:\n|$)", sized)
        if not match:
            raise RuntimeError(f"Could not determine remote result size: {remote}")
        total = int(match.group(1))
        with local.open("wb") as stream:
            for offset in range(0, total, chunk_bytes):
                count = min(chunk_bytes, total - offset)
                raw = session.command(
                    f"echo __NN_CHUNK_BEGIN__; dd if={remote} bs=1 skip={offset} "
                    f"count={count} 2>/dev/null | base64; echo __NN_CHUNK_END__",
                    prompt,
                    120,
                )
                payload = re.search(
                    rb"__NN_CHUNK_BEGIN__\r*\n(.*?)\r*\n__NN_CHUNK_END__",
                    raw,
                    re.S,
                )
                if not payload:
                    raise RuntimeError(
                        f"Could not transfer NN result chunk at offset {offset}"
                    )
                stream.write(
                    base64.b64decode(
                        re.sub(rb"\s+", b"", payload.group(1)), validate=True
                    )
                )
        if local.stat().st_size != total:
            raise RuntimeError(
                f"NN archive size mismatch: expected {total}, got {local.stat().st_size}"
            )

    @staticmethod
    def _summarize_results(results_dir: Path, exp: dict[str, Any]) -> dict:
        expected_levels = [str(profile["name"]) for profile in exp["profiles"]]
        csv_paths = sorted(results_dir.glob("*.csv"))
        hist_paths = sorted(results_dir.glob("cyclictest_*.txt"))
        if len(csv_paths) != len(expected_levels) * 2:
            raise RuntimeError(
                f"Expected {len(expected_levels) * 2} NN CSV files, got {len(csv_paths)}"
            )
        if len(hist_paths) != len(expected_levels) * 2:
            raise RuntimeError(
                f"Expected {len(expected_levels) * 2} cyclictest files, got {len(hist_paths)}"
            )
        requests = 0
        failures = 0
        csv_summary = []
        for path in csv_paths:
            response_times = []
            file_failures = 0
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                for row in csv.DictReader(stream):
                    response_times.append(float(row["response_ms"]))
                    if float(row["inference_ms"]) < 0:
                        file_failures += 1
            response_times.sort()
            requests += len(response_times)
            failures += file_failures
            p99_index = max(0, min(len(response_times) - 1, int(.99 * len(response_times))))
            csv_summary.append(
                {
                    "file": path.name,
                    "requests": len(response_times),
                    "inference_failures": file_failures,
                    "response_ms_p99": response_times[p99_index] if response_times else None,
                    "response_ms_max": response_times[-1] if response_times else None,
                }
            )
        hist_summary = []
        for path in hist_paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            maximum = re.search(r"^# Max Latencies:\s*(\d+)", text, re.M)
            overflow = re.search(r"^# Histogram Overflows:\s*(\d+)", text, re.M)
            hist_summary.append(
                {
                    "file": path.name,
                    "max_latency_us": int(maximum.group(1)) if maximum else None,
                    "histogram_overflows": int(overflow.group(1)) if overflow else None,
                }
            )
        return {
            "csv_files": len(csv_paths),
            "cyclictest_files": len(hist_paths),
            "http_requests": requests,
            "http_inference_failures": failures,
            "csv": csv_summary,
            "cyclictest": hist_summary,
        }
