import tempfile
import unittest
import base64
import io
import json
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import run_bare_jailhouse_600s
import run_full_cyclictest_matrix
import run_full_nn_matrix
import run_e2000q_full_cyclictest_matrix
import run_tl3588_int_latency_bare
import run_tl3588_int_latency_jailhouse
import run_tl3588_int_latency_xen
import run_tl3588_int_latency_xhypass
import run_tl3588_int_latency_matrix
import run_e2000q_int_latency_matrix
import run_tl3588_nn_then_int_latency

from smoke_scripts import (
    run_e2000q_all_smoke,
    run_e2000q_bare_smoke,
    run_e2000q_experiment,
    run_e2000q_jailhouse_smoke,
    run_e2000q_xen_smoke,
    run_experiment,
    run_nn_bare,
    run_nn_jailhouse,
    run_nn_xen_credit2,
    run_nn_xen_credit2_wfx,
    run_nn_xen_null,
    run_nn_xen_null_wfx,
    run_nn_xhypass,
    run_xen_credit2,
    run_xen_credit2_wfx,
    run_xen_null,
    run_xen_null_wfx,
    run_xen_variants_smoke,
    run_xhypass,
    run_xhypass_smoke,
)

from xhypass_lab.config import ConfigError, load_config, resolved_run_config
from xhypass_lab.platforms import (
    ENVIRONMENTS as PLATFORM_ENVIRONMENTS,
    load_platform_config,
    platform_data_root,
)
from xhypass_lab.runner import ExperimentRunner, condition_name, safe_name
from xhypass_lab.nn_runner import NNExperimentRunner
from xhypass_lab.int_latency_runner import IntLatencyRunner
from xhypass_lab.int_latency_config import build_int_latency_run_config
from xhypass_lab.serial_session import (
    SerialSession,
    SerialTimeout,
    write_console_bytes,
)


class FrameworkTests(unittest.TestCase):
    def test_nn_resume_quarantines_stale_completed_directory(self):
        run = run_nn_xen_credit2.build_run_config()
        with tempfile.TemporaryDirectory() as directory:
            runner = NNExperimentRunner(run, Path(directory), dry_run=False)
            output = (
                Path(directory)
                / "NN"
                / "xen_credit2"
                / "condition"
                / "_details"
                / "completed"
                / "run_004"
            )
            output.mkdir(parents=True)
            (output / "serial.log").write_text("partial", encoding="utf-8")
            (output / "metadata.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )

            runner._create_output_directory(output, 1)

            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            failed = list(output.parent.parent.joinpath("failed").iterdir())
            self.assertEqual(len(failed), 1)
            self.assertEqual(
                json.loads(
                    (failed[0] / "metadata.json").read_text(encoding="utf-8")
                )["status"],
                "abandoned",
            )

    def test_int_latency_resume_quarantines_stale_completed_directory(self):
        run = run_tl3588_int_latency_bare.build_run_config("idle")
        with tempfile.TemporaryDirectory() as directory:
            runner = IntLatencyRunner([run], Path(directory), dry_run=False)
            output = (
                Path(directory)
                / "int-latency"
                / "bare"
                / "idle"
                / "_details"
                / "completed"
                / "run_002"
            )
            output.mkdir(parents=True)
            (output / "com10.log").write_text("partial", encoding="utf-8")
            (output / "metadata.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )

            runner._create_output_directory(output, 2)

            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            failed = list(output.parent.parent.joinpath("failed").iterdir())
            self.assertEqual(len(failed), 1)
            self.assertEqual(
                json.loads(
                    (failed[0] / "metadata.json").read_text(encoding="utf-8")
                )["status"],
                "abandoned",
            )

    def test_tl3588_combined_campaign_runs_int_latency_before_nn(self):
        workflow = run_tl3588_nn_then_int_latency
        calls = []
        with patch.object(workflow, "_write_status"):
            with patch.object(
                workflow.run_full_nn_matrix,
                "main",
                side_effect=lambda: calls.append("NN") or 0,
            ):
                with patch.object(
                    workflow.run_tl3588_int_latency_matrix,
                    "main",
                    side_effect=lambda: calls.append("int-latency") or 0,
                ):
                    result = workflow.main()

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["int-latency", "NN"])
        self.assertEqual(workflow.NN_RUNS_PER_ENVIRONMENT, 5)
        self.assertEqual(workflow.INT_LATENCY_RUNS_PER_CONDITION, 5)

    def test_tl3588_combined_campaign_stops_if_int_latency_is_incomplete(self):
        workflow = run_tl3588_nn_then_int_latency
        with patch.object(workflow, "_write_status"):
            with patch.object(
                workflow.run_tl3588_int_latency_matrix, "main", return_value=1
            ):
                with patch.object(
                    workflow.run_full_nn_matrix, "main"
                ) as nn:
                    result = workflow.main()

        self.assertEqual(result, 1)
        nn.assert_not_called()

    def test_reach_uboot_accepts_xen_reboot_and_interrupt_ack(self):
        class XenRebootSession:
            def __init__(self):
                self.buffer = bytearray()
                self.reads = 0
                self.raw = []

            def _read_once(self):
                self.reads += 1
                if self.reads == 1:
                    self.buffer.extend(
                        b"(XEN) Hardware Dom0 shutdown: rebooting machine\n"
                    )
                elif self.reads == 2:
                    self.buffer.extend(b"=> <INTERRUPT>\n")

            def send(self, data):
                self.raw.append(data)

        run = run_tl3588_int_latency_xhypass.build_run_config("stress")
        runner = ExperimentRunner(run, Path("data/RK3588"), True)
        session = XenRebootSession()

        runner._reach_uboot(session, run["environment"]["uboot_prompt"])

        self.assertEqual(session.raw, [b"\x03"])

    def test_tl3588_xhypass_waits_for_com10_reboot_ready_marker(self):
        run = run_tl3588_int_latency_xhypass.build_run_config("idle")
        self.assertEqual(
            run["environment"]["int_latency_reboot_ready_pattern"],
            r"(?:\(XEN\)\s+exit_rto\s+rc=0|\(XEN\)\s+!+set gicv3 interrupt passthrough)",
        )
        self.assertEqual(
            run["environment"]["int_latency_rto_enter_pattern"],
            r"(?:\(XEN\)\s+enter_rto\s+rc=0|\(XEN\)\s+!+set gicv3 interrupt injection)",
        )

        class FakeControl:
            def __init__(self):
                self.buffer = bytearray()
                self.reads = 0

            def _read_once(self):
                self.reads += 1
                if self.reads == 3:
                    raw = (
                        b"(XEN) !!!!!!!!!!!!!set gicv3 interrupt "
                        b"passthrough\n"
                    )
                    self.buffer.extend(raw)
                    return raw
                return b""

        class FakeRtos:
            def __init__(self):
                self.buffer = bytearray()
                self.reads = 0

            def _read_once(self):
                self.reads += 1
                if self.reads == 1:
                    raw = b"PMU diff samples: 600000\n"
                    self.buffer.extend(raw)
                    return raw
                return b""

        control = FakeControl()
        with patch("xhypass_lab.int_latency_runner.time.sleep"):
            IntLatencyRunner._monitor_dual_serial(
                control,
                FakeRtos(),
                completion_pattern=r"PMU diff samples:\s*600000",
                timeout=2,
                completion_settle_seconds=0,
                completion_not_before_seconds=0,
                started_at=time.monotonic(),
                retry_patterns=[],
                control_completion_pattern=run["environment"][
                    "int_latency_reboot_ready_pattern"
                ],
            )
        self.assertGreaterEqual(control.reads, 3)

    def test_tl3588_xhypass_accepts_new_exit_rto_marker(self):
        run = run_tl3588_int_latency_xhypass.build_run_config("idle")

        class FakeControl:
            def __init__(self):
                self.buffer = bytearray()

            def _read_once(self):
                raw = b"(XEN) exit_rto rc=0\n"
                self.buffer.extend(raw)
                return raw

        class FakeRtos:
            def __init__(self):
                self.buffer = bytearray()

            def _read_once(self):
                raw = b"PMU diff samples: 600000\n"
                self.buffer.extend(raw)
                return raw

        with patch("xhypass_lab.int_latency_runner.time.sleep"):
            IntLatencyRunner._monitor_dual_serial(
                FakeControl(),
                FakeRtos(),
                completion_pattern=r"PMU diff samples:\s*600000",
                timeout=2,
                completion_settle_seconds=0,
                completion_not_before_seconds=0,
                started_at=time.monotonic(),
                retry_patterns=[],
                control_completion_pattern=run["environment"][
                    "int_latency_reboot_ready_pattern"
                ],
            )

    def test_e2000q_bare_int_latency_dual_serial_configuration(self):
        idle = build_int_latency_run_config("E2000Q", "bare", "idle")
        stressed = build_int_latency_run_config("E2000Q", "bare", "stress")

        self.assertEqual(idle["control_serial"]["port"], "COM12")
        self.assertEqual(idle["rtos_serial"]["port"], "COM11")
        self.assertEqual(idle["control_serial"]["baudrate"], 115200)
        self.assertEqual(idle["rtos_serial"]["baudrate"], 115200)
        self.assertEqual(idle["environment"]["boot_command"], "run boot_rtos_idle")
        self.assertEqual(
            stressed["environment"]["boot_command"], "run boot_rtos_stress"
        )
        self.assertEqual(idle["experiment"]["duration_seconds"], 600)
        self.assertEqual(idle["experiment"]["timeout_seconds"], 780)
        self.assertEqual(
            idle["experiment"]["completion_pattern"],
            r"taskpendcnt:\s*600000",
        )
        self.assertEqual(
            IntLatencyRunner._serial_log_name(idle["control_serial"], "control"),
            "com12.log",
        )
        self.assertEqual(
            IntLatencyRunner._serial_log_name(idle["rtos_serial"], "rtos"),
            "com11.log",
        )

    def test_e2000q_jailhouse_int_latency_configuration(self):
        idle = build_int_latency_run_config("E2000Q", "jailhouse", "idle")
        stressed = build_int_latency_run_config(
            "E2000Q", "jailhouse", "stress"
        )
        environment = idle["environment"]

        self.assertEqual(environment["boot_command"], "run boot_oee")
        self.assertEqual(environment["module_load_command"], "insmod jailhouse.ko")
        self.assertEqual(
            environment["enable_command"],
            "jailhouse enable /usr/share/jailhouse/cells/e2000q-gsk-8g.cell",
        )
        self.assertEqual(
            environment["int_latency_create_command"],
            "jailhouse cell create e2000q-8g-QSemOS-RT.cell",
        )
        self.assertEqual(
            environment["int_latency_load_commands"]["idle"],
            "jailhouse cell load e2000q-QSemOS-RT "
            "qsemos-rt.bin_idle -a 0xC0000000",
        )
        self.assertEqual(
            stressed["environment"]["int_latency_load_commands"]["stress"],
            "jailhouse cell load e2000q-QSemOS-RT "
            "qsemos-rt.bin_stress -a 0xC0000000",
        )
        self.assertEqual(
            environment["int_latency_start_command"],
            "jailhouse cell start e2000q-QSemOS-RT",
        )
        self.assertEqual(idle["control_serial"]["port"], "COM12")
        self.assertEqual(idle["rtos_serial"]["port"], "COM11")

    def test_e2000q_wfx_int_latency_environments_replace_only_the_dtb(self):
        for environment, dtb in (
            (
                "xen_credit2_WFX",
                "e2000q-gsk-board-xen-dom0less.dtb.credit2.WFX",
            ),
            ("xen_null_WFX", "e2000q-gsk-board-xen-dom0less.dtb.null.WFX"),
        ):
            idle = build_int_latency_run_config("E2000Q", environment, "idle")
            stressed = build_int_latency_run_config(
                "E2000Q", environment, "stress"
            )
            self.assertEqual(
                Path(idle["environment"]["int_latency_boot_files"][0]["source"]).name,
                "qsemos-rt.bin_native",
            )
            self.assertEqual(
                Path(stressed["environment"]["int_latency_boot_files"][0]["source"]).name,
                "qsemos-rt.bin_native_stress",
            )
            self.assertEqual(
                Path(idle["environment"]["int_latency_boot_files"][1]["source"]).name,
                dtb,
            )

    def test_e2000q_xen_int_latency_artifacts_and_pin_mapping(self):
        credit_idle = build_int_latency_run_config(
            "E2000Q", "xen_credit2", "idle"
        )
        credit_stress = build_int_latency_run_config(
            "E2000Q", "xen_credit2", "stress"
        )
        null_idle = build_int_latency_run_config("E2000Q", "xen_null", "idle")

        for run in (credit_idle, credit_stress, null_idle):
            self.assertEqual(
                run["environment"]["boot_command"], "run boot_xen_dom0less"
            )
            self.assertEqual(run["control_serial"]["port"], "COM12")
            self.assertEqual(run["rtos_serial"]["port"], "COM11")
        self.assertEqual(
            credit_idle["environment"]["int_latency_pin_commands"],
            ["xl vcpu-pin 0 1 3", "xl vcpu-pin 1 0 1"],
        )
        self.assertEqual(
            null_idle["environment"]["int_latency_pin_commands"],
            ["xl vcpu-pin 1 0 1", "xl vcpu-pin 0 1 3"],
        )

        def artifact_names(run):
            return [
                (Path(item["source"]).name, Path(item["destination"]).name)
                for item in run["environment"]["int_latency_boot_files"]
            ]

        self.assertEqual(
            artifact_names(credit_idle),
            [
                ("qsemos-rt.bin_native", "qsemos-rt.bin"),
                (
                    "e2000q-gsk-board-xen-dom0less.dtb.credit2",
                    "e2000q-gsk-board-xen-dom0less.dtb",
                ),
            ],
        )
        self.assertEqual(
            artifact_names(credit_stress)[0],
            ("qsemos-rt.bin_native_stress", "qsemos-rt.bin"),
        )
        self.assertEqual(
            artifact_names(null_idle)[1],
            (
                "e2000q-gsk-board-xen-dom0less.dtb.null",
                "e2000q-gsk-board-xen-dom0less.dtb",
            ),
        )
        for run in (credit_idle, credit_stress, null_idle):
            for item in run["environment"]["int_latency_boot_files"]:
                self.assertFalse(Path(item["source"]).is_absolute())
                self.assertFalse(Path(item["destination"]).is_absolute())

    def test_e2000q_xhypass_int_latency_uses_pass_binary_and_credit2_dtb(self):
        idle = build_int_latency_run_config("E2000Q", "XHyPass", "idle")
        stressed = build_int_latency_run_config("E2000Q", "XHyPass", "stress")

        self.assertEqual(
            idle["environment"]["boot_command"], "run boot_xen_dom0less"
        )
        self.assertEqual(
            idle["environment"]["int_latency_pin_commands"],
            ["xl vcpu-pin 0 1 3", "xl vcpu-pin 1 0 1"],
        )
        idle_files = idle["environment"]["int_latency_boot_files"]
        stress_files = stressed["environment"]["int_latency_boot_files"]
        self.assertEqual(Path(idle_files[0]["source"]).name, "qsemos-rt.bin_pass")
        self.assertEqual(
            Path(stress_files[0]["source"]).name,
            "qsemos-rt.bin_pass_stress",
        )
        self.assertEqual(
            Path(idle_files[1]["source"]).name,
            "e2000q-gsk-board-xen-dom0less.dtb.credit2",
        )
        self.assertEqual(Path(idle_files[0]["destination"]).name, "qsemos-rt.bin")
        self.assertEqual(
            Path(idle_files[1]["destination"]).name,
            "e2000q-gsk-board-xen-dom0less.dtb",
        )
        for run in (idle, stressed):
            for item in run["environment"]["int_latency_boot_files"]:
                self.assertFalse(Path(item["source"]).is_absolute())

    def test_e2000q_int_latency_uses_serial_reboot_when_ssh_is_absent(self):
        config = build_int_latency_run_config(
            "E2000Q", "XHyPass", "stress"
        )

        class ProbeSerial:
            def __init__(self):
                self.buffer = bytearray()

            def sendline(self, _line=""):
                return None

            def drain(self, _seconds):
                self.buffer.extend(b"gsk-e2000q login: ")
                return bytes(self.buffer)

        class FakeBaseRunner:
            def __init__(self):
                self.serial_calls = 0
                self.ssh_calls = 0

            def _reboot_serial_to_uboot(self, *_args, **_kwargs):
                self.serial_calls += 1

            def _reboot_jailhouse_rootcell(self, *_args, **_kwargs):
                self.ssh_calls += 1

        base = FakeBaseRunner()
        runner = IntLatencyRunner([config], Path("unused"), dry_run=True)
        with patch.object(runner, "_base_runner", return_value=base):
            runner._ensure_uboot(ProbeSerial(), config)

        self.assertEqual(base.serial_calls, 1)
        self.assertEqual(base.ssh_calls, 0)

    def test_tl3588_bare_int_latency_dual_serial_configuration(self):
        idle = run_tl3588_int_latency_bare.build_run_config("idle")
        stressed = run_tl3588_int_latency_bare.build_run_config("stress")

        self.assertEqual(idle["control_serial"]["port"], "COM10")
        self.assertEqual(idle["rtos_serial"]["port"], "COM14")
        self.assertEqual(idle["control_serial"]["baudrate"], 115200)
        self.assertEqual(idle["rtos_serial"]["baudrate"], 115200)
        self.assertEqual(idle["environment"]["boot_command"], "run boot_rtos_idle")
        self.assertEqual(
            stressed["environment"]["boot_command"], "run boot_rtos_stress"
        )
        self.assertEqual(idle["experiment"]["duration_seconds"], 600)
        self.assertEqual(idle["experiment"]["timeout_seconds"], 780)
        self.assertEqual(
            idle["experiment"]["completion_pattern"],
            r"PMU diff samples:\s*600000",
        )

    def test_tl3588_int_latency_plan_separates_idle_and_stress(self):
        configs = [
            run_tl3588_int_latency_bare.build_run_config("idle"),
            run_tl3588_int_latency_bare.build_run_config("stress"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            planned = IntLatencyRunner(
                configs, Path(directory), dry_run=True
            ).plan()
        self.assertEqual(
            planned[0][2].parts[-6:],
            (
                "int-latency",
                "bare",
                "idle",
                "_details",
                "completed",
                "run_001",
            ),
        )
        self.assertIn("stress", planned[1][2].parts)

    def test_tl3588_jailhouse_int_latency_commands(self):
        idle = run_tl3588_int_latency_jailhouse.build_run_config("idle")
        stressed = run_tl3588_int_latency_jailhouse.build_run_config("stress")
        environment = idle["environment"]

        self.assertEqual(idle["control_serial"]["port"], "COM10")
        self.assertEqual(idle["rtos_serial"]["port"], "COM14")
        self.assertEqual(environment["boot_command"], "run boot_oee")
        self.assertEqual(environment["module_load_command"], "insmod jailhouse.ko")
        self.assertEqual(
            environment["enable_command"],
            "jailhouse enable /usr/share/jailhouse/cells/tl3588.cell",
        )
        self.assertEqual(
            environment["int_latency_create_command"],
            "jailhouse cell create tl3588-QSemOS-RT.cell",
        )
        self.assertEqual(
            environment["int_latency_load_commands"]["idle"],
            "jailhouse cell load tl3588-QSemOS-RT "
            "qsemos-rt.bin_idle -a 0x24000000",
        )
        self.assertEqual(
            stressed["environment"]["int_latency_load_commands"]["stress"],
            "jailhouse cell load tl3588-QSemOS-RT "
            "qsemos-rt.bin_stress -a 0x24000000",
        )
        self.assertEqual(
            environment["int_latency_start_command"],
            "jailhouse cell start tl3588-QSemOS-RT",
        )

    def test_tl3588_jailhouse_int_latency_runs_cell_steps_in_order(self):
        class FakePort:
            def reset_input_buffer(self):
                pass

        class FakeSerial:
            def __init__(self):
                self.commands = []
                self.buffer = bytearray()
                self.port = FakePort()

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                marker = command.split("printf '\\n", 1)[1].split("=%s", 1)[0]
                return f"\n{marker}=0\n__XHYPASS_PROMPT__# ".encode()

        config = run_tl3588_int_latency_jailhouse.build_run_config("idle")
        runner = IntLatencyRunner([config], Path("data/RK3588"), dry_run=True)
        control = FakeSerial()
        rtos = FakeSerial()
        base = unittest.mock.MagicMock()
        with patch.object(runner, "_base_runner", return_value=base):
            with patch.object(runner, "_prepare_rtos_capture"):
                with patch.object(
                    runner,
                    "_background_serial_capture",
                    return_value=nullcontext(),
                ):
                    with patch.object(runner, "_monitor_dual_serial") as monitor:
                        runner._run_jailhouse_trial(control, rtos, config)

        base._boot_and_login.assert_called_once_with(control)
        base._load_jailhouse_module.assert_called_once()
        joined = "\n".join(control.commands)
        self.assertLess(joined.index("jailhouse enable"), joined.index("cell create"))
        self.assertLess(joined.index("cell create"), joined.index("cell load"))
        self.assertLess(joined.index("cell load"), joined.index("cell start"))
        monitor.assert_called_once()

    def test_jailhouse_int_latency_accepts_rc_before_delayed_kernel_log(self):
        class DelayedKernelSerial:
            def __init__(self):
                self.buffer = bytearray()

            def command(self, command, prompt, timeout):
                self.buffer.extend(
                    b"\n__INT_CREATE_RC__=0\n"
                    b"__XHYPASS_PROMPT__# "
                    b"[ 39.556461] Created Jailhouse cell\n"
                )
                raise SerialTimeout("prompt displaced by delayed kernel log")

        output = IntLatencyRunner._checked_serial_command(
            DelayedKernelSerial(),
            "jailhouse cell create tl3588-QSemOS-RT.cell",
            r"__XHYPASS_PROMPT__#\s*$",
            "CREATE",
            timeout=60,
        )
        self.assertIn(b"__INT_CREATE_RC__=0", output)

    def test_int_latency_command_ignores_stale_prompt_before_rc_marker(self):
        class StalePromptSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.patterns = []

            def command(self, command, prompt, timeout):
                self.buffer.extend(b"__XHYPASS_PROMPT__# ")
                return bytes(self.buffer)

            def expect(self, patterns, timeout):
                self.patterns.extend(patterns)
                self.buffer.extend(b"\n__INT_XEN_PIN_1_RC__=0\n")
                return 0, bytes(self.buffer)

        session = StalePromptSerial()
        output = IntLatencyRunner._checked_serial_command(
            session,
            "xl vcpu-pin 1 0 1",
            r"__XHYPASS_PROMPT__#\s*$",
            "XEN_PIN_1",
            timeout=30,
        )

        self.assertIn(b"__INT_XEN_PIN_1_RC__=0", output)
        self.assertIn("__INT_XEN_PIN_1_RC__", session.patterns[0])

    def test_int_latency_command_accepts_xen_log_joined_to_rc_marker(self):
        class InterleavedXenSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.expect_called = False

            def command(self, command, prompt, timeout):
                return (
                    b"xl vcpu-pin 0 1 3; printf marker\n"
                    b"(XEN) common/sched/null.c:389: 1 <-- NULL (d0v1)\n"
                    b"__INT_XEN_PIN_2_RC__=0(XEN) common/sched/null.c:357: "
                    b"3 <-- d0v1\n__XHYPASS_PROMPT__# "
                )

            def expect(self, patterns, timeout):
                self.expect_called = True
                raise AssertionError("valid interleaved RC should be accepted")

        session = InterleavedXenSerial()
        output = IntLatencyRunner._checked_serial_command(
            session,
            "xl vcpu-pin 0 1 3",
            r"__XHYPASS_PROMPT__#\s*$",
            "XEN_PIN_2",
            timeout=30,
        )

        self.assertIn(b"__INT_XEN_PIN_2_RC__=0(XEN)", output)
        self.assertFalse(session.expect_called)

    def test_int_latency_interleaved_rc_still_rejects_command_echo(self):
        echoed = (
            b"printf '__INT_XEN_PIN_2_RC__=0(XEN)'\n"
            b"__XHYPASS_PROMPT__# "
        )
        self.assertFalse(
            IntLatencyRunner._has_checked_rc(
                echoed, "__INT_XEN_PIN_2_RC__", 0
            )
        )

    def test_jailhouse_int_latency_recovers_root_shell_after_auto_logout(self):
        class LoggedOutSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.sent = []
                self.matches = [0, 2, 0]

            def send(self, data):
                self.sent.append(data)

            def sendline(self, line=""):
                self.sent.append(line)

            def expect(self, patterns, timeout):
                return self.matches.pop(0), b""

        session = LoggedOutSerial()
        IntLatencyRunner._recover_rootcell_shell(
            session,
            {
                "login_prompt": r"tl3588 login:\s*$",
                "username": "root",
                "shell_prompts": [r"root@[^\r\n]*#\s*$", r"#\s*$"],
            },
        )
        self.assertIn("\x03", session.sent)
        self.assertIn("root", session.sent)
        self.assertEqual(session.sent[-1], "export PS1='__XHYPASS_PROMPT__# '")

    def test_jailhouse_int_latency_reboots_directly_without_cleanup(self):
        class FinishSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.raw = []

            def sendline(self, line=""):
                self.lines.append(line)

            def send(self, data):
                self.raw.append(data)

            def drain(self, seconds):
                return b"tl3588 ~ # "

        config = run_tl3588_int_latency_jailhouse.build_run_config("idle")
        runner = IntLatencyRunner([config], Path("data/RK3588"), dry_run=True)
        control = FinishSerial()
        base = unittest.mock.MagicMock()
        with patch.object(runner, "_checked_serial_command") as cleanup:
            with patch.object(runner, "_base_runner", return_value=base):
                with patch("xhypass_lab.int_latency_runner.time.sleep"):
                    runner._finish_jailhouse_trial(control, config)

        cleanup.assert_not_called()
        self.assertEqual(control.raw, ["\x03"])
        self.assertEqual(control.lines[-3:], ["root", "", "    /sbin/reboot -f"])
        base._reach_uboot.assert_called_once()

    def test_xen_int_latency_reboots_without_waiting_for_flooded_prompt(self):
        class FinishSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.raw = []

            def sendline(self, line=""):
                self.lines.append(line)

            def send(self, data):
                self.raw.append(data)

            def drain(self, seconds):
                return b"tl3588 ~ # "

        config = run_tl3588_int_latency_xen.build_run_config(
            "xen_credit2", "idle"
        )
        runner = IntLatencyRunner([config], Path("data/RK3588"), dry_run=True)
        control = FinishSerial()
        base = unittest.mock.MagicMock()
        with patch.object(runner, "_base_runner", return_value=base):
            with patch("xhypass_lab.int_latency_runner.time.sleep"):
                runner._finish_xen_dom0less_trial(control, config)

        self.assertEqual(control.raw, ["\x03"])
        self.assertEqual(control.lines[-3:], ["root", "", "    /sbin/reboot -f"])
        base._reach_uboot.assert_called_once()

    def test_tl3588_xen_int_latency_artifact_and_pin_mapping(self):
        credit_idle = run_tl3588_int_latency_xen.build_run_config(
            "xen_credit2", "idle"
        )
        credit_stress = run_tl3588_int_latency_xen.build_run_config(
            "xen_credit2", "stress"
        )
        null_idle = run_tl3588_int_latency_xen.build_run_config(
            "xen_null", "idle"
        )

        self.assertEqual(
            run_tl3588_int_latency_xen.ENVIRONMENTS,
            (
                "xen_credit2",
                "xen_credit2_WFX",
                "xen_null",
                "xen_null_WFX",
            ),
        )
        self.assertEqual(
            credit_idle["environment"]["boot_command"],
            "run boot_xen_dom0less",
        )
        self.assertEqual(
            credit_idle["environment"]["int_latency_pin_commands"],
            ["xl vcpu-pin 0 3 7", "xl vcpu-pin 1 0 3"],
        )
        self.assertEqual(
            null_idle["environment"]["int_latency_dynamic_pin_swap"],
            {
                "moving_domain_id": 1,
                "moving_vcpu_id": 0,
                "target_pcpu": 3,
            },
        )

        def artifact_names(config):
            return [
                (Path(item["source"]).name, Path(item["destination"]).name)
                for item in config["environment"]["int_latency_boot_files"]
            ]

        self.assertEqual(
            artifact_names(credit_idle),
            [
                ("qsemos-rt.bin_native", "qsemos-rt.bin"),
                (
                    "tl3588-evm-xen-dom0less.dtb.credit2",
                    "tl3588-evm-xen-dom0less.dtb",
                ),
            ],
        )
        self.assertEqual(
            artifact_names(credit_stress)[0],
            ("qsemos-rt.bin_native_stress", "qsemos-rt.bin"),
        )
        self.assertEqual(
            artifact_names(null_idle)[1],
            (
                "tl3588-evm-xen-dom0less.dtb.null",
                "tl3588-evm-xen-dom0less.dtb",
            ),
        )
        self.assertEqual(
            artifact_names(
                run_tl3588_int_latency_xen.build_run_config(
                    "xen_credit2_WFX", "idle"
                )
            )[1][0],
            "tl3588-evm-xen-dom0less.dtb.credit2.WFX",
        )
        null_wfx = run_tl3588_int_latency_xen.build_run_config(
            "xen_null_WFX", "stress"
        )
        self.assertEqual(
            artifact_names(null_wfx),
            [
                ("qsemos-rt.bin_native_stress", "qsemos-rt.bin"),
                (
                    "tl3588-evm-xen-dom0less.dtb.null.WFX",
                    "tl3588-evm-xen-dom0less.dtb",
                ),
            ],
        )
        self.assertIn(
            "int_latency_dynamic_pin_swap", null_wfx["environment"]
        )

    def test_tl3588_xen_null_dynamically_swaps_dom1_vcpu0_with_pcpu3(self):
        before = b"""\
Name ID VCPU CPU State Time(s) Affinity
Domain-0 0 0 0 -b- 3.2 all / all
Domain-0 0 2 3 r-- 2.3 all / all
Domain-0 0 3 2 -b- 2.1 all / all
(null) 1 0 4 r-- 30.9 all / all
"""
        after = b"""\
Name ID VCPU CPU State Time(s) Affinity
Domain-0 0 0 0 -b- 3.2 all / all
Domain-0 0 2 4 r-- 2.4 4 / all
Domain-0 0 3 2 -b- 2.1 all / all
(null) 1 0 3 r-- 31.0 3 / all
"""
        config = run_tl3588_int_latency_xen.build_run_config(
            "xen_null", "idle"
        )
        runner = IntLatencyRunner([config], Path("unused"), dry_run=True)
        with patch.object(
            runner,
            "_checked_serial_command",
            side_effect=[before, b"", b"", after],
        ) as command:
            runner._apply_dynamic_vcpu_swap(
                unittest.mock.MagicMock(),
                r"__XHYPASS_PROMPT__#\s*$",
                config["environment"]["int_latency_dynamic_pin_swap"],
            )

        self.assertEqual(
            [call.args[1] for call in command.call_args_list],
            [
                "xl vcpu-list",
                "xl vcpu-pin 1 0 3",
                "xl vcpu-pin 0 2 4",
                "xl vcpu-list",
            ],
        )

    def test_int_latency_boot_artifacts_are_copied_to_explicit_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_bin = root / "qsemos-rt.bin_native"
            source_dtb = root / "board.dtb.credit2"
            target = root / "dom0less"
            target.mkdir()
            source_bin.write_bytes(b"idle-bin")
            source_dtb.write_bytes(b"credit-dtb")
            config = run_tl3588_int_latency_xen.build_run_config(
                "xen_credit2", "idle"
            )
            config["environment"]["int_latency_boot_files"] = [
                {
                    "source": str(source_bin),
                    "destination": str(target / "qsemos-rt.bin"),
                },
                {
                    "source": str(source_dtb),
                    "destination": str(target / "tl3588-evm-xen-dom0less.dtb"),
                },
            ]
            IntLatencyRunner._sync_boot_files(config)
            self.assertEqual((target / "qsemos-rt.bin").read_bytes(), b"idle-bin")
            self.assertEqual(
                (target / "tl3588-evm-xen-dom0less.dtb").read_bytes(),
                b"credit-dtb",
            )

    def test_tl3588_xhypass_int_latency_uses_pass_binary_and_credit2_dtb(self):
        idle = run_tl3588_int_latency_xhypass.build_run_config("idle")
        stressed = run_tl3588_int_latency_xhypass.build_run_config("stress")

        self.assertEqual(idle["environment_name"], "XHyPass")
        self.assertEqual(
            idle["environment"]["boot_command"], "run boot_xen_dom0less"
        )
        self.assertEqual(
            idle["environment"]["int_latency_pin_commands"],
            ["xl vcpu-pin 0 3 7", "xl vcpu-pin 1 0 3"],
        )
        idle_files = idle["environment"]["int_latency_boot_files"]
        stress_files = stressed["environment"]["int_latency_boot_files"]
        self.assertEqual(Path(idle_files[0]["source"]).name, "qsemos-rt.bin_pass")
        self.assertEqual(
            Path(stress_files[0]["source"]).name,
            "qsemos-rt.bin_pass_stress",
        )
        self.assertEqual(
            Path(idle_files[1]["source"]).name,
            "tl3588-evm-xen-dom0less.dtb.credit2",
        )
        self.assertEqual(Path(idle_files[0]["destination"]).name, "qsemos-rt.bin")

    def test_tl3588_int_latency_matrix_has_seven_environments_and_two_conditions(self):
        self.assertGreaterEqual(
            run_tl3588_int_latency_matrix.TARGET_RUNS_PER_CONDITION, 1
        )
        self.assertEqual(
            run_tl3588_int_latency_matrix.ENVIRONMENTS,
            (
                "bare",
                "jailhouse",
                "xen_credit2",
                "xen_credit2_WFX",
                "xen_null",
                "xen_null_WFX",
                "XHyPass",
            ),
        )
        self.assertEqual(
            run_tl3588_int_latency_matrix.CONDITIONS, ("idle", "stress")
        )
        for environment in run_tl3588_int_latency_matrix.ENVIRONMENTS:
            for condition in run_tl3588_int_latency_matrix.CONDITIONS:
                config = run_tl3588_int_latency_matrix.build_run_config(
                    environment, condition
                )
                self.assertEqual(config["environment_name"], environment)
                self.assertEqual(config["experiment"]["condition"], condition)

    def test_tl3588_int_latency_matrix_counts_only_logs_with_completion_marker(self):
        config = run_tl3588_int_latency_matrix.build_run_config("bare", "idle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series = run_tl3588_int_latency_matrix.result_series(config, root)
            series.mkdir(parents=True)
            (series / "rtos_run1.log").write_text(
                "PMU diff samples: 600000\n", encoding="utf-8"
            )
            (series / "rtos_run2.log").write_text(
                "incomplete output\n", encoding="utf-8"
            )
            valid = run_tl3588_int_latency_matrix.successful_results(
                config, root
            )
        self.assertEqual([path.name for path in valid], ["rtos_run1.log"])

    def test_e2000q_int_latency_matrix_definition_and_configs(self):
        matrix = run_e2000q_int_latency_matrix
        self.assertGreaterEqual(matrix.TARGET_RUNS_PER_CONDITION, 1)
        self.assertEqual(
            matrix.ENVIRONMENTS,
            (
                "bare",
                "jailhouse",
                "xen_credit2",
                "xen_credit2_WFX",
                "xen_null",
                "xen_null_WFX",
                "XHyPass",
            ),
        )
        self.assertEqual(matrix.CONDITIONS, ("idle", "stress"))
        self.assertEqual(matrix.DURATION_SECONDS, 600)
        for environment in matrix.ENVIRONMENTS:
            for condition in matrix.CONDITIONS:
                config = matrix.build_run_config(environment, condition)
                self.assertEqual(config["environment_name"], environment)
                self.assertEqual(config["experiment"]["condition"], condition)
                self.assertEqual(config["experiment"]["duration_seconds"], 600)
                self.assertEqual(config["control_serial"]["port"], "COM12")
                self.assertEqual(config["rtos_serial"]["port"], "COM11")
                if config["environment"].get("environment_type") == "xen":
                    expected_pin_commands = (
                        ["xl vcpu-pin 1 0 1", "xl vcpu-pin 0 1 3"]
                        if environment in {"xen_null", "xen_null_WFX"}
                        else ["xl vcpu-pin 0 1 3", "xl vcpu-pin 1 0 1"]
                    )
                    self.assertEqual(
                        config["environment"]["int_latency_pin_commands"],
                        expected_pin_commands,
                    )

    def test_int_latency_waits_five_seconds_after_completion_marker(self):
        self.assertFalse(
            IntLatencyRunner._rtos_run_complete(
                True,
                now=19.9,
                completion_seen_at=15.0,
                completion_settle_seconds=5.0,
            )
        )
        self.assertTrue(
            IntLatencyRunner._rtos_run_complete(
                True,
                now=20.0,
                completion_seen_at=15.0,
                completion_settle_seconds=5.0,
            )
        )
        self.assertFalse(
            IntLatencyRunner._rtos_run_complete(
                False,
                now=100.0,
                completion_seen_at=None,
                completion_settle_seconds=5.0,
            )
        )

    def test_int_latency_rejects_marker_before_current_run_guard(self):
        self.assertFalse(
            IntLatencyRunner._completion_marker_is_eligible(
                elapsed_seconds=15.0,
                completion_not_before_seconds=540.0,
            )
        )
        self.assertTrue(
            IntLatencyRunner._completion_marker_is_eligible(
                elapsed_seconds=600.0,
                completion_not_before_seconds=540.0,
            )
        )

    def test_bare_int_latency_stops_uboot_during_post_marker_capture(self):
        class FakeControl:
            def __init__(self):
                self.buffer = bytearray()
                self.reads = 0
                self.sent = []

            def _read_once(self):
                self.reads += 1
                raw = b"DDR V1.12\n" if self.reads == 1 else b"=> "
                self.buffer.extend(raw)
                return raw

            def send(self, data):
                self.sent.append(data)

        class FakeRtos:
            def __init__(self):
                self.buffer = bytearray()
                self.reads = 0

            def _read_once(self):
                self.reads += 1
                raw = b"PMU diff samples: 600000\n" if self.reads == 1 else b""
                self.buffer.extend(raw)
                return raw

        control = FakeControl()
        IntLatencyRunner._monitor_dual_serial(
            control,
            FakeRtos(),
            completion_pattern=r"PMU diff samples:\s*600000",
            timeout=2,
            completion_settle_seconds=0,
            completion_not_before_seconds=0,
            started_at=time.monotonic(),
            retry_patterns=[],
            stop_bare_autoboot=True,
            uboot_prompt=r"=>\s*$",
            reset_markers=[r"DDR V\d"],
            login_prompt=r"tl3588 login:\s*$",
        )
        self.assertEqual(control.sent, [b"\x03"])

    def test_platform_configs_expose_the_same_seven_environments(self):
        tl3588 = load_platform_config("RK3588")
        e2000q = load_platform_config("E2000Q")
        self.assertEqual(tl3588["platform"]["name"], "RK3588")
        self.assertTrue(tl3588["platform"]["ready"])
        self.assertEqual(e2000q["platform"]["name"], "E2000Q")
        self.assertTrue(e2000q["platform"]["ready"])
        self.assertEqual(set(PLATFORM_ENVIRONMENTS), set(tl3588["environments"]))
        self.assertEqual(set(PLATFORM_ENVIRONMENTS), set(e2000q["environments"]))

    def test_e2000q_bare_cyclictest_settings(self):
        cyclictest = run_e2000q_experiment.build_run_config("cyclictest")
        stressed = run_e2000q_experiment.build_run_config("cyclictest-stress")

        self.assertEqual(cyclictest["platform_name"], "E2000Q")
        self.assertEqual(cyclictest["serial"]["port"], "COM12")
        self.assertEqual(cyclictest["serial"]["baudrate"], 115200)
        self.assertEqual(cyclictest["environment"]["boot_command"], "run boot_oee")
        self.assertEqual(cyclictest["experiment"]["cpu"], 3)
        self.assertEqual(stressed["experiment"]["cpu"], 3)
        self.assertEqual(stressed["experiment"]["stress_cpus"], "3")
        self.assertEqual(stressed["experiment"]["stress_vm_workers"], 1)
        self.assertNotIn("ssh", cyclictest["reboot"])
        self.assertEqual(
            run_e2000q_bare_smoke.EXPERIMENTS,
            ("cyclictest", "cyclictest-stress"),
        )

    def test_all_e2000q_environments_are_configured(self):
        config = load_platform_config("E2000Q")
        for environment in PLATFORM_ENVIRONMENTS:
            with self.subTest(environment=environment):
                run = resolved_run_config(config, environment, "cyclictest", {})
                self.assertFalse(
                    run["environment"]["boot_command"].startswith("TODO_")
                )

    def test_e2000q_four_xen_environments_run_on_serial_cpu3(self):
        expected_sources = {
            "xen_credit2": "Xen_credit2",
            "xen_credit2_WFX": "Xen_credit2_nativeWFX",
            "xen_null": "Xen_null",
            "xen_null_WFX": "Xen_null_nativeWFX",
        }
        self.assertEqual(
            run_e2000q_xen_smoke.ENVIRONMENTS,
            tuple(expected_sources),
        )
        for environment, source_name in expected_sources.items():
            with self.subTest(environment=environment):
                cyclictest = run_e2000q_experiment.build_run_config(
                    "cyclictest", environment
                )
                stressed = run_e2000q_experiment.build_run_config(
                    "cyclictest-stress", environment
                )
                env = cyclictest["environment"]
                self.assertEqual(env["boot_command"], "run boot_xen")
                self.assertTrue(env["xen_serial_only"])
                self.assertEqual(
                    Path(env["local_boot_files"]["source_dir"]).name,
                    source_name,
                )
                self.assertEqual(
                    env["local_boot_files"]["target_dir"],
                    "experiments/deploy/E2000Q",
                )
                self.assertEqual(cyclictest["experiment"]["cpu"], 3)
                self.assertEqual(stressed["experiment"]["stress_cpus"], "3")

    def test_e2000q_xen_serial_only_skips_tl3588_dom1_setup(self):
        run = run_e2000q_experiment.build_run_config(
            "cyclictest", "xen_credit2"
        )
        runner = ExperimentRunner(run, Path("data/E2000Q"), True)
        with patch.object(runner, "_run_cyclictest") as run_cyclictest:
            runner._run_xen_cyclictest(object(), Path("unused"))
        run_cyclictest.assert_called_once()

    def test_e2000q_xhypass_loads_and_removes_module_on_cpu3(self):
        class XHyPassSerial:
            def __init__(self):
                self.commands = []
                self.module_loaded = False

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                if "insmod" in command:
                    self.module_loaded = True
                    return b"XI=0\n"
                elif "rmmod" in command:
                    self.module_loaded = False
                    return b"XR=0\n"
                if "echo XM=$?" in command:
                    return b"XM=0\n" if self.module_loaded else b"XM=1\n"
                return b"__XHYPASS_PROMPT__# "

            def drain(self, seconds):
                return b""

        run = run_e2000q_experiment.build_run_config(
            "cyclictest", "XHyPass"
        )
        env = run["environment"]
        self.assertEqual(env["boot_command"], "run boot_xen")
        self.assertEqual(
            Path(env["local_boot_files"]["source_dir"]).name, "XHyPass"
        )
        self.assertEqual(env["xhypass_module_cpu"], 3)
        self.assertEqual(
            env["xhypass_module_file"], "interrupt_passthrough.ko"
        )
        self.assertNotIn("pre_experiment_commands", env)
        self.assertNotIn("post_experiment_commands", env)

        runner = ExperimentRunner(run, Path("data/E2000Q"), True)
        serial = XHyPassSerial()
        with patch("xhypass_lab.runner.time.sleep"):
            with patch.object(runner, "_run_cyclictest") as run_cyclictest:
                runner._run_xen_cyclictest(serial, Path("unused"))
        run_cyclictest.assert_called_once()
        load = next(command for command in serial.commands if "insmod" in command)
        self.assertIn("taskset -c 3 insmod interrupt_passthrough.ko", load)
        self.assertIn("echo XI=$?", load)
        self.assertIn("", serial.commands)
        unload = next(command for command in serial.commands if "rmmod" in command)
        self.assertIn("taskset -c 3 rmmod interrupt_passthrough.ko", unload)
        self.assertIn("echo XR=$?", unload)
        self.assertTrue(
            all(command != "lsmod" for command in serial.commands)
        )
        self.assertFalse(serial.module_loaded)

    def test_e2000q_reboot_removes_residual_xhypass_module_on_cpu3(self):
        class RebootSerial:
            def __init__(self):
                self.commands = []
                self.module_loaded = True

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                if "rmmod" in command:
                    self.module_loaded = False
                    return b"XR=0\n"
                if "echo XM=$?" in command:
                    return b"XM=0\n" if self.module_loaded else b"XM=1\n"
                return b"__REBOOT_PROBE__# "

            def drain(self, seconds):
                return b""

        run = run_e2000q_experiment.build_run_config(
            "cyclictest", "bare"
        )
        self.assertTrue(run["reboot"]["prepare_serial_environment"])
        self.assertEqual(run["reboot"]["xhypass_module_cpu"], 3)
        serial = RebootSerial()
        ExperimentRunner(run, Path("data/E2000Q"), True)._prepare_xen_reboot_serial(
            serial
        )
        unload = next(command for command in serial.commands if "rmmod" in command)
        self.assertIn(
            "taskset -c 3 rmmod interrupt_passthrough.ko", unload
        )
        self.assertEqual(
            sum("echo XM=$?" in command for command in serial.commands), 2
        )
        self.assertNotIn("lsmod", serial.commands)
        self.assertFalse(serial.module_loaded)

    def test_lsmod_is_the_authoritative_module_state(self):
        loaded = (
            b"Module Size Used by\n"
            b"interrupt_passthrough 16384 0\n"
            b"other_module 4096 1\n"
        )
        unloaded = b"Module Size Used by\nother_module 4096 1\n"
        self.assertTrue(
            ExperimentRunner._module_is_loaded(
                loaded, "interrupt_passthrough"
            )
        )
        self.assertFalse(
            ExperimentRunner._module_is_loaded(
                unloaded, "interrupt_passthrough"
            )
        )

    def test_e2000q_smoke_and_formal_matrix_definitions(self):
        expected = (
            "bare",
            "jailhouse",
            "xen_credit2",
            "xen_credit2_WFX",
            "xen_null",
            "xen_null_WFX",
            "XHyPass",
        )
        self.assertEqual(run_e2000q_all_smoke.ENVIRONMENTS, expected)
        self.assertEqual(run_e2000q_all_smoke.DURATION_SECONDS, 10)
        self.assertEqual(run_e2000q_all_smoke.RUNS_PER_CONDITION, 1)
        self.assertTrue(run_e2000q_all_smoke.SKIP_COMPLETED)
        self.assertEqual(
            run_e2000q_full_cyclictest_matrix.ENVIRONMENTS, expected
        )
        self.assertEqual(
            run_e2000q_full_cyclictest_matrix.EXPERIMENTS,
            ("cyclictest", "cyclictest-stress"),
        )
        self.assertEqual(
            run_e2000q_full_cyclictest_matrix.DURATION_SECONDS, 600
        )
        self.assertEqual(
            run_e2000q_full_cyclictest_matrix.RUNS_PER_CONDITION, 5
        )
        formal = run_e2000q_experiment.build_run_config(
            "cyclictest-stress",
            "XHyPass",
            duration_seconds=600,
            interval_us=1000,
        )
        self.assertEqual(formal["experiment"]["duration_seconds"], 600)
        self.assertEqual(formal["experiment"]["cpu"], 3)
        self.assertEqual(formal["experiment"]["stress_cpus"], "3")

    def test_e2000q_jailhouse_rootcell_settings(self):
        cyclictest = run_e2000q_experiment.build_run_config(
            "cyclictest", "jailhouse"
        )
        stressed = run_e2000q_experiment.build_run_config(
            "cyclictest-stress", "jailhouse"
        )
        env = cyclictest["environment"]

        self.assertEqual(env["boot_command"], "run boot_oee")
        self.assertTrue(env["jailhouse_rootcell_only"])
        self.assertEqual(env["module_load_command"], "insmod jailhouse.ko")
        self.assertEqual(
            env["enable_command"],
            "jailhouse enable /usr/share/jailhouse/cells/e2000q-gsk-8g.cell",
        )
        self.assertNotIn("ssh", env)
        self.assertNotIn("linux_command", env)
        self.assertEqual(cyclictest["experiment"]["cpu"], 3)
        self.assertEqual(stressed["experiment"]["stress_cpus"], "3")
        self.assertEqual(
            run_e2000q_jailhouse_smoke.EXPERIMENTS,
            ("cyclictest", "cyclictest-stress"),
        )

    def test_e2000q_jailhouse_enables_and_tests_on_serial_rootcell(self):
        class RootcellSerial:
            def __init__(self):
                self.commands = []

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                return b"__JH_ENABLE_RC__=0\n__XHYPASS_PROMPT__# "

        run = run_e2000q_experiment.build_run_config(
            "cyclictest", "jailhouse"
        )
        runner = ExperimentRunner(run, Path("data/E2000Q"), True)
        serial = RootcellSerial()
        with patch.object(runner, "_load_jailhouse_module") as load_module:
            with patch.object(runner, "_run_cyclictest") as run_cyclictest:
                runner._run_jailhouse_cyclictest(serial, Path("unused"))
                runner._run_jailhouse_cyclictest(serial, Path("unused"))

        load_module.assert_called_once()
        self.assertEqual(len(serial.commands), 1)
        self.assertIn(
            "jailhouse enable /usr/share/jailhouse/cells/e2000q-gsk-8g.cell",
            serial.commands[0],
        )
        self.assertIn("__JH_ENABLE_RC__", serial.commands[0])
        self.assertEqual(run_cyclictest.call_count, 2)

    def test_platform_data_roots_are_separate(self):
        self.assertEqual(platform_data_root("RK3588").name, "RK3588")
        self.assertEqual(platform_data_root("E2000Q").name, "E2000Q")
        self.assertNotEqual(
            platform_data_root("RK3588"), platform_data_root("E2000Q")
        )

    def test_full_nn_matrix_definition_and_environment_alignment(self):
        self.assertEqual(
            run_full_nn_matrix.ENVIRONMENTS,
            (
                "bare",
                "jailhouse",
                "xen_credit2",
                "xen_credit2_WFX",
                "xen_null",
                "xen_null_WFX",
                "XHyPass",
            ),
        )
        self.assertGreaterEqual(run_full_nn_matrix.RUNS_PER_ENVIRONMENT, 1)
        self.assertEqual(run_full_nn_matrix.DURATION_SECONDS, 600)
        self.assertEqual(
            [profile["name"] for profile in run_full_nn_matrix.LOAD_PROFILES],
            ["light", "medium", "heavy"],
        )
        configs = {
            environment: run_full_nn_matrix.build_formal_config(environment)
            for environment in run_full_nn_matrix.ENVIRONMENTS
        }
        for environment, run in configs.items():
            exp = run["experiment"]
            self.assertEqual(run["environment_name"], environment)
            expected_profile = (
                "dual-tflite-formal-v2-cgroup"
                if environment == "bare"
                else "dual-tflite-formal-v1"
            )
            self.assertEqual(exp["profile_name"], expected_profile)
            self.assertEqual(exp["duration_seconds"], 600)
            self.assertEqual(exp["cyclictest_interval_us"], 1000)
            self.assertEqual(
                [profile["name"] for profile in exp["profiles"]],
                ["light", "medium", "heavy"],
            )
        for environment in ("xen_credit2", "xen_credit2_WFX", "XHyPass"):
            exp = configs[environment]["experiment"]
            self.assertEqual(exp["dom0_cyclictest_cpu"], 6)
            self.assertEqual(exp["dom1_cyclictest_cpu"], 6)
            self.assertEqual(exp["dom0_workload_cpus"], "0-5")
            self.assertEqual(exp["dom1_workload_cpus"], "0-5")
        for environment in ("xen_null", "xen_null_WFX"):
            exp = configs[environment]["experiment"]
            self.assertEqual(exp["dom0_cyclictest_cpu"], 3)
            self.assertEqual(exp["dom1_cyclictest_cpu"], 3)
            self.assertEqual(exp["dom0_workload_cpus"], "0-2")
            self.assertEqual(exp["dom1_workload_cpus"], "0-2")
        self.assertEqual(configs["XHyPass"]["experiment"]["xhypass_module_cpu"], 6)

    def test_full_nn_matrix_schedules_environments_round_robin(self):
        counts = {"env_a": 0, "env_b": 0}
        calls = []

        def config(environment):
            return {"environment_name": environment}

        def archives(run):
            return [object()] * counts[run["environment_name"]]

        class FakeRunner:
            def __init__(self, run, data_root, dry_run=False):
                self.environment = run["environment_name"]

            def run(self, runs, reboot_policy):
                calls.append(self.environment)
                counts[self.environment] += 1

        workflow = run_full_nn_matrix
        with patch.object(workflow, "ENVIRONMENTS", ("env_a", "env_b")):
            with patch.object(workflow, "RUNS_PER_ENVIRONMENT", 2):
                with patch.object(
                    workflow, "BUILDERS", {"env_a": object(), "env_b": object()}
                ):
                    with patch.object(
                        workflow, "build_formal_config", side_effect=config
                    ):
                        with patch.object(
                            workflow, "_valid_result_archives", side_effect=archives
                        ):
                            with patch.object(
                                workflow, "NNExperimentRunner", FakeRunner
                            ):
                                with patch.object(
                                    workflow,
                                    "_write_campaign",
                                    return_value=Path("campaign.json"),
                                ):
                                    result = workflow.main()

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["env_a", "env_b", "env_a", "env_b"])

    def test_full_nn_matrix_continues_after_an_incomplete_round(self):
        counts = {"env_a": 0, "env_b": 0}
        calls = []

        def config(environment):
            return {"environment_name": environment}

        def archives(run):
            return [object()] * counts[run["environment_name"]]

        class FakeRunner:
            def __init__(self, run, data_root, dry_run=False):
                self.environment = run["environment_name"]

            def run(self, runs, reboot_policy):
                calls.append(self.environment)
                if self.environment == "env_b":
                    raise RuntimeError("simulated failure")
                counts[self.environment] += 1

        workflow = run_full_nn_matrix
        with patch.object(workflow, "ENVIRONMENTS", ("env_a", "env_b")):
            with patch.object(workflow, "RUNS_PER_ENVIRONMENT", 2):
                with patch.object(
                    workflow, "BUILDERS", {"env_a": object(), "env_b": object()}
                ):
                    with patch.object(
                        workflow, "build_formal_config", side_effect=config
                    ):
                        with patch.object(
                            workflow, "_valid_result_archives", side_effect=archives
                        ):
                            with patch.object(
                                workflow, "NNExperimentRunner", FakeRunner
                            ):
                                with patch.object(
                                    workflow,
                                    "_write_campaign",
                                    return_value=Path("campaign.json"),
                                ):
                                    with patch.object(workflow.traceback, "print_exc"):
                                        result = workflow.main()

        self.assertEqual(result, 1)
        self.assertEqual(calls, ["env_a", "env_b", "env_a", "env_b"])
        self.assertEqual(counts, {"env_a": 2, "env_b": 0})

    def test_config_override(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest", {"duration_seconds": 10})
        self.assertEqual(run["experiment"]["duration_seconds"], 10)
        self.assertEqual(run["environment"]["boot_command"], "run boot_oee")

    def test_plan_is_unique_per_run(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest", {})
        with tempfile.TemporaryDirectory() as directory:
            paths = ExperimentRunner(run, Path(directory), True).plan(3, "each-run")
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(paths[0].name, "run_001")
        self.assertEqual(paths[2].name, "run_003")
        self.assertEqual(paths[0].parent.name, "completed")
        self.assertEqual(paths[0].parent.parent.name, "_details")
        self.assertEqual(paths[0].parents[2].name, "cpu6_t1_p99_i1000us_d600s_h10000")
        self.assertEqual(paths[0].parents[3].name, "bare")
        self.assertEqual(paths[0].parents[4].name, "cyclictest")

    def test_plan_continues_numbering_for_same_condition(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest", {})
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(run, Path(directory), True)
            old = runner.plan(1, "each-run")[0]
            series = old.parent.parent.parent
            series.mkdir(parents=True)
            (series / "hist_run1.txt").write_text("hist")
            paths = runner.plan(2, "each-run")
        self.assertEqual([path.name for path in paths], ["run_002", "run_003"])

    def test_completed_histogram_is_published_at_condition_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "condition" / "_details" / "completed" / "run_001"
            output.mkdir(parents=True)
            (output / "hist.txt").write_text("000005 000001\n")
            published = ExperimentRunner._publish_histogram(output)
            self.assertEqual(published.name, "hist_run1.txt")
            self.assertEqual(published.parent.name, "condition")
            self.assertEqual(published.read_text(), "000005 000001\n")

    def test_condition_name_includes_stress_settings(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest-stress", {})
        self.assertEqual(
            condition_name(run),
            "cpu6_t1_p99_i1000us_d600s_h10000_stress-cpu6_vm1_256M",
        )

    def test_safe_name(self):
        self.assertEqual(safe_name("XHyPass / cyclictest"), "XHyPass-cyclictest")

    def test_unconfigured_environment_is_rejected(self):
        config = load_config(Path("config/RK3588/lab.json"))
        with self.assertRaises(ConfigError):
            resolved_run_config(config, "xen", "cyclictest", {})

    def test_jailhouse_configuration(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "jailhouse", "cyclictest", {})
        self.assertEqual(run["environment"]["boot_command"], "run boot_oee")
        self.assertEqual(run["environment"]["rootcell_ip"], "202.197.67.50")
        self.assertEqual(
            run["environment"]["module_load_command"], "insmod jailhouse.ko"
        )
        self.assertEqual(run["environment"]["module_settle_seconds"], 20)
        self.assertIn("login", run["environment"]["nonroot_login_prompt"])
        self.assertIn("jailhouse enable", run["environment"]["enable_command"])
        self.assertIn("isolcpus=3 rcu_nocb=3", run["environment"]["linux_command"])
        self.assertFalse(run["reboot"]["allow_serial_fallback"])

    def test_jailhouse_module_loads_from_root_home_before_network_setup(self):
        class ModuleSerial:
            def __init__(self):
                self.commands = []

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                return b"__JH_INSMOD_RC__=0\n__XHYPASS_PROMPT__# "

        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "jailhouse", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        serial = ModuleSerial()
        runner._load_jailhouse_module(
            serial, run["environment"], r"__XHYPASS_PROMPT__#\s*$"
        )
        self.assertEqual(len(serial.commands), 1)
        self.assertIn("grep -qx jailhouse", serial.commands[0])
        self.assertIn("insmod jailhouse.ko", serial.commands[0])
        self.assertIn("sleep 20", serial.commands[0])
        self.assertIn("__JH_INSMOD_RC__", serial.commands[0])

    def test_xen_credit2_configuration_and_command_alignment(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2", "cyclictest-stress", {})
        env = run["environment"]
        exp = run["experiment"]
        self.assertEqual(env["boot_command"], "run boot_xen")
        self.assertTrue(env["local_boot_files"]["source_dir"].endswith("Xen_credit2"))
        self.assertEqual(env["ssh"]["host"], "202.197.67.50")
        self.assertEqual(
            env["dom0_init_commands"],
            ["cd ~/dom-interrupt", "sh set_bridge.sh", "xl create compute_NN_dom1.cfg"],
        )
        self.assertIn("xl vcpu-pin 0 6 6", env["dom0_pin_command"])
        self.assertIn("xl vcpu-pin dom1 6 7", env["dom1_pin_command"])
        self.assertIn("xl vcpu-pin 0 0 0-5", env["dom0_pin_command"])
        self.assertIn("xl vcpu-pin dom1 0 0-5", env["dom1_pin_command"])
        self.assertNotIn(" all", env["dom0_pin_command"])
        self.assertNotIn(" all", env["dom1_pin_command"])
        self.assertEqual(env["dom1_console_command"], "xl console dom1")
        self.assertEqual(env["dom1_username"], "root")
        self.assertEqual(env["dom1_boot_timeout"], 180)
        self.assertEqual(env["post_dom1_login_delay_seconds"], 10)
        self.assertEqual(
            env["pin_verify_commands"], ["xl vcpu-list 0", "xl vcpu-list dom1"]
        )
        self.assertEqual(exp["cpu"], 6)
        self.assertEqual(exp["stress_cpus"], "6")
        self.assertEqual(exp["interval_us"], 1000)

    def test_xen_credit2_entrypoint(self):
        run = run_xen_credit2.build_run_config()
        self.assertEqual(run["environment_name"], "xen_credit2")
        self.assertEqual(run["experiment_name"], run_xen_credit2.EXPERIMENT)
        self.assertEqual(run["experiment"]["duration_seconds"], 10)

    def test_xen_credit2_wfx_inherits_xen_workflow(self):
        config = load_config(Path("config/RK3588/lab.json"))
        base = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        native = resolved_run_config(
            config, "xen_credit2_WFX", "cyclictest", {}
        )
        self.assertEqual(native["environment"]["environment_type"], "xen")
        self.assertEqual(
            native["environment"]["boot_command"],
            base["environment"]["boot_command"],
        )
        self.assertEqual(
            native["environment"]["dom0_pin_command"],
            base["environment"]["dom0_pin_command"],
        )
        self.assertEqual(
            native["environment"]["dom1_pin_command"],
            base["environment"]["dom1_pin_command"],
        )
        self.assertTrue(
            native["environment"]["local_boot_files"]["source_dir"].endswith(
                "Xen_credit2_nativeWFX"
            )
        )
        self.assertEqual(
            native["environment"]["local_boot_files"]["target_dir"],
            base["environment"]["local_boot_files"]["target_dir"],
        )

    def test_xen_credit2_wfx_entrypoint(self):
        run = run_xen_credit2_wfx.build_run_config()
        self.assertEqual(run["environment_name"], "xen_credit2_WFX")
        self.assertEqual(run["environment"]["environment_type"], "xen")
        self.assertTrue(
            run["environment"]["local_boot_files"]["source_dir"].endswith(
                "Xen_credit2_nativeWFX"
            )
        )

    def test_xhypass_inherits_credit2_and_adds_rto_module_hooks(self):
        config = load_config(Path("config/RK3588/lab.json"))
        credit2 = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        xhypass = resolved_run_config(config, "XHyPass", "cyclictest", {})
        self.assertEqual(xhypass["environment"]["dom0_pin_command"], credit2["environment"]["dom0_pin_command"])
        self.assertEqual(xhypass["environment"]["dom1_pin_command"], credit2["environment"]["dom1_pin_command"])
        self.assertTrue(
            xhypass["environment"]["local_boot_files"]["source_dir"].endswith("XHyPass")
        )
        command = xhypass["environment"]["pre_experiment_commands"][0]
        self.assertIn("cd ~ && test -f interrupt_passthrough.ko", command)
        self.assertIn(
            "taskset -c 6 insmod ./interrupt_passthrough.ko "
            "rto_cpu=6 event_sgi=7",
            command,
        )
        self.assertEqual(
            xhypass["environment"]["xhypass_module_file"],
            "./interrupt_passthrough.ko",
        )
        self.assertEqual(
            xhypass["environment"]["xhypass_module_load_args"],
            "rto_cpu=6 event_sgi=7",
        )
        self.assertEqual(
            xhypass["environment"]["pre_experiment_success_marker"],
            "__XHYPASS_RTO_READY__",
        )
        self.assertEqual(
            xhypass["environment"]["post_module_load_delay_seconds"], 2
        )
        self.assertEqual(run_xhypass.build_run_config()["environment_name"], "XHyPass")
        self.assertEqual(run_xhypass_smoke.EXPERIMENTS, ("cyclictest", "cyclictest-stress"))

    def test_xhypass_module_is_removed_before_xen_domain(self):
        class LoadedModuleSSH(FakeRebootSSH):
            def run(self, command, **kwargs):
                self.commands.append(command)
                if "__XHYPASS_MODULE_LOADED__" in command:
                    return 0, b"__XHYPASS_MODULE_LOADED__\n", b""
                if command.startswith("if command -v xl"):
                    return 0, b"__XEN_ENV__\ndom1 1 1 7 b 0\n", b""
                return 0, b"", b""

        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "XHyPass", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        ssh = LoadedModuleSSH(b"")
        runner._prepare_xen_reboot_ssh(ssh, Path("unused.log"))
        unload = ssh.commands.index("taskset -c 6 rmmod interrupt_passthrough.ko")
        destroy = ssh.commands.index("xl destroy dom1")
        self.assertLess(unload, destroy)

    def test_serial_marker_must_be_a_complete_output_line(self):
        echoed_only = (
            b"if loaded; then echo __XHYPASS_MODULE_LOADED__; fi\r\n"
            b"__REBOOT_PROBE__# "
        )
        actual_output = echoed_only + b"\r\n__XHYPASS_MODULE_LOADED__\r\n"
        self.assertFalse(
            ExperimentRunner._has_output_line(
                echoed_only, "__XHYPASS_MODULE_LOADED__"
            )
        )
        self.assertTrue(
            ExperimentRunner._has_output_line(
                actual_output, "__XHYPASS_MODULE_LOADED__"
            )
        )

    def test_xen_null_configuration_and_entrypoint(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_null", "cyclictest-stress", {})
        env = run["environment"]
        exp = run["experiment"]
        self.assertEqual(env["environment_type"], "xen")
        self.assertEqual(env["boot_command"], "run boot_xen")
        self.assertTrue(env["local_boot_files"]["source_dir"].endswith("Xen_null"))
        self.assertEqual(
            env["dom0_init_commands"],
            [
                "cd ~/dom-interrupt",
                "xl vcpu-set 0 4",
                "xl vcpu-pin 0 0 0; xl vcpu-pin 0 1 1; xl vcpu-pin 0 2 4; xl vcpu-pin 0 3 6",
                "sh set_bridge.sh",
                "xl create compute_NN_dom1_nullsched.cfg",
            ],
        )
        self.assertEqual(env["dom0_pin_command"], "")
        self.assertEqual(
            env["dom0_init_post_delays_seconds"], {"xl vcpu-set 0 4": 2}
        )
        self.assertEqual(
            env["dom1_pin_command"],
            "xl vcpu-pin dom1 0 2; xl vcpu-pin dom1 1 3; xl vcpu-pin dom1 2 5; xl vcpu-pin dom1 3 7",
        )
        self.assertEqual(exp["cpu"], 3)
        self.assertEqual(exp["stress_cpus"], "3")
        self.assertEqual(exp["stress_vm_workers"], 1)
        direct = run_xen_null.build_run_config()
        self.assertEqual(direct["environment_name"], "xen_null")
        self.assertEqual(direct["experiment"]["cpu"], 3)
        self.assertEqual(direct["experiment"]["stress_cpus"], "3")

    def test_xen_null_wfx_only_changes_boot_artifact_source(self):
        config = load_config(Path("config/RK3588/lab.json"))
        base = resolved_run_config(config, "xen_null", "cyclictest-stress", {})
        wfx = resolved_run_config(config, "xen_null_WFX", "cyclictest-stress", {})
        for key in (
            "boot_command",
            "dom0_init_commands",
            "dom0_pin_command",
            "dom1_pin_command",
            "dom1_console_command",
            "post_dom1_login_delay_seconds",
        ):
            self.assertEqual(wfx["environment"][key], base["environment"][key])
        self.assertTrue(
            wfx["environment"]["local_boot_files"]["source_dir"].endswith(
                "Xen_null_nativeWFX"
            )
        )
        self.assertEqual(wfx["experiment"]["cpu"], 3)
        self.assertEqual(wfx["experiment"]["stress_cpus"], "3")
        direct = run_xen_null_wfx.build_run_config()
        self.assertEqual(direct["environment_name"], "xen_null_WFX")
        self.assertEqual(direct["experiment"]["cpu"], 3)

    def test_xen_variants_smoke_campaign(self):
        self.assertEqual(run_xen_variants_smoke.DURATION_SECONDS, 10)
        self.assertEqual(run_xen_variants_smoke.INTERVAL_US, 1000)
        self.assertEqual(run_xen_variants_smoke.RUNS_PER_CASE, 1)
        self.assertEqual(len(run_xen_variants_smoke.CAMPAIGN), 8)
        self.assertEqual(
            set(run_xen_variants_smoke.CAMPAIGN),
            {
                (environment, experiment)
                for environment in run_xen_variants_smoke.ENVIRONMENTS
                for experiment in run_xen_variants_smoke.EXPERIMENTS
            },
        )
        for environment, experiment in run_xen_variants_smoke.CAMPAIGN:
            config = run_xen_variants_smoke.build_case_config(
                environment, experiment
            )
            self.assertEqual(config["experiment"]["duration_seconds"], 10)
            self.assertEqual(config["experiment"]["interval_us"], 1000)
            expected_cpu = 3 if environment.startswith("xen_null") else 6
            self.assertEqual(config["experiment"]["cpu"], expected_cpu)
            if experiment == "cyclictest-stress":
                self.assertEqual(
                    config["experiment"]["stress_cpus"], str(expected_cpu)
                )

    def test_full_cyclictest_matrix_definition(self):
        matrix = run_full_cyclictest_matrix
        self.assertEqual(matrix.DURATION_SECONDS, 600)
        self.assertEqual(matrix.INTERVALS_US, (1_000, 10_000))
        self.assertEqual(matrix.RUNS_PER_CONDITION, 5)
        self.assertEqual(matrix.REBOOT_POLICY, "each-run")
        self.assertEqual(matrix.ENVIRONMENTS, ("bare", "jailhouse"))
        self.assertEqual(
            len(matrix.ENVIRONMENTS)
            * len(matrix.EXPERIMENTS)
            * len(matrix.INTERVALS_US)
            * matrix.RUNS_PER_CONDITION,
            40,
        )
        for environment in matrix.ENVIRONMENTS:
            for experiment in matrix.EXPERIMENTS:
                for interval_us in matrix.INTERVALS_US:
                    config = matrix.build_case_config(
                        environment, experiment, interval_us
                    )
                    self.assertEqual(
                        config["experiment"]["duration_seconds"], 600
                    )
                    self.assertEqual(
                        config["experiment"]["interval_us"], interval_us
                    )

    def test_nn_bare_configuration_and_loadgen_mapping(self):
        run = run_nn_bare.build_run_config()
        self.assertEqual(run["environment_name"], "bare")
        self.assertEqual(run["environment"]["boot_command"], "run boot_oee")
        self.assertEqual(run["experiment_name"], "NN")
        self.assertEqual(run["experiment"]["duration_seconds"], 30)
        self.assertEqual(len(run["experiment"]["profiles"]), 3)
        assignments = NNExperimentRunner._environment_assignments(run["experiment"])
        self.assertIn("DURATION=30", assignments)
        self.assertIn("HEAVY_PEAK_QPS=20", assignments)
        self.assertIn("MEDIUM_POISSON_QPS=6", assignments)
        self.assertIn("WORKLOAD_CPUS=0-5", assignments)
        self.assertIn("CGROUP_WEIGHT=100", assignments)
        self.assertIn("CGROUP_SHARES=1024", assignments)
        self.assertEqual(
            condition_name(run),
            "dual-tflite-v2-cgroup_d30s_seed12345_workers8_ct-i1000us_h10000",
        )

    def test_nn_summary_requires_six_csv_and_six_cyclictest_files(self):
        run = run_nn_bare.build_run_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in ("light", "medium", "heavy"):
                for model in ("inception", "mnasnet"):
                    (root / f"{model}_{profile}.csv").write_text(
                        "timestamp,response_ms,inference_ms,request_id\n"
                        "1.0,2.5,2.0,1\n",
                        encoding="utf-8",
                    )
                for cpu in (6, 7):
                    (root / f"cyclictest_{profile}_cpu{cpu}.txt").write_text(
                        "# Histogram\n000003 10\n"
                        "# Max Latencies: 00003\n"
                        "# Histogram Overflows: 00000\n",
                        encoding="utf-8",
                    )
            summary = NNExperimentRunner._summarize_results(
                root, run["experiment"]
            )
        self.assertEqual(summary["csv_files"], 6)
        self.assertEqual(summary["cyclictest_files"], 6)
        self.assertEqual(summary["http_requests"], 6)
        self.assertEqual(summary["http_inference_failures"], 0)

    def test_nn_jailhouse_configuration(self):
        run = run_nn_jailhouse.build_run_config()
        exp = run["experiment"]
        self.assertEqual(run["environment_name"], "jailhouse")
        self.assertEqual(run["environment"]["boot_command"], "run boot_oee")
        self.assertEqual(run["environment"]["module_load_command"], "insmod jailhouse.ko")
        self.assertEqual(exp["rootcell_internal_ip"], "202.197.10.50")
        self.assertEqual(exp["nonroot_internal_ip"], "202.197.10.51")
        self.assertEqual(exp["rootcell_cyclictest_cpu"], 6)
        self.assertEqual(exp["nonroot_cyclictest_cpu"], 3)
        self.assertIn("./tl3588-linux-demo.cell ./Image", exp["jailhouse_linux_command"])
        runner = NNExperimentRunner(run, Path("data"), True)
        self.assertIn("MODEL_THREADS=3", runner._jailhouse_assignments(exp, "rootcell"))
        self.assertIn("CPU_CYCLIC=3", runner._jailhouse_assignments(exp, "nonroot"))

    def test_nn_xen_credit2_configuration(self):
        run = run_nn_xen_credit2.build_run_config()
        env = run["environment"]
        exp = run["experiment"]
        self.assertEqual(run["environment_name"], "xen_credit2")
        self.assertEqual(env["boot_command"], "run boot_xen")
        self.assertEqual(env["xen_boot_health_attempts"], 3)
        self.assertEqual(env["xen_control_ready_timeout_seconds"], 20)
        self.assertEqual(env["dom1_start_attempts"], 3)
        self.assertEqual(env["dom1_console_ready_timeout_seconds"], 60)
        self.assertTrue(env["local_boot_files"]["source_dir"].endswith("Xen_credit2"))
        self.assertIn("xl create compute_NN_dom1.cfg", env["dom0_init_commands"])
        self.assertIn("xl vcpu-pin 0 6 6", env["dom0_pin_command"])
        self.assertIn("xl vcpu-pin dom1 6 7", env["dom1_pin_command"])
        self.assertEqual(exp["dom0_cyclictest_cpu"], 6)
        self.assertEqual(exp["dom1_cyclictest_cpu"], 6)
        self.assertEqual(exp["remote_xen_dir"], "~/NN/xen_credit2")
        self.assertEqual(exp["dom0_bridge_ip"], "202.197.67.50")
        self.assertEqual(exp["dom1_interface"], "enX0")
        self.assertEqual(exp["dom1_ip"], "202.197.67.51")
        self.assertEqual(exp["dom1_netmask"], "255.255.255.0")
        self.assertEqual(exp["dom1_network_settle_seconds"], 5)
        self.assertTrue(Path(exp["local_dom0_script"]).is_file())
        self.assertTrue(Path(exp["local_dom1_script"]).is_file())
        runner = NNExperimentRunner(run, Path("data"), True)
        self.assertIn("MODEL_THREADS=6", runner._xen_assignments(exp, "dom0"))
        self.assertIn("MODEL_THREADS=3", runner._xen_assignments(exp, "dom1"))
        self.assertIn("CPU_CYCLIC=6", runner._xen_assignments(exp, "dom1"))

    def test_nn_xen_credit2_wfx_inherits_credit2_nn_layout(self):
        credit2 = run_nn_xen_credit2.build_run_config()
        wfx = run_nn_xen_credit2_wfx.build_run_config()
        self.assertEqual(wfx["environment_name"], "xen_credit2_WFX")
        self.assertTrue(
            wfx["environment"]["local_boot_files"]["source_dir"].endswith(
                "Xen_credit2_nativeWFX"
            )
        )
        self.assertEqual(
            wfx["environment"]["local_boot_files"]["target_dir"],
            credit2["environment"]["local_boot_files"]["target_dir"],
        )
        self.assertEqual(
            wfx["environment"]["dom0_init_commands"],
            credit2["environment"]["dom0_init_commands"],
        )
        self.assertEqual(wfx["experiment"], credit2["experiment"])
        self.assertEqual(wfx["experiment"]["duration_seconds"], 30)
        self.assertEqual(
            [profile["name"] for profile in wfx["experiment"]["profiles"]],
            ["heavy"],
        )
        runner = NNExperimentRunner(wfx, Path("data"), True)
        with patch.object(runner, "_run_nn_xen_credit2") as run_xen:
            runner._run_experiment(object(), Path("output"))
        run_xen.assert_called_once()

    def test_nn_xen_null_uses_null_topology_and_four_vcpu_nn_layout(self):
        run = run_nn_xen_null.build_run_config()
        env = run["environment"]
        exp = run["experiment"]
        self.assertEqual(run["environment_name"], "xen_null")
        self.assertTrue(env["local_boot_files"]["source_dir"].endswith("Xen_null"))
        self.assertIn("xl vcpu-set 0 4", env["dom0_init_commands"])
        self.assertIn(
            "xl create compute_NN_dom1_nullsched.cfg", env["dom0_init_commands"]
        )
        self.assertIn("xl vcpu-pin dom1 3 7", env["dom1_pin_command"])
        commands = env["dom0_init_commands"]
        resize_index = commands.index("xl vcpu-set 0 4")
        dom0_pin_index = next(
            index
            for index, command in enumerate(commands)
            if "xl vcpu-pin 0 0 0" in command
        )
        bridge_index = commands.index("sh set_bridge.sh")
        create_index = commands.index("xl create compute_NN_dom1_nullsched.cfg")
        dom1_pin_index = commands.index(env["dom1_pin_command"])
        self.assertLess(resize_index, dom0_pin_index)
        self.assertLess(dom0_pin_index, bridge_index)
        self.assertLess(bridge_index, create_index)
        self.assertEqual(dom1_pin_index, create_index + 1)
        self.assertTrue(env["dom1_pin_during_init"])
        self.assertEqual(exp["dom0_cyclictest_cpu"], 3)
        self.assertEqual(exp["dom1_cyclictest_cpu"], 3)
        self.assertEqual(exp["dom0_workload_cpus"], "0-2")
        self.assertEqual(exp["dom1_workload_cpus"], "0-2")
        runner = NNExperimentRunner(run, Path("data"), True)
        with patch.object(runner, "_run_nn_xen_credit2") as run_xen:
            runner._run_experiment(object(), Path("output"))
        run_xen.assert_called_once()

    def test_nn_xen_null_wfx_only_replaces_null_boot_artifacts(self):
        null = run_nn_xen_null.build_run_config()
        wfx = run_nn_xen_null_wfx.build_run_config()
        self.assertEqual(wfx["environment_name"], "xen_null_WFX")
        self.assertTrue(
            wfx["environment"]["local_boot_files"]["source_dir"].endswith(
                "Xen_null_nativeWFX"
            )
        )
        self.assertEqual(
            wfx["environment"]["dom0_init_commands"],
            null["environment"]["dom0_init_commands"],
        )
        self.assertEqual(
            wfx["environment"]["dom1_pin_command"],
            null["environment"]["dom1_pin_command"],
        )
        self.assertTrue(wfx["environment"]["dom1_pin_during_init"])
        self.assertEqual(wfx["experiment"], null["experiment"])
        runner = NNExperimentRunner(wfx, Path("data"), True)
        with patch.object(runner, "_run_nn_xen_credit2") as run_xen:
            runner._run_experiment(object(), Path("output"))
        run_xen.assert_called_once()

    def test_nn_xhypass_inherits_credit2_and_uses_rto_module_on_cpu6(self):
        credit2 = run_nn_xen_credit2.build_run_config()
        xhypass = run_nn_xhypass.build_run_config()
        self.assertEqual(xhypass["environment_name"], "XHyPass")
        self.assertTrue(
            xhypass["environment"]["local_boot_files"]["source_dir"].endswith(
                "XHyPass"
            )
        )
        self.assertEqual(
            xhypass["environment"]["dom0_init_commands"],
            credit2["environment"]["dom0_init_commands"],
        )
        self.assertEqual(xhypass["experiment"]["dom0_cyclictest_cpu"], 6)
        self.assertEqual(xhypass["experiment"]["dom1_cyclictest_cpu"], 6)
        self.assertEqual(xhypass["experiment"]["dom0_workload_cpus"], "0-5")
        self.assertEqual(xhypass["experiment"]["dom1_workload_cpus"], "0-5")
        self.assertEqual(xhypass["experiment"]["xhypass_module_cpu"], 6)
        runner = NNExperimentRunner(xhypass, Path("data"), True)
        with patch.object(runner, "_run_nn_xen_credit2") as run_xen:
            runner._run_experiment(object(), Path("output"))
        run_xen.assert_called_once()

    def test_nn_xhypass_module_commands_are_exact_and_reconnect_if_needed(self):
        class FakeSSH:
            def __init__(self):
                self.active = False
                self.reconnects = 0
                self.commands = []

            def is_active(self):
                return self.active

            def reconnect(self):
                self.active = True
                self.reconnects += 1

            def run(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return 0, b"", b""

        ssh = FakeSSH()
        log = Path("xhypass-module.log")
        with patch("xhypass_lab.nn_runner.time.sleep") as sleep:
            NNExperimentRunner._set_xhypass_nn_module(
                ssh,
                log,
                "dom0",
                6,
                load=True,
                settle_seconds=2,
                module_file="./interrupt_passthrough.ko",
                load_args="rto_cpu=6 event_sgi=7",
            )
        sleep.assert_called_once_with(2)
        NNExperimentRunner._set_xhypass_nn_module(
            ssh, log, "dom0", 6, load=False
        )
        self.assertEqual(ssh.reconnects, 1)
        self.assertEqual(
            [command for command, _ in ssh.commands],
            [
                "cd ~ && taskset -c 6 insmod ./interrupt_passthrough.ko "
                "rto_cpu=6 event_sgi=7",
                "cd ~ && taskset -c 6 rmmod interrupt_passthrough.ko",
            ],
        )
        self.assertTrue(all(options["check"] for _, options in ssh.commands))

    def test_nn_xen_recreates_dom1_after_initial_console_timeout(self):
        class FakeSSH:
            def __init__(self):
                self.commands = []

            def is_active(self):
                return True

            def run(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return 0, b"", b""

        run = run_nn_xhypass.build_run_config()
        env = run["environment"]
        env["dom1_destroy_settle_seconds"] = 0
        env["dom1_recreate_settle_seconds"] = 0
        runner = NNExperimentRunner(run, Path("data"), True)
        ssh = FakeSSH()
        with patch.object(
            runner,
            "_configure_xen_dom1_via_console",
            side_effect=[TimeoutError("no login prompt"), None],
        ) as configure:
            runner._start_xen_dom1_with_retry(
                ssh, env, run["experiment"], Path("dom0.log")
            )

        self.assertEqual(configure.call_count, 2)
        commands = [command for command, _ in ssh.commands]
        self.assertEqual(commands[0], "xl destroy dom1")
        self.assertIn("xl create compute_NN_dom1.cfg", commands)
        self.assertIn(env["dom1_pin_command"], commands)
        self.assertIn("xl vcpu-list dom1", commands)
        destroy_options = ssh.commands[0][1]
        self.assertFalse(destroy_options["check"])

    def test_nn_xen_dom1_network_is_stabilized_before_ssh(self):
        class FakeSerial:
            def __init__(self):
                self.commands = []
                self.drains = []

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                if "__NN_DOM1_NET_STABLE_RC__" in command:
                    return b"__NN_DOM1_NET_STABLE_RC__=0\n"
                return b"__NN_DOM1_NET_RC__=0\n"

            def drain(self, seconds):
                self.drains.append(seconds)

        run = run_nn_xen_credit2.build_run_config()
        runner = NNExperimentRunner(run, Path("data"), True)
        serial = FakeSerial()
        runner._configure_xen_dom1_network(
            serial, run["experiment"], r"__XEN_DOM1_PROMPT__#\s*$"
        )
        self.assertIn("NetworkManager.service", serial.commands[0])
        self.assertIn("network-manager.service", serial.commands[0])
        self.assertIn("networking.service", serial.commands[0])
        self.assertIn("systemd-networkd.service", serial.commands[0])
        self.assertIn("[u]dhcpc.*enX0", serial.commands[0])
        self.assertIn("ifconfig enX0 202.197.67.51", serial.commands[0])
        self.assertEqual(serial.drains, [5.0])

    def test_nn_xen_result_collection_does_not_touch_healthy_dom1_network(self):
        class HealthySSH:
            def __init__(self):
                self.reconnects = 0

            def is_active(self):
                return True

            def tcp_available(self, timeout):
                return True

            def reconnect(self):
                self.reconnects += 1

        class UntouchedSerial:
            def __getattr__(self, name):
                raise AssertionError(f"healthy dom1 must not use serial.{name}")

        run = run_nn_xen_credit2.build_run_config()
        runner = NNExperimentRunner(run, Path("data"), True)
        dom0 = HealthySSH()
        dom1 = HealthySSH()
        runner._refresh_xen_result_connections(
            UntouchedSerial(), dom0, dom1,
            run["environment"], run["experiment"],
            Path("dom0.log"),
        )
        self.assertEqual(dom0.reconnects, 0)
        self.assertEqual(dom1.reconnects, 0)

    def test_nn_xen_dom1_ip_repair_is_lightweight_serial_ifconfig(self):
        class FakeSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.commands = []

            def sendline(self, line=""):
                self.lines.append(line)

            def expect(self, patterns, timeout):
                return 0, b"__XEN_DOM1_PROMPT__# "

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                return b"__NN_DOM1_REPAIR_RC__=0\n"

        run = run_nn_xen_credit2.build_run_config()
        runner = NNExperimentRunner(run, Path("data"), True)
        serial = FakeSerial()
        runner._repair_xen_dom1_network(
            serial, run["experiment"], r"__XEN_DOM1_PROMPT__#\s*$"
        )
        self.assertIn("ifconfig enX0 202.197.67.51", serial.commands[0])
        self.assertNotIn("systemctl", serial.commands[0])

    def test_nn_xen_dom1_ip_repair_enters_and_leaves_com10_console(self):
        class FakeSerial:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.raw = []
                self.commands = []
                self.expect_count = 0

            def sendline(self, line=""):
                self.lines.append(line)

            def send(self, raw):
                self.raw.append(raw)

            def expect(self, patterns, timeout):
                self.expect_count += 1
                if self.expect_count == 1:
                    return 1, b"phytiumpi login: "
                if self.expect_count == 2:
                    return 1, b"root@phytiumpi:~# "
                return 0, b"prompt"

            def command(self, command, prompt, timeout):
                self.commands.append(command)
                if "__NN_DOM1_REPAIR_RC__" in command:
                    return b"__NN_DOM1_REPAIR_RC__=0\n"
                return b"__XEN_DOM1_PROMPT__# "

        run = run_nn_xen_credit2.build_run_config()
        runner = NNExperimentRunner(run, Path("data"), True)
        serial = FakeSerial()
        runner._repair_xen_dom1_network_via_serial_console(
            serial, run["environment"], run["experiment"]
        )
        self.assertEqual(serial.lines[0], "xl console dom1")
        self.assertEqual(serial.lines[1], "")
        self.assertEqual(serial.lines[2], "root")
        self.assertIn("ifconfig enX0 202.197.67.51", serial.commands[-1])
        self.assertEqual(serial.raw, [b"\x1d"])

    def test_nn_xen_control_plane_accepts_healthy_xl_list(self):
        class FakeSerial:
            def command(self, command, prompt, timeout):
                return b"__NN_XEN_CONTROL_READY__\n__NN_XEN_KERNEL__=5.10.0+\n"

        run = run_nn_xen_credit2.build_run_config()
        runner = NNExperimentRunner(run, Path("data"), True)
        runner._ensure_xen_control_plane(
            FakeSerial(), run["environment"], r"__XHYPASS_PROMPT__#\s*$"
        )

    def test_nn_xen_preflight_accepts_parent_dependency_directory(self):
        class FakeSSH:
            def __init__(self):
                self.command = ""

            def run(self, command, **kwargs):
                self.command = command
                return 0, b"__NN_BASE__=..\n", b""

        ssh = FakeSSH()
        base = NNExperimentRunner._resolve_xen_nn_base(
            ssh,
            "~/NN/xen_credit2",
            "mnasnet_1.3_224.tflite",
            Path("dom1.log"),
            "dom1",
        )
        self.assertEqual(base, "..")
        self.assertIn("chmod +x", ssh.command)
        self.assertIn("benchmark_http_infer", ssh.command)

    def test_full_matrix_resumes_latest_compatible_incomplete_campaign(self):
        matrix = run_full_cyclictest_matrix
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "campaign_id": "resume-me",
                "campaign_type": "full-cyclictest-matrix",
                "status": "completed_with_failures",
                "environments": list(matrix.ENVIRONMENTS),
                "experiments": list(matrix.EXPERIMENTS),
                "intervals_us": list(matrix.INTERVALS_US),
                "duration_seconds": matrix.DURATION_SECONDS,
                "runs_per_condition": matrix.RUNS_PER_CONDITION,
                "reboot_policy": matrix.REBOOT_POLICY,
                "conditions": [],
            }
            path = (
                root
                / "campaigns"
                / "full-cyclictest-matrix-20260814-000000"
                / "campaign.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary), encoding="utf-8")
            with patch.object(matrix, "DATA_ROOT", root):
                resumed = matrix.find_incomplete_campaign()
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed[1]["campaign_id"], "resume-me")

    def test_full_matrix_recognizes_control_plane_failures(self):
        matrix = run_full_cyclictest_matrix
        self.assertEqual(matrix.MAX_CONSECUTIVE_INFRASTRUCTURE_FAILURES, 2)
        self.assertTrue(
            matrix._is_infrastructure_failure(
                RuntimeError(
                    "Timed out after 10.0s waiting for login or shell prompt"
                )
            )
        )
        self.assertFalse(
            matrix._is_infrastructure_failure(
                RuntimeError("Could not transfer remote histogram file")
            )
        )

    def test_xen_domain_detection_uses_complete_domain_name(self):
        output = (
            b"__XEN_ENV__\n"
            b"Name ID Mem VCPUs State Time(s)\n"
            b"Domain-0 0 4096 7 r----- 10.0\n"
            b"dom1 1 2048 7 -b---- 1.0\n"
        )
        self.assertTrue(ExperimentRunner._xl_list_has_domain(output, "dom1"))
        self.assertFalse(ExperimentRunner._xl_list_has_domain(output, "dom"))

    def test_ssh_reboot_destroys_dom1_only_on_xen(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        xen = FakeRebootSSH(b"__XEN_ENV__\nDomain-0 0 1 7 r 0\ndom1 1 1 7 b 0\n")
        runner._prepare_xen_reboot_ssh(xen, Path("unused.log"))
        self.assertEqual(xen.commands[-1], "xl destroy dom1")
        linux = FakeRebootSSH(b"")
        runner._prepare_xen_reboot_ssh(linux, Path("unused.log"))
        self.assertNotIn("xl destroy dom1", linux.commands)

    def test_xen_waits_for_dom1_then_runs_experiment_over_serial(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        serial = FakeXenSerial()
        FakeXenSSH.instances.clear()
        with patch("xhypass_lab.runner.SSHSession", FakeXenSSH):
            with patch("xhypass_lab.runner.time.sleep") as sleep:
                with patch.object(runner, "_run_cyclictest") as serial_experiment:
                    runner._run_xen_cyclictest(serial, Path("output"))
        ssh = FakeXenSSH.instances[0]
        self.assertEqual(ssh.console_command, "xl console dom1")
        self.assertEqual(ssh.console_username, "root")
        sleep.assert_called_once_with(10.0)
        serial_experiment.assert_called_once_with(serial, Path("output"))

    def test_xen_null_drains_two_seconds_after_vcpu_set(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_null", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        serial = FakeXenSerial()
        FakeXenSSH.instances.clear()
        with patch("xhypass_lab.runner.SSHSession", FakeXenSSH):
            with patch("xhypass_lab.runner.time.sleep"):
                with patch.object(runner, "_run_cyclictest"):
                    runner._run_xen_cyclictest(serial, Path("output"))
        self.assertEqual(serial.drains, [2.0])
        self.assertIn("xl vcpu-set 0 4", serial.commands)
        ssh = FakeXenSSH.instances[0]
        self.assertIn("xl vcpu-list 0", ssh.commands)
        self.assertIn("xl vcpu-list dom1", ssh.commands)

    def test_serial_experiment_returns_to_root_home_before_preflight(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        run["experiment"]["duration_seconds"] = 0
        run["experiment"]["cleanup_remote"] = False
        runner = ExperimentRunner(run, Path("data"), True)
        serial = FakeCyclictestSerial()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "_download_base64"):
                runner._run_cyclictest(serial, Path(directory))
        self.assertEqual(serial.commands[0], "cd ~")
        self.assertIn("test -x ./cyclictest", serial.commands[1])

    def test_serial_preflight_uses_short_independent_probes(self):
        config = load_config(Path("config/E2000Q/lab.json"))
        run = resolved_run_config(config, "XHyPass", "cyclictest-stress", {})
        run["experiment"]["duration_seconds"] = 0
        run["experiment"]["cleanup_remote"] = False
        runner = ExperimentRunner(run, Path("data"), True)
        serial = FakeCyclictestSerial()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "_download_base64"):
                runner._run_cyclictest(serial, Path(directory))

        probes = [
            command
            for command in serial.commands
            if any(
                f"echo {marker}=$?" in command
                for marker in ("XYCY", "XYST", "XYTS", "XYB6")
            )
        ]
        self.assertEqual(len(probes), 4)
        self.assertTrue(
            all("__XY_PREFLIGHT__" not in command for command in probes)
        )
        self.assertTrue(all("&&" not in command for command in probes))

    def test_serial_preflight_reports_the_failed_dependency(self):
        class MissingStressSerial(FakeCyclictestSerial):
            def command(self, command, prompt, timeout):
                if "echo XYST=$?" in command:
                    self.commands.append(command)
                    return b"XYST=1\n__XHYPASS_PROMPT__# "
                return super().command(command, prompt, timeout)

        config = load_config(Path("config/E2000Q/lab.json"))
        run = resolved_run_config(config, "XHyPass", "cyclictest-stress", {})
        runner = ExperimentRunner(run, Path("data"), True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError, r"missing or unusable: stress-ng$"
            ):
                runner._run_cyclictest(MissingStressSerial(), Path(directory))

    def test_base64_download_accepts_xen_hvc_crcrlf(self):
        payload = b"# Histogram\n000003 000001\n"
        serial = FakeBase64Serial(base64.b64encode(payload))
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "hist.txt"
            ExperimentRunner._download_base64(
                serial, "/tmp/hist.txt", local, r"__XHYPASS_PROMPT__#\s*$"
            )
            self.assertEqual(local.read_bytes(), payload)

    def test_base64_download_retries_a_corrupted_serial_marker(self):
        payload = b"stress-ng output\n"

        class FlakySerial:
            def __init__(self):
                self.calls = 0

            def command(self, command, prompt, timeout):
                self.calls += 1
                if self.calls == 1:
                    return b"XYB64BEGIN\nYmFk\nXYB64EN\n__XHYPASS_PROMPT__# "
                return (
                    b"XYB64BEGIN\r\n"
                    + base64.b64encode(payload)
                    + b"\r\nXYB64END\r\n__XHYPASS_PROMPT__# "
                )

        serial = FlakySerial()
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "stress-ng.log"
            ExperimentRunner._download_base64(
                serial,
                "/tmp/stress-ng.log",
                local,
                r"__XHYPASS_PROMPT__#\s*$",
            )
            self.assertEqual(local.read_bytes(), payload)
        self.assertEqual(serial.calls, 2)

    def test_xen_hvc_cmdline_accepts_crcrlf(self):
        raw = (
            b"__CMDLINE_BEGIN__\r\r\n"
            b"earlycon=xenboot console=ttyS2 isolcpus=3\r\r\n"
            b"__CMDLINE_END__\r\r\n"
        )
        self.assertEqual(
            ExperimentRunner._extract_cmdline(raw),
            "earlycon=xenboot console=ttyS2 isolcpus=3",
        )

    def test_device_output_does_not_crash_gbk_console(self):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")
        write_console_bytes(stream, "© replacement �\n".encode("utf-8"))
        stream.flush()
        self.assertIn(b"?", buffer.getvalue())

    def test_stress_command_competes_with_cyclictest_on_cpu6(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest-stress", {})
        run["experiment"]["cleanup_remote"] = False
        runner = ExperimentRunner(run, Path("data"), True)
        serial = FakeCyclictestSerial()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "_download_base64"):
                runner._run_cyclictest(serial, Path(directory))
        launch = next(command for command in serial.commands if "XY_STRESS_PID" in command)
        self.assertIn("taskset -c 6 ./stress-ng --vm 1 --vm-bytes 256M", launch)
        self.assertIn("--timeout 600s", launch)
        self.assertIn("taskset -c 6 ./cyclictest", launch)

    def test_local_boot_files_are_overlaid_without_removing_other_files(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2", "cyclictest", {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "xen-uImage.tl3588").write_bytes(b"new")
            (target / "xen-uImage.tl3588").write_bytes(b"old")
            (target / "keep.txt").write_bytes(b"keep")
            run["environment"]["local_boot_files"] = {
                "source_dir": str(source),
                "target_dir": str(target),
            }
            ExperimentRunner(run, root / "data", True)._sync_local_boot_files()
            self.assertEqual((target / "xen-uImage.tl3588").read_bytes(), b"new")
            self.assertEqual((target / "keep.txt").read_bytes(), b"keep")

    def test_metadata_secrets_are_redacted(self):
        redacted = ExperimentRunner._redact_secrets(
            {"password": "secret", "nested": {"nonroot_password": "secret2"}}
        )
        self.assertEqual(redacted["password"], "<redacted>")
        self.assertEqual(redacted["nested"]["nonroot_password"], "<redacted>")

    def test_cyclictest_stress_configuration(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest-stress", {})
        exp = run["experiment"]
        self.assertEqual(exp["stress_cpus"], "6")
        self.assertEqual(exp["stress_vm_workers"], 1)
        self.assertEqual(exp["stress_vm_bytes"], "256M")
        self.assertNotIn("stress_timeout", exp)

    def test_direct_entry_selects_stress_experiment(self):
        run = run_experiment.build_run_config()
        self.assertEqual(run["experiment_name"], "cyclictest-stress")
        self.assertEqual(run["experiment"]["duration_seconds"], 10)

    def test_fixed_600s_campaign_definition(self):
        self.assertEqual(run_bare_jailhouse_600s.DURATION_SECONDS, 600)
        self.assertEqual(run_bare_jailhouse_600s.RUNS_PER_CASE, 5)
        self.assertEqual(
            run_bare_jailhouse_600s.CAMPAIGN,
            (
                ("bare", "cyclictest"),
                ("bare", "cyclictest-stress"),
                ("jailhouse", "cyclictest"),
                ("jailhouse", "cyclictest-stress"),
            ),
        )
        config = run_bare_jailhouse_600s.build_case_config(
            "jailhouse", "cyclictest-stress"
        )
        self.assertEqual(config["experiment"]["duration_seconds"], 600)
        self.assertEqual(config["environment"]["boot_command"], "run boot_oee")

    def test_code_settings_entrypoint(self):
        run = run_experiment.build_run_config()
        self.assertEqual(run["environment_name"], run_experiment.ENVIRONMENT)
        self.assertEqual(run["serial"]["port"], run_experiment.SERIAL_PORT)
        self.assertEqual(run["serial"]["baudrate"], 115_200)
        self.assertTrue(run["serial"]["show_output"])
        self.assertEqual(
            run["experiment"]["duration_seconds"], run_experiment.DURATION_SECONDS
        )

    def test_uboot_stop_uses_only_raw_ctrl_c(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest", {})
        fake = FakeUbootSession()
        ExperimentRunner(run, Path("data"), True)._reach_uboot(fake, r"=>\s*$")
        self.assertGreaterEqual(len(fake.sent), 3)
        self.assertTrue(all(item == b"\x03" for item in fake.sent))
        self.assertTrue(fake.reset_observed_before_first_send)

    def test_serial_transcript_can_switch_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "run001" / "serial.log"
            second = Path(directory) / "run002" / "serial.log"
            session = SerialSession({"port": "COM10"}, first)
            session._open_log(first)
            session.log.write(b"first")
            session.log.flush()
            session.switch_transcript(second)
            session.log.write(b"second")
            session.log.close()
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_boot_command_retries_mmc_stop_error(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "bare", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        fake = FakeBootRetrySession()
        runner._boot_command_until_login(
            fake,
            run["environment"],
            run["environment"]["login_prompt"],
            run["environment"]["uboot_prompt"],
        )
        self.assertEqual(
            fake.lines,
            ["run boot_oee", "run boot_oee"],
        )

    def test_xen_boot_command_retries_wrong_image_type_error(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "xen_credit2_WFX", "cyclictest", {})
        runner = ExperimentRunner(run, Path("data"), True)
        fake = FakeBootRetrySession("Wrong Image Type for bootm command")
        runner._boot_command_until_login(
            fake,
            run["environment"],
            run["environment"]["login_prompt"],
            run["environment"]["uboot_prompt"],
        )
        self.assertEqual(fake.lines, ["run boot_xen", "run boot_xen"])

    def test_e2000q_xen_resets_uboot_after_retry_cycle_is_exhausted(self):
        class ResetBootSession:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.run_attempts = 0

            def sendline(self, line=""):
                self.lines.append(line)
                if line == "run boot_xen":
                    self.run_attempts += 1

            def send(self, data):
                raise AssertionError(f"Unexpected raw send: {data!r}")

            def expect(self, patterns, timeout, clear=False):
                if patterns == [r"=>\s*$"]:
                    return 0, b"=> "
                if self.run_attempts <= 2:
                    error = "Wrong Image Type for bootm command"
                    return patterns.index(error), error.encode("utf-8")
                return 0, b"gsk-e2000q login: "

        run = run_e2000q_experiment.build_run_config(
            "cyclictest", "xen_credit2"
        )
        env = run["environment"]
        self.assertEqual(env["boot_reset_cycles"], 3)
        env["boot_max_attempts"] = 2
        env["boot_reset_cycles"] = 2
        fake = ResetBootSession()
        runner = ExperimentRunner(run, Path("data/E2000Q"), True)
        with patch.object(runner, "_reach_uboot") as reach_uboot:
            with patch("xhypass_lab.runner.time.sleep"):
                runner._boot_command_until_login(
                    fake,
                    env,
                    env["login_prompt"],
                    env["uboot_prompt"],
                )
        self.assertEqual(
            fake.lines,
            ["run boot_xen", "run boot_xen", "reset", "run boot_xen"],
        )
        reach_uboot.assert_called_once_with(fake, env["uboot_prompt"])

    def test_e2000q_int_latency_arp_failure_resets_uboot_immediately(self):
        class ArpFailureSession:
            def __init__(self):
                self.buffer = bytearray()
                self.lines = []
                self.boot_attempts = 0

            def sendline(self, line=""):
                self.lines.append(line)
                if line == "run boot_xen_dom0less":
                    self.boot_attempts += 1

            def send(self, data):
                raise AssertionError(f"Unexpected raw send: {data!r}")

            def expect(self, patterns, timeout, clear=False):
                if patterns == [r"=>\s*$"]:
                    return 0, b"=> "
                if self.boot_attempts == 1:
                    error = "ARP Retry count exceeded"
                    return patterns.index(error), error.encode("utf-8")
                return 0, b"gsk-e2000q login: "

        run = run_e2000q_int_latency_matrix.build_run_config(
            "xen_null_WFX", "idle"
        )
        env = run["environment"]
        self.assertIn(
            "ARP Retry count exceeded", env["boot_immediate_reset_patterns"]
        )
        fake = ArpFailureSession()
        runner = ExperimentRunner(run, Path("data/E2000Q"), True)
        with patch.object(runner, "_reach_uboot") as reach_uboot:
            with patch("xhypass_lab.runner.time.sleep"):
                runner._boot_command_until_login(
                    fake,
                    env,
                    env["login_prompt"],
                    env["uboot_prompt"],
                )

        self.assertEqual(
            fake.lines,
            ["run boot_xen_dom0less", "reset", "run boot_xen_dom0less"],
        )
        reach_uboot.assert_called_once_with(fake, env["uboot_prompt"])

    def test_e2000q_matrix_pauses_after_exhausted_boot_recovery(self):
        self.assertTrue(
            run_e2000q_full_cyclictest_matrix.PAUSE_ON_INFRASTRUCTURE_FAILURE
        )
        self.assertTrue(
            run_e2000q_full_cyclictest_matrix._is_infrastructure_failure(
                RuntimeError(
                    "Boot failed after 3 reset cycles x 3 attempts: "
                    "Wrong Image Type for bootm command"
                )
            )
        )

    def test_nonroot_forced_password_change(self):
        config = load_config(Path("config/RK3588/lab.json"))
        run = resolved_run_config(config, "jailhouse", "cyclictest", {})
        fake = FakeForcedPasswordSession(run["environment"]["nonroot_password"])
        ExperimentRunner(run, Path("data"), True)._login_nonroot_cell(
            fake, run["environment"]
        )
        self.assertEqual(
            fake.lines,
            [
                "root",
                run["environment"]["nonroot_password"],
                run["environment"]["nonroot_password"],
                "",
                "export PS1='__JH_NONROOT_PROMPT__# '",
            ],
        )


class FakeUbootSession:
    def __init__(self):
        self.buffer = bytearray()
        self.sent = []
        self.reset_observed = False
        self.reset_observed_before_first_send = False

    def send(self, data):
        if not self.sent:
            self.reset_observed_before_first_send = self.reset_observed
        self.sent.append(data)

    def _read_once(self):
        if not self.sent:
            self.buffer.extend(b"DDR V1.12\r\n")
            self.reset_observed = True
        elif len(self.sent) >= 3:
            self.buffer.extend(b"\r\n=> ")
        else:
            self.buffer.extend(b"booting\r\n")
        return b""


class FakeBootRetrySession:
    def __init__(self, first_error="mmc fail to send stop cmd"):
        self.buffer = bytearray()
        self.lines = []
        self.boot_attempts = 0
        self.first_error = first_error

    def sendline(self, line=""):
        self.lines.append(line)
        self.boot_attempts += 1

    def send(self, data):
        raise AssertionError(f"Unexpected raw send during retry: {data!r}")

    def expect(self, patterns, timeout, clear=False):
        if patterns == [r"=>\s*$"]:
            self.buffer.extend(b"\r\n=> ")
            return 0, bytes(self.buffer)
        if self.boot_attempts == 1:
            self.buffer.extend(self.first_error.encode("utf-8") + b"\r\n=> ")
            return patterns.index(self.first_error), bytes(self.buffer)
        self.buffer.extend(b"tl3588 login: ")
        return 0, bytes(self.buffer)


class FakeForcedPasswordSession:
    def __init__(self, password):
        self.buffer = bytearray()
        self.lines = []
        self.password = password
        self.stage = "login"

    def sendline(self, line=""):
        self.lines.append(line)

    def expect(self, patterns, timeout, clear=False):
        if self.stage == "login":
            self.stage = "new_password"
            return 0, b"phytiumpi login: "
        if self.stage == "new_password":
            self.stage = "retype"
            return 3, b"New password: "
        if self.stage == "retype":
            self.stage = "shell"
            return 0, b"Retype new password: "
        if self.stage == "shell":
            self.stage = "prompt"
            return 0, b"root@phytiumpi:~# "
        if self.stage == "prompt":
            return 0, b"__JH_NONROOT_PROMPT__# "
        raise AssertionError(self.stage)

    def command(self, command, prompt, timeout):
        self.sendline(command)
        return b"__JH_NONROOT_PROMPT__# "


class FakeRebootSSH:
    def __init__(self, probe_output):
        self.probe_output = probe_output
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("if command -v xl"):
            return 0, self.probe_output, b""
        return 0, b"", b""


class FakeXenSerial:
    def __init__(self):
        self.commands = []
        self.drains = []

    def command(self, command, prompt, timeout):
        self.commands.append(command)
        return b"__XHYPASS_PROMPT__# "

    def drain(self, seconds):
        self.drains.append(seconds)
        return b""


class FakeXenSSH:
    instances = []

    def __init__(self, settings):
        self.settings = settings
        self.commands = []
        self.console_command = None
        self.console_username = None
        self.instances.append(self)

    def connect(self):
        return None

    def close(self):
        return None

    def run(self, command, **kwargs):
        self.commands.append(command)
        return 0, b"", b""

    def wait_for_console_login(self, command, **kwargs):
        self.console_command = command
        self.console_username = kwargs["username"]


class FakeCyclictestSerial:
    def __init__(self):
        self.commands = []

    def command(self, command, prompt, timeout):
        self.commands.append(command)
        for marker in ("XYCY", "XYST", "XYTS", "XYB6"):
            if f"echo {marker}=$?" in command:
                return f"{marker}=0\n__XHYPASS_PROMPT__# ".encode()
        if "__XY_DONE__" in command:
            return b"__XY_DONE__=0\n__XY_STRESS_DONE__=0\n__XHYPASS_PROMPT__# "
        return b"__XHYPASS_PROMPT__# "


class FakeBase64Serial:
    def __init__(self, payload):
        self.payload = payload

    def command(self, command, prompt, timeout):
        return (
            b"XYB64BEGIN\r\r\n"
            + self.payload
            + b"\r\r\nXYB64END\r\r\n__XHYPASS_PROMPT__# "
        )


if __name__ == "__main__":
    unittest.main()
