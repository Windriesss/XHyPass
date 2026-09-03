from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_module(
    "transition_analysis",
    PROJECT_ROOT / "experiments" / "transition" / "analyze_transition.py",
)
correctness = load_module(
    "transition_correctness",
    PROJECT_ROOT / "experiments" / "transition" / "validate_correctness.py",
)
campaign = load_module(
    "transition_campaign",
    PROJECT_ROOT / "experiments" / "transition" / "dom0_campaign_serial.py",
)


class TransitionAnalysisTests(unittest.TestCase):
    @staticmethod
    def write_attempts(root: Path) -> Path:
        path = root / "idle" / "run_001" / "transition_attempts.csv"
        path.parent.mkdir(parents=True)
        rows = [
            [1, "run-001", "idle", 0, "DYN-to-RTO", 0, -16, 10, 100, 100],
            [1, "run-001", "idle", 0, "DYN-to-RTO", 1, 0, 20, 200, 350],
            [1, "run-001", "idle", 0, "RTO-to-DYN", 0, 0, 30, 300, 300],
            [1, "run-001", "idle", 1, "DYN-to-RTO", 0, 0, 40, 400, 400],
            [1, "run-001", "idle", 1, "RTO-to-DYN", 0, 0, 50, 500, 500],
        ]
        columns = [
            "schema_version",
            "run_id",
            "condition",
            "iteration",
            "direction",
            "attempt",
            "rc",
            "counter_cycles",
            "duration_ns",
            "request_elapsed_ns",
        ]
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path

    def test_run_level_retry_and_latency_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_attempts(root)
            run_path, summary_path = analysis.analyze(
                "RK3588", root, root / "analysis"
            )
            frame = pd.read_csv(run_path)
            enter = frame[frame["direction"] == "DYN-to-RTO"].iloc[0]
            self.assertEqual(int(enter["requests"]), 2)
            self.assertEqual(int(enter["hvc_attempts"]), 3)
            self.assertEqual(int(enter["retries"]), 1)
            self.assertEqual(int(enter["busy_returns"]), 1)
            self.assertAlmostEqual(float(enter["first_try_success_rate"]), 0.5)
            self.assertAlmostEqual(float(enter["p50_us"]), 0.375)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["platform"], "RK3588")
            self.assertEqual(
                summary["latency_definition"],
                "caller-observed complete request latency from request_elapsed_ns",
            )
            self.assertEqual(len(summary["groups"]), 2)

class TransitionCorrectnessTests(unittest.TestCase):
    def test_campaign_accepts_platform_spi_source(self):
        self.assertEqual(campaign.parse_sources("sgi,spi"), ["sgi", "spi"])

    def test_example_record_passes(self):
        example = (
            PROJECT_ROOT
            / "experiments"
            / "transition"
            / "correctness_schema.example.json"
        )
        self.assertEqual(correctness.validate_record(example), [])

    def test_lost_notification_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "correctness.json"
            payload = {
                "schema_version": 1,
                "watchdog_timeouts": 0,
                "final_mode": "DYN",
                "notifications": {
                    "SPI": {
                        "produced": 2,
                        "consumed": 1,
                        "lost": 1,
                        "duplicates": 0,
                        "unexpected": 0,
                    }
                },
                "state_cases": [
                    {"passed": True, "final_mode": "DYN"}
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            errors = correctness.validate_record(path)
            self.assertTrue(any("lost" in error for error in errors))
            self.assertTrue(any("produced != consumed" in error for error in errors))


class TransitionAbiTests(unittest.TestCase):
    def test_guest_and_xen_hvc_constants_match(self):
        guest = (
            PROJECT_ROOT
            / "guest"
            / "linux"
            / "interrupt_passthrough"
            / "interrupt_passthrough.c"
        ).read_text(encoding="utf-8")
        xen = (
            PROJECT_ROOT
            / "hypervisor"
            / "xen"
            / "include"
            / "public"
            / "arch-arm.h"
        ).read_text(encoding="utf-8")
        self.assertIn("#define XHYPASS_HVC_IMM          0x4a48", guest)
        self.assertIn("#define XEN_HYPERCALL_XHYPASS   0x4A48", xen)
        self.assertIn("#define XHYPASS_HVC_ENTER_RTO    12", guest)
        self.assertIn("#define XEN_XHYPASS_ENTER_RTO   12", xen)
        self.assertIn("#define XHYPASS_HVC_EXIT_RTO     13", guest)
        self.assertIn("#define XEN_XHYPASS_EXIT_RTO    13", xen)


if __name__ == "__main__":
    unittest.main()
