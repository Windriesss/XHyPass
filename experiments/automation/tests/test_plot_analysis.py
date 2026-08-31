import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PLOT_ROOT = Path(__file__).resolve().parents[3] / "plots"
sys.path.insert(0, str(PLOT_ROOT))

from xhypass_plot.model import RunRecord, combine, weighted_quantile
from xhypass_plot.parser import parse_histogram, parse_stress_log, scan_runs
from xhypass_plot.report import (
    _latest_batches,
    _platform_label_from_groups,
    _select_expected_runs,
    _stress_summary,
)
from analyze import build_platform_config
from analyze_nn import (
    _available_cpus,
    _available_cyclic_series,
    _combine_cyclic_series,
    _completion_rate,
    _is_formal_configuration,
    _representative_cyclic_series,
    _weighted_ccdf,
)
from plot_nn_scatter_2x3 import _paired_cyclic_filename
from plot_interrupt_latency_tail_percentiles import run_metrics


def record(path: Path, batch: str = "20260814-000000") -> RunRecord:
    return RunRecord("RK3588", "bare", "cyclictest-stress", batch, 1, 600, path, np.array([2., 3., 7.]), np.array([8, 1, 1]), 0)


class PlotAnalysisTests(unittest.TestCase):
    def test_interrupt_tail_metrics_are_computed_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "rtos_run1.log"
            second = root / "rtos_run2.log"
            first.write_text(
                "t0_region>\n[100]:99\n[200]:1\nt1_region>\n",
                encoding="utf-8",
            )
            second.write_text(
                "t0_region>\n[300]:99\n[500]:1\nt1_region>\n",
                encoding="utf-8",
            )

            first_metrics = run_metrics(first)
            second_metrics = run_metrics(second)
            self.assertEqual(first_metrics["P99"], 1.0)
            self.assertEqual(first_metrics["Max"], 2.0)
            self.assertEqual(second_metrics["P99"], 3.0)
            self.assertEqual(second_metrics["Max"], 5.0)

    def test_e2000q_plot_profile_is_independent_and_uses_formal_interval(self):
        config = build_platform_config("E2000Q")
        self.assertEqual(tuple(config.data_sources), ("E2000Q",))
        self.assertEqual(config.data_sources["E2000Q"].name, "E2000Q")
        self.assertEqual(config.intervals_us, (1000,))
        self.assertEqual(config.duration_seconds, 600)
        self.assertEqual(config.expected_runs_per_condition, 5)

    def test_plot_labels_derive_platform_instead_of_hardcoding_rk3588(self):
        groups = {
            ("E2000Q", "bare", "cyclictest", 1000): [
                record(Path("unused"))
            ]
        }
        self.assertEqual(_platform_label_from_groups(groups), "E2000Q")

    def test_backup_fboot_directory_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "backup_fboot" / "cyclictest" / "bare" / "run_001"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "condition": "archived",
                        "configuration": {
                            "environment_name": "bare",
                            "experiment_name": "cyclictest",
                            "experiment": {
                                "duration_seconds": 600,
                                "interval_us": 1000,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "hist.txt").write_text("1 1\n", encoding="utf-8")

            self.assertEqual(scan_runs("RK3588", root), [])

    def test_histogram_and_weighted_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hist.txt"
            path.write_text("# Histogram\n2 8\n3 1\n7 1\n# Histogram Overflows: 4\n", encoding="utf-8")
            values, counts, overflow = parse_histogram(path)
            self.assertEqual(overflow, 4)
            self.assertEqual(weighted_quantile(values, counts, .5), 2)
            merged_values, merged_counts = combine([record(Path(directory)), record(Path(directory))])
            self.assertEqual(merged_values.tolist(), [2, 3, 7])
            self.assertEqual(merged_counts.tolist(), [16, 2, 2])

    def test_latest_batch_is_selected_per_group(self):
        old = record(Path("old"), "20260813-000000")
        new = record(Path("new"), "20260814-000000")
        self.assertEqual(_latest_batches([old, new]), [new])

    def test_plot_selects_exactly_first_five_runs_per_condition(self):
        records = []
        for run_index in range(1, 7):
            item = record(Path(f"run_{run_index}"))
            item.run = run_index
            records.append(item)
        selected = _select_expected_runs(records, 5)
        self.assertEqual([item.run for item in selected], [1, 2, 3, 4, 5])

    def test_stress_log_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "stress-ng.log").write_text(
                "stress-ng: info:  [123] vm 1000 10.00 29.00 1.00 100.00 33.33\n",
                encoding="utf-8",
            )
            parsed = parse_stress_log(record(path))
            self.assertIsNotNone(parsed)
            summary = _stress_summary([parsed])
            self.assertEqual(summary[0]["bogo_ops_per_real_second_mean"], 100.0)
            self.assertEqual(summary[0]["coefficient_of_variation_percent"], 0.0)

    def test_nn_completion_rate_uses_completion_timestamp_bins(self):
        time_points, rates = _completion_rate(
            np.asarray([10.0, 10.2, 11.1]), bin_seconds=1.0
        )
        self.assertEqual(time_points.tolist(), [0.5, 1.5])
        self.assertEqual(rates.tolist(), [2.0, 1.0])

    def test_nn_plot_discovers_environment_specific_cpus(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            results = run / "results"
            results.mkdir()
            (results / "cyclictest_light_cpu3.txt").touch()
            (results / "cyclictest_light_cpu6.txt").touch()
            self.assertEqual(_available_cpus([run], "light"), [3, 6])

    def test_nn_plot_discovers_xen_domain_vcpus(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            results = run / "results"
            results.mkdir()
            (results / "cyclictest_light_vcpu6_vm1.txt").touch()
            (results / "cyclictest_light_vcpu6_vm2.txt").touch()
            self.assertEqual(
                _available_cyclic_series([run], "light"),
                [
                    ("cyclictest_light_vcpu6_vm1.txt", "dom0"),
                    ("cyclictest_light_vcpu6_vm2.txt", "dom1"),
                ],
            )

    def test_nn_plot_uses_cpu6_or_dom0_as_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bare = root / "bare"
            (bare / "results").mkdir(parents=True)
            (bare / "results" / "cyclictest_heavy_cpu7.txt").touch()
            (bare / "results" / "cyclictest_heavy_cpu6.txt").touch()
            self.assertEqual(
                _representative_cyclic_series([bare], "heavy"),
                [("cyclictest_heavy_cpu6.txt", "Cyclictest")],
            )

            xen = root / "xen"
            (xen / "results").mkdir(parents=True)
            (xen / "results" / "cyclictest_heavy_vcpu3_vm2.txt").touch()
            (xen / "results" / "cyclictest_heavy_vcpu3_vm1.txt").touch()
            self.assertEqual(
                _representative_cyclic_series([xen], "heavy"),
                [("cyclictest_heavy_vcpu3_vm1.txt", "dom0")],
            )

    def test_nn_analysis_merges_both_cyclictest_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            results = run / "results"
            results.mkdir()
            (results / "cyclictest_light_vcpu6_vm1.txt").write_text(
                "2 8\n7 1\n# Histogram Overflows: 1\n", encoding="utf-8"
            )
            (results / "cyclictest_light_vcpu6_vm2.txt").write_text(
                "2 2\n9 3\n# Histogram Overflows: 2\n", encoding="utf-8"
            )
            values, counts, overflows, streams = _combine_cyclic_series(
                [run], "light"
            )
            self.assertEqual(values.tolist(), [2.0, 7.0, 9.0])
            self.assertEqual(counts.tolist(), [10, 1, 3])
            self.assertEqual(overflows, 3)
            self.assertEqual(streams, 2)

            _, ccdf = _weighted_ccdf(values, counts)
            np.testing.assert_allclose(ccdf, [1.0, 4 / 14, 3 / 14])

    def test_nn_scatter_pairs_models_with_their_own_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "bare": ("cpu7.txt", "cpu6.txt"),
                "jailhouse": ("cpu3.txt", "cpu6.txt"),
                "xen_credit2": ("_vm2.txt", "_vm1.txt"),
            }
            for environment, (mnas_suffix, inception_suffix) in cases.items():
                run = root / environment
                results = run / "results"
                results.mkdir(parents=True)
                if environment == "bare":
                    filenames = (
                        "cyclictest_light_cpu6.txt",
                        "cyclictest_light_cpu7.txt",
                    )
                elif environment == "jailhouse":
                    filenames = (
                        "cyclictest_light_cpu3.txt",
                        "cyclictest_light_cpu6.txt",
                    )
                else:
                    filenames = (
                        "cyclictest_light_vcpu6_vm1.txt",
                        "cyclictest_light_vcpu6_vm2.txt",
                    )
                for filename in filenames:
                    (results / filename).touch()
                self.assertTrue(
                    _paired_cyclic_filename(
                        environment, "light", "mnas", [run]
                    ).endswith(mnas_suffix)
                )
                self.assertTrue(
                    _paired_cyclic_filename(
                        environment, "light", "inception", [run]
                    ).endswith(inception_suffix)
                )

    def test_nn_analysis_accepts_only_complete_600s_formal_profile(self):
        metadata = {
            "condition": "dual-tflite-formal-v1_d600s_seed12345"
        }
        config = {
            "experiment": {
                "profile_name": "dual-tflite-formal-v1",
                "duration_seconds": 600,
                "profiles": [
                    {"name": "light"},
                    {"name": "medium"},
                    {"name": "heavy"},
                ],
            }
        }
        self.assertTrue(_is_formal_configuration(metadata, config))
        config["experiment"]["duration_seconds"] = 30
        self.assertFalse(_is_formal_configuration(metadata, config))

    def test_nn_analysis_uses_v2_cgroup_only_for_bare(self):
        metadata = {
            "condition": "dual-tflite-formal-v2-cgroup_d600s_seed12345"
        }
        config = {
            "environment_name": "bare",
            "experiment": {
                "profile_name": "dual-tflite-formal-v2-cgroup",
                "duration_seconds": 600,
                "profiles": [
                    {"name": "light"},
                    {"name": "medium"},
                    {"name": "heavy"},
                ],
            },
        }
        self.assertTrue(_is_formal_configuration(metadata, config))

        metadata["condition"] = "dual-tflite-formal-v1_d600s_seed12345"
        config["experiment"]["profile_name"] = "dual-tflite-formal-v1"
        self.assertFalse(_is_formal_configuration(metadata, config))


if __name__ == "__main__":
    unittest.main()
