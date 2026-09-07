"""Isolated audit probes. Failures demonstrate current defects, not fixes."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT.parent.parent / "spring_lite"))
import analyze_result9_multiobjective as metrics
import finalize_exp8_block_run as finalizer


class AuditMetrics(unittest.TestCase):
    def flatten(self, effective, rows, valid):
        run = AUDIT / "ppo_top7_w5530_pareto_16s_test2M_i250_seed7_20260905_000000"
        complete = {"state_check": "passed", "latency_details": {"row_count": rows, "valid_latency_count": valid}}
        with patch.object(metrics, "read_json", return_value=complete), patch.object(
            metrics, "recompute_all_active", return_value={"effective_total": effective}
        ):
            return metrics.flatten_run(run)

    def test_incomplete_latency_coverage_cannot_pass_protocol(self):
        row = self.flatten(2_000_000, 1, 1)
        self.assertFalse(row["protocol_ok"], "one valid latency row passed a two-million-transaction protocol")

    def test_overcounted_transactions_cannot_pass_protocol(self):
        row = self.flatten(2_001_000, 2_001_000, 2_001_000)
        self.assertFalse(row["protocol_ok"], "completion_ratio above 1 still passed protocol")

    def test_failed_protocol_run_excluded_from_comparison_mean(self):
        good = self.flatten(2_000_000, 2_000_000, 2_000_000)
        bad = self.flatten(1_000_000, 1_000_000, 1_000_000)
        good["weighted_cross_ratio"] = 0.8
        bad["weighted_cross_ratio"] = 0.0
        group = metrics.group_rows([good, bad])[0]
        self.assertEqual(group["weighted_cross_ratio_mean"], 0.8,
                         "failed run changed official comparison mean")

    def test_skipped_state_check_not_marked_passed(self):
        with tempfile.TemporaryDirectory(prefix="finalizer-probe-", dir=AUDIT) as tmp:
            root = Path(tmp)
            (root / "run_context.json").write_text(json.dumps({
                "params_snapshot": str(root / "snapshot.json"),
                "exp_test": str(root / "expTest"), "experiment": "isolated-audit"
            }), encoding="utf-8")

            def fake_run(command, output_path, env):
                if command[0] == "go":
                    return subprocess.CompletedProcess(command, 0, stdout="--- SKIP: TestFinalResult (0.00s)\nPASS\n")
                (root / "analysis.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="{}")

            with patch.object(finalizer, "parse_args", return_value=SimpleNamespace(run_root=str(root))), \
                 patch.object(finalizer, "ensure_results8_run_root", return_value=root), \
                 patch.object(finalizer, "running_block_emulator_processes", return_value=[]), \
                 patch.object(finalizer, "sha256_file", return_value="same"), \
                 patch.object(finalizer, "archive_spring_io", return_value=root / "spring_io"), \
                 patch.object(finalizer, "run_and_record", side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                try:
                    finalizer.main()
                except RuntimeError:
                    return
            complete = json.loads((root / "run_complete.json").read_text(encoding="utf-8"))
            self.assertNotEqual(complete["state_check"], "passed", "SKIP and empty analysis were finalized as passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
