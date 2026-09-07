import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_result9_multiobjective import group_rows, parse_run_name  # noqa: E402


class Result9MultiobjectiveTest(unittest.TestCase):
    def test_parse_run_name_handles_proposed_and_baseline(self):
        proposed = parse_run_name(
            "ppo_top7_w5530_pareto_16s_test2M_i250_seed17_20260813_151207"
        )
        baseline = parse_run_name(
            "nsshard_adapted_cap12_16s_validation444k_i250_seed7_20260807_214042"
        )

        self.assertEqual(proposed["method"], "Proposed-TopK7")
        self.assertEqual(proposed["seed"], 17)
        self.assertEqual(baseline["method"], "NSshard-adapted")
        self.assertEqual(baseline["window"], "validation444k")

    def test_group_rows_uses_sample_sd_only_for_multiple_runs(self):
        rows = []
        for seed, value in [(7, 0.7), (17, 0.8), (27, 0.9)]:
            row = {
                "method": "Proposed-TopK7",
                "window": "test2M",
                "inject_tps": 250,
                "seed": seed,
                "protocol_ok": True,
            }
            for field in __import__(
                "analyze_result9_multiobjective"
            ).NUMERIC_GROUP_FIELDS:
                row[field] = value
            rows.append(row)

        group = group_rows(rows)[0]

        self.assertEqual(group["n"], 3)
        self.assertEqual(group["seeds"], [7, 17, 27])
        self.assertAlmostEqual(group["weighted_cross_ratio_mean"], 0.8)
        self.assertAlmostEqual(group["weighted_cross_ratio_sd"], 0.1)


if __name__ == "__main__":
    unittest.main()
