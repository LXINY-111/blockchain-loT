import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_block_eval import analyze_decisions, build_arg_parser  # noqa: E402


class AnalyzeBlockEvalTest(unittest.TestCase):
    def test_parser_defaults_to_configured_16_shards(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.shards, 16)

    def test_analyze_decisions_reports_action_mask_and_fallback_ratios(self):
        records = [
            {
                "shard": 0,
                "source": "python_ppo",
                "confidence": 0.8,
                "entropy": 0.3,
                "action_mask": [1, 0, 1, 0],
            },
            {
                "shard": 2,
                "source": "python_ppo_capacity_guard",
                "confidence": 0.9,
                "entropy": 0.2,
                "action_mask": [0, 0, 1, 0],
            },
            {
                "shard": 1,
                "source": "heuristic_fallback",
                "confidence": 0.1,
                "entropy": 1.0,
                "action_mask": [1, 1, 0, 1],
            },
        ]

        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "spring_io.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "decision_records.jsonl",
                    "\n".join(json.dumps(record) for record in records) + "\n",
                )

            result = analyze_decisions(zip_path, shards=4)

        self.assertEqual(result["decision_count"], 3)
        self.assertEqual(result["action_mask_size_hist"], {1: 1, 2: 1, 3: 1})
        self.assertAlmostEqual(result["action_mask_size_mean"], 2.0)
        self.assertAlmostEqual(result["single_candidate_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(result["multi_candidate_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(result["python_ppo_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(result["fallback_ratio"], 1.0 / 3.0)
        self.assertEqual(result["fallback_source_hist"], {"heuristic_fallback": 1})


if __name__ == "__main__":
    unittest.main()
