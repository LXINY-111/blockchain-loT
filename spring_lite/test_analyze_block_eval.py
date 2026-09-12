import json
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_block_eval import (  # noqa: E402
    analyze_decisions,
    analyze_injection_pace,
    analyze_tx_latency_details,
    build_arg_parser,
    read_csv_source,
    summarize_load_balance,
)


class AnalyzeBlockEvalTest(unittest.TestCase):
    def test_parser_defaults_to_configured_16_shards(self):
        parser = build_arg_parser()
        args = parser.parse_args([])

        self.assertEqual(args.shards, 16)
        self.assertEqual(args.result_path, "expTest/result")
        self.assertEqual(args.spring_io_path, "spring_io")

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

    def test_analyze_decisions_accepts_unpacked_directory(self):
        record = {
            "shard": 3,
            "source": "python_ppo",
            "confidence": 0.7,
            "entropy": 0.4,
            "action_mask": [0, 0, 0, 1],
        }
        with TemporaryDirectory() as tmp:
            spring_io = Path(tmp) / "spring_io"
            spring_io.mkdir()
            (spring_io / "decision_records.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            result = analyze_decisions(spring_io, shards=4)

        self.assertEqual(result["decision_count"], 1)
        self.assertEqual(result["action_hist"], [0, 0, 0, 1])
        self.assertEqual(result["python_ppo_ratio"], 1.0)

    def test_analyze_decisions_distinguishes_major_and_any_related_shard(self):
        records = []
        cases = [
            (2, [0.7, 0.0, 0.3, 0.0]),
            (0, [0.6, 0.4, 0.0, 0.0]),
            (3, [0.5, 0.5, 0.0, 0.0]),
        ]
        for shard, sender_pos in cases:
            state = [0.0] * 40 + sender_pos
            records.append(
                {
                    "shard": shard,
                    "source": "python_ppo",
                    "state": state,
                }
            )

        with TemporaryDirectory() as tmp:
            spring_io = Path(tmp) / "spring_io"
            spring_io.mkdir()
            (spring_io / "decision_records.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = analyze_decisions(spring_io, shards=4)

        self.assertEqual(result["major_related_follow_count"], 1)
        self.assertAlmostEqual(result["major_related_follow_ratio"], 1.0 / 3.0)
        self.assertEqual(result["selected_related_mass_count"], 2)
        self.assertAlmostEqual(result["same_as_related_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(result["chosen_related_mass_mean"], 0.3)

    def test_analyze_decisions_reports_candidate_coverage_and_conditional_choice(self):
        cases = [
            # Major shard 0 is available and chosen; all related shards covered.
            (0, [0.7, 0.0, 0.3, 0.0], [1, 0, 1, 0]),
            # Major shard 1 is absent; only 40% of relation mass is covered.
            (2, [0.0, 0.6, 0.4, 0.0], [0, 0, 1, 1]),
            # Major shard 0 is available but PPO chooses another related shard.
            (1, [0.5, 0.5, 0.0, 0.0], [1, 1, 0, 0]),
        ]
        records = []
        for shard, sender_pos, mask in cases:
            records.append(
                {
                    "shard": shard,
                    "source": "python_ppo",
                    "state": [0.0] * 40 + sender_pos,
                    "action_mask": mask,
                }
            )

        with TemporaryDirectory() as tmp:
            spring_io = Path(tmp) / "spring_io"
            spring_io.mkdir()
            (spring_io / "decision_records.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            result = analyze_decisions(spring_io, shards=4)

        self.assertEqual(result["candidate_diagnostic_seen"], 3)
        self.assertAlmostEqual(result["candidate_major_anchor_coverage_ratio"], 2 / 3)
        self.assertAlmostEqual(result["candidate_major_anchor_absent_ratio"], 1 / 3)
        self.assertAlmostEqual(result["candidate_all_related_coverage_ratio"], 2 / 3)
        self.assertAlmostEqual(
            result["candidate_related_mass_coverage_mean"],
            0.8,
        )
        self.assertAlmostEqual(
            result["major_anchor_choice_when_available_ratio"],
            0.5,
        )

    def test_summarize_load_balance_reports_scale_free_metrics(self):
        rows = {
            "1": {
                "EpochID": "1",
                "Shard_0_Load": "8",
                "Shard_1_Load": "2",
                "Shard_2_Load": "0",
                "Shard_3_Load": "0",
            },
            "2": {
                "EpochID": "2",
                "Shard_0_Load": "4",
                "Shard_1_Load": "4",
                "Shard_2_Load": "4",
                "Shard_3_Load": "4",
            },
        }

        summary = summarize_load_balance(["1", "2"], rows, shards=4)

        self.assertEqual(summary["load_epoch_count"], 2)
        self.assertAlmostEqual(summary["mean_max_shard_load_share"], 0.525)
        self.assertAlmostEqual(summary["mean_active_shards"], 3.0)
        self.assertAlmostEqual(summary["mean_jain_fairness"], (25 / 68 + 1) / 2)
        self.assertAlmostEqual(summary["aggregate_max_shard_load_share"], 12 / 26)

    def test_read_csv_source_accepts_unpacked_exp_test_or_result_directory(self):
        with TemporaryDirectory() as tmp:
            exp_test = Path(tmp) / "expTest"
            output_dir = exp_test / "result" / "supervisor_measureOutput"
            output_dir.mkdir(parents=True)
            csv_path = output_dir / "Tx_number.csv"
            csv_path.write_text("EpochID,Total tx # in this epoch\n1,1000\n", encoding="utf-8")

            from_exp_test = read_csv_source(
                exp_test,
                "supervisor_measureOutput/Tx_number.csv",
            )
            from_result = read_csv_source(
                exp_test / "result",
                "supervisor_measureOutput/Tx_number.csv",
            )

        self.assertEqual(from_exp_test, from_result)
        self.assertEqual(from_result[0]["EpochID"], "1")

    def test_analyze_injection_pace_reports_actual_offered_tps(self):
        lines = [
            (
                "[INJECTION PACE] batchTx=1000 cumulativeTx=1000 "
                "targetTPS=300 elapsedSec=3.333400 "
                "actualOfferedTPS=299.994000 scheduleLagMs=0.067"
            ),
            (
                "[INJECTION PACE] batchTx=1000 cumulativeTx=2000 "
                "targetTPS=300 elapsedSec=6.666700 "
                "actualOfferedTPS=299.998500 scheduleLagMs=0.033"
            ),
        ]
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "supervisor.log"
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            result = analyze_injection_pace(log_path)

        self.assertEqual(result["pace_record_count"], 2)
        self.assertEqual(result["final_cumulative_tx"], 2000)
        self.assertEqual(result["target_tps"], 300.0)
        self.assertAlmostEqual(result["actual_offered_tps"], 299.9985)
        self.assertAlmostEqual(result["attainment_ratio"], 0.999995)
        self.assertAlmostEqual(result["final_schedule_lag_ms"], 0.033)

    def test_analyze_tx_latency_details_reports_tail_quantiles(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp) / "result"
            output = result / "supervisor_measureOutput"
            output.mkdir(parents=True)
            (output / "Tx_Details.csv").write_text(
                "TxHash,Confirmed latency of this tx (ms)\n"
                "a,1000\n"
                "b,2000\n"
                "c,10000\n"
                "d,4000\n",
                encoding="utf-8",
            )

            summary = analyze_tx_latency_details(result)

        self.assertEqual(summary["row_count"], 4)
        self.assertEqual(summary["valid_latency_count"], 4)
        self.assertEqual(summary["valid_latency_ratio"], 1.0)
        self.assertEqual(summary["unique_tx_hash_count"], 4)
        self.assertEqual(summary["duplicate_tx_hash_count"], 0)
        self.assertEqual(summary["missing_tx_hash_count"], 0)
        self.assertAlmostEqual(summary["mean_sec"], 4.25)
        self.assertAlmostEqual(summary["p50_sec"], 2.0)
        self.assertAlmostEqual(summary["p95_sec"], 10.0)
        self.assertAlmostEqual(summary["p99_sec"], 10.0)
        self.assertAlmostEqual(summary["max_sec"], 10.0)


if __name__ == "__main__":
    unittest.main()
