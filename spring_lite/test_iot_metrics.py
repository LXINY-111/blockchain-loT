"""测量的数值、唯一关联、零分母、参考段隔离和旧入口回归。"""
import contextlib
import copy
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iot_metrics import (HASH_COLUMN, META_COLUMNS, IoTMetrics, Scene,
                         analyze_chain_details, evaluation_context, nearest_rank)
from test_prepare_iot_v2_dataset import generate, sources


def manifest(start=5, count=3):
    return dict(dataset_window=dict(start_tx=start, end_tx_exclusive=start + count),
                thresholds=dict(high_cost_ge=0.8, high_payload_ge=900,
                                long_distance_ge=30, weak_link_le=0.2))


def fixture():
    # 全局场景序号故意不同于文件行号，且 CSV 顺序随后被打乱。
    scenes = [Scene(5, 105, "a", "b", 0, 0, 0, 1, False),
              Scene(6, 106, "c", "d", 0.2, 100, 12, 0.5, False),
              Scene(7, 107, "e", "f", 0.8, 900, 40, 0.1, True)]
    rows = []
    for nonce, scene in enumerate(scenes):
        cross = int(nonce == 1)
        row = {HASH_COLUMN: str(nonce + 100),
               "Tx propose timestamp": "1000000",
               "Tx finally commit timestamp": str(1000000 + 1000 * (nonce + 1)),
               "Confirmed latency of this tx (ms)": str(1000 * (nonce + 1)),
               "Relay1 Tx commit timestamp (not a relay tx -> nil)": "1000500" if cross else "",
               "Relay2 Tx commit timestamp (not a relay tx -> nil)": "1002000" if cross else ""}
        row.update(dict(zip(META_COLUMNS, map(str, (nonce, scene.dataset_row,
            "0x" + scene.sender, "0x" + scene.recipient, 0, cross, cross)))))
        rows.append(row)
    return scenes, rows


class IoTMetricTests(unittest.TestCase):
    def test_stage_load_counts_each_relay_phase_once(self):
        scenes, rows = fixture()
        result = analyze_chain_details(rows, scenes, manifest(), 2)
        self.assertEqual(result['execution_load']['loads'], [3,1])
        self.assertEqual(result['effective_load']['loads'], [2.5,.5])
        self.assertAlmostEqual(result['execution_load']['amplification'], 4/3)
        self.assertAlmostEqual(result['execution_load']['jain_fairness'], .8)

    def test_weighted_ratios_are_ratios_of_sums_and_empty_groups_are_null(self):
        scenes, _ = fixture()
        metric = IoTMetrics(manifest(), "offline")
        for i, scene in enumerate(scenes):
            metric.add(scene, i == 1)
        result = metric.summary()["groups"]
        self.assertAlmostEqual(result["all"]["cross_ratio"], 1 / 3)
        self.assertAlmostEqual(result["all"]["cost_weighted_cross_ratio"], 0.2)
        self.assertAlmostEqual(result["all"]["payload_weighted_cross_ratio"], 0.1)
        self.assertAlmostEqual(result["all"]["cross_cost_per_tx"], 0.2 / 3)
        self.assertIsNone(result["all"]["mean_latency_sec"])
        self.assertEqual(result["high_cost"]["cross_ratio"], 0)

    def test_zero_denominator_and_empty_group_not_reported_as_success(self):
        scene = Scene(5, 105, "a", "b", 0, 0, 0, 1, False)
        metric = IoTMetrics(manifest(count=1), "offline")
        metric.add(scene, True)
        result = metric.summary()["groups"]
        self.assertIsNone(result["all"]["cost_weighted_cross_ratio"])
        self.assertIsNone(result["all"]["payload_weighted_cross_ratio"])
        self.assertEqual(result["high_cost"]["count"], 0)
        self.assertIsNone(result["high_cost"]["cross_ratio"])

    def test_chain_join_uses_keys_not_row_order_and_matches_offline(self):
        scenes, rows = fixture()
        result = analyze_chain_details(list(reversed(rows)), scenes, manifest(), 2)
        all_rows = result["groups"]["all"]
        self.assertEqual(result["join"]["matched_count"], 3)
        self.assertAlmostEqual(all_rows["cost_weighted_cross_ratio"], 0.2)
        self.assertEqual(all_rows["mean_latency_sec"], 2)
        self.assertEqual(all_rows["p95_latency_sec"], 3)
        self.assertEqual(result["groups"]["high_cost"]["mean_latency_sec"], 3)
        legacy = [{key: value for key, value in row.items() if key not in META_COLUMNS} for row in rows]
        recovered = analyze_chain_details(legacy, scenes, manifest(), 2, list(reversed(rows)))
        self.assertEqual(recovered["groups"], result["groups"])
        with self.assertRaisesRegex(ValueError, "metadata"):
            analyze_chain_details(legacy, scenes, manifest(), 2)

        contradictory = copy.deepcopy(rows)
        contradictory[0]["Sender address"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Recovered metadata disagrees"):
            analyze_chain_details(rows, scenes, manifest(), 2, contradictory)
        invalid_relay = copy.deepcopy(rows)
        invalid_relay[1]["Relay1 Tx commit timestamp (not a relay tx -> nil)"] = "nan"
        with self.assertRaisesRegex(ValueError, "relay-stage"):
            analyze_chain_details(invalid_relay, scenes, manifest(), 2)

    def test_wrong_duplicate_missing_or_incomplete_records_fail(self):
        scenes, rows = fixture()
        for field, wrong in (("Sender address", "wrong"), ("Tx local nonce", "500"),
                ("Dataset row index", "105"), ("Sender shard", "99"),
                ("Cross shard", "1"), ("Tx finally commit timestamp", ""),
                ("Confirmed latency of this tx (ms)", "nan"),
                ("Confirmed latency of this tx (ms)", "1000000"),
                ("Relay1 Tx commit timestamp (not a relay tx -> nil)", "1000010")):
            with self.subTest(field=field, wrong=wrong):
                invalid = copy.deepcopy(rows)
                invalid[0][field] = wrong
                with self.assertRaises(ValueError):
                    analyze_chain_details(invalid, scenes, manifest(), 2)
        for invalid in (rows[:-1], rows + [rows[0]], [rows[0], rows[0], rows[2]]):
            with self.assertRaises(ValueError):
                analyze_chain_details(invalid, scenes, manifest(), 2)
        metric = IoTMetrics(manifest(), "offline")
        metric.add(scenes[0], False)
        with self.assertRaises(ValueError):
            metric.add(scenes[0], False)
        with self.assertRaises(ValueError):
            metric.summary()

    def test_reference_window_separate_and_global_chunk_indices_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, chain = generate(root, sources(root), "chunk", count=12, offset=100)
            sidecar = dataset / "transaction_scene.csv"
            scenes, info = evaluation_context(chain, sidecar, 5, 7, 0, 5)
            self.assertEqual((scenes[0].dataset_row, scenes[0].tx_index), (5, 105))
            self.assertEqual(info["reference_window"]["end_tx_exclusive"], 5)
            with self.assertRaisesRegex(ValueError, "reference window"):
                evaluation_context(chain, sidecar, 5, 7, 0, 6)
            for scene in scenes:
                expected = min(scene.distance / 60, 1) * (1 - scene.quality) * max(
                    0.05, min(math.log1p(scene.payload) / math.log1p(1000000), 1))
                self.assertAlmostEqual(scene.cost, expected)

    def test_opt_in_offline_metrics_do_not_change_original_outputs(self):
        import eval_offline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, chain = generate(root, sources(root), "eval")
            argv = ["eval_offline.py", "--csv", str(chain), "--sidecar", str(dataset / "transaction_scene.csv"),
                    "--tx_identity", "iot_v2", "--policy", "hash", "--start_tx", "5", "--max_txs", "7",
                    "--iot_reference_start_tx", "0", "--iot_reference_max_txs", "5", "--log_jsonl", ""]
            results = []
            for extra in ([], ["--iot_metrics"]):
                with patch.object(sys, "argv", argv + extra), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(eval_offline, "write_jsonl") as output:
                    eval_offline.main()
                    results.append(output.call_args.args[1])
            self.assertNotIn("iot_metrics", results[0])
            self.assertEqual(results[0]["summary"], results[1]["summary"])
            self.assertEqual(results[1]["iot_metrics"]["groups"]["all"]["count"], 7)

    def test_nearest_rank_and_inclusive_ties(self):
        self.assertEqual(nearest_rank([1, 2, 3, 4, 5], 0.8), 4)
        self.assertEqual(nearest_rank([2, 2, 2], 0.2), 2)
        self.assertIsNone(nearest_rank([], 0.95))


if __name__ == "__main__":
    unittest.main()
