import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_pareto_checkpoint import (  # noqa: E402
    select_candidate,
    select_candidate_decision,
)


class SelectParetoCheckpointTest(unittest.TestCase):
    def test_selects_lowest_cross_candidate_within_load_guardrails(self):
        manifest = {
            "pareto_frontier": [
                {
                    "epoch": 1,
                    "checkpoint_path": "epoch_001.pt",
                    "cross_ratio": 0.75,
                    "stage_max_load_share": 0.31,
                    "active_shards_mean": 9.3,
                },
                {
                    "epoch": 12,
                    "checkpoint_path": "epoch_012.pt",
                    "cross_ratio": 0.66,
                    "stage_max_load_share": 0.33,
                    "active_shards_mean": 8.1,
                },
                {
                    "epoch": 6,
                    "checkpoint_path": "epoch_006.pt",
                    "cross_ratio": 0.50,
                    "stage_max_load_share": 0.37,
                    "active_shards_mean": 6.5,
                },
            ]
        }

        selected = select_candidate(
            manifest,
            min_active_shards=8.0,
            max_stage_hotspot=0.35,
            max_active_deficit_ratio=0.05,
        )

        self.assertEqual(selected["epoch"], 12)

    def test_uses_bounded_active_tolerance_when_strict_set_is_empty(self):
        manifest = {
            "pareto_frontier": [
                {
                    "epoch": 5,
                    "checkpoint_path": "epoch_005.pt",
                    "cross_ratio": 0.70,
                    "stage_max_load_share": 0.32,
                    "active_shards_mean": 7.82,
                },
                {
                    "epoch": 6,
                    "checkpoint_path": "epoch_006.pt",
                    "cross_ratio": 0.62,
                    "stage_max_load_share": 0.33,
                    "active_shards_mean": 7.70,
                },
            ]
        }

        decision = select_candidate_decision(
            manifest,
            min_active_shards=8.0,
            max_stage_hotspot=0.35,
            max_active_deficit_ratio=0.05,
        )

        self.assertEqual(decision["selection_tier"], "active_tolerance")
        self.assertFalse(decision["strict_feasible"])
        self.assertEqual(decision["selected"]["epoch"], 5)
        self.assertAlmostEqual(
            decision["constraint_audit"]["active_deficit_ratio"],
            0.0225,
        )
        self.assertFalse(
            decision["constraint_audit"]["strict_constraints_satisfied"]
        )
        self.assertTrue(
            decision["constraint_audit"]["fallback_constraints_satisfied"]
        )

    def test_fallback_never_relaxes_stage_hotspot_cap(self):
        manifest = {
            "pareto_frontier": [
                {
                    "epoch": 5,
                    "checkpoint_path": "epoch_005.pt",
                    "cross_ratio": 0.55,
                    "stage_max_load_share": 0.351,
                    "active_shards_mean": 7.99,
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "bounded active-shard"):
            select_candidate(
                manifest,
                min_active_shards=8.0,
                max_stage_hotspot=0.35,
                max_active_deficit_ratio=0.05,
            )

    def test_rejects_candidate_outside_active_tolerance(self):
        manifest = {
            "pareto_frontier": [
                {
                    "epoch": 6,
                    "checkpoint_path": "epoch_006.pt",
                    "cross_ratio": 0.50,
                    "stage_max_load_share": 0.33,
                    "active_shards_mean": 7.59,
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "bounded active-shard"):
            select_candidate(
                manifest,
                min_active_shards=8.0,
                max_stage_hotspot=0.35,
                max_active_deficit_ratio=0.05,
            )


if __name__ == "__main__":
    unittest.main()
