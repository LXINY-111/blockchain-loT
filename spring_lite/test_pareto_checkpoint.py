import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_offline import (  # noqa: E402
    checkpoint_manifest_view,
    checkpoint_pareto_frontier,
)


def record(epoch, cross_ratio, stage_hotspot, active_shards):
    return {
        "event": "validation",
        "epoch": epoch,
        "checkpoint_path": f"epoch_{epoch:03d}.pt",
        "score": -float(epoch),
        "summary": {
            "cross_ratio": cross_ratio,
            "relation_cross_ratio": cross_ratio,
            "active_shards_mean": active_shards,
            "owner_max_load_share": 0.55,
            "stage_max_load_share": stage_hotspot,
        },
        "score_parts": {"system_stage_tps_ceiling": 350.0},
    }


class ParetoCheckpointTest(unittest.TestCase):
    def test_frontier_preserves_distinct_cross_load_tradeoffs(self):
        balanced = record(1, 0.75, 0.31, 9.3)
        compromise = record(12, 0.66, 0.33, 8.1)
        cross_focused = record(6, 0.50, 0.37, 6.5)
        dominated = record(2, 0.80, 0.40, 7.0)

        frontier = checkpoint_pareto_frontier(
            [balanced, compromise, cross_focused, dominated]
        )

        self.assertEqual(
            {item["epoch"] for item in frontier},
            {1, 6, 12},
        )

    def test_manifest_view_keeps_selection_metrics_and_path(self):
        candidate = record(12, 0.66, 0.33, 8.1)

        view = checkpoint_manifest_view(candidate)

        self.assertEqual(view["epoch"], 12)
        self.assertEqual(view["checkpoint_path"], "epoch_012.pt")
        self.assertAlmostEqual(view["cross_ratio"], 0.66)
        self.assertAlmostEqual(view["stage_max_load_share"], 0.33)
        self.assertAlmostEqual(view["active_shards_mean"], 8.1)
        self.assertAlmostEqual(view["system_stage_tps_ceiling"], 350.0)


if __name__ == "__main__":
    unittest.main()
