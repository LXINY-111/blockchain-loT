import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CHECKPOINT_DIR,
    BATCH_SIZE,
    DEFAULT_CAPACITY_GUARD,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CAPACITY_GUARD_FACTOR,
    DEFAULT_EVAL_LOG_JSONL,
    DEFAULT_IOT_REWARD_MODE,
    DEFAULT_MAX_TXS,
    DEFAULT_OFFLINE_EPOCHS,
    DEFAULT_SHARD_NUM,
    DEFAULT_TRAIN_LOG_JSONL,
    DEFAULT_TX_BATCH_SIZE,
    EXPERIMENT_ROOT,
    IOT_FEATURE_DIM,
    MODEL_PATH,
    state_dim,
)


class Exp7Shard16ConfigTest(unittest.TestCase):
    def test_defaults_target_exp7_16_shard_iot_mainline(self):
        self.assertEqual(DEFAULT_SHARD_NUM, 16)
        self.assertEqual(state_dim(DEFAULT_SHARD_NUM, IOT_FEATURE_DIM), 203)
        self.assertIn("实验结果7", str(EXPERIMENT_ROOT))
        self.assertEqual(CHECKPOINT_DIR, EXPERIMENT_ROOT / "models")
        self.assertEqual(MODEL_PATH, CHECKPOINT_DIR / "ppo_top8_guard13_w4540_16s_seed7.pt")
        self.assertEqual(
            DEFAULT_TRAIN_LOG_JSONL,
            EXPERIMENT_ROOT / "offline_training" / "ppo_top8_guard13_w4540_16s_seed7_train.jsonl",
        )
        self.assertEqual(
            DEFAULT_EVAL_LOG_JSONL,
            EXPERIMENT_ROOT / "offline_eval" / "ppo_top8_guard13_w4540_16s_seed7_eval.jsonl",
        )
        self.assertEqual(DEFAULT_CANDIDATE_TOP_K, 8)
        self.assertEqual(DEFAULT_CAPACITY_GUARD, 1)
        self.assertEqual(DEFAULT_CAPACITY_GUARD_FACTOR, 1.3)
        self.assertEqual(DEFAULT_IOT_REWARD_MODE, "iot_dense_balanced")
        self.assertEqual(BATCH_SIZE, 1024)
        self.assertEqual(DEFAULT_OFFLINE_EPOCHS, 15)
        self.assertEqual(DEFAULT_MAX_TXS, 2_000_000)
        self.assertEqual(DEFAULT_TX_BATCH_SIZE, 1000)


if __name__ == "__main__":
    unittest.main()
