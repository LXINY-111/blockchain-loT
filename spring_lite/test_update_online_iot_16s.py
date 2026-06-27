import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IOT_FEATURE_DIM, state_dim  # noqa: E402
from update_online import build_agent, build_buffer_from_update  # noqa: E402


class UpdateOnlineIoTShard16Test(unittest.TestCase):
    def test_build_buffer_accepts_16_shard_iot_state(self):
        iot_state_dim = state_dim(16, IOT_FEATURE_DIM)
        data = {
            "shards": 16,
            "iot_feature_dim": IOT_FEATURE_DIM,
            "reward": 0.25,
            "actions": [
                {
                    "state": [0.0 for _ in range(iot_state_dim)],
                    "next_state": [0.1 for _ in range(iot_state_dim)],
                    "action": 4,
                    "log_prob": -1.25,
                    "value": 0.5,
                    "reward": 0.2,
                    "done": True,
                    "related_known": True,
                    "related_shard": 4,
                    "related_weight": 0.8,
                    "action_mask": [1 if sid < 4 else 0 for sid in range(16)],
                }
            ],
        }

        buffer, meta = build_buffer_from_update(data)

        self.assertEqual(len(buffer), 1)
        self.assertEqual(meta["skipped"], 0)
        self.assertEqual(meta["iot_feature_dim"], IOT_FEATURE_DIM)
        self.assertEqual(meta["state_dim"], iot_state_dim)
        self.assertEqual(meta["action_hist"][4], 1)

    def test_build_agent_uses_iot_state_dim_for_16_shards(self):
        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "ppo_top4_16s_seed7.pt"
            agent, _payload, source = build_agent(
                16,
                model_path,
                iot_feature_dim=IOT_FEATURE_DIM,
            )

        self.assertEqual(agent.state_dim, state_dim(16, IOT_FEATURE_DIM))
        self.assertEqual(agent.action_dim, 16)
        self.assertEqual(source, "new_model")


if __name__ == "__main__":
    unittest.main()
