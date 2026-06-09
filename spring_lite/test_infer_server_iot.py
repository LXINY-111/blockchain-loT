import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HIDDEN_DIM, IOT_FEATURE_DIM, state_dim  # noqa: E402
from infer_server import AgentCache, infer_items  # noqa: E402
from ppo import PPOAgent  # noqa: E402


class InferServerIoTTest(unittest.TestCase):
    def test_infer_items_accepts_iot_state_dim_and_model(self):
        shards = 4
        iot_dim = state_dim(shards, IOT_FEATURE_DIM)

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "spring_iot_ppo.pt"
            agent = PPOAgent(
                state_dim=iot_dim,
                action_dim=shards,
                hidden_dim=HIDDEN_DIM,
                device="cpu",
            )
            agent.save(model_path, extra={"model_source": "unit_test_iot"})

            cache = AgentCache()
            outputs = infer_items(
                items=[
                    {
                        "address": "iot_state:camera",
                        "related": "iot_peer:cloud",
                        "state": [0.0 for _ in range(iot_dim)],
                    }
                ],
                shards=shards,
                sample=False,
                request_id=7,
                model_path=model_path,
                iot_feature_dim=IOT_FEATURE_DIM,
                cache=cache,
            )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["source"], "python_ppo")
        self.assertEqual(outputs[0]["batch_id"], 7)
        self.assertGreaterEqual(outputs[0]["shard"], 0)
        self.assertLess(outputs[0]["shard"], shards)

    def test_missing_iot_model_does_not_auto_init_during_evaluation(self):
        shards = 4
        iot_dim = state_dim(shards, IOT_FEATURE_DIM)

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "missing_iot_ppo.pt"
            outputs = infer_items(
                items=[
                    {
                        "address": "iot_state:camera",
                        "related": "iot_peer:cloud",
                        "state": [0.0 for _ in range(iot_dim)],
                    }
                ],
                shards=shards,
                sample=False,
                request_id=8,
                model_path=model_path,
                iot_feature_dim=IOT_FEATURE_DIM,
                allow_model_init=False,
                cache=AgentCache(),
            )

            self.assertFalse(model_path.exists())

        self.assertEqual(outputs[0]["source"], "python_heuristic_no_model")
        self.assertEqual(outputs[0]["batch_id"], 8)


if __name__ == "__main__":
    unittest.main()
