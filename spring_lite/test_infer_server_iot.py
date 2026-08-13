import sys
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


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

    def test_infer_items_respects_action_mask(self):
        shards = 4
        iot_dim = state_dim(shards, IOT_FEATURE_DIM)

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "masked_iot_ppo.pt"
            agent = PPOAgent(
                state_dim=iot_dim,
                action_dim=shards,
                hidden_dim=HIDDEN_DIM,
                device="cpu",
            )
            with torch.no_grad():
                for param in agent.net.parameters():
                    param.zero_()
                agent.net.actor[-1].bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
            agent.save(model_path, extra={"model_source": "unit_test_mask"})

            outputs = infer_items(
                items=[
                    {
                        "address": "iot_state:camera",
                        "related": "iot_peer:cloud",
                        "state": [0.0 for _ in range(iot_dim)],
                        "action_mask": [0, 1, 0, 0],
                    }
                ],
                shards=shards,
                sample=False,
                request_id=9,
                model_path=model_path,
                iot_feature_dim=IOT_FEATURE_DIM,
                cache=AgentCache(),
            )

        self.assertEqual(outputs[0]["source"], "python_ppo")
        self.assertEqual(outputs[0]["shard"], 1)

    def test_infer_server_reads_utf8_json_from_go_pipe_with_chinese_model_path(self):
        shards = 4
        iot_dim = state_dim(shards, IOT_FEATURE_DIM)

        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "实验结果9" / "models"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "spring_iot_ppo.pt"
            agent = PPOAgent(
                state_dim=iot_dim,
                action_dim=shards,
                hidden_dim=HIDDEN_DIM,
                device="cpu",
            )
            agent.save(model_path, extra={"model_source": "unit_test_utf8_pipe"})

            req = {
                "request_id": 10,
                "shards": shards,
                "iot_feature_dim": IOT_FEATURE_DIM,
                "sample": False,
                "model": str(model_path),
                "allow_model_init": False,
                "items": [
                    {
                        "address": "iot_state:camera",
                        "related": "iot_peer:cloud",
                        "state": [0.0 for _ in range(iot_dim)],
                    }
                ],
            }
            payload = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
            script = Path(__file__).resolve().parent / "infer_server.py"

            completed = subprocess.run(
                [sys.executable, "-u", str(script)],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
        resp = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(resp["items"][0]["source"], "python_ppo")

    def test_checkpoint_metadata_mismatch_falls_back_with_explicit_source(self):
        shards = 4
        iot_dim = state_dim(shards, IOT_FEATURE_DIM)

        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "wrong_topk.pt"
            agent = PPOAgent(
                state_dim=iot_dim,
                action_dim=shards,
                hidden_dim=HIDDEN_DIM,
                device="cpu",
            )
            agent.save(model_path, extra={"candidate_top_k": 8})

            with self.assertRaisesRegex(RuntimeError, "checkpoint config mismatch"):
                infer_items(
                    items=[
                        {
                            "address": "iot_state:camera",
                            "related": "iot_peer:cloud",
                            "state": [0.0 for _ in range(iot_dim)],
                        }
                    ],
                    shards=shards,
                    sample=False,
                    request_id=11,
                    model_path=model_path,
                    iot_feature_dim=IOT_FEATURE_DIM,
                    allow_model_init=False,
                    expected_checkpoint={"candidate_top_k": 7},
                    cache=AgentCache(),
                )


if __name__ == "__main__":
    unittest.main()
