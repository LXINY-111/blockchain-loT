"""新数据接入的边界测试：身份、窗口、模型隔离和真实账户放置。"""
import argparse
import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

from config import LOAD_SEMANTICS_VERSION
from infer_server import AgentCache, infer_items
from iot_v2 import V2_LOAD_SEMANTICS, batch_account_scenes, load_semantics, scene_for_legacy_features
from offline_env import PolicyOutput, SpringOfflineEnv, load_iot_transactions
from ppo import PPOAgent
from test_prepare_iot_v2_dataset import generate, sources
from train_offline import new_summary, update_summary
from update_online import run_update
from prepare_iot_v2_run import prepare, RUNTIME_FILES


class V2IntegrationTests(unittest.TestCase):
    def test_run_directory_references_shared_resources_without_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = generate(root, sources(root), "dataset")
            binary = root / "simulator.exe"
            binary.write_bytes(b"fixture, never executed")
            destination = root / "run"
            args = argparse.Namespace(output_dir=destination, dataset_dir=dataset, binary=binary,
                model=None, max_txs=3, start_tx=0, inject_speed=250, base_port=43000, policy="hash", seed=7)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
            self.assertEqual({p.name for p in destination.iterdir()},
                             {"paramsConfig.json", "ipTable.json", "run_context.json", "run.ps1"})
            context = json.loads((destination / "run_context.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(context["binary"]), binary)
            self.assertEqual(Path(context["python_dir"]), Path(__file__).resolve().parent)
            self.assertEqual(set(context["python_files_sha256"]), set(RUNTIME_FILES))
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_identity_amount_chunk_offset_and_both_account_placements(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, chain = generate(root, sources(root), "chunk", count=7, offset=5)
            txs = load_iot_transactions(chain, output / "transaction_scene.csv", identity="iot_v2")
            self.assertEqual(txs[0].sender, f"{6:040x}")
            self.assertEqual(txs[0].value, 900719925474099312345)
            sliced = load_iot_transactions(chain, output / "transaction_scene.csv", 3, 2, identity="iot_v2")
            self.assertEqual(sliced, txs[2:5])
            env = SpringOfflineEnv(shards=16, iot_feature_dim=10, candidate_top_k=7)
            def policy(state, address, related, info):
                return PolicyOutput(action=env._best_candidate_action(address, info.sender_pos, info.action_mask))
            actions, metrics = env.run_batch(txs, policy)
            addresses = {a for tx in txs for a in (tx.sender, tx.recipient)}
            self.assertEqual({a.address for a in actions}, addresses)
            self.assertEqual(sum(env.shard_load), len(addresses))
            self.assertTrue(all(len(a.state) == 203 for a in actions))
            summary = new_summary(16)
            update_summary(summary, metrics)
            self.assertEqual(summary["load_semantics_version"], V2_LOAD_SEMANTICS)

    def test_wrong_identity_short_window_and_changed_addresses_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, chain = generate(root, sources(root), "small")
            scene = output / "transaction_scene.csv"
            with self.assertRaises(ValueError):
                load_iot_transactions(chain, scene)
            with self.assertRaises(ValueError):
                load_iot_transactions(chain, scene, 20, identity="iot_v2")
            original = scene.read_text(encoding="utf-8")
            scene.write_text(original.replace(f"0x{1:040x}", f"0x{999:040x}", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_iot_transactions(chain, scene, identity="iot_v2")

    def test_reverse_zero_quality_and_nonfinite_fields(self):
        row = dict(distance_m="12", link_quality_forward="0", link_quality_reverse="0.7",
                   flowDuration="0", srcNumPackets="1", dstNumPackets="2", srcPayloadSize="30",
                   dstPayloadSize="40", from_actor_type="device", to_actor_type="device")
        self.assertEqual(scene_for_legacy_features(row)["link_quality"], "0")
        self.assertEqual(scene_for_legacy_features(row, reverse=True)["link_quality"], "0.7")
        row["flowDuration"] = "nan"
        with self.assertRaises(ValueError):
            scene_for_legacy_features(row)

    def test_same_dimension_old_model_rejected_in_inference_and_online_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "old.pt"
            PPOAgent(state_dim=203, action_dim=16).save(model, extra={"load_semantics_version": LOAD_SEMANTICS_VERSION})
            expected = {"load_semantics_version": V2_LOAD_SEMANTICS}
            with self.assertRaises(RuntimeError):
                infer_items([], 16, False, 1, model, 10, False, expected, AgentCache())
            request = root / "update.json"
            request.write_text(json.dumps(dict(shards=16, iot_feature_dim=10,
                                               expected_checkpoint=expected, actions=[])), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_update(request, model, root / "log.jsonl")
            self.assertFalse((root / "log.jsonl").exists())
            self.assertEqual(load_semantics(argparse.Namespace(tx_identity="iot_v2")), V2_LOAD_SEMANTICS)


if __name__ == "__main__":
    unittest.main()
