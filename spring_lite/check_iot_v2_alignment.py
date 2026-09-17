"""用同一真实数据窗口逐项核对 Go/Python；临时比较文件不写入数据目录。"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from itertools import islice
from pathlib import Path

from config import ROOT_DIR
from iot_v2 import V2_DIRECTORY, batch_account_scenes
from offline_env import PolicyOutput, SpringOfflineEnv, iter_batches, load_iot_transactions
from mechanism import add_mechanism_args, settings, metadata, input_dim


def make_fixture(directory, start, count, mode, batch_size, mechanism=None):
    txs = load_iot_transactions(directory / "selectedTxs_iot_v2.csv",
                               directory / "transaction_scene.csv", count, start, identity="iot_v2")
    with (directory / "selectedTxs_iot_v2.csv").open(encoding="utf-8", newline="") as handle:
        raw = list(islice(csv.reader(handle), start, start + len(txs)))
    env = SpringOfflineEnv(shards=16, tx_batch_size=batch_size, max_block_size=1000,
                           iot_feature_dim=10, sender_pos_mode=mode,
                           candidate_top_k=7, reward_mode="iot_dense_balanced", **(mechanism or {}))
    def policy(state, address, related, info):
        return PolicyOutput(action=env._best_candidate_action(address, info.sender_pos, info.action_mask))
    batches = []
    cursor = 0
    for batch in iter_batches(txs, batch_size):
        scenes = batch_account_scenes(batch)
        actions, metrics = env.run_batch(batch, policy)
        batches.append(dict(
            Rows=raw[cursor:cursor+len(batch)],
            Forward=[list(tx.iot_features) for tx in batch],
            Reverse=[list(tx.recipient_iot_features) for tx in batch],
            Costs=[tx.communication_cost_weight for tx in batch],
            Actions=[dict(Address=a.address, Related=a.related, State=a.state,
                          ActionMask=a.action_mask, Shard=a.action, LocalReward=a.local_reward,
                          Cost=scenes[a.address][1], CostByShard=a.cost_by_shard) for a in actions], Reward=metrics.reward))
        cursor += len(batch)
    return dict(ScenePath=str((directory / "transaction_scene.csv").resolve()), StartTx=start,
                Count=len(txs), SenderPosMode=mode, BatchSize=batch_size, Batches=batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=V2_DIRECTORY)
    parser.add_argument("--start-tx", type=int, default=0)
    parser.add_argument("--max-txs", type=int, default=2001)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sender-pos-mode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--model", type=Path, help="also compare actual Go-to-Python PPO inference")
    add_mechanism_args(parser)
    args = parser.parse_args()
    fixture = make_fixture(args.dataset_dir, args.start_tx, args.max_txs,
                           args.sender_pos_mode, args.batch_size, settings(args))
    fixture.update(MechanismVersion=settings(args)['mechanism_version'], FeatureMode=settings(args)['feature_mode'],
                   SceneCostMode=settings(args)['scene_cost_mode'])
    if args.model:
        from infer_server import AgentCache, infer_items
        items = [dict(address=a["Address"], related=a["Related"], state=a["State"],
                      action_mask=a["ActionMask"]) for b in fixture["Batches"] for a in b["Actions"]][:16]
        outputs = infer_items(items, 16, False, 1, args.model.resolve(), 10, False,
                              {"load_semantics_version": "real_accounts_execution_v2_1", **metadata(args)}, AgentCache())
        fixture.update(Model=str(args.model.resolve()), PythonExecutable=sys.executable,
                       PythonDirectory=str(Path(__file__).resolve().parent),
                       InferenceItems=items, InferenceExpected=outputs)
    with tempfile.TemporaryDirectory(prefix="iot_v2_parity_") as temp:
        path = Path(temp) / "comparison.json"
        path.write_text(json.dumps(fixture, ensure_ascii=True), encoding="utf-8")
        env = dict(os.environ, IOT_V2_PARITY_INPUT=str(path))
        subprocess.run(["go", "test", "./supervisor/committee", "-run", "^TestIOTV2CrossLanguage$",
                        "-count=1", "-v"], cwd=ROOT_DIR, env=env, check=True)
    print(json.dumps(dict(status="passed", transactions=fixture["Count"],
                          batches=len(fixture["Batches"]),
                          actions=sum(len(b["Actions"]) for b in fixture["Batches"]),
                          state_dim=input_dim(16, 10, settings(args)['mechanism_version']), sender_pos_mode=args.sender_pos_mode,
                          ppo_inference_checks=len(fixture.get("InferenceItems", [])))))


if __name__ == "__main__":
    main()
