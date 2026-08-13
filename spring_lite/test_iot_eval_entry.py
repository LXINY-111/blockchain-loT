import argparse
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_IOT_REWARD_MODE, IOT_FEATURE_DIM, REWARD_MODE, state_dim  # noqa: E402
from eval_offline import (  # noqa: E402
    eval_state_dim,
    load_eval_transactions,
    make_policy,
    normalize_reward_mode,
)


class IoTEvalEntryTest(unittest.TestCase):
    def test_eval_state_dim_switches_with_mdp_mode(self):
        self.assertEqual(eval_state_dim(argparse.Namespace(mdp_mode="spring"), 4), state_dim(4))
        self.assertEqual(
            eval_state_dim(argparse.Namespace(mdp_mode="iot"), 4),
            state_dim(4, IOT_FEATURE_DIM),
        )

    def test_iot_default_reward_mode_uses_dense_balanced_mainline(self):
        args = argparse.Namespace(mdp_mode="iot", reward_mode=REWARD_MODE)

        normalize_reward_mode(args)

        self.assertEqual(args.reward_mode, DEFAULT_IOT_REWARD_MODE)

    def test_load_eval_transactions_uses_iot_sidecar(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            from_addr = "0x" + "1" * 40
            to_addr = "0x" + "2" * 40

            with tx_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                row = [""] * 18
                row[3] = from_addr
                row[4] = to_addr
                row[6] = "0"
                row[7] = "0"
                row[8] = "1"
                writer.writerow(row)

            with sidecar_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "tx_index",
                        "device_label",
                        "device_mac",
                        "srcIp",
                        "dstIp",
                        "srcPort",
                        "dstPort",
                        "protocol",
                        "srcNumPackets",
                        "dstNumPackets",
                        "srcPayloadSize",
                        "dstPayloadSize",
                        "from_address",
                        "to_address",
                        "distance",
                        "link_quality",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tx_index": "0",
                        "device_label": "TPLinkSmartPlug",
                        "device_mac": "50:c7:bf:00:56:39",
                        "srcIp": "192.168.1.227",
                        "dstIp": "54.254.250.149",
                        "srcPort": "40000",
                        "dstPort": "443",
                        "protocol": "tls",
                        "srcNumPackets": "2",
                        "dstNumPackets": "2",
                        "srcPayloadSize": "100",
                        "dstPayloadSize": "200",
                        "from_address": from_addr,
                        "to_address": to_addr,
                        "distance": "25",
                        "link_quality": "0.5",
                    }
                )

            args = argparse.Namespace(mdp_mode="iot", sidecar=str(sidecar_csv), max_txs=1)
            txs = load_eval_transactions(args, tx_csv)

            self.assertEqual(len(txs), 1)
            self.assertEqual(txs[0].sender, to_addr)
            self.assertEqual(txs[0].recipient, from_addr)

    def test_original_spring_eval_can_use_iot_identity_without_iot_feature_dim(self):
        self.assertEqual(
            eval_state_dim(argparse.Namespace(mdp_mode="spring", tx_identity="iot"), 4),
            state_dim(4),
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            from_addr = "0x" + "1" * 40
            to_addr = "0x" + "2" * 40

            with tx_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                row = [""] * 18
                row[3] = from_addr
                row[4] = to_addr
                row[6] = "0"
                row[7] = "0"
                row[8] = "1"
                writer.writerow(row)

            with sidecar_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "tx_index",
                        "device_label",
                        "device_mac",
                        "srcIp",
                        "dstIp",
                        "dstPort",
                        "protocol",
                        "srcNumPackets",
                        "dstNumPackets",
                        "srcPayloadSize",
                        "dstPayloadSize",
                        "from_address",
                        "to_address",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tx_index": "0",
                        "device_label": "Camera",
                        "device_mac": "aa:bb:cc:dd:ee:ff",
                        "srcIp": "192.168.1.2",
                        "dstIp": "8.8.8.8",
                        "dstPort": "443",
                        "protocol": "tls",
                        "srcNumPackets": "1",
                        "dstNumPackets": "1",
                        "srcPayloadSize": "10",
                        "dstPayloadSize": "20",
                        "from_address": from_addr,
                        "to_address": to_addr,
                    }
                )

            args = argparse.Namespace(
                mdp_mode="spring",
                tx_identity="iot",
                sidecar=str(sidecar_csv),
                max_txs=1,
            )
            txs = load_eval_transactions(args, tx_csv)

            self.assertEqual(len(txs), 1)
            self.assertEqual(txs[0].sender, to_addr)
            self.assertEqual(txs[0].recipient, from_addr)

    def test_random_baseline_policy_is_seeded(self):
        args = argparse.Namespace(policy="random", seed=7, shards=4)
        first_policy = make_policy(args, agent=None)
        second_policy = make_policy(args, agent=None)

        first_actions = [
            first_policy([], f"state-{idx}", "", None).action
            for idx in range(6)
        ]
        second_actions = [
            second_policy([], f"state-{idx}", "", None).action
            for idx in range(6)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertTrue(all(0 <= action < args.shards for action in first_actions))


if __name__ == "__main__":
    unittest.main()
