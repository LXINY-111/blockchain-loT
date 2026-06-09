import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IOT_FEATURE_DIM, state_dim  # noqa: E402
from heuristic import addr2shard, heuristic_from_state  # noqa: E402
from offline_env import (  # noqa: E402
    PolicyOutput,
    SpringOfflineEnv,
    iot_state_object_key,
    load_iot_transactions,
    spring_reward,
)


def write_tx_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for from_addr, to_addr, value in rows:
            row = [""] * 18
            row[3] = from_addr
            row[4] = to_addr
            row[6] = "0"
            row[7] = "0"
            row[8] = str(value)
            writer.writerow(row)


def write_sidecar_csv(path: Path, rows):
    fieldnames = [
        "tx_index",
        "time",
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
        "flowDuration",
        "from_address",
        "to_address",
        "mapped_mote_id",
        "mapped_x",
        "mapped_y",
        "related_mote_id",
        "related_x",
        "related_y",
        "distance",
        "link_quality",
        "tx_batch_id",
        "anchor_addresses",
        "anchor_weights",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IoTMDPTest(unittest.TestCase):
    def test_iot_state_object_key_groups_same_device_peer_and_protocol(self):
        first = {
            "device_label": "AugustDoorBell",
            "device_mac": "e0:76:d0:3f:00:ae",
            "srcIp": "192.168.1.216",
            "dstIp": "54.249.82.177",
            "protocol": "tls",
        }
        second = dict(first)
        second["srcPayloadSize"] = "900"

        self.assertEqual(iot_state_object_key(first), iot_state_object_key(second))

        changed_protocol = dict(first)
        changed_protocol["protocol"] = "http"
        self.assertNotEqual(iot_state_object_key(first), iot_state_object_key(changed_protocol))

    def test_iot_state_object_key_prefers_sidecar_to_address(self):
        row = {
            "device_label": "AugustDoorBell",
            "device_mac": "e0:76:d0:3f:00:ae",
            "dstIp": "54.249.82.177",
            "dstPort": "443",
            "protocol": "tls",
            "state_object_key": "iot_state:readable-label",
            "to_address": "0x" + "a" * 40,
        }

        self.assertEqual(iot_state_object_key(row), "0x" + "a" * 40)

    def test_load_iot_transactions_uses_state_object_instead_of_each_flow(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            from_addr = "0x" + "1" * 40
            to_addr = "0x" + "2" * 40
            write_tx_csv(tx_csv, [(from_addr, to_addr, 100), (from_addr, to_addr, 120)])
            write_sidecar_csv(
                sidecar_csv,
                [
                    {
                        "tx_index": "0",
                        "time": "2017-01-01 00:00:00",
                        "device_label": "AugustDoorBell",
                        "device_mac": "e0:76:d0:3f:00:ae",
                        "srcIp": "192.168.1.216",
                        "dstIp": "54.249.82.177",
                        "srcPort": "50000",
                        "dstPort": "443",
                        "protocol": "tls",
                        "srcNumPackets": "5",
                        "dstNumPackets": "5",
                        "srcPayloadSize": "100",
                        "dstPayloadSize": "200",
                        "flowDuration": "10",
                        "from_address": from_addr,
                        "to_address": to_addr,
                        "mapped_mote_id": "1",
                        "mapped_x": "0",
                        "mapped_y": "0",
                        "related_mote_id": "2",
                        "related_x": "30",
                        "related_y": "40",
                        "distance": "50",
                        "link_quality": "0.2",
                        "tx_batch_id": "0",
                    },
                    {
                        "tx_index": "1",
                        "time": "2017-01-01 00:00:01",
                        "device_label": "AugustDoorBell",
                        "device_mac": "e0:76:d0:3f:00:ae",
                        "srcIp": "192.168.1.216",
                        "dstIp": "54.249.82.177",
                        "srcPort": "50001",
                        "dstPort": "443",
                        "protocol": "tls",
                        "srcNumPackets": "5",
                        "dstNumPackets": "5",
                        "srcPayloadSize": "120",
                        "dstPayloadSize": "220",
                        "flowDuration": "11",
                        "from_address": from_addr,
                        "to_address": to_addr,
                        "mapped_mote_id": "1",
                        "mapped_x": "0",
                        "mapped_y": "0",
                        "related_mote_id": "2",
                        "related_x": "30",
                        "related_y": "40",
                        "distance": "50",
                        "link_quality": "0.2",
                        "tx_batch_id": "0",
                    },
                ],
            )

            txs = load_iot_transactions(tx_csv, sidecar_csv)

            self.assertEqual(len(txs), 2)
            self.assertEqual(txs[0].sender, txs[1].sender)
            self.assertEqual(txs[0].sender, to_addr)
            self.assertEqual(txs[0].recipient, from_addr)
            self.assertEqual(len(txs[0].iot_features), IOT_FEATURE_DIM)
            self.assertGreater(txs[0].communication_cost_weight, 0.0)

    def test_load_iot_transactions_preserves_multi_anchor_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            primary_anchor = "0x" + "1" * 40
            peer_anchor = "0x" + "2" * 40
            state_object = "0x" + "3" * 40
            write_tx_csv(tx_csv, [(primary_anchor, state_object, 100)])
            write_sidecar_csv(
                sidecar_csv,
                [
                    {
                        "tx_index": "0",
                        "time": "2017-01-01 00:00:00",
                        "device_label": "AugustDoorBell",
                        "device_mac": "e0:76:d0:3f:00:ae",
                        "srcIp": "192.168.1.216",
                        "dstIp": "192.168.1.1",
                        "srcPort": "50000",
                        "dstPort": "53",
                        "protocol": "dns",
                        "srcNumPackets": "1",
                        "dstNumPackets": "1",
                        "srcPayloadSize": "37",
                        "dstPayloadSize": "242",
                        "flowDuration": "285",
                        "from_address": primary_anchor,
                        "to_address": state_object,
                        "mapped_mote_id": "1",
                        "mapped_x": "0",
                        "mapped_y": "0",
                        "related_mote_id": "2",
                        "related_x": "30",
                        "related_y": "40",
                        "distance": "50",
                        "link_quality": "0.2",
                        "tx_batch_id": "0",
                        "anchor_addresses": f"{primary_anchor};{peer_anchor};{primary_anchor}",
                        "anchor_weights": "0.5;0.5;0.5",
                    },
                ],
            )

            txs = load_iot_transactions(tx_csv, sidecar_csv)

            self.assertEqual(txs[0].recipient, primary_anchor)
            self.assertEqual(txs[0].related_addresses, (primary_anchor, peer_anchor))
            self.assertEqual(txs[0].related_weights, (0.5, 0.5))

    def test_iot_env_builds_sender_pos_from_all_multi_anchors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            primary_anchor = "0x" + "1" * 40
            peer_anchor = "0x" + "2" * 40
            state_object = "0x" + "3" * 40
            write_tx_csv(tx_csv, [(primary_anchor, state_object, 100)])
            write_sidecar_csv(
                sidecar_csv,
                [
                    {
                        "tx_index": "0",
                        "time": "2017-01-01 00:00:00",
                        "device_label": "AugustDoorBell",
                        "device_mac": "e0:76:d0:3f:00:ae",
                        "srcIp": "192.168.1.216",
                        "dstIp": "192.168.1.1",
                        "srcPort": "50000",
                        "dstPort": "53",
                        "protocol": "dns",
                        "srcNumPackets": "1",
                        "dstNumPackets": "1",
                        "srcPayloadSize": "37",
                        "dstPayloadSize": "242",
                        "flowDuration": "285",
                        "from_address": primary_anchor,
                        "to_address": state_object,
                        "mapped_mote_id": "1",
                        "mapped_x": "0",
                        "mapped_y": "0",
                        "related_mote_id": "2",
                        "related_x": "30",
                        "related_y": "40",
                        "distance": "50",
                        "link_quality": "0.2",
                        "tx_batch_id": "0",
                        "anchor_addresses": f"{primary_anchor};{peer_anchor}",
                        "anchor_weights": "0.5;0.5",
                    },
                ],
            )
            txs = load_iot_transactions(tx_csv, sidecar_csv)

        env = SpringOfflineEnv(shards=4, iot_feature_dim=IOT_FEATURE_DIM, reward_mode="iot")
        expected_shards = {
            addr2shard(primary_anchor, 4),
            addr2shard(peer_anchor, 4),
        }

        def inspect_policy(state, _address, _related, info):
            self.assertEqual(info.related_count, 2)
            observed_shards = {idx for idx, value in enumerate(info.sender_pos) if value > 0.0}
            self.assertEqual(observed_shards, expected_shards)
            return PolicyOutput(action=0)

        actions, metrics = env.run_batch(txs, inspect_policy)

        self.assertEqual(len(actions), 1)
        self.assertEqual(metrics.related_known_count, 1)

    def test_iot_batch_loads_use_weighted_multi_anchor_cross_rate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            primary_anchor = "0x" + "1" * 40
            peer_anchor = "0x" + "2" * 40
            state_object = "0x" + "3" * 40
            primary_shard = addr2shard(primary_anchor, 4)
            peer_shard = addr2shard(peer_anchor, 4)
            self.assertNotEqual(primary_shard, peer_shard)
            write_tx_csv(tx_csv, [(primary_anchor, state_object, 100)])
            write_sidecar_csv(
                sidecar_csv,
                [
                    {
                        "tx_index": "0",
                        "time": "2017-01-01 00:00:00",
                        "device_label": "AugustDoorBell",
                        "device_mac": "e0:76:d0:3f:00:ae",
                        "srcIp": "192.168.1.216",
                        "dstIp": "192.168.1.1",
                        "srcPort": "50000",
                        "dstPort": "53",
                        "protocol": "dns",
                        "srcNumPackets": "1",
                        "dstNumPackets": "1",
                        "srcPayloadSize": "37",
                        "dstPayloadSize": "242",
                        "flowDuration": "285",
                        "from_address": primary_anchor,
                        "to_address": state_object,
                        "mapped_mote_id": "1",
                        "mapped_x": "0",
                        "mapped_y": "0",
                        "related_mote_id": "2",
                        "related_x": "30",
                        "related_y": "40",
                        "distance": "50",
                        "link_quality": "0.2",
                        "tx_batch_id": "0",
                        "anchor_addresses": f"{primary_anchor};{peer_anchor}",
                        "anchor_weights": "0.5;0.5",
                    },
                ],
            )
            txs = load_iot_transactions(tx_csv, sidecar_csv)

        env = SpringOfflineEnv(shards=4, iot_feature_dim=IOT_FEATURE_DIM, reward_mode="iot")

        def primary_policy(_state, _address, _related, _info):
            return PolicyOutput(action=primary_shard)

        _actions, metrics = env.run_batch(txs, primary_policy)

        self.assertAlmostEqual(metrics.cross_rate, 0.5)
        self.assertAlmostEqual(metrics.cross_tx, 0.5)

    def test_heuristic_uses_fractional_multi_anchor_sender_pos(self):
        shards = 4
        state = [0.0 for _ in range(11 * shards + 1)]
        for window in range(5):
            state[window * shards + 0] = 1.0
        sender_pos_offset = 10 * shards
        state[sender_pos_offset + 0] = 0.34
        state[sender_pos_offset + 1] = 0.33
        state[sender_pos_offset + 2] = 0.33

        self.assertEqual(heuristic_from_state(state, shards), 0)

    def test_iot_env_places_repeated_state_object_once_and_appends_features(self):
        txs = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tx_csv = root / "tx.csv"
            sidecar_csv = root / "sidecar.csv"
            from_addr = "0x" + "1" * 40
            to_addr = "0x" + "2" * 40
            write_tx_csv(tx_csv, [(from_addr, to_addr, 100), (from_addr, to_addr, 120)])
            common = {
                "device_label": "AugustDoorBell",
                "device_mac": "e0:76:d0:3f:00:ae",
                "srcIp": "192.168.1.216",
                "dstIp": "54.249.82.177",
                "srcPort": "50000",
                "dstPort": "443",
                "protocol": "tls",
                "srcNumPackets": "5",
                "dstNumPackets": "5",
                "srcPayloadSize": "100",
                "dstPayloadSize": "200",
                "flowDuration": "10",
                "from_address": from_addr,
                "to_address": to_addr,
                "mapped_mote_id": "1",
                "mapped_x": "0",
                "mapped_y": "0",
                "related_mote_id": "2",
                "related_x": "30",
                "related_y": "40",
                "distance": "50",
                "link_quality": "0.2",
                "tx_batch_id": "0",
            }
            rows = [dict(common, tx_index="0"), dict(common, tx_index="1")]
            write_sidecar_csv(sidecar_csv, rows)
            txs = load_iot_transactions(tx_csv, sidecar_csv)

        env = SpringOfflineEnv(shards=2, iot_feature_dim=IOT_FEATURE_DIM, reward_mode="iot")
        peer_shard = addr2shard(txs[0].recipient, 2)

        def cross_policy(state, _address, _related, _info):
            self.assertEqual(len(state), state_dim(2, IOT_FEATURE_DIM))
            return PolicyOutput(action=1 - peer_shard)

        actions, metrics = env.run_batch(txs, cross_policy)

        self.assertEqual(len(actions), 1)
        self.assertEqual(metrics.action_count, 1)
        self.assertGreater(metrics.communication_cost, 0.0)

    def test_iot_reward_penalizes_communication_cost(self):
        low_cost, _ = spring_reward(
            effective_loads=[1.0, 1.0],
            cross_loads=[0.5, 0.5],
            reward_mode="iot",
            communication_cost=0.0,
        )
        high_cost, _ = spring_reward(
            effective_loads=[1.0, 1.0],
            cross_loads=[0.5, 0.5],
            reward_mode="iot",
            communication_cost=0.8,
        )

        self.assertLess(high_cost, low_cost)


if __name__ == "__main__":
    unittest.main()
