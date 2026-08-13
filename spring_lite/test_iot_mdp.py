import csv
import math
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
    Tx,
    iot_dense_action_reward,
    iot_feature_vector,
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
        "anchor_types",
        "anchor_weights",
        "anchor_distances",
        "anchor_link_qualities",
        "anchor_count",
        "flow_direction",
        "relation_type",
        "is_control_protocol",
        "is_multicast_or_broadcast",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IoTMDPTest(unittest.TestCase):
    def test_iot_feature_vector_keeps_lite_multi_anchor_scene_features(self):
        row = {
            "protocol": "tls",
            "srcNumPackets": "5",
            "dstNumPackets": "7",
            "srcPayloadSize": "100",
            "dstPayloadSize": "300",
            "flowDuration": "1000",
            "anchor_count": "3",
            "anchor_types": "device;gateway;cloud_endpoint",
            "anchor_weights": "0.2;0.3;0.5",
            "anchor_distances": "10;30;50",
            "anchor_link_qualities": "0.9;0.6;0.3",
            "flow_direction": "outbound",
            "relation_type": "device_to_cloud",
            "is_control_protocol": "0",
            "is_multicast_or_broadcast": "0",
        }

        features = iot_feature_vector(row, prior_frequency=9)

        self.assertEqual(len(features), IOT_FEATURE_DIM)
        self.assertEqual(IOT_FEATURE_DIM, 10)
        self.assertAlmostEqual(features[0], 0.75)
        self.assertAlmostEqual(features[1], 36.0 / 60.0)
        self.assertAlmostEqual(features[2], 10.0 / 60.0)
        self.assertAlmostEqual(features[3], 0.51)
        self.assertAlmostEqual(features[4], math.log1p(400.0 / 1000.0) / math.log1p(10000.0))
        self.assertAlmostEqual(features[5], 0.2)
        self.assertAlmostEqual(features[6], 0.3)
        self.assertAlmostEqual(features[7], 0.5)
        self.assertAlmostEqual(features[8], 0.0)
        self.assertEqual(features[9], 0.0)

    def test_iot_state_appends_current_load_before_lite_features(self):
        env = SpringOfflineEnv(shards=4, iot_feature_dim=IOT_FEATURE_DIM, reward_mode="iot")
        env.shard_load = [1, 2, 1, 0]
        iot_features = [float(idx + 1) / 10.0 for idx in range(IOT_FEATURE_DIM)]

        state = env.build_state_from_sender_pos(
            sender_pos=[0.25, 0.25, 0.25, 0.25],
            flag=1.0,
            iot_features=iot_features,
        )

        current_load_offset = 11 * 4 + 1
        self.assertEqual(len(state), state_dim(4, IOT_FEATURE_DIM))
        self.assertEqual(state[current_load_offset : current_load_offset + 4], [0.25, 0.5, 0.25, 0.0])
        self.assertEqual(state[-IOT_FEATURE_DIM:], iot_features)

    def test_iot_dense_action_reward_prefers_anchor_mass_and_load_balance(self):
        sender_pos = [0.05, 0.80, 0.10, 0.05]

        good, _ = iot_dense_action_reward(
            chosen_shard=1,
            sender_pos=sender_pos,
            communication_cost_weight=0.4,
            shard_load_before=2,
            load_mean_before=2.0,
        )
        bad, _ = iot_dense_action_reward(
            chosen_shard=3,
            sender_pos=sender_pos,
            communication_cost_weight=0.4,
            shard_load_before=2,
            load_mean_before=2.0,
        )
        overloaded, _ = iot_dense_action_reward(
            chosen_shard=1,
            sender_pos=sender_pos,
            communication_cost_weight=0.4,
            shard_load_before=8,
            load_mean_before=2.0,
        )

        self.assertGreater(good, bad)
        self.assertLess(overloaded, good)

    def test_iot_dense_balanced_low_load_bonus_is_bounded(self):
        sender_pos = [0.0, 0.50, 0.50, 0.0]

        low_load, _ = iot_dense_action_reward(
            chosen_shard=1,
            sender_pos=sender_pos,
            communication_cost_weight=0.0,
            shard_load_before=0,
            load_mean_before=4.0,
            low_load_bonus_weight=0.25,
            low_load_bonus_cap=0.18,
        )
        average_load, _ = iot_dense_action_reward(
            chosen_shard=1,
            sender_pos=sender_pos,
            communication_cost_weight=0.0,
            shard_load_before=4,
            load_mean_before=4.0,
            low_load_bonus_weight=0.25,
            low_load_bonus_cap=0.18,
        )
        empty_anchor_low_load, _ = iot_dense_action_reward(
            chosen_shard=0,
            sender_pos=sender_pos,
            communication_cost_weight=0.0,
            shard_load_before=0,
            load_mean_before=4.0,
            low_load_bonus_weight=0.25,
            low_load_bonus_cap=0.18,
        )

        self.assertGreater(low_load, average_load)
        self.assertLessEqual(low_load - average_load, 0.30)
        self.assertLess(empty_anchor_low_load, average_load)

    def test_iot_dense_balanced_uses_same_global_iot_reward(self):
        iot_reward, _ = spring_reward(
            effective_loads=[1.0, 2.0, 1.0, 2.0],
            cross_loads=[0.2, 0.2, 0.1, 0.1],
            reward_mode="iot",
            communication_cost=0.2,
        )
        balanced_reward, parts = spring_reward(
            effective_loads=[1.0, 2.0, 1.0, 2.0],
            cross_loads=[0.2, 0.2, 0.1, 0.1],
            reward_mode="iot_dense_balanced",
            communication_cost=0.2,
        )

        self.assertAlmostEqual(balanced_reward, iot_reward)
        self.assertEqual(parts["reward_mode"], "iot_dense_balanced")

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

    def test_iot_batch_loads_separate_primary_execution_from_multi_anchor_relation(self):
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

        # The state object is placed with the primary owner, so the actual
        # BlockEmulator transaction is inner-shard. The peer anchor remains a
        # weighted communication diagnostic and contributes 0.5 relation cross.
        self.assertAlmostEqual(metrics.cross_rate, 0.0)
        self.assertAlmostEqual(metrics.cross_tx, 0.0)
        self.assertAlmostEqual(metrics.relation_cross_rate, 0.5)
        self.assertAlmostEqual(sum(metrics.owner_loads), 1.0)
        self.assertEqual(metrics.owner_loads[primary_shard], 1)
        self.assertEqual(metrics.loads[primary_shard], 1)
        self.assertAlmostEqual(metrics.owner_anchor_tps_ceiling, 200.0)
        self.assertAlmostEqual(metrics.system_stage_tps_ceiling, 200.0)

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

    def test_iot_cross_transaction_uses_full_stage_capacity_on_both_shards(self):
        shards = 2
        primary_anchor = "0x" + "1" * 40
        state_object = "0x" + "2" * 40
        primary_shard = addr2shard(primary_anchor, shards)
        sender_shard = 1 - primary_shard
        tx = Tx(
            sender=state_object,
            recipient=primary_anchor,
            related_addresses=(primary_anchor,),
            related_weights=(1.0,),
            iot_features=tuple(0.0 for _ in range(IOT_FEATURE_DIM)),
            communication_cost_weight=0.4,
        )
        env = SpringOfflineEnv(
            shards=shards,
            iot_feature_dim=IOT_FEATURE_DIM,
            reward_mode="iot",
        )

        _actions, metrics = env.run_batch(
            [tx],
            lambda _state, _address, _related, _info: PolicyOutput(
                action=sender_shard
            ),
        )

        self.assertAlmostEqual(metrics.cross_rate, 1.0)
        self.assertEqual(metrics.loads, [1, 1])
        self.assertAlmostEqual(metrics.effective_loads[primary_shard], 0.5)
        self.assertAlmostEqual(metrics.effective_loads[sender_shard], 0.5)
        self.assertAlmostEqual(metrics.communication_cost, 0.4)
        self.assertAlmostEqual(metrics.stage_max_load_per_tx, 1.0)

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

    def test_candidate_filter_capacity_guard_blocks_overloaded_related_shard(self):
        env = SpringOfflineEnv(
            shards=4,
            iot_feature_dim=IOT_FEATURE_DIM,
            reward_mode="iot",
            candidate_top_k=2,
            capacity_guard=1,
            capacity_guard_factor=1.2,
        )
        anchor = "0x" + "a" * 40
        env.addr_shard[anchor] = 0
        env.shard_load = [30, 1, 1, 1]
        tx = Tx(
            sender="0x" + "b" * 40,
            recipient=anchor,
            related_addresses=(anchor,),
            related_weights=(1.0,),
            iot_features=tuple(0.0 for _ in range(IOT_FEATURE_DIM)),
        )
        observed_masks = []

        def overloaded_policy(_state, _address, _related, info):
            observed_masks.append(list(info.action_mask))
            return PolicyOutput(action=0, source="test_policy")

        actions, _metrics = env.run_batch([tx], overloaded_policy)

        self.assertEqual(len(actions), 1)
        self.assertEqual(observed_masks[0][0], 0)
        self.assertEqual(sum(observed_masks[0]), 2)
        self.assertNotEqual(actions[0].action, 0)
        self.assertTrue(actions[0].source.endswith("capacity_guard"))


if __name__ == "__main__":
    unittest.main()
