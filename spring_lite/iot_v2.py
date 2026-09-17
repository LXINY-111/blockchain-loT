"""真实账户场景的字段适配；不更改数据文件、奖励权重或策略网络。"""
import csv
import math
from collections import defaultdict
from pathlib import Path

from config import ROOT_DIR, LOAD_SEMANTICS_VERSION
from prepare_iot_v2_dataset import check_chain_row, read_dataset_metadata

V2_IDENTITY = "iot_v2"
V2_LOAD_SEMANTICS = "real_accounts_execution_v2_1"
V2_DIRECTORY = ROOT_DIR / "data_iot_v2"


def resolved_identity(args):
    identity = str(getattr(args, "tx_identity", "auto"))
    if identity == V2_IDENTITY:
        return V2_IDENTITY
    return "iot" if identity == "iot" or (
        identity == "auto" and getattr(args, "mdp_mode", "spring") == "iot") else "raw"


def load_semantics(args):
    return V2_LOAD_SEMANTICS if resolved_identity(args) == V2_IDENTITY else LOAD_SEMANTICS_VERSION


def scene_for_legacy_features(row, reverse=False):
    """复用原 10 维特征公式，但端点类型与有向质量来自真实账户附表。

    flowDuration 保留原始数值；第 5 维沿用原归一化强度指标，不能解释为
    已验证单位的字节/秒。零时长沿用原有负载量回退，避免除零。
    """
    result = dict(row)
    result.update(anchor_count="1", anchor_weights="1", distance=row["distance_m"],
                  link_quality=row["link_quality_reverse" if reverse else "link_quality_forward"],
                  anchor_types=row["from_actor_type" if reverse else "to_actor_type"])
    for name in ("distance_m", "link_quality_forward", "link_quality_reverse", "flowDuration",
                 "srcNumPackets", "dstNumPackets", "srcPayloadSize", "dstPayloadSize"):
        value = float(row[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid v2 field {name}: {row[name]!r}")
        if name.startswith("link_quality") and value > 1:
            raise ValueError(f"Invalid reception probability: {name}")
    if row["from_actor_type"] != "device" or row["to_actor_type"] != "device":
        raise ValueError("This v2 contract requires logical device accounts")
    return result


def load_rows(chain_path, scene_path, start_tx=0, max_txs=0):
    """物理交易行与场景行严格配对；窗口局部序号独立于分块的全局序号。"""
    if start_tx < 0 or max_txs < 0:
        raise ValueError("Negative transaction window")
    summary, _ = read_dataset_metadata(Path(scene_path).parent)
    offset = int(summary["index_offset"])
    if offset < 0:
        raise ValueError("Negative global transaction offset")
    loaded = 0
    with Path(chain_path).open(encoding="utf-8-sig", newline="") as chain, Path(
        scene_path).open(encoding="utf-8-sig", newline="") as scene:
        companion = csv.DictReader(scene)
        for index, raw in enumerate(csv.reader(chain)):
            a, b = check_chain_row(raw, index + 1)
            row = next(companion, None)
            if row is None or int(row["tx_index"]) != offset + index or int(row["source_row"]) != index + 1:
                raise ValueError(f"Missing/misaligned v2 scene at row {index + 1}")
            if (row["from_address"], row["to_address"]) != (a, b):
                raise ValueError(f"V2 transaction identity mismatch at row {index + 1}")
            if index < start_tx:
                continue
            yield raw, row
            loaded += 1
            if max_txs and loaded >= max_txs:
                return
        if next(companion, None) is not None:
            raise ValueError("Extra v2 scene rows")
    if max_txs and loaded != max_txs:
        raise ValueError(f"Incomplete v2 window: loaded={loaded}, requested={max_txs}")


def batch_account_scenes(txs):
    """仅聚合当前可见批次的双向通信；不读取未来账户表的统计。

    维度仍为：关联数、平均/最小距离、平均有向质量、平均强度、四类占比、
    控制通信占比。均值按交易次数计权，关联数按不同通信对端计数。
    """
    records = defaultdict(list)
    peers = defaultdict(set)
    for tx in txs:
        for address, peer, features, cost in (
            (tx.sender, tx.recipient, tx.iot_features, tx.communication_cost_weight),
            (tx.recipient, tx.sender, tx.recipient_iot_features, tx.recipient_communication_cost_weight),
        ):
            records[address].append((features, cost))
            peers[address].add(peer)
    result = {}
    for address, values in records.items():
        features = [sum(v[0][i] for v in values) / len(values) for i in range(10)]
        features[0] = min(1.0, len(peers[address]) / 4.0)
        features[2] = min(v[0][2] for v in values)
        result[address] = (features, sum(v[1] for v in values) / len(values))
    return result
