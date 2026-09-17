"""IoT 评估口径：只测量，不改变状态、奖励、模型或策略。

每笔原始交易计一次。成本和字节加权比例使用全窗口的分子/分母之比，
不能平均批次比例，也不能把零分母或缺失链上时延伪装成零。
"""
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from iot_v2 import load_rows, scene_for_legacy_features
from offline_env import iot_communication_cost_weight
from mechanism import load_statistics

METRIC_VERSION = "iot_evaluation_v1"
HASH_COLUMN = "TxHash (Byte -> Big Int)"
META_COLUMNS = ("Tx local nonce", "Dataset row index", "Sender address",
                "Recipient address", "Sender shard", "Recipient shard", "Cross shard")


def address_key(address):
    value = str(address).strip().lower()
    return value[2:] if value.startswith("0x") else value


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def nearest_rank(values, probability):
    """最近秩分位数；与已有链上 P95 口径一致，不插值。"""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))]


@dataclass(frozen=True)
class Scene:
    dataset_row: int
    tx_index: int
    sender: str
    recipient: str
    cost: float
    payload: float
    distance: float
    quality: float
    control: bool

    def __post_init__(self):
        if (not all(math.isfinite(v) and v >= 0 for v in (self.cost, self.payload, self.distance, self.quality))
                or self.cost > 1 or self.quality > 1 or self.dataset_row < 0 or self.tx_index < 0):
            raise ValueError("Invalid derived IoT scene values")


def load_scenes(chain, sidecar, start, count):
    if start < 0 or count <= 0:
        raise ValueError("IoT evaluation requires a positive explicit transaction window")
    scenes = []
    for local_index, (_, row) in enumerate(load_rows(Path(chain), Path(sidecar), start, count)):
        # 复用训练代码的正向场景成本公式，不另写一个近似公式。
        legacy = scene_for_legacy_features(row)
        scenes.append(Scene(start + local_index, int(row["tx_index"]),
            address_key(row["from_address"]), address_key(row["to_address"]),
            iot_communication_cost_weight(legacy),
            float(row["srcPayloadSize"]) + float(row["dstPayloadSize"]),
            float(row["distance_m"]), float(row["link_quality_forward"]),
            str(row["is_control_protocol"]).strip() == "1"))
    return scenes


def evaluation_context(chain, sidecar, start, count, reference_start=200000, reference_count=50000):
    # 分组阈值只能来自测试窗口之前的参考段，避免策略结果或测试分布参与选阈值。
    if reference_start < 0 or reference_count <= 0 or reference_start + reference_count > start:
        raise ValueError("IoT reference window must be positive and entirely before the evaluated window")
    reference = load_scenes(chain, sidecar, reference_start, reference_count)
    thresholds = {
        "high_cost_ge": nearest_rank([s.cost for s in reference], 0.8),
        "high_payload_ge": nearest_rank([s.payload for s in reference], 0.8),
        "long_distance_ge": nearest_rank([s.distance for s in reference], 0.8),
        "weak_link_le": nearest_rank([s.quality for s in reference], 0.2),
    }
    with Path(sidecar).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "version": METRIC_VERSION,
        "dataset_window": {"start_tx": start, "end_tx_exclusive": start + count},
        "reference_window": {"start_tx": reference_start,
                             "end_tx_exclusive": reference_start + reference_count},
        "thresholds": thresholds,
        "quantile_rule": "nearest_rank; inclusive thresholds; ties may exceed 20%; groups overlap",
        "scene_sha256": digest.hexdigest(),
        "cost_rule": "clip(d/60,0,1)*(1-q_forward)*max(0.05,clip(log1p(bytes)/log1p(1000000),0,1))",
        "cost_scope": "forward scene cost, matching existing reward bookkeeping; not measured energy or network bytes",
        "duration_scope": "flowDuration has unresolved units and is not used as latency or bytes/second",
    }
    return load_scenes(chain, sidecar, start, count), manifest


@dataclass
class Totals:
    count: int = 0
    cross: int = 0
    cost: float = 0.0
    cross_cost: float = 0.0
    payload: float = 0.0
    cross_payload: float = 0.0
    latencies: list = field(default_factory=list)

    def add(self, scene, cross, latency):
        self.count += 1
        self.cross += int(cross)
        self.cost += scene.cost
        self.cross_cost += scene.cost * int(cross)
        self.payload += scene.payload
        self.cross_payload += scene.payload * int(cross)
        if latency is not None:
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Invalid confirmed latency")
            self.latencies.append(latency)

    def summary(self):
        return dict(count=self.count, cross_count=self.cross,
            cross_ratio=ratio(self.cross, self.count),
            scene_cost_sum=self.cost, cross_cost_sum=self.cross_cost,
            scene_cost_mean=ratio(self.cost, self.count),
            cost_weighted_cross_ratio=ratio(self.cross_cost, self.cost),
            cross_cost_per_tx=ratio(self.cross_cost, self.count),
            payload_bytes_sum=self.payload, cross_payload_bytes_sum=self.cross_payload,
            payload_weighted_cross_ratio=ratio(self.cross_payload, self.payload),
            latency_count=len(self.latencies),
            mean_latency_sec=ratio(sum(self.latencies), len(self.latencies)),
            p95_latency_sec=nearest_rank(self.latencies, 0.95))


class IoTMetrics:
    def __init__(self, manifest, latency_scope):
        self.manifest = manifest
        self.latency_scope = latency_scope
        self.seen = set()
        self.groups = {name: Totals() for name in ("all", "high_cost", "high_payload",
            "long_distance", "weak_link", "control_protocol", "other_protocol")}

    def add(self, scene, cross, latency=None):
        if scene.dataset_row in self.seen:
            raise ValueError(f"Duplicate scene contribution: {scene.dataset_row}")
        self.seen.add(scene.dataset_row)
        if cross not in (0, 1, False, True):
            raise ValueError("Cross-shard indicator must be 0 or 1")
        t = self.manifest["thresholds"]
        selected = ["all", "control_protocol" if scene.control else "other_protocol"]
        for name, match in (("high_cost", scene.cost >= t["high_cost_ge"]),
                ("high_payload", scene.payload >= t["high_payload_ge"]),
                ("long_distance", scene.distance >= t["long_distance_ge"]),
                ("weak_link", scene.quality <= t["weak_link_le"])):
            if match:
                selected.append(name)
        for name in selected:
            self.groups[name].add(scene, cross, latency)

    def summary(self):
        window = self.manifest["dataset_window"]
        if self.seen != set(range(window["start_tx"], window["end_tx_exclusive"])):
            raise ValueError("IoT metrics require complete, unique coverage of the requested window")
        return dict(self.manifest, status="complete", latency_scope=self.latency_scope,
                    groups={name: value.summary() for name, value in self.groups.items()})


def _unique_by_hash(rows, label):
    result = {}
    for row in rows:
        value = row.get(HASH_COLUMN, "")
        if not value or not value.isdigit():
            raise ValueError(f"Missing/invalid transaction hash in {label}")
        key = str(int(value))
        if key in result:
            raise ValueError(f"Duplicate transaction hash in {label}")
        result[key] = row
    return result


def analyze_chain_details(detail_rows, scenes, manifest, shards, metadata_rows=None):
    """以运行内哈希连接元数据，再按文件行号和双方地址核对场景。

    旧 CSV 必须提供从原数据库读取的元数据；禁止按 CSV 行序或时间猜配。
    不完整、重复、地址错配、阶段/分片矛盾均报错，不输出貌似完整的比例。
    """
    details = _unique_by_hash(detail_rows, "Tx_Details")
    metadata = details if metadata_rows is None else _unique_by_hash(metadata_rows, "metadata")
    if set(details) != set(metadata) or len(details) != len(scenes):
        raise ValueError("Transaction detail/metadata/window coverage mismatch")
    by_row = {s.dataset_row: s for s in scenes}
    metric = IoTMetrics(manifest, "measured_chain_confirmation_seconds")
    start = manifest["dataset_window"]["start_tx"]
    stage_loads = [0] * shards
    effective_loads = [0.] * shards
    for key, detail in details.items():
        meta = metadata[key]
        if any(str(meta.get(name, "")) == "" for name in META_COLUMNS):
            raise ValueError("Missing transaction metadata; legacy results require --iot_tx_metadata")
        if metadata_rows is not None and all(detail.get(name, "") != "" for name in META_COLUMNS):
            for name in META_COLUMNS:
                normalizer = address_key if name.endswith("address") else int
                if normalizer(detail[name]) != normalizer(meta[name]):
                    raise ValueError("Recovered metadata disagrees with measured transaction metadata")
        index = int(meta["Dataset row index"])
        scene = by_row.get(index)
        if scene is None or int(meta["Tx local nonce"]) != index - start:
            raise ValueError("Transaction window/nonce mismatch")
        if (address_key(meta["Sender address"]), address_key(meta["Recipient address"])) != (scene.sender, scene.recipient):
            raise ValueError(f"Transaction address/scene mismatch at dataset row {index}")
        sender, recipient = int(meta["Sender shard"]), int(meta["Recipient shard"])
        if not (0 <= sender < shards and 0 <= recipient < shards):
            raise ValueError("Shard index outside configured range")
        cross = int(meta["Cross shard"])
        if cross not in (0, 1) or cross != int(sender != recipient):
            raise ValueError("Cross indicator disagrees with committed shard IDs")
        relay1 = detail.get("Relay1 Tx commit timestamp (not a relay tx -> nil)", "")
        relay2 = detail.get("Relay2 Tx commit timestamp (not a relay tx -> nil)", "")
        if bool(relay1) != bool(cross) or bool(relay2) != bool(cross):
            raise ValueError("Relay completion disagrees with committed shard IDs")
        for name in ("Broker1 Tx commit timestamp (not a broker tx -> nil)",
                     "Broker2 Tx commit timestamp (not a broker tx -> nil)"):
            if detail.get(name):
                raise ValueError("IoT v2 measurement currently supports original/relay transactions, not broker splits")
        proposed = float(detail.get("Tx propose timestamp") or "nan")
        committed = float(detail.get("Tx finally commit timestamp") or "nan")
        latency_ms = float(detail.get("Confirmed latency of this tx (ms)") or "nan")
        if not all(math.isfinite(v) for v in (proposed, committed, latency_ms)) or proposed <= 0 or committed < proposed or latency_ms < 0:
            raise ValueError("Missing/invalid final confirmation timestamp or latency")
        # UnixMilli 相减和持续时间截断最多相差 1 毫秒。
        if abs((committed - proposed) - latency_ms) > 1.01:
            raise ValueError("Confirmation latency disagrees with timestamps")
        if cross:
            r1, r2 = float(relay1), float(relay2)
            if not (math.isfinite(r1) and math.isfinite(r2) and proposed <= r1 <= r2 and r2 == committed):
                raise ValueError("Invalid relay-stage timestamps")
        metric.add(scene, cross, latency_ms / 1000.0)
        stage_loads[sender] += 1
        if cross:
            stage_loads[recipient] += 1
            effective_loads[sender] += .5
            effective_loads[recipient] += .5
        else:
            effective_loads[sender] += 1
    result = metric.summary()
    # 按完整已确认交易队列还原实际执行阶段，不用异步 epoch 的局部数量替代。
    result['execution_load'] = dict(load_statistics(stage_loads),
        amplification=ratio(sum(stage_loads), len(scenes)),
        scope='complete_confirmed_transaction_window; intra=1, cross=2')
    result['effective_load'] = load_statistics(effective_loads)
    result["join"] = dict(method="run_tx_hash_then_dataset_row_and_both_addresses",
        metadata_source="archived_chain_database" if metadata_rows is not None else "Tx_Details",
        expected_count=len(scenes), matched_count=len(details), coverage_ratio=1.0)
    result["metadata_scope"] = (
        "original archive files read-only; exporter uses system-temporary database snapshots"
        if metadata_rows is not None else "captured from committed block stages in Tx_Details")
    return result


def read_metadata(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def chain_context(args):
    run_dir = Path(args.iot_run_dir)
    config = json.loads((run_dir / "paramsConfig.json").read_text(encoding="utf-8-sig"))
    if config.get("SpringIOTIdentityMode") != 2 or config.get("SpringIOTMode") != 1:
        raise ValueError("IoT metrics require the real-account v2 dataset mode")
    directory = getattr(args, "iot_dataset_dir", "")
    chain = Path(directory) / "selectedTxs_iot_v2.csv" if directory else Path(config["DatasetFile"])
    sidecar = Path(directory) / "transaction_scene.csv" if directory else Path(config["SpringIOTSidecarFile"])
    return evaluation_context(chain, sidecar, int(config["DatasetStartTx"]),
        int(config["TotalDataSize"]), args.iot_reference_start_tx, args.iot_reference_max_txs)
