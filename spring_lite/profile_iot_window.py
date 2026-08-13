import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from config import (
    DEFAULT_BLOCK_INTERVAL_MS,
    DEFAULT_IOT_SIDECAR_PATH,
    DEFAULT_SHARD_NUM,
    DEFAULT_TEST_MAX_TXS,
    DEFAULT_TEST_START_TX,
    LOAD_SEMANTICS_VERSION,
)
from heuristic import addr2shard
from offline_env import iot_anchor_key


def profile_iot_window(
    sidecar_path: Path,
    start_tx: int,
    max_txs: int,
    shards: int,
    block_size: int,
    block_interval_ms: int,
) -> Dict[str, object]:
    if start_tx < 0:
        raise ValueError("start_tx must be non-negative")
    if max_txs <= 0:
        raise ValueError("max_txs must be positive")
    if shards <= 0:
        raise ValueError("shards must be positive")
    if block_size <= 0 or block_interval_ms <= 0:
        raise ValueError("block_size and block_interval_ms must be positive")

    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"IoT sidecar not found: {sidecar_path}")

    owner_shard_loads = [0 for _ in range(shards)]
    owner_counts: Counter = Counter()
    relation_counts: Counter = Counter()
    loaded = 0
    first_time: Optional[str] = None
    last_time: Optional[str] = None

    with sidecar_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_index, row in enumerate(reader):
            raw_index = str(row.get("tx_index", "")).strip()
            if raw_index and int(raw_index) != source_index:
                raise ValueError(
                    "sidecar tx_index mismatch: "
                    f"row={source_index}, tx_index={raw_index}"
                )
            if source_index < start_tx:
                continue
            if loaded >= max_txs:
                break

            owner = iot_anchor_key(row)
            sid = addr2shard(owner, shards)
            owner_shard_loads[sid] += 1
            owner_counts[owner] += 1
            relation = str(row.get("relation_type", "")).strip() or "unknown"
            relation_counts[relation] += 1

            timestamp = str(row.get("time", "")).strip()
            if first_time is None:
                first_time = timestamp
            last_time = timestamp
            loaded += 1

    if loaded != max_txs:
        raise RuntimeError(
            f"window incomplete: loaded={loaded}, requested={max_txs}"
        )

    owner_load_dist = [
        float(value) / float(loaded) if loaded else 0.0
        for value in owner_shard_loads
    ]
    owner_max_load_share = max(owner_load_dist) if owner_load_dist else 0.0
    hottest_owner_shard = (
        max(range(shards), key=lambda sid: owner_shard_loads[sid])
        if owner_shard_loads
        else -1
    )
    per_shard_capacity_tps = (
        float(block_size) * 1000.0 / float(block_interval_ms)
    )
    owner_anchor_tps_ceiling = (
        per_shard_capacity_tps / owner_max_load_share
        if owner_max_load_share > 0
        else 0.0
    )

    return {
        "sidecar": str(sidecar_path.resolve()),
        "start_tx": start_tx,
        "max_txs": max_txs,
        "end_tx_exclusive": start_tx + loaded,
        "first_time": first_time,
        "last_time": last_time,
        "shards": shards,
        "block_size": block_size,
        "block_interval_ms": block_interval_ms,
        "per_shard_capacity_tps": per_shard_capacity_tps,
        "owner_anchor_count": len(owner_counts),
        "owner_shard_loads": owner_shard_loads,
        "owner_shard_load_dist": owner_load_dist,
        "owner_max_load_share": owner_max_load_share,
        "hottest_owner_shard": hottest_owner_shard,
        "owner_anchor_tps_ceiling": owner_anchor_tps_ceiling,
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "load_semantics_version": LOAD_SEMANTICS_VERSION,
        "ceiling_scope": (
            "primary-owner fixed-shard upper bound; sender and relay pressure "
            "can only lower the sustainable TPS"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar",
        default=str(DEFAULT_IOT_SIDECAR_PATH),
    )
    parser.add_argument("--start_tx", type=int, default=DEFAULT_TEST_START_TX)
    parser.add_argument("--max_txs", type=int, default=DEFAULT_TEST_MAX_TXS)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--block_size", type=int, default=1000)
    parser.add_argument(
        "--block_interval_ms",
        type=int,
        default=DEFAULT_BLOCK_INTERVAL_MS,
    )
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = profile_iot_window(
        sidecar_path=Path(args.sidecar),
        start_tx=args.start_tx,
        max_txs=args.max_txs,
        shards=args.shards,
        block_size=args.block_size,
        block_interval_ms=args.block_interval_ms,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
