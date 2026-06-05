import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def as_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_csv_from_zip(zip_path: Path, name: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(name).decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(data)))


def keyed(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(row.get("EpochID", "")): row for row in rows}


def total_tx(row: Dict[str, str]) -> float:
    return as_float(row.get("Total tx # in this epoch")) or 0.0


def normal_tx(row: Dict[str, str]) -> float:
    return as_float(row.get("Normal tx # in this epoch")) or 0.0


def relay1_tx(row: Dict[str, str]) -> float:
    return as_float(row.get("Relay1 tx # in this epoch")) or 0.0


def relay2_tx(row: Dict[str, str]) -> float:
    return as_float(row.get("Relay2 tx # in this epoch")) or 0.0


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil(q * len(ordered))) - 1
    idx = min(max(idx, 0), len(ordered) - 1)
    return ordered[idx]


def summarize_period(
    name: str,
    rows: List[Dict[str, str]],
    cross_by_epoch: Dict[str, Dict[str, str]],
    tps_by_epoch: Dict[str, Dict[str, str]],
    latency_by_epoch: Dict[str, Dict[str, str]],
    variance_by_epoch: Dict[str, Dict[str, str]],
    max_epoch_tps: float,
) -> Dict[str, object]:
    if not rows:
        return {
            "name": name,
            "epoch_count": 0,
            "effective_total": 0.0,
        }

    epoch_ids = [str(row["EpochID"]) for row in rows]
    effective_total = sum(total_tx(row) for row in rows)
    normal_total = sum(normal_tx(row) for row in rows)
    relay1_total = sum(relay1_tx(row) for row in rows)
    relay2_total = sum(relay2_tx(row) for row in rows)
    cross_total = sum(
        as_float(cross_by_epoch[eid].get("CTX # in this epoch")) or 0.0
        for eid in epoch_ids
        if eid in cross_by_epoch
    )
    latency_total_sec = sum(
        as_float(latency_by_epoch[eid].get("Sum of All Tx TCL (sec.)")) or 0.0
        for eid in epoch_ids
        if eid in latency_by_epoch
    )
    start_times = [
        as_float(tps_by_epoch[eid].get("Epoch start time"))
        for eid in epoch_ids
        if eid in tps_by_epoch
    ]
    end_times = [
        as_float(tps_by_epoch[eid].get("Epoch end time"))
        for eid in epoch_ids
        if eid in tps_by_epoch
    ]
    start_times = [value for value in start_times if value and value > 0]
    end_times = [value for value in end_times if value and value > 0]
    wall_seconds = (
        (max(end_times) - min(start_times)) / 1000.0
        if start_times and end_times
        else 0.0
    )

    tps_values = [
        as_float(tps_by_epoch[eid].get("Avg. TPS of this epoch"))
        for eid in epoch_ids
        if eid in tps_by_epoch
    ]
    tps_values = [
        value
        for value in tps_values
        if value is not None and 0.0 <= value <= max_epoch_tps
    ]
    variance_values = [
        as_float(variance_by_epoch[eid].get("ShardLoadVariance"))
        for eid in epoch_ids
        if eid in variance_by_epoch
    ]
    variance_values = [value for value in variance_values if value is not None]

    return {
        "name": name,
        "first_epoch": int(epoch_ids[0]),
        "last_epoch": int(epoch_ids[-1]),
        "epoch_count": len(rows),
        "effective_total": effective_total,
        "normal_total": normal_total,
        "relay1_total": relay1_total,
        "relay2_total": relay2_total,
        "cross_total": cross_total,
        "weighted_cross_ratio": cross_total / effective_total
        if effective_total > 0
        else 0.0,
        "avg_confirm_latency_sec": latency_total_sec / effective_total
        if effective_total > 0
        else 0.0,
        "wall_tps": effective_total / wall_seconds if wall_seconds > 0 else 0.0,
        "mean_epoch_tps": statistics.mean(tps_values) if tps_values else 0.0,
        "median_epoch_tps": statistics.median(tps_values) if tps_values else 0.0,
        "mean_load_variance": statistics.mean(variance_values)
        if variance_values
        else 0.0,
        "median_load_variance": statistics.median(variance_values)
        if variance_values
        else 0.0,
        "p95_load_variance": percentile(variance_values, 0.95),
    }


def analyze_decisions(zip_path: Path, shards: int) -> Dict[str, object]:
    if not zip_path.exists():
        return {}

    with zipfile.ZipFile(zip_path) as zf:
        if "decision_records.jsonl" not in zf.namelist():
            return {}
        action_hist = [0 for _ in range(shards)]
        related_shard_hist = [0 for _ in range(shards)]
        same_related_by_shard = [0 for _ in range(shards)]
        confidence_sum = 0.0
        entropy_sum = 0.0
        sender_pos_nonzero = 0
        count = 0
        source_hist: Dict[str, int] = {}
        sender_pos_start = 10 * shards
        sender_pos_end = sender_pos_start + shards

        for line in zf.open("decision_records.jsonl"):
            rec = json.loads(line)
            count += 1
            shard = int(rec.get("shard", -1))
            if 0 <= shard < shards:
                action_hist[shard] += 1
            confidence_sum += float(rec.get("confidence", 0.0) or 0.0)
            entropy_sum += float(rec.get("entropy", 0.0) or 0.0)
            source = str(rec.get("source", "unknown"))
            source_hist[source] = source_hist.get(source, 0) + 1

            state = rec.get("state") or []
            if len(state) >= sender_pos_end:
                sender_pos = [
                    float(value) for value in state[sender_pos_start:sender_pos_end]
                ]
                if any(abs(value) > 1e-12 for value in sender_pos):
                    sender_pos_nonzero += 1
                    related_shard = max(range(shards), key=lambda sid: sender_pos[sid])
                    related_shard_hist[related_shard] += 1
                    if shard == related_shard:
                        same_related_by_shard[related_shard] += 1

    action_total = sum(action_hist)
    related_total = sum(related_shard_hist)
    related_follow_by_shard = [
        (
            same_related_by_shard[sid] / related_shard_hist[sid]
            if related_shard_hist[sid]
            else 0.0
        )
        for sid in range(shards)
    ]
    observed_follow = [
        related_follow_by_shard[sid]
        for sid in range(shards)
        if related_shard_hist[sid] > 0
    ]

    return {
        "decision_count": count,
        "action_hist": action_hist,
        "action_dist": [
            value / action_total if action_total else 0.0 for value in action_hist
        ],
        "confidence_mean": confidence_sum / count if count else 0.0,
        "entropy_mean": entropy_sum / count if count else 0.0,
        "source_hist": source_hist,
        "sender_pos_nonzero_ratio": sender_pos_nonzero / count if count else 0.0,
        "related_shard_hist": related_shard_hist,
        "same_related_by_shard": same_related_by_shard,
        "related_follow_by_shard": related_follow_by_shard,
        "same_as_related_ratio": sum(same_related_by_shard) / related_total
        if related_total
        else 0.0,
        "min_related_follow_ratio": min(observed_follow) if observed_follow else 0.0,
    }


def analyze_send_log(log_path: Path, shards: int) -> Dict[str, object]:
    if not log_path.exists():
        return {}
    pattern = re.compile(
        r"\[SPRING SEND\].*?nowData=(\d+).*?counts=\[([0-9 ]+)\]"
    )
    rows: List[List[int]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        counts = [int(value) for value in match.group(2).split()]
        if len(counts) == shards:
            rows.append(counts)

    totals = [0 for _ in range(shards)]
    max_shares: List[float] = []
    for counts in rows:
        for sid, value in enumerate(counts):
            totals[sid] += value
        total = sum(counts)
        if total > 0:
            max_shares.append(max(counts) / total)

    total_all = sum(totals)
    return {
        "send_batch_count": len(rows),
        "tx_placement_totals": totals,
        "tx_placement_dist": [
            value / total_all if total_all else 0.0 for value in totals
        ],
        "mean_batch_max_share": statistics.mean(max_shares) if max_shares else 0.0,
        "p95_batch_max_share": percentile(max_shares, 0.95),
        "extreme_batch_count_ge_0_8": sum(1 for value in max_shares if value >= 0.8),
    }


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    result_zip = Path(args.result_zip)
    tx_rows = read_csv_from_zip(result_zip, "supervisor_measureOutput/Tx_number.csv")
    cross_rows = read_csv_from_zip(
        result_zip, "supervisor_measureOutput/CrossTransaction_ratio.csv"
    )
    tps_rows = read_csv_from_zip(result_zip, "supervisor_measureOutput/Average_TPS.csv")
    latency_rows = read_csv_from_zip(
        result_zip, "supervisor_measureOutput/Transaction_Confirm_Latency.csv"
    )
    variance_rows = read_csv_from_zip(
        result_zip, "supervisor_measureOutput/Shard_Load_Variance.csv"
    )

    active_rows = [row for row in tx_rows if total_tx(row) > 0]
    injection_rows = [
        row for row in tx_rows if total_tx(row) > 0 and normal_tx(row) > 0
    ]
    capacity_rows = [
        row for row in tx_rows if total_tx(row) >= float(args.min_effective_tx)
    ]

    cross_by_epoch = keyed(cross_rows)
    tps_by_epoch = keyed(tps_rows)
    latency_by_epoch = keyed(latency_rows)
    variance_by_epoch = keyed(variance_rows)

    periods = {
        "all_active": summarize_period(
            "all_active",
            active_rows,
            cross_by_epoch,
            tps_by_epoch,
            latency_by_epoch,
            variance_by_epoch,
            args.max_epoch_tps,
        ),
        "main_injection": summarize_period(
            "main_injection",
            injection_rows,
            cross_by_epoch,
            tps_by_epoch,
            latency_by_epoch,
            variance_by_epoch,
            args.max_epoch_tps,
        ),
        "capacity_like": summarize_period(
            "capacity_like",
            capacity_rows,
            cross_by_epoch,
            tps_by_epoch,
            latency_by_epoch,
            variance_by_epoch,
            args.max_epoch_tps,
        ),
    }

    result = {
        "result_zip": str(result_zip),
        "spring_io_zip": str(args.spring_io_zip) if args.spring_io_zip else "",
        "log": str(args.log) if args.log else "",
        "shards": args.shards,
        "periods": periods,
        "decisions": analyze_decisions(Path(args.spring_io_zip), args.shards)
        if args.spring_io_zip
        else {},
        "send_log": analyze_send_log(Path(args.log), args.shards)
        if args.log
        else {},
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_zip", default="expTest/result.zip")
    parser.add_argument("--spring_io_zip", default="spring_io.zip")
    parser.add_argument("--log", default="")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--min_effective_tx", type=float, default=1000.0)
    parser.add_argument("--max_epoch_tps", type=float, default=5000.0)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
