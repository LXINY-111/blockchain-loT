import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import DEFAULT_SHARD_NUM


@contextmanager
def open_text_source(source_path: Path, name: str):
    if source_path.is_file() and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as zf:
            with zf.open(name) as raw:
                with io.TextIOWrapper(
                    raw,
                    encoding="utf-8-sig",
                    errors="replace",
                    newline="",
                ) as handle:
                    yield handle
        return

    candidates = [source_path / name, source_path / "result" / name]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as handle:
                yield handle
            return
    raise FileNotFoundError(
        f"result file {name!r} not found under {source_path}"
    )


def as_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def read_text_source(source_path: Path, name: str) -> str:
    with open_text_source(source_path, name) as handle:
        return handle.read()


def read_csv_source(source_path: Path, name: str) -> List[Dict[str, str]]:
    data = read_text_source(source_path, name)
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


def shard_load_values(row: Dict[str, str], shards: int) -> List[float]:
    """Return one epoch's per-shard load vector in stable shard order."""
    values: List[float] = []
    for sid in range(shards):
        value = as_float(row.get(f"Shard_{sid}_Load"))
        values.append(value if value is not None and value >= 0.0 else 0.0)
    return values


def jain_fairness(values: List[float]) -> float:
    """Jain's fairness index over all configured shards, including zero load."""
    if not values:
        return 0.0
    total = sum(values)
    sum_squares = sum(value * value for value in values)
    if total <= 0.0 or sum_squares <= 0.0:
        return 0.0
    return total * total / (len(values) * sum_squares)


def load_cv(values: List[float]) -> float:
    """Population coefficient of variation for one shard-load vector."""
    if not values:
        return 0.0
    mean_value = statistics.fmean(values)
    if mean_value <= 0.0:
        return 0.0
    return statistics.pstdev(values) / mean_value


def summarize_load_balance(
    epoch_ids: List[str],
    variance_by_epoch: Dict[str, Dict[str, str]],
    shards: int,
) -> Dict[str, object]:
    """Compute scale-free load metrics from the archived per-shard loads."""
    max_shares: List[float] = []
    active_counts: List[float] = []
    fairness_values: List[float] = []
    cv_values: List[float] = []
    aggregate_loads = [0.0 for _ in range(shards)]

    for epoch_id in epoch_ids:
        row = variance_by_epoch.get(epoch_id)
        if row is None:
            continue
        loads = shard_load_values(row, shards)
        total = sum(loads)
        if total <= 0.0:
            continue
        max_shares.append(max(loads) / total)
        active_counts.append(float(sum(1 for value in loads if value > 1e-12)))
        fairness_values.append(jain_fairness(loads))
        cv_values.append(load_cv(loads))
        for sid, value in enumerate(loads):
            aggregate_loads[sid] += value

    aggregate_total = sum(aggregate_loads)
    return {
        "load_epoch_count": len(max_shares),
        "mean_max_shard_load_share": statistics.fmean(max_shares)
        if max_shares
        else 0.0,
        "p95_max_shard_load_share": percentile(max_shares, 0.95),
        "mean_active_shards": statistics.fmean(active_counts)
        if active_counts
        else 0.0,
        "min_active_shards": min(active_counts) if active_counts else 0.0,
        "mean_jain_fairness": statistics.fmean(fairness_values)
        if fairness_values
        else 0.0,
        "p05_jain_fairness": percentile(fairness_values, 0.05),
        "mean_load_cv": statistics.fmean(cv_values) if cv_values else 0.0,
        "p95_load_cv": percentile(cv_values, 0.95),
        "aggregate_shard_loads": aggregate_loads,
        "aggregate_max_shard_load_share": (
            max(aggregate_loads) / aggregate_total
            if aggregate_total > 0.0
            else 0.0
        ),
        "aggregate_jain_fairness": jain_fairness(aggregate_loads),
    }


def analyze_tx_latency_details(result_path: Path) -> Dict[str, object]:
    name = "supervisor_measureOutput/Tx_Details.csv"
    try:
        with open_text_source(result_path, name) as handle:
            reader = csv.DictReader(handle)
            values_ms = []
            row_count = 0
            seen_tx_hashes = set()
            duplicate_tx_hash_count = 0
            missing_tx_hash_count = 0
            for row in reader:
                row_count += 1
                tx_hash = ""
                for key, raw_value in row.items():
                    if key and key.strip().lower().startswith("txhash"):
                        tx_hash = str(raw_value or "").strip()
                        break
                if not tx_hash:
                    missing_tx_hash_count += 1
                elif tx_hash in seen_tx_hashes:
                    duplicate_tx_hash_count += 1
                else:
                    seen_tx_hashes.add(tx_hash)
                value = as_float(row.get("Confirmed latency of this tx (ms)"))
                if value is not None and value >= 0:
                    values_ms.append(value)
    except FileNotFoundError:
        return {}

    if not values_ms:
        return {
            "row_count": row_count,
            "valid_latency_count": 0,
            "invalid_latency_count": row_count,
            "valid_latency_ratio": 0.0,
            "unique_tx_hash_count": len(seen_tx_hashes),
            "duplicate_tx_hash_count": duplicate_tx_hash_count,
            "missing_tx_hash_count": missing_tx_hash_count,
        }

    values_ms.sort()

    def ordered_percentile(q: float) -> float:
        idx = int(math.ceil(q * len(values_ms))) - 1
        idx = min(max(idx, 0), len(values_ms) - 1)
        return values_ms[idx] / 1000.0

    return {
        "row_count": row_count,
        "valid_latency_count": len(values_ms),
        "invalid_latency_count": row_count - len(values_ms),
        "valid_latency_ratio": len(values_ms) / row_count if row_count else 0.0,
        "unique_tx_hash_count": len(seen_tx_hashes),
        "duplicate_tx_hash_count": duplicate_tx_hash_count,
        "missing_tx_hash_count": missing_tx_hash_count,
        "mean_sec": statistics.fmean(values_ms) / 1000.0,
        "p50_sec": ordered_percentile(0.50),
        "p95_sec": ordered_percentile(0.95),
        "p99_sec": ordered_percentile(0.99),
        "max_sec": values_ms[-1] / 1000.0,
    }


def summarize_period(
    name: str,
    rows: List[Dict[str, str]],
    cross_by_epoch: Dict[str, Dict[str, str]],
    tps_by_epoch: Dict[str, Dict[str, str]],
    latency_by_epoch: Dict[str, Dict[str, str]],
    variance_by_epoch: Dict[str, Dict[str, str]],
    shards: int,
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
    load_balance = summarize_load_balance(
        epoch_ids,
        variance_by_epoch,
        shards,
    )

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
        "p05_epoch_tps": percentile(tps_values, 0.05),
        "p95_epoch_tps": percentile(tps_values, 0.95),
        "mean_load_variance": statistics.mean(variance_values)
        if variance_values
        else 0.0,
        "median_load_variance": statistics.median(variance_values)
        if variance_values
        else 0.0,
        "p95_load_variance": percentile(variance_values, 0.95),
        "load_balance": load_balance,
    }


@contextmanager
def open_decision_lines(source_path: Path):
    if source_path.is_file() and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as zf:
            if "decision_records.jsonl" not in zf.namelist():
                yield None
            else:
                with zf.open("decision_records.jsonl") as handle:
                    yield handle
        return

    candidates = [
        source_path / "decision_records.jsonl",
        source_path / "spring_io" / "decision_records.jsonl",
    ]
    decision_file = next((path for path in candidates if path.is_file()), None)
    if decision_file is None:
        yield None
        return
    with decision_file.open("rb") as handle:
        yield handle


def analyze_decisions(source_path: Path, shards: int) -> Dict[str, object]:
    if not source_path.exists():
        return {}

    with open_decision_lines(source_path) as decision_lines:
        if decision_lines is None:
            return {}
        action_hist = [0 for _ in range(shards)]
        related_shard_hist = [0 for _ in range(shards)]
        same_related_by_shard = [0 for _ in range(shards)]
        selected_related_mass_count = 0
        selected_related_mass_sum = 0.0
        confidence_sum = 0.0
        entropy_sum = 0.0
        sender_pos_nonzero = 0
        count = 0
        source_hist: Dict[str, int] = {}
        action_mask_size_hist: Dict[int, int] = {}
        action_mask_size_sum = 0.0
        action_mask_seen = 0
        identifiable_mask_seen = 0
        major_related_in_mask_count = 0
        all_related_in_mask_count = 0
        candidate_related_mass_sum = 0.0
        major_related_chosen_when_in_mask_count = 0
        major_related_chosen_when_in_mask_seen = 0
        sender_pos_start = 10 * shards
        sender_pos_end = sender_pos_start + shards

        for line in decision_lines:
            rec = json.loads(line)
            count += 1
            shard = int(rec.get("shard", -1))
            if 0 <= shard < shards:
                action_hist[shard] += 1
            confidence_sum += float(rec.get("confidence", 0.0) or 0.0)
            entropy_sum += float(rec.get("entropy", 0.0) or 0.0)
            source = str(rec.get("source", "unknown"))
            source_hist[source] = source_hist.get(source, 0) + 1
            raw_mask = rec.get("action_mask") or []
            identifiable_mask = (
                isinstance(raw_mask, list) and len(raw_mask) >= shards
            )
            mask_size = 0
            if isinstance(raw_mask, list) and raw_mask:
                mask_size = sum(1 for value in raw_mask if int(value or 0) > 0)
            else:
                mask_size = int(rec.get("action_mask_count", 0) or 0)
            if mask_size > 0:
                action_mask_seen += 1
                action_mask_size_sum += float(mask_size)
                action_mask_size_hist[mask_size] = (
                    action_mask_size_hist.get(mask_size, 0) + 1
                )

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
                    if 0 <= shard < shards and sender_pos[shard] > 1e-12:
                        selected_related_mass_count += 1
                        selected_related_mass_sum += sender_pos[shard]
                    if identifiable_mask:
                        identifiable_mask_seen += 1
                        allowed = [
                            int(raw_mask[sid] or 0) > 0
                            for sid in range(shards)
                        ]
                        related_shards = [
                            sid
                            for sid, value in enumerate(sender_pos)
                            if value > 1e-12
                        ]
                        major_in_mask = allowed[related_shard]
                        if major_in_mask:
                            major_related_in_mask_count += 1
                            major_related_chosen_when_in_mask_seen += 1
                            if shard == related_shard:
                                major_related_chosen_when_in_mask_count += 1
                        if related_shards and all(
                            allowed[sid] for sid in related_shards
                        ):
                            all_related_in_mask_count += 1
                        total_related_mass = sum(sender_pos)
                        if total_related_mass > 1e-12:
                            candidate_related_mass_sum += sum(
                                sender_pos[sid]
                                for sid in related_shards
                                if allowed[sid]
                            ) / total_related_mass

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
    single_candidate_count = action_mask_size_hist.get(1, 0)
    multi_candidate_count = sum(
        value for size, value in action_mask_size_hist.items() if size > 1
    )
    python_ppo_count = sum(
        value for source, value in source_hist.items() if source.startswith("python_ppo")
    )
    fallback_source_hist = {
        source: value
        for source, value in source_hist.items()
        if "fallback" in source.lower()
    }
    fallback_count = sum(fallback_source_hist.values())
    candidate_only_count = source_hist.get("go_candidate_only", 0)

    return {
        "decision_count": count,
        "action_hist": action_hist,
        "action_dist": [
            value / action_total if action_total else 0.0 for value in action_hist
        ],
        "confidence_mean": confidence_sum / count if count else 0.0,
        "entropy_mean": entropy_sum / count if count else 0.0,
        "source_hist": source_hist,
        "python_ppo_count": python_ppo_count,
        "python_ppo_ratio": python_ppo_count / count if count else 0.0,
        "fallback_count": fallback_count,
        "fallback_ratio": fallback_count / count if count else 0.0,
        "fallback_source_hist": fallback_source_hist,
        "candidate_only_count": candidate_only_count,
        "candidate_only_ratio": candidate_only_count / count if count else 0.0,
        "action_mask_seen": action_mask_seen,
        "action_mask_seen_ratio": action_mask_seen / count if count else 0.0,
        "action_mask_size_hist": dict(sorted(action_mask_size_hist.items())),
        "action_mask_size_mean": action_mask_size_sum / action_mask_seen
        if action_mask_seen
        else 0.0,
        "single_candidate_count": single_candidate_count,
        "single_candidate_ratio": single_candidate_count / action_mask_seen
        if action_mask_seen
        else 0.0,
        "multi_candidate_count": multi_candidate_count,
        "multi_candidate_ratio": multi_candidate_count / action_mask_seen
        if action_mask_seen
        else 0.0,
        "sender_pos_nonzero_ratio": sender_pos_nonzero / count if count else 0.0,
        "candidate_diagnostic_seen": identifiable_mask_seen,
        "candidate_diagnostic_seen_ratio": identifiable_mask_seen / related_total
        if related_total
        else 0.0,
        "candidate_major_anchor_coverage_count": major_related_in_mask_count,
        "candidate_major_anchor_coverage_ratio": major_related_in_mask_count
        / identifiable_mask_seen
        if identifiable_mask_seen
        else 0.0,
        "candidate_major_anchor_absent_ratio": (
            identifiable_mask_seen - major_related_in_mask_count
        )
        / identifiable_mask_seen
        if identifiable_mask_seen
        else 0.0,
        "candidate_all_related_coverage_count": all_related_in_mask_count,
        "candidate_all_related_coverage_ratio": all_related_in_mask_count
        / identifiable_mask_seen
        if identifiable_mask_seen
        else 0.0,
        "candidate_related_mass_coverage_mean": candidate_related_mass_sum
        / identifiable_mask_seen
        if identifiable_mask_seen
        else 0.0,
        "major_anchor_choice_when_available_count": (
            major_related_chosen_when_in_mask_count
        ),
        "major_anchor_choice_when_available_ratio": (
            major_related_chosen_when_in_mask_count
            / major_related_chosen_when_in_mask_seen
            if major_related_chosen_when_in_mask_seen
            else 0.0
        ),
        "major_anchor_not_chosen_when_available_ratio": (
            (
                major_related_chosen_when_in_mask_seen
                - major_related_chosen_when_in_mask_count
            )
            / major_related_chosen_when_in_mask_seen
            if major_related_chosen_when_in_mask_seen
            else 0.0
        ),
        "related_shard_hist": related_shard_hist,
        "same_related_by_shard": same_related_by_shard,
        "major_related_follow_by_shard": same_related_by_shard,
        "related_follow_by_shard": related_follow_by_shard,
        "major_related_follow_count": sum(same_related_by_shard),
        "major_related_follow_ratio": sum(same_related_by_shard) / related_total
        if related_total
        else 0.0,
        # Match the offline environment metric: the selected shard has any
        # positive sender/anchor relation, not necessarily the largest one.
        "selected_related_mass_count": selected_related_mass_count,
        "same_as_related_ratio": selected_related_mass_count / related_total
        if related_total
        else 0.0,
        "chosen_related_mass_mean": selected_related_mass_sum / related_total
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


def analyze_injection_pace(log_path: Path) -> Dict[str, object]:
    if not log_path.exists():
        return {}
    pattern = re.compile(
        r"\[INJECTION PACE\].*?"
        r"batchTx=(\d+).*?"
        r"cumulativeTx=(\d+).*?"
        r"targetTPS=(\d+).*?"
        r"elapsedSec=([0-9.]+).*?"
        r"actualOfferedTPS=([0-9.]+).*?"
        r"scheduleLagMs=(-?[0-9.]+)"
    )
    rows: List[Dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "batch_tx": float(match.group(1)),
                "cumulative_tx": float(match.group(2)),
                "target_tps": float(match.group(3)),
                "elapsed_sec": float(match.group(4)),
                "actual_offered_tps": float(match.group(5)),
                "schedule_lag_ms": float(match.group(6)),
            }
        )

    if not rows:
        return {}
    final = rows[-1]
    return {
        "pace_record_count": len(rows),
        "final_cumulative_tx": int(final["cumulative_tx"]),
        "target_tps": final["target_tps"],
        "elapsed_sec": final["elapsed_sec"],
        "actual_offered_tps": final["actual_offered_tps"],
        "attainment_ratio": (
            final["actual_offered_tps"] / final["target_tps"]
            if final["target_tps"] > 0
            else 0.0
        ),
        "final_schedule_lag_ms": final["schedule_lag_ms"],
        "max_abs_schedule_lag_ms": max(
            abs(row["schedule_lag_ms"]) for row in rows
        ),
    }


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    if (getattr(args, "iot_tx_metadata", "") or getattr(args, "iot_dataset_dir", "")) and not getattr(args, "iot_run_dir", ""):
        raise ValueError("IoT metadata/dataset options require --iot_run_dir")
    result_value = getattr(args, "result_path", getattr(args, "result_zip", ""))
    spring_io_value = getattr(
        args,
        "spring_io_path",
        getattr(args, "spring_io_zip", ""),
    )
    result_path = Path(result_value)
    tx_rows = read_csv_source(result_path, "supervisor_measureOutput/Tx_number.csv")
    cross_rows = read_csv_source(
        result_path, "supervisor_measureOutput/CrossTransaction_ratio.csv"
    )
    tps_rows = read_csv_source(result_path, "supervisor_measureOutput/Average_TPS.csv")
    latency_rows = read_csv_source(
        result_path, "supervisor_measureOutput/Transaction_Confirm_Latency.csv"
    )
    variance_rows = read_csv_source(
        result_path, "supervisor_measureOutput/Shard_Load_Variance.csv"
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
            args.shards,
            args.max_epoch_tps,
        ),
        "main_injection": summarize_period(
            "main_injection",
            injection_rows,
            cross_by_epoch,
            tps_by_epoch,
            latency_by_epoch,
            variance_by_epoch,
            args.shards,
            args.max_epoch_tps,
        ),
        "capacity_like": summarize_period(
            "capacity_like",
            capacity_rows,
            cross_by_epoch,
            tps_by_epoch,
            latency_by_epoch,
            variance_by_epoch,
            args.shards,
            args.max_epoch_tps,
        ),
    }

    result = {
        "result_path": str(result_path),
        "period_load_basis": "effective transactions: intra=1, cross=0.5 per endpoint; full-window stages are in iot_metrics.execution_load",
        "spring_io_path": str(spring_io_value) if spring_io_value else "",
        "result_zip": str(result_path)
        if result_path.is_file() and zipfile.is_zipfile(result_path)
        else "",
        "spring_io_zip": str(spring_io_value)
        if spring_io_value
        and Path(spring_io_value).is_file()
        and zipfile.is_zipfile(Path(spring_io_value))
        else "",
        "log": str(args.log) if args.log else "",
        "shards": args.shards,
        "periods": periods,
        "decisions": analyze_decisions(Path(spring_io_value), args.shards)
        if spring_io_value
        else {},
        "send_log": analyze_send_log(Path(args.log), args.shards)
        if args.log
        else {},
        "injection_pace": analyze_injection_pace(Path(args.log))
        if args.log
        else {},
        "latency_details": analyze_tx_latency_details(result_path),
    }

    if getattr(args, "iot_run_dir", ""):
        # 显式启用 IoT 评估；旧数据入口和原有指标保持兼容。
        from iot_metrics import analyze_chain_details, chain_context, read_metadata
        scenes, manifest = chain_context(args)
        metadata_path = getattr(args, "iot_tx_metadata", "")
        iot = analyze_chain_details(
            read_csv_source(result_path, "supervisor_measureOutput/Tx_Details.csv"),
            scenes, manifest, args.shards,
            read_metadata(metadata_path) if metadata_path else None)
        all_iot = iot["groups"]["all"]
        all_chain = periods["all_active"]
        if (abs(all_chain["effective_total"] - all_iot["count"]) > 1e-6 or
                abs(all_chain["cross_total"] - all_iot["cross_count"]) > 1e-6):
            raise ValueError("IoT transaction-level totals disagree with chain aggregate metrics")
        iot["metadata_path"] = str(metadata_path)
        result["iot_metrics"] = iot

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        "--result_zip",
        dest="result_path",
        default="expTest/result",
        help="unpacked result directory or legacy result ZIP",
    )
    parser.add_argument(
        "--spring_io_path",
        "--spring_io_zip",
        dest="spring_io_path",
        default="spring_io",
        help="unpacked spring_io directory or legacy spring_io ZIP",
    )
    parser.add_argument("--log", default="")
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--min_effective_tx", type=float, default=1000.0)
    parser.add_argument("--max_epoch_tps", type=float, default=5000.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--iot_run_dir", default="", help="enable IoT metrics using this run's paramsConfig.json")
    parser.add_argument("--iot_dataset_dir", default="", help="optional dataset path override after moving files")
    parser.add_argument("--iot_tx_metadata", default="", help="legacy runs: CSV exported from their read-only chain databases")
    parser.add_argument("--iot_reference_start_tx", type=int, default=200000)
    parser.add_argument("--iot_reference_max_txs", type=int, default=50000)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
