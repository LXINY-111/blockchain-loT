"""Read-only audit of the PPT's archived experiment metrics and legacy input flag."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import csv
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit_run(run):
    report = read_json(run / "analysis.json")
    period = report["periods"]["all_active"]
    stage_totals = []
    queue_max = 0
    for shard in range(16):
        path = run / f"expTest/result/pbft_shardNum=16/Shard{shard}16.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        stage_totals.append(sum(float(r["# of all Txs in this block"]) for r in rows))
        queue_max = max(queue_max, max((int(r["TxPool Size"]) for r in rows), default=0))
    stage_total = sum(stage_totals)
    variance_path = run / "expTest/result/supervisor_measureOutput/Shard_Load_Variance.csv"
    tx_path = run / "expTest/result/supervisor_measureOutput/Tx_number.csv"
    with tx_path.open(encoding="utf-8-sig", newline="") as handle:
        active = {int(r["EpochID"]) for r in csv.DictReader(handle)
                  if float(r["Total tx # in this epoch"]) > 0}
    with variance_path.open(encoding="utf-8-sig", newline="") as handle:
        variances = [statistics.pvariance([float(r[f"Shard_{s}_Load"]) for s in range(16)])
                     for r in csv.DictReader(handle) if int(r["EpochID"]) in active]
    result = {
        "run": str(run), "completed": (run / "run_complete.json").exists(),
        "cross_pct": 100 * period["weighted_cross_ratio"],
        "cross_total": period["cross_total"], "effective_total": period["effective_total"],
        "mean_latency_sec": period["avg_confirm_latency_sec"],
        "p99_sec": report["latency_details"]["p99_sec"],
        "stage_totals": stage_totals, "stage_total": stage_total,
        "aggregate_stage_jain": stage_total**2 / (16 * sum(x*x for x in stage_totals)),
        "mean_effective_load_variance": statistics.fmean(variances),
        "variance_report_difference": statistics.fmean(variances) - period["mean_load_variance"],
        "stage_identity_difference": stage_total - period["effective_total"] - period["cross_total"],
        "max_observed_txpool": queue_max,
    }
    if "experiment_manifest.json" in {p.name for p in run.iterdir()}:
        manifest = read_json(run / "experiment_manifest.json")
        result["window"] = manifest.get("data_window")
    decisions = run / "spring_io/decision_records.jsonl"
    if decisions.exists():
        flags = Counter()
        anchor_features = Counter()
        with decisions.open(encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                state = row.get("state", [])
                if len(state) == 203:
                    flags[str(state[176])] += 1
                    anchor_features[str(state[193])] += 1
        result["decision_flag_counts"] = dict(flags)
        result["decision_anchor_count_feature_counts"] = dict(anchor_features)
    return result


def audit_archives():
    results = []
    for number in (9, 10):
        parent = ROOT.parent / ("\u5b9e\u9a8c\u7ed3\u679c" + str(number)) / "block_eval"
        for run in sorted(parent.iterdir()):
            relevant = ("test2M_i250" in run.name or "test2M_i300_seed7" in run.name
                        or (number == 10 and "validation444k" in run.name))
            if relevant and (run / "analysis.json").is_file():
                result = audit_run(run)
                results.append(result)
                print(json.dumps({k: result[k] for k in ("run", "aggregate_stage_jain", "mean_effective_load_variance")}, ensure_ascii=True), flush=True)
    (OUT / "archive_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_legacy_dataset():
    windows = [("train", 0, 2500000), ("validation", 2500000, 2944019), ("test", 2944019, 4944019)]
    output = []
    path = ROOT / "data_iot/iot_flow_sidecar_multi_anchor_full.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = {name: i for i, name in enumerate(next(reader))}
        for name, start, end in windows:
            seen = set()
            owners = defaultdict(set)
            related = defaultdict(set)
            flags = Counter()
            degree_counts = Counter()
            batch_new_counts = []
            later_batch_tx = 0
            first_batch_tx = 0
            start_clock = time.monotonic()
            for index in range(start, end):
                row = next(reader)
                assert int(row[columns["tx_index"]]) == index
                state = row[columns["to_address"]]
                owners[state].add(row[columns["from_address"]])
                if state in seen:
                    later_batch_tx += 1
                else:
                    first_batch_tx += 1
                related[state].update(a for a in row[columns["anchor_addresses"]].split(";") if a and a != state)
                if (index - start + 1) % 1000 == 0 or index == end - 1:
                    new = [state for state in related if state not in seen]
                    batch_new_counts.append(len(new))
                    for state in new:
                        degree = len(related[state])
                        flags[str(int(degree >= 4))] += 1
                        degree_counts[str(degree)] += 1
                    seen.update(related)
                    related.clear()
            result = {
                "split": name, "start": start, "end": end, "rows": end-start,
                "unique_states": len(seen), "first_placement_flag_counts": dict(flags),
                "first_placement_related_count_distribution": dict(degree_counts),
                "batches": len(batch_new_counts), "no_new_action_batches": batch_new_counts.count(0),
                "transactions_in_state_first_batch": first_batch_tx,
                "transactions_after_state_first_batch": later_batch_tx,
                "later_batch_tx_fraction": later_batch_tx/(end-start),
                "states_with_multiple_owners": sum(len(value)>1 for value in owners.values()),
                "elapsed_sec": time.monotonic()-start_clock,
            }
            output.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    (OUT / "legacy_dataset_flag_and_revisits.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if "--dataset" in sys.argv:
        audit_legacy_dataset()
    else:
        audit_archives()
