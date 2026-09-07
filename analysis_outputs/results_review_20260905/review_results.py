"""Read completed archived runs only; never inspect the excluded live run."""
from __future__ import annotations
import csv
import json
import math
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOTS = [Path(r"E:\project_iot\实验结果9"), Path(r"E:\project_iot\实验结果10")]
EXCLUDED = "ppo_top7_w5530_pareto_16s_test2M_i250_seed27_20260905_202840"
PATTERN = re.compile(r"^(.*)_16s_(test2M|validation444k)_i(\d+)_seed(\d+)_(\d{8}_\d{6})$")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"nonfinite value: {value}")
    return x

def q(values, p):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None

def extract(run, result_set):
    match = PATTERN.fullmatch(run.name)
    if not match:
        return {"result_set": result_set, "run": run.name, "status": "unrecognized_name"}
    method, window, rate, seed, timestamp = match.groups()
    complete = read_json(run / "run_complete.json")
    analysis = read_json(run / "analysis.json")
    cfg = read_json(run / "paramsConfig.snapshot.json")
    ctx = read_json(run / "run_context.json")
    expected = 2_000_000 if window == "test2M" else 444_019
    row = dict(result_set=result_set, run=run.name, run_path=str(run), method=method,
               window=window, target_tps=int(rate), seed=int(seed), timestamp=timestamp,
               expected_total=expected, status="completed_marker", issues=[])
    row["config"] = {k: cfg.get(k) for k in [
        "ConsensusMethod", "SpringMode", "SpringRandomSeed", "SpringIOTMode", "SpringIOTIdentityMode",
        "SpringCandidateTopK", "SpringCapacityGuard", "SpringCandidateLoadWeight", "SpringModelFile",
        "DatasetStartTx", "TotalDataSize", "InjectSpeed", "TxBatchSize", "BlockSize", "Block_Interval",
        "SpringIOTCSTRWeight", "SpringIOTBalanceWeight", "SpringIOTCommCostWeight", "SpringIOTHotspotWeight",
        "Delay", "JitterRange", "Bandwidth", "SpringNSShardCapacityFactor"]}
    expected_start = 2_944_019 if window == "test2M" else 2_500_000
    for k, want in [("DatasetStartTx",expected_start),("TotalDataSize",expected),("InjectSpeed",int(rate)),
                    ("SpringRandomSeed",int(seed)),("BlockSize",1000),("Block_Interval",5000)]:
        if cfg.get(k) != want:
            row["issues"].append(f"config {k}={cfg.get(k)}, expected {want}")
    row["context_shards"] = ctx.get("shards")
    state_path = run / "state_check.txt"
    state_text = state_path.read_text(encoding="utf-8-sig", errors="replace") if state_path.is_file() else ""
    row["state_actual_pass"] = "--- PASS: TestFinalResult" in state_text and "--- SKIP:" not in state_text
    for field, pat in [
        ("expected_accounts",r"Results from dataset file: # of accounts\s+(\d+)"),
        ("correct_accounts",r"# of correct accounts\s+(\d+)"),
        ("found_accounts",r"# of found accounts\s+(\d+)"),
        ("duplicate_accounts",r"# of duplicate accounts\s+(\d+)")]:
        m = re.search(pat,state_text)
        row[field] = int(m.group(1)) if m else None
    row["state_shards_reported"] = len(set(re.findall(r"Shard (\d+) newest block=", state_text)))
    if not row["state_actual_pass"]: row["issues"].append("no explicit executed state PASS")
    if row["expected_accounts"] != row["correct_accounts"]: row["issues"].append("account balance count mismatch")
    if row["duplicate_accounts"] not in (None,0): row["issues"].append("duplicate accounts")
    latency = complete.get("latency_details") or analysis.get("latency_details") or {}
    pace = complete.get("injection_pace") or analysis.get("injection_pace") or {}
    row["latency_rows"] = latency.get("row_count")
    row["valid_latency_count"] = latency.get("valid_latency_count")
    for key in ["mean_sec","p50_sec","p95_sec","p99_sec","max_sec"]:
        row["latency_"+key] = latency.get(key)
    row["actual_offered_tps"] = pace.get("actual_offered_tps")
    row["injected_total"] = pace.get("final_cumulative_tx")
    row["max_abs_schedule_lag_ms"] = pace.get("max_abs_schedule_lag_ms")
    if row["latency_rows"] != expected or row["valid_latency_count"] != expected:
        row["issues"].append("archived latency coverage mismatch")
    if row["injected_total"] != expected: row["issues"].append("injected total mismatch")
    result = run / "expTest" / "result" / "supervisor_measureOutput"
    filenames = {"tx":"Tx_number.csv","cross":"CrossTransaction_ratio.csv","tps":"Average_TPS.csv",
                 "variance":"Shard_Load_Variance.csv","latency":"Transaction_Confirm_Latency.csv"}
    tables = {}
    for key, filename in filenames.items():
        path = result / filename
        if not path.is_file():
            row["issues"].append(f"missing {filename}")
            return row
        tables[key] = read_csv(path)
        ids = [r["EpochID"] for r in tables[key]]
        if len(ids) != len(set(ids)): row["issues"].append(f"duplicate epochs {key}")
    active = [r for r in tables["tx"] if num(r["Total tx # in this epoch"])>0]
    active_ids = {r["EpochID"] for r in active}
    row["active_epochs"] = len(active)
    for short, col in [("effective_total","Total tx # in this epoch"),("normal_total","Normal tx # in this epoch"),
                       ("relay1_total","Relay1 tx # in this epoch"),("relay2_total","Relay2 tx # in this epoch")]:
        row[short] = sum(num(r[col]) for r in active)
    row["cross_total"] = sum(num(r["CTX # in this epoch"]) for r in tables["cross"] if r["EpochID"] in active_ids)
    row["cross_ratio"] = row["cross_total"] / row["effective_total"]
    if row["effective_total"] != expected: row["issues"].append("confirmed effective count mismatch")
    if row["relay1_total"] != row["relay2_total"]: row["issues"].append("relay1 relay2 mismatch")
    if row["normal_total"]+row["relay1_total"] != expected: row["issues"].append("unique-stage total mismatch")
    tps_rows = [r for r in tables["tps"] if r["EpochID"] in active_ids]
    begin = min(num(r["Epoch start time"]) for r in tps_rows)
    end = max(num(r["Epoch end time"]) for r in tps_rows)
    row["wall_seconds"] = (end-begin)/1000
    row["wall_tps"] = row["effective_total"] / row["wall_seconds"]
    row["first_epoch"] = min(int(r["EpochID"]) for r in active)
    row["last_epoch"] = max(int(r["EpochID"]) for r in active)
    row["raw_epoch_tps_mean"] = st.fmean(num(r["Avg. TPS of this epoch"]) for r in tps_rows)
    loads, aggregate = [], [0.0]*16
    for r in tables["variance"]:
        if r["EpochID"] not in active_ids: continue
        values = [num(r[f"Shard_{sid}_Load"]) for sid in range(16)]
        total = sum(values)
        if total <= 0: continue
        if any(x<0 for x in values): row["issues"].append("negative shard load")
        fair = total**2/(16*sum(x*x for x in values))
        loads.append({"epoch":int(r["EpochID"]),"max_share":max(values)/total,"jain":fair,
                      "active_shards":sum(x>0 for x in values),"cv":st.pstdev(values)/(total/16),
                      "variance":st.pvariance(values),"loads":values})
        aggregate=[a+b for a,b in zip(aggregate,values)]
    if len(loads) != len(active): row["issues"].append("missing active epoch load coverage")
    for key in ["max_share","jain","active_shards","cv","variance"]:
        row["mean_"+key] = st.fmean(r[key] for r in loads)
    row["p95_max_share"] = q([r["max_share"] for r in loads],.95)
    row["aggregate_loads"] = aggregate
    row["aggregate_load_total"] = sum(aggregate)
    row["aggregate_max_share"] = max(aggregate)/sum(aggregate)
    row["aggregate_jain"] = sum(aggregate)**2/(16*sum(x*x for x in aggregate))
    if sum(aggregate) != expected: row["issues"].append("load sum does not equal expected transaction total")
    row["archived_cross_difference"] = row["cross_ratio"] - complete.get("all_active",{}).get("weighted_cross_ratio",row["cross_ratio"])
    row["archived_wall_tps_difference"] = row["wall_tps"] - complete.get("all_active",{}).get("wall_tps",row["wall_tps"])
    decisions = analysis.get("decisions",{})
    row["decisions"] = decisions
    row["quality_ok"] = len(row["issues"])==0
    return row

def main():
    rows, inventory = [], []
    for root in ROOTS:
        for run in sorted((root / "block_eval").iterdir()):
            if run.name == EXCLUDED:
                inventory.append(dict(result_set=root.name,run=run.name,status="excluded_by_user_do_not_read"))
                continue
            if not run.is_dir(): continue
            if not (run / "run_complete.json").is_file():
                inventory.append(dict(result_set=root.name,run=run.name,status="no_completion_marker_not_inspected"))
                continue
            record = extract(run,root.name)
            rows.append(record)
            inventory.append(dict(result_set=root.name,run=run.name,status=record["status"]))
    (OUT / "run_metrics.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT / "inventory.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding="utf-8")
    cols=["result_set","method","window","target_tps","seed","cross_ratio","mean_max_share","mean_jain",
          "mean_active_shards","latency_mean_sec","latency_p95_sec","latency_p99_sec","wall_tps","quality_ok","run"]
    with (OUT / "run_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"completed":len(rows),"inventory":len(inventory),
                      "issues":[{"run":r["run"],"issues":r.get("issues")} for r in rows if r.get("issues")]},ensure_ascii=True))

if __name__=="__main__":main()
