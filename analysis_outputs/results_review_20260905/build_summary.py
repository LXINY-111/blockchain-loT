import csv, hashlib, json, math, statistics as st
from collections import defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent
rows=json.loads((OUT/"run_metrics.json").read_text(encoding="utf-8"))
fields=["cross_ratio","mean_max_share","mean_jain","mean_active_shards","mean_variance","latency_mean_sec",
        "latency_p95_sec","latency_p99_sec","wall_tps"]
labels={"anchoronly":"AnchorOnly","hash":"Hash","heuristic_pure":"Heuristic","minstate":"MinState",
        "nsshard_adapted_cap12":"NSshard-adapted","ppo_top7_w5530_pareto":"PPO TopK7",
        "ppo_top8_w5530_pareto":"PPO TopK8","random":"Random","candidate_only_top7_w5530":"Candidate-Only"}
groups=defaultdict(list)
lineage=[]; queues=[]
freeze=json.loads(Path(r"E:\project_iot\实验结果9\protocol_freeze_current_top7_w5530_16s.json").read_text(encoding="utf-8-sig"))
modelhash={x["seed"]:x["model_sha256"].lower() for x in freeze["model_selections"]}
for r in rows:
    groups[(r["result_set"],r["method"],r["window"],r["target_tps"])].append(r)
    root=Path(r["run_path"])
    manifest=json.loads((root/"experiment_manifest.json").read_text(encoding="utf-8-sig"))
    analysis=json.loads((root/"analysis.json").read_text(encoding="utf-8-sig"))
    files=manifest.get("files",{});dec=analysis.get("decisions",{})
    meta={"run":r["run"],"result_set":r["result_set"],"method":r["method"],"seed":r["seed"],
          "dataset_sha256":files.get("dataset",{}).get("sha256"),"sidecar_sha256":files.get("sidecar",{}).get("sha256"),
          "model_sha256":(files.get("model") or {}).get("sha256"),"git_commit":manifest.get("git",{}).get("commit"),
          "dirty":bool(manifest.get("git",{}).get("status_porcelain")),"model_matches_frozen":None,
          "decision_count":dec.get("decision_count"),"python_ppo_ratio":dec.get("python_ppo_ratio"),
          "fallback_ratio":dec.get("fallback_ratio"),"candidate_only_ratio":dec.get("candidate_only_ratio"),
          "source_hist":dec.get("source_hist",dec.get("source_histogram")),
          "action_mask_size_mean":dec.get("action_mask_size_mean"),
          "candidate_major_anchor_coverage_ratio":dec.get("candidate_major_anchor_coverage_ratio"),
          "major_related_follow_ratio":dec.get("major_related_follow_ratio"),
          "candidate_related_mass_coverage_mean":dec.get("candidate_related_mass_coverage_mean"),
          "major_anchor_choice_when_available_ratio":dec.get("major_anchor_choice_when_available_ratio")}
    if r["method"]=="ppo_top7_w5530_pareto" and meta["model_sha256"]:
        meta["model_matches_frozen"] = meta["model_sha256"].lower()==modelhash.get(r["seed"])
    lineage.append(meta)
    if (r["result_set"].endswith("9") and r["window"]=="test2M") or r["result_set"].endswith("10"):
        sharded=root/"expTest"/"result"/"pbft_shardNum=16"
        if not sharded.is_dir():continue
        peak={"queue":-1};totals=[0.0]*16;full=[0]*16;per_epoch=defaultdict(lambda:[0.0]*16)
        for sid in range(16):
            path=sharded/f"Shard{sid}16.csv"
            if not path.is_file():continue
            with path.open(encoding="utf-8-sig",newline="") as f:
                for b in csv.DictReader(f):
                    amount=float(b["# of all Txs in this block"]);epoch=int(b["Block Height"])
                    totals[sid]+=amount;per_epoch[epoch][sid]+=amount
                    if amount>=1000:full[sid]+=1
                    queue=int(b["TxPool Size"])
                    if queue>peak["queue"]:peak={"queue":queue,"shard":sid,"epoch":epoch}
        vectors=[v for v in per_epoch.values() if sum(v)>0]
        fair=lambda v:sum(v)**2/(16*sum(x*x for x in v))
        record={"run":r["run"],"result_set":r["result_set"],"method":r["method"],"seed":r["seed"],
                "window":r["window"],"target_tps":r["target_tps"],"peak_queue":peak,
                "physical_stages":sum(totals),"expected_physical_stages":r["normal_total"]+r["relay1_total"]+r["relay2_total"],
                "physical_aggregate_jain":fair(totals),"physical_aggregate_maxshare":max(totals)/sum(totals),
                "physical_epoch_jain_mean":st.fmean(fair(v) for v in vectors),
                "physical_epoch_maxshare_mean":st.fmean(max(v)/sum(v) for v in vectors),
                "physical_loads":totals,"full_blocks":full}
        queues.append(record)

group_rows=[]
for (result_set,method,window,rate),members in groups.items():
    row={"result_set":result_set,"method":method,"label":labels.get(method,method),"window":window,"target_tps":rate,
         "n":len(members),"seeds":sorted(x["seed"] for x in members)}
    for f in fields:
        row[f]=st.fmean(x[f] for x in members)
        row[f+"_sd"]=st.stdev(x[f] for x in members) if len(members)>1 else None
    group_rows.append(row)
paired=[]
for seed in [7,17,27]:
    c=next(r for r in rows if r["result_set"].endswith("10") and r["method"].startswith("candidate_only") and r["seed"]==seed)
    p=next(r for r in rows if r["result_set"].endswith("10") and r["method"]=="ppo_top7_w5530_pareto" and r["seed"]==seed)
    row={"seed":seed}
    for f in fields:row.update({f+"_candidate":c[f],f+"_ppo":p[f],f+"_delta":p[f]-c[f]})
    paired.append(row)
changes=[]
for seed in [7,17,27]:
    old=next(r for r in rows if r["result_set"].endswith("9") and r["method"]=="ppo_top7_w5530_pareto" and r["seed"]==seed and r["target_tps"]==250 and r["window"]=="validation444k")
    new=next(r for r in rows if r["result_set"].endswith("10") and r["method"]=="ppo_top7_w5530_pareto" and r["seed"]==seed)
    changes.append({"seed":seed,"cross_pp_delta":(new["cross_ratio"]-old["cross_ratio"])*100,
                    "latency_mean_relative":new["latency_mean_sec"]/old["latency_mean_sec"]-1,
                    "latency_p95_relative":new["latency_p95_sec"]/old["latency_p95_sec"]-1})
frozen_hash_checks=[]
for x in freeze["model_selections"]:
    p=Path(x["model_path"])
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    frozen_hash_checks.append({"seed":x["seed"],"model_sha256":digest,"matches":digest==x["model_sha256"].lower()})
summary={"groups":group_rows,"paired_result10":paired,"r9_to_r10_same_model_reruns":changes,"lineage":lineage,
         "queue_and_physical_load":queues,"frozen_model_hash_checks":frozen_hash_checks}
(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"hash_checks":frozen_hash_checks,
                  "dataset_hashes":list(set(x["dataset_sha256"] for x in lineage)),
                  "sidecar_hashes":list(set(x["sidecar_sha256"] for x in lineage)),
                  "unknown_hash_count":sum(not x["dataset_sha256"] for x in lineage),
                  "bad_frozen_hash_count":sum(x["model_matches_frozen"] is False for x in lineage),
                  "dirty_count":sum(x["dirty"] for x in lineage),"queues":[{k:r[k] for k in ["method","seed","target_tps","window","peak_queue","physical_aggregate_jain","physical_aggregate_maxshare"]} for r in queues],
                  "reruns":changes},ensure_ascii=True))
