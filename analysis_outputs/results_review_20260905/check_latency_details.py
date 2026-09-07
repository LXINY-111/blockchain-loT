"""Bounded sequential reads of five completed runs, using one CSV chunk at a time."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
rows=json.loads((OUT/"run_metrics.json").read_text(encoding="utf-8"))
selected=[r for r in rows if r["seed"]==7 and r["target_tps"]==250 and (
    (r["result_set"].endswith("9") and r["window"]=="test2M" and r["method"] in ["ppo_top7_w5530_pareto","hash","nsshard_adapted_cap12"])
    or (r["result_set"].endswith("10") and r["window"]=="validation444k"))]
out=[]
for r in selected:
    path=Path(r["run_path"])/"expTest"/"result"/"supervisor_measureOutput"/"Tx_Details.csv"
    latency=[]; hashes=[]; count=invalid=bad_timestamps=0; offsets={}; parts=[]
    for chunk in pd.read_csv(path,usecols=[0,1,3,8],chunksize=200000):
        numeric=chunk.iloc[:,1:].apply(pd.to_numeric,errors="coerce")
        values=numeric.iloc[:,2].to_numpy(dtype=float)
        valid=np.isfinite(values)&(values>=0)
        count+=len(values); invalid+=int((~valid).sum())
        diff=numeric.iloc[:,1].to_numpy()-numeric.iloc[:,0].to_numpy()-values
        bad_timestamps+=int((~np.isfinite(diff)|(diff<0)|(diff>1)).sum())
        u,c=np.unique(diff,return_counts=True)
        for a,b in zip(u,c):offsets[str(a)]=offsets.get(str(a),0)+int(b)
        latency.append(values[valid]/1000)
        hashes.append(pd.util.hash_pandas_object(chunk.iloc[:,0],index=False).to_numpy())
        parts.append({"first_row":count-len(values),"rows":len(values),"mean_sec":float(np.mean(values[valid]/1000))})
    values=np.concatenate(latency); h=np.concatenate(hashes)
    unique_hashes=len(np.unique(h))
    quant=np.quantile(values,[.5,.95,.99],method="inverted_cdf")
    record={"run":r["run"],"path":str(path),"rows":count,"invalid":invalid,"hash_fingerprint_duplicates":count-unique_hashes,
            "mean_sec":float(np.mean(values)),"p50_sec":float(quant[0]),"p95_sec":float(quant[1]),"p99_sec":float(quant[2]),
            "max_sec":float(np.max(values)),"gt30_sec_ratio":float(np.mean(values>30)),"gt60_sec_ratio":float(np.mean(values>60)),
            "gt120_sec_ratio":float(np.mean(values>120)),"timestamp_outside_rounding_tolerance":bad_timestamps,
            "timestamp_offsets_ms":offsets,"mean_vs_archived_delta":float(np.mean(values))-r["latency_mean_sec"]}
    out.append(record)
    (OUT/"latency_raw_checks.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(record,ensure_ascii=True),flush=True)
    del values,h,latency,hashes
