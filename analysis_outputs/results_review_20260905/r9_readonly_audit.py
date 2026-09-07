"""Read completed Result9 archived CSV/JSON only; write audit outputs here."""
from pathlib import Path
import json, re, csv, math
import numpy as np
import pandas as pd

ROOT = Path(r'E:\project_iot\实验结果9\block_eval')
OUT = Path(__file__).parent

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def jain(x):
    x = np.asarray(x, dtype=float)
    return float(x.sum() ** 2 / (len(x) * (x*x).sum())) if x.sum() else None

runs=[]
for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue
    if not (d/'run_complete.json').is_file():
        runs.append({'run':d.name,'status':'unfinished_presence_only'})
        continue
    a=read_json(d/'analysis.json')
    c=read_json(d/'run_complete.json')
    ctx=read_json(d/'run_context.json')
    m=read_json(d/'experiment_manifest.json')
    state=(d/'state_check.txt').read_text(encoding='utf-8-sig')
    q=d/'expTest'/'result'/'supervisor_measureOutput'
    cross=pd.read_csv(q/'CrossTransaction_ratio.csv')
    tps=pd.read_csv(q/'Average_TPS.csv')
    load=pd.read_csv(q/'Shard_Load_Variance.csv')
    active=cross.iloc[:,1]>0
    first,last=int(cross.loc[active].iloc[0,0]),int(cross.loc[active].iloc[-1,0])
    loads=load[load.iloc[:,0].isin(cross.loc[active].iloc[:,0])].iloc[:,2:].to_numpy(float)
    totals=loads.sum(axis=0)
    epoch_sum=loads.sum(axis=1)
    jj=epoch_sum**2/(16*(loads**2).sum(axis=1))
    xx=loads.max(axis=1)/epoch_sum
    tps_active=tps[tps.iloc[:,1]>0]
    total=float(cross.iloc[:,1].sum())
    raw_cross=float(cross.iloc[:,2].sum()/total)
    wall=float(total/((tps_active.iloc[:,6].max()-tps_active.iloc[:,5].min())/1000))
    placement=a.get('send_log',{}).get('tx_placement_totals',[])
    acc={key: re.search(pattern,state).group(1) if re.search(pattern,state) else None for key,pattern in {
        'dataset_accounts':r'Results from dataset file: # of accounts (\d+)',
        'correct_accounts':r'# of correct accounts (\d+)',
        'found_accounts':r'# of found accounts (\d+)',
        'duplicate_accounts':r'# of duplicate accounts (\d+)'
    }.items()}
    row={'run':d.name,'path':str(d),'scheme':ctx['scheme'],'seed':ctx['seed'],
        'phase':ctx['phase'],'inject_speed':ctx['inject_speed'],'expected_tx':ctx['total_data_size'],
        'data_start':ctx['dataset_start_tx'],'actual_state_pass':'--- PASS: TestFinalResult' in state and 'test pass' in state,
        'state_accounts':acc,'declared_state':c.get('state_check'),
        'effective_total_raw':total,'normal_raw':float(cross.iloc[:,3].sum()),
        'relay1_raw':float(cross.iloc[:,4].sum()),'relay2_raw':float(cross.iloc[:,5].sum()),
        'cross_ratio_raw':raw_cross,'wall_tps_raw':wall,'wall_tps_analysis':a['periods']['all_active']['wall_tps'],
        'load_total_raw':float(totals.sum()),'load_jain_aggregate':jain(totals),
        'load_jain_epoch_mean':float(jj.mean()),'load_max_share_aggregate':float(totals.max()/totals.sum()),
        'load_max_share_epoch_mean':float(xx.mean()),'load_max_share_epoch_p95':float(np.percentile(xx,95)),
        'load_totals_by_shard':totals.tolist(),'zero_total_load_shards':int((totals==0).sum()),
        'mean_load_variance':float(load[load.iloc[:,0].isin(cross.loc[active].iloc[:,0])].iloc[:,1].mean()),
        'placement_jain':jain(placement) if placement else None,
        'placement_max_share':max(placement)/sum(placement) if placement and sum(placement) else None,
        'placement_sum':sum(placement),'analysis_latency':a['latency_details'],
        'analysis_offered':a.get('injection_pace'),
        'cross_analysis_diff':raw_cross-a['periods']['all_active']['weighted_cross_ratio'],
        'fallback_ratio':a.get('decisions',{}).get('fallback_ratio'),
        'python_ppo_ratio':a.get('decisions',{}).get('python_ppo_ratio'),
        'dataset_hash':m.get('files',{}).get('dataset',{}).get('sha256'),
        'sidecar_hash':m.get('files',{}).get('sidecar',{}).get('sha256'),
        'model_hash':m.get('files',{}).get('model',{}).get('sha256'),
        'git_commit':m.get('git',{}).get('commit'),'git_dirty':bool(m.get('git',{}).get('status_porcelain')),
        'params':m.get('params_snapshot'),
        'first_epoch':first,'last_epoch':last,'active_epochs':int(active.sum()),
    }
    if ctx['phase']=='test':
        # Numeric columns only; strictly sequential reads limit memory and IO.
        tx=pd.read_csv(q/'Tx_Details.csv',usecols=[1,3,8])
        latency=tx.iloc[:,2].to_numpy(dtype=float)
        valid=np.isfinite(latency)&(latency>=0)
        v=latency[valid]/1000
        diff=tx.iloc[:,1]-tx.iloc[:,0]-tx.iloc[:,2]
        row['raw_latency']={'rows':len(tx),'valid':int(valid.sum()),'invalid':int((~valid).sum()),
            'coverage':len(tx)/ctx['total_data_size'],'mean':float(v.mean()),
            'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),
            'p99':float(np.percentile(v,99)),'max':float(v.max()),
            'timestamp_latency_mismatches':int((diff!=0).sum()),
            'gt30sec':int((v>30).sum()),'gt60sec':int((v>60).sum()),'gt120sec':int((v>120).sum())}
        del tx
        print(d.name, 'raw latency checked', row['raw_latency'],flush=True)
    runs.append(row)

(OUT/'notes_r9.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
fields=['run','scheme','phase','seed','inject_speed','actual_state_pass','effective_total_raw','cross_ratio_raw','wall_tps_raw','load_jain_aggregate','load_jain_epoch_mean','load_max_share_aggregate','load_max_share_epoch_mean','load_max_share_epoch_p95','mean_load_variance','placement_jain','placement_max_share']
with (OUT/'r9_block_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(runs)
print('SAVED',len(runs),'completed runs')
