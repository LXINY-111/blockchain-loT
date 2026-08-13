# 实验结果9：16分片主锚点负载对齐实验手册

本手册对应负载语义版本 `primary_owner_execution_v1`。旧模型
`ppo_top7_w5530_16s_seed7.pt` 只作为历史先导结果保留，不能与本手册的
对齐模型混用。

## 1. 固定数据窗口

- 训练集：`[0, 2500000)`，共 2,500,000 笔有效交易。
- 验证集：`[2500000, 2944019)`，共 444,019 笔有效交易。
- 测试集：`[2944019, 4944019)`，共 2,000,000 笔有效交易。
- 训练、checkpoint 选择、TopK、奖励权重和注入速率标定只能使用训练集与验证集。
- 测试集只能在所有超参数和统一注入速率冻结后运行，不能反复查看后调参。

方法学说明：2026-07-24 的旧模型曾运行相同的最后 2,000,000 笔窗口，并用于发现
主锚点热点。当前 Pareto 模型尚未运行该窗口，但该窗口不能再称为“全项目从未观察
过的严格盲测集”。论文中应称为固定时间顺序测试窗口；若要报告严格盲测结果，应再
增加一个此前未观察过的时间窗口或外部数据集。

## 2. 当前候选模型

- 模型：`E:\project_iot\实验结果9\models\ppo_top7_w5530_pareto_16s_seed7.pt`
- SHA-256：`81CC835E7CF111150B5B528AFDBA6750B83B7EC0921BE682F308025ED3207620`
- 来源：15 个逐 epoch checkpoint 的 Pareto 前沿。
- 一级严格约束：平均活跃分片数不低于 8，系统阶段最大负载占比不高于 35%。
- 二级有界容差：仅当一级无候选时，阶段热点上限仍为 35%，活跃分片阈值最多
  允许 5% 相对缺口；超过该容差仍判定选择失败。
- 当前选择：epoch 12；验证集跨片率 66.2794%，系统阶段最大负载占比 33.2508%。

这是已完成系统标定的候选模型，不代表 TopK 与奖励权重已经最终冻结。完成
TopK6/7/8 和 w50-35/w55-30/w60-25 验证筛选后，才确定正式主方法。

这里的 Pareto 选择替代“只按单一 validation score 选择 best checkpoint”的做法。
固定主锚点占比 `owner_max_load_share` 是数据窗口属性，不能被分配策略直接改变；
模型选择使用可被策略影响的 `stage_max_load_share`。
所有 PPO 随机种子必须统一使用 `hierarchical_constrained_pareto_v1` 规则。选择记录中的
`selection_tier` 必须区分 `strict`（严格层）与 `active_tolerance`（活跃分片容差层），
不能把二级选择写成满足了一级严格约束。

## 3. 构建与测试

```powershell
Set-Location "E:\project_iot\block-emulator-main-iot"
python -m unittest discover -s spring_lite -p "test_*.py"
go test ./...
go build -o .\blockEmulator_Windows_Precompile.exe .
```

## 4. 离线训练与逐 epoch 归档

下面是可复现实验模板。每次重跑必须使用新的 `$trainRun`，不能覆盖已有目录。

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trainRun = "E:\project_iot\实验结果9\offline_training\ppo_top7_w5530_pareto_16s_seed7_$stamp"

python .\spring_lite\train_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode iot --tx_identity iot `
  --model "$trainRun\best_scalar.pt" `
  --last_model "$trainRun\last.pt" `
  --epoch_checkpoint_dir "$trainRun\checkpoints" `
  --pareto_manifest "$trainRun\pareto_manifest.json" `
  --shards 16 --epochs 15 `
  --start_tx 0 --max_txs 2500000 `
  --validation_start_tx 2500000 --validation_max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --block_interval_ms 5000 `
  --sender_pos_mode 1 `
  --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
  --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
  --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
  --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --eval_every_epochs 1 `
  --seed 7 --device cpu `
  --log_jsonl "$trainRun\train.jsonl"
```

## 5. Pareto checkpoint 选择

```powershell
python .\spring_lite\select_pareto_checkpoint.py `
  --manifest "$trainRun\pareto_manifest.json" `
  --output_model "E:\project_iot\实验结果9\models\ppo_top7_w5530_pareto_16s_seed7.pt" `
  --selection_json "E:\project_iot\实验结果9\models\ppo_top7_w5530_pareto_16s_seed7_selection.json" `
  --min_active_shards 8 `
  --max_stage_hotspot 0.35 `
  --max_active_deficit_ratio 0.05
```

选择顺序固定如下：

1. 一级严格层：同时满足 `active_shards_mean >= 8` 和
   `stage_max_load_share <= 0.35`，然后选择实际跨片率最低者。
2. 一级为空时进入二级容差层：热点上限不放宽，且
   `(8 - active_shards_mean) / 8 <= 0.05`；先选择活跃分片缺口最小者，再比较
   跨片率、热点、epoch 和候选事件名。
3. 二级仍为空时明确失败，不继续放宽约束。

现有 seed7、seed17、seed27 已保存完整 checkpoint，无需重新训练。统一重选前先归档旧的
稳定模型与选择记录：

```powershell
$resultsRoot = "E:\project_iot\实验结果9"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$selectionArchive = "$resultsRoot\models\archive\before_hierarchical_$stamp"
New-Item -ItemType Directory -Force -Path $selectionArchive | Out-Null

Get-ChildItem "$resultsRoot\models" -File |
  Where-Object Name -Like "ppo_top7_w5530_pareto_16s_seed*" |
  Copy-Item -Destination $selectionArchive

foreach ($seed in 7,17,27) {
  $trainRun = Get-ChildItem "$resultsRoot\offline_training" -Directory |
    Where-Object Name -Like "ppo_top7_w5530_pareto_16s_seed${seed}_*" |
    Where-Object { Test-Path (Join-Path $_.FullName "pareto_manifest.json") } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $trainRun) { throw "Missing training run for seed $seed" }

  python .\spring_lite\select_pareto_checkpoint.py `
    --manifest "$($trainRun.FullName)\pareto_manifest.json" `
    --output_model "$resultsRoot\models\ppo_top7_w5530_pareto_16s_seed${seed}.pt" `
    --selection_json "$resultsRoot\models\ppo_top7_w5530_pareto_16s_seed${seed}_selection.json" `
    --min_active_shards 8 `
    --max_stage_hotspot 0.35 `
    --max_active_deficit_ratio 0.05 2>&1 |
      Tee-Object "$($trainRun.FullName)\hierarchical_selection_stdout.log"
  if ($LASTEXITCODE -ne 0) { throw "Selection failed for seed $seed" }
}
```

## 6. Python 独立验证

```powershell
python .\spring_lite\eval_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode iot --tx_identity iot `
  --model "E:\project_iot\实验结果9\models\ppo_top7_w5530_pareto_16s_seed7.pt" `
  --policy ppo --shards 16 `
  --start_tx 2500000 --max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --block_interval_ms 5000 `
  --sender_pos_mode 1 `
  --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
  --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
  --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
  --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --seed 7 --device cpu `
  --log_jsonl "E:\project_iot\实验结果9\offline_eval\ppo_top7_w5530_pareto_16s_seed7_validation.jsonl"
```

### 6.1 三个随机种子的统一独立验证与归档

完成第 5 节的统一重选后，使用下面的循环分别验证三个冻结模型。每个目录都保存模型
哈希、选择记录、完整终端输出、JSONL 指标和完成摘要；不会读取最后 2,000,000 笔
测试窗口。

```powershell
$resultsRoot = "E:\project_iot\实验结果9"
$evalRows = foreach ($seed in 7,17,27) {
  $model = "$resultsRoot\models\ppo_top7_w5530_pareto_16s_seed${seed}.pt"
  $selectionPath = "$resultsRoot\models\ppo_top7_w5530_pareto_16s_seed${seed}_selection.json"
  $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
  $epoch = [int]$selection.selected.epoch
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $evalRoot = "$resultsRoot\offline_eval\ppo_top7_w5530_pareto_epoch${epoch}_16s_seed${seed}_validation_hierarchical_$stamp"
  New-Item -ItemType Directory -Force -Path $evalRoot | Out-Null
  Copy-Item $selectionPath "$evalRoot\selection.json"

  $modelHash = (Get-FileHash -Algorithm SHA256 $model).Hash
  [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString("o")
    experiment = Split-Path $evalRoot -Leaf
    model = $model
    model_sha256 = $modelHash
    selection_record = $selectionPath
    selection_tier = $selection.selection_tier
    strict_feasible = $selection.strict_feasible
    window = [ordered]@{ start_tx = 2500000; max_txs = 444019 }
  } | ConvertTo-Json -Depth 8 |
    Set-Content "$evalRoot\run_context.json" -Encoding UTF8

  python .\spring_lite\eval_offline.py `
    --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
    --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
    --mdp_mode iot --tx_identity iot `
    --model $model --policy ppo --shards 16 `
    --start_tx 2500000 --max_txs 444019 `
    --tx_batch_size 1000 --max_block_size 1000 --block_interval_ms 5000 `
    --sender_pos_mode 1 `
    --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
    --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
    --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
    --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
    --candidate_load_weight 1.0 --seed $seed --device cpu `
    --log_jsonl "$evalRoot\eval.jsonl" 2>&1 |
      Tee-Object "$evalRoot\stdout.log"
  if ($LASTEXITCODE -ne 0) { throw "Offline evaluation failed for seed $seed" }

  $summaryRecord = Get-Content "$evalRoot\eval.jsonl" |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Where-Object { $null -ne $_.summary } |
    Select-Object -Last 1
  if (-not $summaryRecord) { throw "Missing evaluation summary for seed $seed" }

  [ordered]@{
    schema_version = 1
    completed_at = (Get-Date).ToString("o")
    status = "passed"
    model = $model
    model_sha256 = $modelHash
    selection_tier = $selection.selection_tier
    strict_feasible = $selection.strict_feasible
    summary = $summaryRecord.summary
  } | ConvertTo-Json -Depth 12 |
    Set-Content "$evalRoot\run_complete.json" -Encoding UTF8

  [PSCustomObject]@{
    Seed = $seed
    Tier = $selection.selection_tier
    Epoch = $epoch
    CrossPct = [math]::Round(100 * $summaryRecord.summary.cross_ratio, 4)
    ActiveShards = [math]::Round($summaryRecord.summary.active_shards_mean, 4)
    StageHotspotPct = [math]::Round(100 * $summaryRecord.summary.stage_max_load_share, 4)
    EvalRoot = $evalRoot
  }
}
$evalRows | Format-Table -AutoSize
```

必须核对：

- `load_semantics_version=primary_owner_execution_v1`
- `cross_ratio`：按真实主锚点执行语义统计的跨片率。
- `relation_cross_ratio`：多锚点关系图的跨片率。
- `owner_max_load_share`：验证窗口固有的主锚点热点。
- `stage_max_load_share`：加入发送端与跨片中继后的可控阶段热点。

### 6.1 TopK7 奖励权重筛选

TopK6/7/8 的 Python 验证筛选完成后固定 `candidate_top_k=7`。通信成本权重保持
0.10，热点权重保持 0.05，只比较 `w50-35`、`w55-30` 和 `w60-25`。其中
`w55-30` 已按同一协议完成，因此不重复训练；下面的入口顺序补跑另外两组，自动执行
15 epoch 训练、Pareto checkpoint 选择、冻结模型独立验证和三组结果汇总：

```powershell
python .\spring_lite\run_weight_screen.py `
  --results_root "E:\project_iot\实验结果9" `
  --weights w5035 w6025 `
  --seed 7 `
  --epochs 15
```

训练器保存 15 个逐 epoch checkpoint。若最后仍有不足一个 PPO batch 的样本，还会
执行一次 final update，并把更新后的 `last.pt` 作为额外 `final_validation` 候选写入
Pareto 清单。因此清单包含 15 个 `validation` 候选和 1 个可能不同的 final 候选；
选择约束和排序规则对它们完全相同。

2026-07-31 的验证结果：

| 权重 | 选中 epoch | 实际跨片率 | 关系跨片率 | 活跃分片 | 阶段热点 |
|---|---:|---:|---:|---:|---:|
| w50-35 | 1 | 70.0875% | 57.1200% | 8.4135 | 32.5060% |
| w55-30 | 12 | **66.2794%** | 62.9900% | 8.1169 | 33.2508% |
| w60-25 | 1 | 73.8313% | 56.1643% | 8.8584 | 31.8154% |

三组均满足活跃分片和阶段热点约束。`w55-30` 的实际跨片率最低，因此当前奖励权重
筛选保留 `w55-30`。本阶段只使用训练集和验证集，没有运行最后 2M 测试窗口或
BlockEmulator。

## 7. BlockEmulator 验证集速率标定

修复后的注入器采用累计单调时钟节拍，不再受 `TxBatchSize=1000` 的整秒取整影响。
准备入口会根据实际参数自动生成：

- 方案名：`ppo_top{K}_w{跨片权重百分数}{均衡权重百分数}_pareto`
- 默认模型名：`{方案名}_16s_seed{seed}.pt`
- 归档名：同时包含方案名、数据窗口、注入速率和随机种子

当前小规模网格固定通信成本权重为 0.10、热点权重为 0.05，只开放
`CandidateTopK`、`IOTCSTRWeight` 和 `IOTBalanceWeight`。四项奖励权重总和必须为 1。
下面显式运行 TopK7-w5530 的 200、250、300、350 TPS：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

.\scripts\run_exp8_validation_rates.ps1 `
  -ModelPath "E:\project_iot\实验结果9\models\ppo_top7_w5530_pareto_16s_seed7.pt" `
  -Rates 200,250,300,350 `
  -Seed 7 `
  -CandidateTopK 7 `
  -IOTCSTRWeight 0.55 `
  -IOTBalanceWeight 0.30
```

例如，TopK6-w50-35 在 250 TPS 验证窗口的单次准备命令是：

```powershell
$run = & .\scripts\prepare_exp8_block_run.ps1 `
  -Window validation `
  -InjectSpeed 250 `
  -Seed 7 `
  -CandidateTopK 6 `
  -IOTCSTRWeight 0.50 `
  -IOTBalanceWeight 0.35 `
  -ModelPath "E:\project_iot\实验结果9\models\ppo_top6_w5035_pareto_16s_seed7.pt"
```

准备后必须先核对 `$run.Scheme`、`$run.CandidateTopK` 和两个权重，再调用
`run_exp8_16s.ps1`。模型不存在或配置与 checkpoint 不兼容时不得启动。

2026-07-28 至 2026-07-29 的 seed 7 标定结果：

| 目标 TPS | 实际注入 TPS | 完整 TPS | 跨片率 | 平均延迟 | P95 延迟 |
|---:|---:|---:|---:|---:|---:|
| 200 | 199.9999 | 199.479 | 66.355% | 6.438 s | 11.756 s |
| 250 | 249.9999 | 249.178 | 66.425% | 5.967 s | 9.849 s |
| 300 | 299.9999 | 298.917 | 66.429% | 5.876 s | 9.761 s |
| 350 | 349.9998 | 348.226 | 66.424% | 7.431 s | 20.168 s |

结论：

- 300 TPS 是当前验证集上的最大稳定标定点。
- 350 TPS 已出现尾延迟拐点，不能作为默认负载。
- 250 TPS 作为保守的同速率主对比点；300 TPS 可作为高负载补充对比点。
- 所有方案必须使用相同数据窗口和相同注入速率，不能为每个方法选择有利速率。

## 8. 后续顺序

1. TopK 与奖励权重的 seed 7 Python 验证筛选已经完成，当前候选为 TopK7-w55-30。
2. seed 17 和 seed 27 的训练已经完成；按两级 Pareto 规则统一重选 seed7/17/27，
   再分别运行 Python 独立验证，并记录严格可行率、均值与标准差。
3. 先用验证窗口确认三个冻结模型没有动作坍塌；不得根据最后测试窗口修改选择规则。
4. 所有选择完成后，才允许对最后 2,000,000 笔测试集运行一次正式评估。
5. 正式 proposed 与所有 baseline 使用同一个冻结速率和测试窗口。

当前最后 2,000,000 笔测试集尚未使用本模型运行。旧模型对同一窗口的历史运行目录是
`ppo_top7_w5530_16s_test2M_seed7_20260724_105413`，只能作为诊断记录。不要执行下面
命令，直到步骤 1 和步骤 2 完成并书面冻结配置：

```powershell
$formalRate = 250
$run = & .\scripts\prepare_exp8_block_run.ps1 `
  -Window test `
  -InjectSpeed $formalRate `
  -Seed 7 `
  -CandidateTopK 7 `
  -IOTCSTRWeight 0.55 `
  -IOTBalanceWeight 0.30
$started = & .\scripts\run_exp8_16s.ps1 -RunRoot $run.RunRoot
```
