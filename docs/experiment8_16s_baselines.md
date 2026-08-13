# 实验结果9：16 分片正式 baseline 与消融实验手册

## 1. 对照组定义

核心 baseline 共 6 个：

- `hash`：原始 Hash/Monoxide 哈希分片，`SpringMode=0`。
- `heuristic`：SPRING-Heuristic 启发式方法，`SpringMode=1`。
- `original_spring_ppo`：原始 MDP 与 paper reward 的 SPRING-style PPO，`SpringMode=2`。
- `random`：随机放置状态对象，`SpringMode=3`。
- `minstate`：放到当前状态数最少的分片，`SpringMode=4`。
- `nsshard_adapted`：IoT 加权通信图与容量约束的 NSshard-adapted，`SpringMode=6`。

消融实验另有 1 个：

- `anchoronly`：只跟随 IoT 锚点，不加负载惩罚，`SpringMode=5`。

加上最终冻结的 proposed（提出方法），正式 seed=7 主表共有 8 个方案；主方法已经冻结为
`ppo_top7_w5530`。所有对照组固定使用
有效交易 `[2944019, 4944019)`、16 分片、2,000,000 笔交易和窗口内冷启动。

## 2. 只执行一次的构建检查

必须在准备任一 baseline 之前执行。准备脚本会修改 `paramsConfig.json`，所以完整配置
单元测试不要放到准备之后执行。

```powershell
Set-Location "E:\project_iot\block-emulator-main-iot"

python -m unittest discover -s spring_lite -p "test_*.py"
go test ./...
go build -o .\blockEmulator_Windows_Precompile.exe .
```

## 3. 先训练 Original SPRING-PPO

其余 6 个对照项不训练模型。Original SPRING-PPO 必须基于相同训练/验证划分重新训练，
不能直接复用实验结果7中的模型。

论文中只能将其称为“Original SPRING-PPO adapted to our IoT BlockEmulator setting”或
“SPRING-style PPO baseline with original MDP and paper reward”。本项目离线训练并冻结模型，
不能声称完整复现 SPRING 在线 A-Shard 训练协议。

训练和 checkpoint 选择只使用训练集 `[0, 2500000)` 与验证集
`[2500000, 2944019)`。下面的入口会保存 15 个逐 epoch checkpoint，并额外记录可能不同的
final post-update 候选；不得把单一 scalar best 直接当成部署模型。先从 seed7 开始：

```powershell
$ErrorActionPreference = "Stop"
$resultsRoot = "E:\project_iot\实验结果9"
$seed = 7
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trainRun = "$resultsRoot\offline_training\original_spring_ppo_pareto_16s_seed${seed}_$stamp"

if (
  (Test-Path "$resultsRoot\models\original_spring_ppo_16s_seed${seed}.pt") -or
  (Test-Path "$resultsRoot\models\original_spring_ppo_16s_seed${seed}_selection.json")
) {
  throw "Frozen model already exists for seed $seed; do not overwrite it."
}
if (-not (Test-Path .\data_iot\selectedTxs_iot_multi_anchor_full.csv)) {
  throw "Dataset is missing."
}
if (-not (Test-Path .\data_iot\iot_flow_sidecar_multi_anchor_full.csv)) {
  throw "IoT sidecar is missing."
}
New-Item -ItemType Directory -Path $trainRun | Out-Null

[ordered]@{
  schema_version = 1
  created_at = (Get-Date).ToString("o")
  method_label = "Original SPRING-PPO adapted to our IoT BlockEmulator setting"
  seed = $seed
  shards = 16
  state_dim = 177
  iot_feature_dim = 0
  mdp_mode = "spring"
  tx_identity = "iot"
  reward_mode = "paper"
  lambda_weight = 0.5
  beta = 0.1
  candidate_top_k = 0
  capacity_guard = 0
  capacity_guard_factor = 1.5
  train_window = @{ start_tx = 0; max_txs = 2500000 }
  validation_window = @{ start_tx = 2500000; max_txs = 444019 }
  test_window_accessed = $false
} | ConvertTo-Json -Depth 6 |
  Set-Content "$trainRun\run_context.json" -Encoding UTF8

python -u .\spring_lite\train_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode spring --tx_identity iot `
  --model "$trainRun\best_scalar.pt" `
  --last_model "$trainRun\last.pt" `
  --epoch_checkpoint_dir "$trainRun\checkpoints" `
  --pareto_manifest "$trainRun\pareto_manifest.json" `
  --shards 16 --epochs 15 `
  --start_tx 0 --max_txs 2500000 `
  --validation_start_tx 2500000 --validation_max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --block_interval_ms 5000 `
  --sender_pos_mode 1 `
  --temporal_top_k 8 `
  --reward_mode paper --lambda_weight 0.5 --beta 0.1 `
  --candidate_top_k 0 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --eval_every_epochs 1 `
  --seed $seed --device cpu `
  --log_jsonl "$trainRun\train.jsonl" 2>&1 |
    Tee-Object "$trainRun\console.log"

if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }

$epochCheckpoints = @(Get-ChildItem "$trainRun\checkpoints\epoch_*.pt" -File)
if ($epochCheckpoints.Count -ne 15) {
  throw "Incomplete training audit chain: expected 15 epoch checkpoints, got $($epochCheckpoints.Count)."
}
$manifest = Get-Content "$trainRun\pareto_manifest.json" -Raw | ConvertFrom-Json
if (@($manifest.all_checkpoints).Count -ne 16) {
  throw "Expected 15 epoch candidates plus one final post-update candidate."
}
```

按已经冻结的分层规则选择 checkpoint。严格层要求
`active_shards_mean >= 8` 且 `stage_max_load_share <= 0.35`；严格层为空时，容差层只允许
活跃分片相对缺口不超过 5%，热点上限仍为 0.35。容差层仍为空时命令必须失败，不得继续放宽：

```powershell
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  # Windows PowerShell 5.1 converts native stderr into ErrorRecord objects.
  # Capture first and inspect Python's exit code before restoring Stop mode.
  $selectionOutput = python .\spring_lite\select_pareto_checkpoint.py `
    --manifest "$trainRun\pareto_manifest.json" `
    --output_model "$resultsRoot\models\original_spring_ppo_16s_seed${seed}.pt" `
    --selection_json "$resultsRoot\models\original_spring_ppo_16s_seed${seed}_selection.json" `
    --min_active_shards 8 `
    --max_stage_hotspot 0.35 `
    --max_active_deficit_ratio 0.05 2>&1
  $selectionExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $previousErrorActionPreference
}
$selectionOutput |
  Tee-Object "$trainRun\hierarchical_selection_stdout.log"

if ($selectionExitCode -ne 0) {
  $validationRecords = @(Get-Content "$trainRun\train.jsonl" |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Where-Object { $_.event -in @("validation", "final_validation") })
  $actionByCandidate = @{}
  foreach ($record in $validationRecords) {
    $actionByCandidate["$($record.event)|$($record.epoch)"] = @(
      $record.summary.action_dist
    )
  }
  $candidateAudit = @($manifest.all_checkpoints | ForEach-Object {
    $active = [double]$_.active_shards_mean
    $hotspot = [double]$_.stage_max_load_share
    [ordered]@{
      event = $_.event
      epoch = $_.epoch
      cross_ratio = [double]$_.cross_ratio
      active_shards_mean = $active
      active_deficit_ratio = [math]::Max([double]0.0, (8.0 - $active) / 8.0)
      stage_max_load_share = $hotspot
      strict_eligible = ($active -ge 8.0 -and $hotspot -le 0.35)
      tolerance_eligible = (
        $hotspot -le 0.35 -and
        [math]::Max([double]0.0, (8.0 - $active) / 8.0) -le 0.05
      )
      action_dist = $actionByCandidate["$($_.event)|$($_.epoch)"]
      checkpoint_path = $_.checkpoint_path
    }
  })
  [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString("o")
    status = "selection_failed"
    method_label = "Original SPRING-PPO adapted to our IoT BlockEmulator setting"
    seed = $seed
    selection_policy = "hierarchical_constrained_pareto_v1"
    constraints = [ordered]@{
      min_active_shards = 8.0
      max_stage_hotspot = 0.35
      max_active_deficit_ratio = 0.05
    }
    strict_candidate_count = @($candidateAudit | Where-Object strict_eligible).Count
    tolerance_candidate_count = @($candidateAudit | Where-Object tolerance_eligible).Count
    reason = "No checkpoint satisfies the frozen strict or active-tolerance layer."
    test_window_accessed = $false
    candidates = $candidateAudit
  } | ConvertTo-Json -Depth 12 |
    Set-Content "$trainRun\selection_failed.json" -Encoding UTF8
  $candidateAudit |
    Select-Object event,epoch,cross_ratio,active_shards_mean,active_deficit_ratio,stage_max_load_share,action_dist |
    Format-Table -Wrap -AutoSize
  throw "Checkpoint selection failed for seed $seed; do not relax constraints or continue to independent validation."
}

$selection = Get-Content `
  "$resultsRoot\models\original_spring_ppo_16s_seed${seed}_selection.json" `
  -Raw | ConvertFrom-Json
$selection | Select-Object selection_tier,strict_feasible,constraint_audit,selected |
  Format-List
```

选择成功后，只在验证集上独立复核冻结模型；本阶段不得读取正式测试窗口：

```powershell
$selectedEpoch = [int]$selection.selected.epoch
$selectedEvent = [string]$selection.selected.event
$evalStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalRoot = "$resultsRoot\offline_eval\original_spring_ppo_${selectedEvent}_epoch${selectedEpoch}_16s_seed${seed}_validation_$evalStamp"
New-Item -ItemType Directory -Force -Path $evalRoot | Out-Null
Copy-Item `
  "$resultsRoot\models\original_spring_ppo_16s_seed${seed}_selection.json" `
  "$evalRoot\selection.json"

python -u .\spring_lite\eval_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode spring --tx_identity iot `
  --model "$resultsRoot\models\original_spring_ppo_16s_seed${seed}.pt" `
  --policy ppo --shards 16 `
  --start_tx 2500000 --max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --block_interval_ms 5000 `
  --sender_pos_mode 1 `
  --temporal_top_k 8 `
  --reward_mode paper --lambda_weight 0.5 --beta 0.1 `
  --candidate_top_k 0 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --seed $seed --device cpu `
  --log_jsonl "$evalRoot\validation.jsonl" 2>&1 |
    Tee-Object "$evalRoot\console.log"

if ($LASTEXITCODE -ne 0) { throw "Validation failed for seed $seed" }

$validation = Get-Content "$evalRoot\validation.jsonl" -Tail 1 |
  ConvertFrom-Json
if (
  $validation.loaded_txs -ne 444019 -or
  $validation.dataset_start_tx -ne 2500000 -or
  $validation.dataset_end_tx_exclusive -ne 2944019 -or
  $validation.state_dim -ne 177 -or
  $validation.iot_feature_dim -ne 0 -or
  $validation.candidate_top_k -ne 0 -or
  $validation.capacity_guard -ne 0 -or
  $validation.sample
) {
  throw "Validation protocol audit failed for seed $seed."
}
$validation.summary | Select-Object `
  cross_ratio,active_shards_mean,stage_max_load_share,action_dist |
  Format-List
```

seed7 完成并通过审计后，将 `$seed` 依次改为 `17`、`27`，其余训练、选择和验证参数
必须完全相同。三个种子不得并行占用同一机器的正式系统评估资源。

若 seed7 在冻结选择规则下失败，保留完整训练目录并停止 seed7 的独立验证。项目提供的
剩余种子入口只接受 seed17 和 seed27；它会在选择失败时写出 `selection_failed.json`，
成功时才继续独立验证。两次必须串行执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_phase6_original_spring_offline.ps1 `
  -Seed 17 -PreflightOnly
if ($LASTEXITCODE -ne 0) { throw "Seed17 preflight failed." }

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_phase6_original_spring_offline.ps1 -Seed 17
if ($LASTEXITCODE -notin @(0, 2)) { throw "Seed17 execution error." }

# 只有 seed17 进程完全结束后才能启动 seed27。退出码 2 表示冻结规则下选择失败，
# 不是训练过程损坏；失败审计已经保存在该 seed 的训练目录。
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_phase6_original_spring_offline.ps1 -Seed 27
if ($LASTEXITCODE -notin @(0, 2)) { throw "Seed27 execution error." }
```

退出码 `0` 表示 checkpoint 选择和独立验证均成功；退出码 `2` 表示训练完整，但严格层与
容差层均无候选。不得为了消除退出码 `2` 而替换种子、使用 `best_scalar.pt`、放宽阈值或
读取正式测试窗口。

## 4. 每个 BlockEmulator 实验的公共命令

在同一个 PowerShell 窗口先定义三个辅助函数。`Window` 和 `InjectSpeed`
必须显式传入，防止误用验证窗口或旧的 1000 TPS 过载配置：

```powershell
function Prepare-Exp8Run {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
      "hash",
      "heuristic",
      "original_spring_ppo",
      "random",
      "minstate",
      "anchoronly",
      "nsshard_adapted"
    )]
    [string]$Scheme,
    [Parameter(Mandatory = $true)]
    [ValidateSet("validation", "test")]
    [string]$Window,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000000)]
    [int]$InjectSpeed,
    [int]$Seed = 7
  )
  $json = & python .\spring_lite\prepare_exp8_baseline.py `
    --scheme $Scheme `
    --seed $Seed `
    --window $Window `
    --inject_speed $InjectSpeed
  if ($LASTEXITCODE -ne 0) { throw "准备实验失败：$Scheme" }
  return ($json | ConvertFrom-Json)
}

function Start-Exp8Run {
  param([Parameter(Mandatory = $true)]$Run)
  & powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\run_exp8_16s.ps1 -RunRoot $Run.run_root
  if ($LASTEXITCODE -ne 0) { throw "启动实验失败：$($Run.scheme)" }
}

function Finish-Exp8Run {
  param([Parameter(Mandatory = $true)]$Run)
  $running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -like "blockEmulator*" }
  if ($running) { throw "BlockEmulator 尚未全部退出，暂不能归档。" }
  $json = & python .\spring_lite\finalize_exp8_block_run.py `
    --run_root $Run.run_root
  if ($LASTEXITCODE -ne 0) { throw "归档或结果校验失败：$($Run.scheme)" }
  return ($json | ConvertFrom-Json)
}
```

正式 seed=7 主比较统一定义为：

```powershell
$formalWindow = "test"
$formalRate = 250
$formalSeed = 7
```

启动后可在另一个 PowerShell 窗口查看 supervisor 输出：

```powershell
Get-Content -LiteralPath "$($run.run_root)\process_logs\supervisor.stdout.log" `
  -Tail 30 -Wait
```

`Ctrl+C` 只停止日志查看，不会停止 BlockEmulator。必须等待所有 BlockEmulator 进程自然结束，
再执行 `Finish-Exp8Run`。收尾工具会归档 `spring_io`、运行最终链状态检查并生成
`analysis.json`。

## 5. 七个对照项的逐项命令

每项必须完整执行准备、启动、等待结束、收尾后，才能开始下一项。建议顺序如下。

### 5.1 Hash

```powershell
$run = Prepare-Exp8Run `
  -Scheme "hash" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.2 Random

```powershell
$run = Prepare-Exp8Run `
  -Scheme "random" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.3 MinState

```powershell
$run = Prepare-Exp8Run `
  -Scheme "minstate" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.4 Heuristic

```powershell
$run = Prepare-Exp8Run `
  -Scheme "heuristic" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.5 NSshard-adapted

```powershell
$run = Prepare-Exp8Run `
  -Scheme "nsshard_adapted" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.6 AnchorOnly 消融

```powershell
$run = Prepare-Exp8Run `
  -Scheme "anchoronly" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

### 5.7 Original SPRING-PPO

确认第 3 节生成的模型存在后执行：

```powershell
$run = Prepare-Exp8Run `
  -Scheme "original_spring_ppo" `
  -Window $formalWindow `
  -InjectSpeed $formalRate `
  -Seed $formalSeed
Start-Exp8Run $run
# 等待全部 BlockEmulator 进程自然结束
$result = Finish-Exp8Run $run
$result.all_active | Format-List
```

## 6. 每轮必须核对

- `effective_total=2000000`：有效交易完整。
- `DatasetStartTx=2944019`：确实使用最后 200 万测试窗口。
- `state_check=passed`：最终链状态、状态根和账户唯一性检查通过。
- PPO 方案还需满足 `python_ppo_ratio=1`、`fallback_ratio=0`。
- Original SPRING-PPO 的 `action_mask_size_mean` 应为 16；提出的方法应等于书面冻结的
  `CandidateTopK`，当前候选值为 7。
- 正式比较只使用 `analysis.json -> periods -> all_active` 的统一口径。

第一轮 seed=7 是完整性与趋势确认。论文正式表格还应至少补 3 个独立运行种子并报告
均值、标准差或 95% 置信区间；不同方案不可并行运行在同一台机器上。
