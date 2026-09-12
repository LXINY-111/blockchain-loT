# 实验结果9：16 分片正式实验流程

> **旧版说明：** 本文档保留用于追溯旧实验，其中的模型名和负载语义已经停用，请勿继续执行。新的正式流程请使用
> [`experiment8_16s_aligned_protocol.md`](./experiment8_16s_aligned_protocol.md)。

## 固定数据划分

- 训练集：有效交易 `[0, 2500000)`，共 2,500,000 笔。
- 验证集：有效交易 `[2500000, 2944019)`，共 444,019 笔，只用于选择最佳检查点。
- 测试集：有效交易 `[2944019, 4944019)`，共 2,000,000 笔，只用于冻结模型后的最终评估。
- 三个窗口都以窗口内冷启动方式重置状态历史；Python 与 Go 使用同一规则。

## 1. 创建结果目录

```powershell
$root = "E:\project_iot\实验结果9"
New-Item -ItemType Directory -Force -Path "$root\models", "$root\offline_training", "$root\offline_eval", "$root\block_eval" | Out-Null
```

## 2. 离线训练并用验证集选择最佳检查点

正式 seed=7 首跑前，确认目标模型和日志不存在，防止覆盖已有结果。

```powershell
Set-Location "E:\project_iot\block-emulator-main-iot"

python .\spring_lite\train_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode iot --tx_identity iot `
  --model "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt" `
  --shards 16 --epochs 15 `
  --start_tx 0 --max_txs 2500000 `
  --validation_start_tx 2500000 --validation_max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --sender_pos_mode 1 `
  --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
  --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
  --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
  --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --eval_every_epochs 1 `
  --seed 7 --device cpu `
  --log_jsonl "E:\project_iot\实验结果9\offline_training\ppo_top7_w5530_16s_seed7_train.jsonl"

if (-not (Test-Path -LiteralPath "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt")) {
  throw "离线训练结束，但最佳检查点不在实验结果9的 models 目录；请先检查训练命令中的输出路径。"
}
```

## 3. Python 独立验证集评估

```powershell
Set-Location "E:\project_iot\block-emulator-main-iot"

if (-not (Test-Path -LiteralPath "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt")) {
  throw "未找到最佳检查点：E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt"
}

python .\spring_lite\eval_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode iot --tx_identity iot `
  --model "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt" `
  --policy ppo --shards 16 `
  --start_tx 2500000 --max_txs 444019 `
  --tx_batch_size 1000 --max_block_size 1000 --sender_pos_mode 1 `
  --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
  --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
  --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
  --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --seed 7 --device cpu `
  --log_jsonl "E:\project_iot\实验结果9\offline_eval\ppo_top7_w5530_16s_seed7_validation.jsonl"
```

## 4. 冻结模型后做 Python 测试集评估

测试集结果不能再用于调 TopK、奖励权重或检查点。

```powershell
python .\spring_lite\eval_offline.py `
  --csv .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --mdp_mode iot --tx_identity iot `
  --model "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt" `
  --policy ppo --shards 16 `
  --start_tx 2944019 --max_txs 2000000 `
  --tx_batch_size 1000 --max_block_size 1000 --sender_pos_mode 1 `
  --reward_mode iot_dense_balanced --lambda_weight 0.5 --beta 0.1 `
  --iot_cstr_weight 0.55 --iot_balance_weight 0.30 `
  --iot_comm_cost_weight 0.10 --iot_hotspot_weight 0.05 `
  --candidate_top_k 7 --capacity_guard 0 --capacity_guard_factor 1.5 `
  --candidate_load_weight 1.0 --seed 7 --device cpu `
  --log_jsonl "E:\project_iot\实验结果9\offline_eval\ppo_top7_w5530_16s_seed7_test2M.jsonl"
```

## 5. 构建并准备 BlockEmulator

```powershell
python -m unittest discover -s spring_lite -p "test_*.py"
go test ./...
go build -o .\blockEmulator_Windows_Precompile.exe .

# 仅对当前 PowerShell 窗口临时允许执行本项目脚本；关闭窗口后自动恢复。
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$run = & .\scripts\prepare_exp8_block_run.ps1
$run | Format-List
```

准备脚本会设置最后 2,000,000 笔测试窗口、生成时间戳目录、快照 `paramsConfig.json`，并保存已有 `spring_io`。

## 6. 写入实验清单并启动 16 分片评估

```powershell
python .\spring_lite\write_experiment_manifest.py `
  --output "$($run.RunRoot)\experiment_manifest.json" `
  --experiment ppo_top7_w5530_16s_test2M_seed7 `
  --phase block_eval --seed 7 --shards 16 `
  --dataset .\data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar .\data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --model "E:\project_iot\实验结果9\models\ppo_top7_w5530_16s_seed7.pt" `
  --params "$($run.RunRoot)\paramsConfig.snapshot.json" `
  --start_tx 2944019 --max_txs 2000000

.\windows_exe_run_IpAddr=127_0_0_1.bat
```

## 7. 归档推理日志并统一分析

所有进程正常结束后执行：

```powershell
Copy-Item -LiteralPath .\spring_io -Destination "$($run.RunRoot)\spring_io" -Recurse
$env:BLOCKEMULATOR_CHECK_FINAL_RESULT = "1"
go test ./query -run TestFinalResult -count=1
Remove-Item Env:\BLOCKEMULATOR_CHECK_FINAL_RESULT
python .\spring_lite\analyze_block_eval.py `
  --result_path "$($run.ExpTest)\result" `
  --spring_io_path "$($run.RunRoot)\spring_io" `
  --shards 16 `
  --output_json "$($run.RunRoot)\analysis.json"
```

正式结果至少核对：`python_ppo_ratio=1`、`fallback_ratio=0`、动作候选数为 7、有效交易总数为 2,000,000，并确认 `paramsConfig.snapshot.json` 中 `DatasetStartTx=2944019`。
