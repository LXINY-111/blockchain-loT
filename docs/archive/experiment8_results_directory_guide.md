# 实验结果8目录与新增机制说明

> 历史审计说明：本文记录迁移前的 Result8 目录状态，所列文件数和路径仅用于追溯，
> 不是后续实验的默认输出位置。当前统一结果根目录为
> `E:\project_iot\实验结果9`，可执行命令以另外三份 16 分片实验手册为准。

## 1. 当前目录状态

审计时间：2026-07-31。结果根目录为 `E:\project_iot\实验结果8`。

| 目录 | 文件数 | 当前大小 | 作用 |
|---|---:|---:|---|
| `models` | 14 | 1.33 MiB | 保存可部署模型及模型选择记录 |
| `offline_training` | 121 | 32.74 MiB | 保存 Python 离线训练全过程 |
| `offline_eval` | 31 | 0.23 MiB | 保存 Python 独立评估结果 |
| `block_eval` | 176 | 334.43 MiB | 保存 BlockEmulator 系统级评估 |
| 根目录清理清单 | 3 | 约 0.25 MiB | 记录删除内容、原因和原文件哈希 |

总计 345 个文件，约 368.98 MiB。文件类型包括 111 个 CSV、70 个 JSON、
23 个 JSONL、37 个日志、94 个模型和 10 个文本文件。K6/K7/K8 的终端汇总表未
自动保存；2026-07-31 的 TopK7 奖励权重汇总已经同时保存为 JSON 和 CSV。

## 2. models

- `ppo_top7_w5530_16s_seed7.pt`：早期未做主锚点负载对齐的历史 best 模型。
- `ppo_top7_w5530_16s_seed7_last.pt`：上述早期训练的最后一轮模型。
- `ppo_top7_w5530_aligned_16s_seed7.pt`：已对齐负载语义、但按旧单一分数选择的模型。
- `ppo_top7_w5530_aligned_16s_seed7_last.pt`：上述对齐训练的最后一轮模型。
- `ppo_top6_w5530_pareto_16s_seed7.pt`：TopK6 Pareto 候选模型，来自 epoch 14。
- `ppo_top6_w5530_pareto_16s_seed7_selection.json`：TopK6 的来源 checkpoint、
  约束和验证指标，跨片率 67.8732%，活跃分片 8.0000，阶段热点 32.9349%。
- `ppo_top7_w5530_pareto_16s_seed7.pt`：TopK7 Pareto 候选模型，来自 epoch 12。
- `ppo_top7_w5530_pareto_16s_seed7_selection.json`：记录 epoch 12 的来源、
  选择约束、指标和输出模型路径；跨片率 66.2794%，活跃分片 8.1169，
  阶段热点 33.2508%。
- `ppo_top8_w5530_pareto_16s_seed7.pt`：TopK8 Pareto 候选模型，来自 epoch 9。
- `ppo_top8_w5530_pareto_16s_seed7_selection.json`：TopK8 的来源 checkpoint、
  约束和验证指标，跨片率 66.4228%，活跃分片 8.2202，阶段热点 34.0846%。
- `ppo_top7_w5035_pareto_16s_seed7.pt`：TopK7-w50-35 候选模型，来自 epoch 1。
- `ppo_top7_w5035_pareto_16s_seed7_selection.json`：w50-35 的来源、选择约束和
  验证指标，跨片率 70.0875%，活跃分片 8.4135，阶段热点 32.5060%。
- `ppo_top7_w6025_pareto_16s_seed7.pt`：TopK7-w60-25 候选模型，来自 epoch 1。
- `ppo_top7_w6025_pareto_16s_seed7_selection.json`：w60-25 的来源、选择约束和
  验证指标，跨片率 73.8313%，活跃分片 8.8584，阶段热点 31.8154%。

每个 Pareto 模型都与相应训练目录中被选中的 `epoch_XXX.pt` 内容相同。这不是无用
重复：`models` 中的文件是稳定部署入口，`checkpoints` 中的文件是训练审计链。
三个 `selection.json` 是本轮 K6/K7/K8 汇总的原始来源之一，必须保留。

## 3. offline_training

### ppo_top7_w5530_16s_seed7_train.jsonl

早期未对齐训练日志。保留用于解释旧模型来源，不进入当前论文主结果。

### ppo_top7_w5530_aligned_16s_seed7_20260727_184109

第一次主锚点负载对齐训练。按单一 validation score 保存 best 模型。

- `train.jsonl`：每个 epoch 的训练和验证指标。
- `stdout.log`、`stderr.log`：训练终端输出与错误输出。
- `run_context.json`：数据窗口、参数和命令。
- `run_complete.json`：训练完成状态与最终摘要。

### 三个 Pareto 训练目录

- `ppo_top6_w5530_pareto_16s_seed7_20260730_115329`
- `ppo_top7_w5530_pareto_16s_seed7_20260728_213904`
- `ppo_top8_w5530_pareto_16s_seed7_20260730_122922`

三者使用相同训练集 2,500,000、验证集 444,019、15 epoch、seed 7 和 w55-30，
仅 `candidate_top_k` 分别为 6、7、8。每个目录中的文件逐项含义如下：

- `checkpoints\epoch_001.pt`：第 1 轮验证后的模型。
- `checkpoints\epoch_002.pt` 到 `epoch_014.pt`：对应第 2 到第 14 轮模型。
- `checkpoints\epoch_015.pt`：第 15 轮模型。15 个文件缺一不可，否则训练不完整。
- `pareto_manifest.json`：15 个逐 epoch checkpoint、一个可能不同的 final post-update
  候选、三个优化目标和 Pareto 前沿。
  K6、K7、K8 的前沿分别包含 14、15、13 个候选。
- `best_scalar.pt`：旧单一综合分数会选择的模型，仅作对照。
- `last.pt`：第 15 轮模型，不能自动等同于最佳模型。
- `train.jsonl`：逐 batch、逐 epoch、逐验证记录。
- `run_context.json`：完整训练命令与数据边界。
- `run_complete.json`：完成状态、模型哈希和选中 epoch。
- `stdout.log`：完整训练终端输出。
- `selection_stdout.log`：K6/K8 的 Pareto 选择器终端输出；K7 的同类信息已经写入
  `run_complete.json` 和 models 下的 `selection.json`。
- `stderr.log`：K7 运行的独立错误流；大小为 0 表示没有错误输出。K6/K8 使用合并
  输出，所以没有单独的 `stderr.log`。

实际选中结果为 K6 epoch 14、K7 epoch 12、K8 epoch 9。`best_scalar.pt` 和
`last.pt` 都不能替代 `models` 中的最终 Pareto 候选。

### 两个 TopK7 奖励权重训练目录

- `ppo_top7_w5035_pareto_16s_seed7_20260731_210731`
- `ppo_top7_w6025_pareto_16s_seed7_20260731_212557`

它们与上述 Pareto 训练目录使用相同文件结构、训练/验证窗口、TopK7、15 epoch 和
seed 7，仅跨片/均衡权重分别为 0.50/0.35 与 0.60/0.25。两次训练状态均为
`passed`，分别选择 epoch 1；现有 w55-30 训练作为权重筛选中间点直接复用。

## 4. offline_eval

- `ppo_top7_w5530_16s_seed7_validation.jsonl`：早期模型在验证窗口的 Python 结果。
- `ppo_top7_w5530_16s_seed7_test2M.jsonl`：早期模型在最后 2M 窗口的 Python
  诊断结果。该窗口已被历史实验观察过。
- `ppo_top7_w5530_aligned_16s_seed7_validation_20260727_185653`：旧单一分数
  对齐模型的独立验证，跨片率约 75.692%。
- `ppo_top7_w5530_pareto_epoch12_16s_seed7_validation_20260728_221030`：
  TopK7 epoch 12 模型的独立验证，跨片率 66.279%，阶段最大负载占比
  33.251%，平均活跃分片 8.117。
- `ppo_top6_w5530_pareto_epoch14_16s_seed7_validation_20260730_120516`：
  TopK6 epoch 14 模型的独立验证。
- `ppo_top8_w5530_pareto_epoch9_16s_seed7_validation_20260730_124110`：
  TopK8 epoch 9 模型的独立验证。
- `ppo_top7_w5035_pareto_epoch1_16s_seed7_validation_20260731_212455`：
  TopK7-w50-35 epoch 1 的独立验证。
- `ppo_top7_w6025_pareto_epoch1_16s_seed7_validation_20260731_213840`：
  TopK7-w60-25 epoch 1 的独立验证。
- `top7_weight_screen_16s_seed7_20260731_213942.json`：三组权重、模型哈希、来源目录、
  Pareto 规则和完整验证摘要的机器可读汇总。
- `top7_weight_screen_16s_seed7_20260731_213942.csv`：相同汇总的 Excel 友好版本。

每个新式评估目录中的全部文件为：

- `eval.jsonl`：完整机器可读评估记录，包含模型、窗口、MDP 参数和全部 summary。
- `run_context.json`：本次验证的模型、参数和完整命令。
- `run_complete.json`：验证成功证明、模型哈希及 summary 副本。
- `stdout.log`：人可读终端输出。
- `stderr.log`：旧 aligned 和 TopK7 目录中的独立错误输出，均为空；K6/K8 将输出
  合并到 `stdout.log`，所以没有该文件。

当前三个 Pareto 独立验证结果为：

| TopK | 选中 epoch | 实际跨片率 | 关系跨片率 | 活跃分片 | 阶段热点 |
|---:|---:|---:|---:|---:|---:|
| 6 | 14 | 67.8732% | 58.8589% | 8.0000 | 32.9349% |
| 7 | 12 | 66.2794% | 62.9900% | 8.1169 | 33.2508% |
| 8 | 9 | 66.4228% | 61.3776% | 8.2202 | 34.0846% |

固定 TopK7 后的奖励权重结果为：

| 权重 | 选中 epoch | 实际跨片率 | 关系跨片率 | 活跃分片 | 阶段热点 |
|---|---:|---:|---:|---:|---:|
| w50-35 | 1 | 70.0875% | 57.1200% | 8.4135 | 32.5060% |
| w55-30 | 12 | 66.2794% | 62.9900% | 8.1169 | 33.2508% |
| w60-25 | 1 | 73.8313% | 56.1643% | 8.8584 | 31.8154% |

## 5. block_eval

### ppo_top7_w5530_16s_test2M_seed7_20260724_105413

早期模型在最后 2M 时间窗口的 BlockEmulator 诊断运行，不是当前模型的正式测试：

- 完整 TPS 313.357；
- 跨片率 62.962%；
- 平均确认延迟 1414.895 秒；
- P95 延迟 3962.243 秒。

该目录已压缩为 8.54 MiB。逐交易和逐动作大型明细已删除，汇总、聚合 CSV、
配置、模型哈希、状态检查与 Supervisor 日志均保留。论文中只能把它作为发现负载
语义不一致和固定锚点热点的历史诊断，不能与当前验证结果直接比较。

### 四个 ppo_top7_w5530_pareto_16s_validation444k 目录

这四次都是完整 BlockEmulator 系统运行，但数据窗口是验证集 444,019 笔，不是最后
2M 测试集。区别仅是目标注入速率为 200、250、300、350 TPS。

| 目标 TPS | 实际注入 TPS | 完整 TPS | 跨片率 | 平均延迟 | P95 延迟 |
|---:|---:|---:|---:|---:|---:|
| 200 | 199.9999 | 199.479 | 66.355% | 6.438 s | 11.756 s |
| 250 | 249.9999 | 249.178 | 66.425% | 5.967 s | 9.849 s |
| 300 | 299.9999 | 298.917 | 66.429% | 5.876 s | 9.761 s |
| 350 | 349.9998 | 348.226 | 66.424% | 7.431 s | 20.168 s |

200/250/300/350 不是四个随机种子，也不能取平均；它们构成同一模型的系统负载曲线。
250 TPS 是保守主对比速率，300 TPS 是高负载补充点，350 TPS 已出现尾延迟拐点。

### ppo_top7_w5530_pareto_16s_validation_rate_calibration_seed7_20260729.json

四档速率的统一汇总，保存模型哈希、数据窗口、实际注入速率、完整 TPS、跨片率、
平均/P50/P95/P99/最大延迟、负载方差、PPO 调用比例和速率选择结论。

## 6. 根目录三个清理审计 JSON

- `cleanup_manifest_20260728.json`：第一次结果瘦身的完整清单。记录 455 个删除目标、
  释放 7.628 GiB、保留规则，以及每个数据库、节点日志、重复目录的原路径、大小和
  删除原因。它不包含性能指标，也不是实验结果。
- `cleanup_manifest_20260729_paced_validation.json`：四次精确节拍验证运行的清理清单。
  每个运行删除 145 个可再生节点数据库/日志，共 580 个目标、约 4.74 GiB；同时明确
  列出每轮保留的论文分析证据。
- `cleanup_manifest_20260729_results_audit.json`：旧 2M 诊断结果的二次压缩审计。
  记录删除的 `Tx_Details.csv`、`decision_records.jsonl`、`feedback_records.jsonl`
  和空推理错误日志，共约 413.85 MiB，并保存每个原文件 SHA-256。该文件证明旧诊断
  虽被压缩，但 `analysis.json` 和聚合 CSV 仍可审计。

这三个文件必须保留。它们解释“大文件为什么不在目录里”，避免以后误认为实验缺失。

## 7. 单次 BlockEmulator 目录内部文件

四个现代验证运行各有 37 个文件。逐项如下：

- `analysis.json`：论文分析优先读取的机器可读汇总。
- `analyzer_output.txt`：便于人工查看的分析器输出。
- `run_complete.json`：完整运行、状态检查和关键指标。
- `run_context.json`：方案、窗口、速率、模型和输出路径。
- `experiment_manifest.json`：数据、模型、配置文件哈希，保证可复现。
- `paramsConfig.snapshot.json`：本次运行使用的参数快照。
- `workload_profile.json`：窗口主锚点分布、热点和理论容量上限。
- `processes.json`：64 个 PBFT 节点及 1 个 Supervisor 的 PID 和启动记录。
- `state_check.txt`：交易数量和分片状态一致性检查。
- `expTest\log\Supervisor.log`：BlockEmulator 自身的 Supervisor 运行日志。
- `expTest\result\supervisor_measureOutput\Average_TPS.csv`：逐 epoch TPS。
- `CrossTransaction_ratio.csv`：逐 epoch 跨片数量和比例。
- `Shard_Load_Variance.csv`：逐 epoch 分片负载方差。
- `Transaction_Confirm_Latency.csv`：逐 epoch 平均确认延迟。
- `Tx_number.csv`：普通交易、Relay1、Relay2 数量。
- `Tx_Details.csv`：逐交易确认延迟，用于 P50/P95/P99/最大值。
- `expTest\result\pbft_shardNum=16\Shard016.csv` 到 `Shard916.csv`：分片 0 到 9。
- `Shard1016.csv`、`Shard1116.csv`、`Shard1216.csv`、`Shard1316.csv`、
  `Shard1416.csv`、`Shard1516.csv`：分片 10 到 15。共 16 个分片测量文件。
- `spring_io\decision_records.jsonl`：每次 PPO 决策、候选掩码和动作。
- `spring_io\feedback_records.jsonl`：系统反馈与奖励记录。
- `spring_io\infer_server_stderr.log`：Python 推理服务错误输出；四次验证均为空。
- `process_logs\supervisor.stdout.log`：Supervisor 标准输出。
- `process_logs\supervisor.stderr.log`：Supervisor 重定向错误流和少量运行诊断。

旧 `ppo_top7_w5530_16s_test2M_seed7_20260724_105413` 只有 27 个保留文件：
`analysis.json`、`experiment_manifest.json`、`paramsConfig.snapshot.json`、
`state_check.txt`、`supervisor_terminal_output.txt`、`Supervisor.log`、16 个
`Shard*.csv` 以及 5 个聚合 CSV。它没有现代的 `run_context.json`、
`run_complete.json`、`workload_profile.json`、`processes.json` 和逐交易/逐动作明细，
因此只能作历史诊断。

节点数据库、每个 PBFT 节点的重复日志和节点标准输出在运行完成并通过状态检查后可
再生成，因此已经删除；上述论文数据全部保留。

## 8. Pareto 对齐模型

“Pareto 对齐模型”不是一种新的神经网络结构，而是两个性质的合称：

1. 对齐：Python 环境和 BlockEmulator 使用相同的主锚点执行负载语义。
2. Pareto：从 15 个 epoch checkpoint 中用多目标规则选择部署模型。

对齐前，多锚点关系权重曾被当作实际执行负载；但 BlockEmulator 实际把一笔 IoT
交易重写为 `state_object -> primary owner anchor`。对齐后明确区分：

- `owner_loads`：数据本身的主锚点分布，策略不能改变。
- `stage_loads`：发送分片和接收分片实际消耗的执行阶段，策略可以影响。
- `relation_*`：多锚点关系，仅用于候选、通信和诊断，不再冒充系统执行交易。

验证窗口的主锚点最大占比为 55.288%，这是数据热点，不代表策略把 55% 动作选到同一
分片。当前模型把可控的系统阶段最大负载占比限制到 33.251%。

## 9. Pareto 前沿和选择约束

模型选择同时考虑三个方向：

- 跨片率越低越好；
- 系统阶段最大负载占比越低越好；
- 平均活跃分片数越高越好。

如果模型 A 在三个指标上都不差于 B，并且至少一个指标严格更好，则 A 支配 B。
所有不被其他模型支配的 checkpoint 构成 Pareto 前沿。

当前选择协议是 `hierarchical_constrained_pareto_v1`，包含两个层级：

- 一级严格层要求 `active_shards_mean >= 8`：16 个分片中平均至少使用一半，防止
  动作坍塌；同时要求 `stage_max_load_share <= 0.35`，使任何单一分片不承担超过
  35% 的阶段负载。满足后优先选择实际跨片率最低者。
- 一级无候选时才进入二级容差层。阶段热点 35% 上限保持为硬约束，活跃分片相对缺口
  `max(0, (8 - active_shards_mean) / 8)` 必须不超过 5%。二级先选择缺口最小者，
  再依次比较跨片率、热点、epoch 和事件名。
- 二级仍无候选就停止，不能继续放宽约束或手工按 seed 挑模型。

选择 JSON 的 `selection_tier` 记录 `strict` 或 `active_tolerance`，`strict_feasible`
记录严格可行集是否存在，`constraint_audit` 保存活跃分片缺口、相对缺口、热点超额以及
两级规则的通过状态。这些字段用于复现实验和统计严格可行率。K6、K7、K8 的历史
seed7 严格选择分别是 epoch 14、12、9；历史记录不会被删除。阈值和 5% 容差都是
验证阶段的工程 guardrail，不是普适定理。

## 10. P95 延迟

把所有交易确认延迟从小到大排序，P95 是第 95 百分位：95% 的交易延迟不超过它，
最慢的 5% 超过它。平均值可能掩盖少量严重排队，因此系统容量判断必须同时看
平均值、P95 和 P99。

本次 300 TPS 的 P95 为 9.761 秒；350 TPS 的平均值只增加到 7.431 秒，但 P95
升到 20.168 秒，说明尾部队列已经出现明显拐点。

## 11. 累计单调时钟注入节拍器

旧实现按每个发送块固定等待，`TxBatchSize=1000` 时会产生整块取整误差。例如目标
300 TPS 会拆成 300+300+300+100，若每块都等待 1 秒，实际只有 250 TPS。

新节拍器从运行开始累计已发送交易数，并计算：

`目标经过时间 = 累计发送交易数 / 目标 TPS`

发送后只等待到这个累计时间点。300 TPS 下，发送 1000 笔后的目标时间是 3.333 秒，
下一批继续沿同一时间线，不会在 batch 边界重新计时。单调时钟不受系统时间校准影响。

系统同时记录目标 TPS、累计交易、经过时间、实际 offered TPS 和调度滞后。四次
验证的实际 offered TPS 与目标误差小于 0.0001%，说明速率标定可信。

## 12. 本轮新增内容和原因

- 精确注入节拍：修复 300/400 TPS 被 batch 取整的问题。
- 实际 offered TPS 日志：区分“想注入多少”和“真正注入多少”。
- 逐 epoch checkpoint：不再只保留 best/last，避免错过更好折中模型。
- Pareto 前沿与选择脚本：把跨片、热点和分片利用率分开审计。
- 主锚点/阶段/关系负载拆分：让离线 MDP 与系统执行语义一致。
- P50/P95/P99/最大延迟：识别平均延迟看不到的排队尾部。
- 自动四档速率运行脚本：顺序启动、等待、归档，防止配置和结果串目录。
- `run_context`、`manifest`、模型哈希和状态检查：保证论文结果可复现。
- 自动清理清单：删除可再生大文件时记录路径、大小、原因和 SHA-256。

## 13. 测试集方法学边界

当前 Pareto epoch 12 模型没有运行最后 2M 窗口。刚完成的四次运行都使用验证窗口
`[2500000, 2944019)`。最后 2M 窗口是 `[2944019, 4944019)`。

但是旧模型已经在 2026-07-24 观察过最后 2M，且该结果帮助发现语义错位和热点。
因此它可作为固定时间顺序测试窗口，但不能再声称是全项目未观察过的严格 blind test。
若论文需要严格盲测，应增加新的时间窗口或外部数据集。
