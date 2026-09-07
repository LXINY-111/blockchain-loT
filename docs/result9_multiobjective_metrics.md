# Result9 完整多目标指标口径

本口径用于重新分析所有已归档的 Result9 BlockEmulator 结果。任何方法都必须先通过
协议与完整性检查，再比较跨片、负载、延迟与系统性能。不得用测试集反向选择模型，
也不得把当前归档中无法统一回算的字段补成估计值。

## 1. 协议与数据有效性

- `state_check`（状态检查）：最终链状态、状态根与账户唯一性检查结果。
- `effective_total`（有效交易总数）：统一活动区间内实际确认的有效交易数。
- `completion_ratio`（完成比例）：有效交易数除以协议要求的交易数。
- `valid_latency_ratio`（有效延迟比例）：具有合法确认延迟的交易数占比。
- `actual_offered_tps`（实际注入 TPS）：注入器最终记录的实际提供速率。
- `offered_attainment_ratio`（注入达成比例）：实际注入 TPS 除以目标 TPS。
- `schedule_lag_ms`（调度滞后毫秒）：注入计划相对目标时间表的偏离。

## 2. 吞吐与时延

- `wall_tps`（墙钟吞吐量）：有效交易数除以首个活动 epoch 开始到最后活动 epoch
  结束的墙钟时间。
- `wall_to_offered_ratio`（完成/注入比例）：墙钟吞吐量除以实际注入 TPS。
- `mean/median/p05/p95_epoch_tps`（epoch 吞吐统计）：活动 epoch 的 TPS 分布。
- `mean/p50/p95/p99/max_latency_sec`（确认延迟统计）：逐交易确认延迟的均值、
  中位数、尾延迟与最大值。

## 3. 跨片通信

- `weighted_cross_ratio`（按交易计数加权的跨片率）：所有活动 epoch 的跨片交易总数
  除以有效交易总数。这里的“加权”只表示按各 epoch 的交易数汇总，不表示 IoT
  关系重要性加权。
- `cross_total`（跨片交易总数）：统一活动区间内的跨片交易数。

## 4. 负载均衡

- `mean/p95_load_variance`（原始负载方差）：保留原有量纲指标，用于兼容旧报告。
- `mean/p95_max_shard_load_share`（最大分片负载占比）：逐 epoch 最热分片负载除以
  当期总负载，再取均值或 P95；越低越好。
- `mean/min_active_shards`（活跃分片数）：逐 epoch 负载大于零的分片数量。
- `mean/p05_jain_fairness`（Jain 公平指数）：
  \((\sum_i x_i)^2/(N\sum_i x_i^2)\)，包含零负载分片；越接近 1 越均衡。
- `mean/p95_load_cv`（负载变异系数）：分片负载总体标准差除以均值；越低越好。
- `aggregate_max_shard_load_share`（聚合热点占比）：整个活动区间累计负载中最热分片
  的占比。
- `aggregate_jain_fairness`（聚合 Jain 公平指数）：对整个活动区间的累计分片负载
  计算 Jain 指数。

## 5. 候选集与动作机制（仅有 decision records 的方法）

- `python_ppo_ratio`（Python PPO 决策比例）与 `fallback_ratio`（回退比例）。
- `action_mask_size_mean`（平均候选集大小）。
- `candidate_major_anchor_coverage_ratio`（主要关系分片候选覆盖率）：关系质量最大的
  分片进入候选集的比例。
- `candidate_all_related_coverage_ratio`（全部关系分片候选覆盖率）：所有正关系质量
  分片都进入候选集的比例。
- `candidate_related_mass_coverage_mean`（候选关系质量覆盖）：候选集中关系质量之和
  除以全部关系质量之和的均值。
- `major_related_follow_ratio`（主要关系分片选择率）与
  `chosen_any_related_ratio`（任一关系分片选择率）。
- `chosen_related_mass_mean`（所选分片关系质量均值）。
- `major_anchor_choice_when_available_ratio`（主要关系分片可用时的选择率）：只在
  主要关系分片已经进入候选集的记录中统计，用于区分候选生成问题与 PPO 动作问题。

## 6. 已定义但当前归档不能统一回算

- `relation_weighted_cross_ratio`（关系重要性加权跨片率）。
- `weighted_communication_cost`（加权通信成本）。
- `overload_avoided_ratio`（反事实过载避免率）。
- `backlog_growth_rate`（积压增长率）。
- `unconfirmed_transaction_count_at_cutoff`（截止时未确认交易数）。

这些字段需要在未来实验中为所有方法统一保存逐交易关系权重、反事实候选负载和队列时间序列。
当前历史结果只能标记为 unavailable（不可用），不能由其他指标代替。
