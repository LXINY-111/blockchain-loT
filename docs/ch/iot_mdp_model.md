# 物联网场景下的 MDP 建模说明

本文记录当前 `block-emulator-main-iot` 项目中的 IoT MDP（马尔可夫决策过程）建模。实验仍分成两阶段：

1. Python offline training（离线训练）
2. BlockEmulator evaluation（区块链测试平台评估）

## 1. 数据集与状态对象

主实验使用冻结后的 multi-anchor full 数据集：

```text
data_iot/selectedTxs_iot_multi_anchor_full.csv
data_iot/iot_flow_sidecar_multi_anchor_full.csv
```

当前优化对象不是单条 flow（通信流），也不是 27 个物理设备，而是 IoT communication state object（物联网通信状态对象）：

```text
state object = device + peer_anchor_identity + dstPort/service + protocol
```

`peer_anchor_identity` 不是简单 `dstIp`，而是可解释的 anchor（锚点）身份，例如 cloud endpoint（云端端点）、gateway（网关）、local endpoint（本地端点）、private endpoint（私有端点）、service group（服务组）或 device（设备）。

数据字段语义：

```text
to_address        = 状态对象账户，由 PPO（近端策略优化）决策放到哪个 shard（分片）
from_address      = 所属设备锚点账户，通过 hash（哈希）固定分片
anchor_addresses  = 状态对象相关的多个 anchor（锚点）
state_object_key  = 人类可读的状态对象标签
```

## 2. State（状态）

基础状态沿用 SPRING：

```text
11k + 1
```

其中 `k` 是 shard（分片）数量：

| 部分 | 维度 | 中文含义 |
| --- | ---: | --- |
| `num_tx_window` | `5k` | 最近 5 个窗口每个分片的交易负载 |
| `cross_tx_window` | `5k` | 最近 5 个窗口每个分片的跨分片负载 |
| `anchor_pos` / `sender_pos` | `k` | 相关锚点当前所在分片的分布 |
| `flag` | `1` | 相关对象是否足够明确 |

IoT-MDP-B-lite 在此基础上追加：

```text
current_load[k] + IoT_feature[10]
```

10 个 IoT-aware feature（物联网感知特征）只保留主要场景信息：

| 特征 | 中文含义 |
| --- | --- |
| `anchor_count` | 相关锚点数量 |
| `avg_distance` | 到相关锚点的平均距离 |
| `min_distance` | 到最近锚点的距离 |
| `avg_link_quality` | 平均链路质量 |
| `traffic_rate` | 通信速率，按载荷/持续时间归一化 |
| `device_share` | 设备锚点权重占比 |
| `edge_share` | 网关/本地/私有端点权重占比 |
| `cloud_share` | 云端锚点权重占比 |
| `service_share` | 服务组锚点权重占比 |
| `is_control_protocol` | 是否控制/发现类协议 |

因此当前状态维度是：

```text
state_dim = 11k + 1 + k + 10
```

当 `k = 4`：

```text
11 * 4 + 1 + 4 + 10 = 59
```

旧 IoT-51 和上一版 IoT-77 checkpoint（检查点）都不能作为当前主实验模型继续使用。
加入 `traffic_rate` 后，旧 IoT-59 checkpoint（检查点）的输入维度虽然相同，
但第 5 个 IoT feature（物联网特征）的语义已经变化，因此也建议重新训练。

## 3. Action（动作）

当前不扩到 16 分片，先固定 `k = 4`，把 PPO 在 4 分片下跑稳。16 分片适合作为后续 scalability（可扩展性）实验。

```text
action in {0, 1, ..., k-1}
```

## 4. Reward（奖励）

B-lite 默认使用全局 IoT reward，不启用 dense reward（稠密奖励）：

```text
R_global
= w_cstr * (1 - CSTR)
+ w_balance * WLB
- w_comm * CommunicationCost
- w_hotspot * HotspotPenalty
```

C-lite 使用同一套 state（状态），但通过命令显式打开：

```powershell
--reward_mode iot_dense
```

`iot_dense` 会给每个 placement action（放置动作）额外局部反馈：

```text
R_action
= anchor_colocation_reward
- communication_cost_penalty
- shard_overload_penalty
```

C-lite+ 继续使用同一套 59 维 state（状态），但额外打开低负载 shard（分片）
鼓励，用来缓解 shard 0 几乎不用的问题：

```powershell
--reward_mode iot_dense_balanced
```

`iot_dense_balanced` 的全局 reward（奖励）仍与 `iot_dense` 一致，只在
action-level dense reward（动作级稠密奖励）里增加一个有上限的
low-load bonus（低负载奖励项）。这个 bonus 默认权重为 `0.25`，上限为
`0.18`，因此不会把完全不贴近 anchor（锚点）的 shard 硬抬成最优。

这样可以单独比较：

```text
B-lite = 轻量 IoT state + 全局 reward
C-lite = 轻量 IoT state + action-level dense reward（动作级稠密奖励）
C-lite+ = 轻量 IoT state + dense reward + bounded low-load bonus（有上限低负载奖励）
```

Baseline（基线）保留四类：

```text
hash      = 原始哈希放置
random    = 随机放置，使用 seed（随机种子）保证可复现
heuristic = 启发式多锚点贴近放置
ppo       = PPO（近端策略优化）模型放置
```

## 5. 当前入口

B-lite 训练示例：

```powershell
python spring_lite\train_offline.py `
  --mdp_mode iot `
  --reward_mode iot `
  --shards 4 `
  --batch_size 1024
```

C-lite 训练示例：

```powershell
python spring_lite\train_offline.py `
  --mdp_mode iot `
  --reward_mode iot_dense `
  --shards 4 `
  --batch_size 1024
```

C-lite+ 训练示例：

```powershell
python spring_lite\train_offline.py `
  --mdp_mode iot `
  --reward_mode iot_dense_balanced `
  --shards 4 `
  --batch_size 1024
```

离线 baseline（基线）评估示例：

```powershell
python spring_lite\eval_offline.py `
  --mdp_mode iot `
  --reward_mode iot_dense_balanced `
  --policy random `
  --seed 7 `
  --shards 4
```

BlockEmulator 配置需要保持：

```text
SpringIOTMode = 1
SpringIOTFeatureDim = 10
SpringRandomSeed = 7
DatasetFile = ./data_iot/selectedTxs_iot_multi_anchor_full.csv
SpringIOTSidecarFile = ./data_iot/iot_flow_sidecar_multi_anchor_full.csv
```

BlockEmulator baseline（区块链测试平台基线）对应的 `SpringMode`：

```text
0 = hash
1 = heuristic
2 = ppo
3 = random
```

Python 和 Go 两侧都会构造同样的 59 维 state（状态），保证 offline training（离线训练）和 BlockEmulator evaluation（区块链测试平台评估）一致。
