# 物联网场景下的 MDP 建模说明

本文档记录当前 BlockEmulator + SPRING/PPO 项目里的 IoT MDP v1 建模方案。MDP 是 Markov Decision Process（马尔可夫决策过程），PPO 是 Proximal Policy Optimization（近端策略优化）。本阶段目标是先跑通“Python offline training（离线训练）+ BlockEmulator evaluation（区块链测试平台评估）”两阶段流程，再在稳定基线（baseline，基线）上继续加入候选分片过滤等创新点。

## 1. 优化对象

### v2 multi-anchor（多锚点）更新

当前增强版 full 数据集已经从单一设备锚点扩展为 multi-anchor interaction graph（多锚点交互图）。每个 `to_address` 仍然表示 IoT communication state object（物联网通信状态对象），但 sidecar（旁路特征表）新增：

```text
anchor_addresses = 状态对象关联的多个 anchor（锚点）账户
anchor_types     = device/gateway/cloud_endpoint/service_group/...
anchor_weights   = 多锚点通信代价权重
relation_type    = device_to_device/device_to_gateway/device_to_cloud/...
```

MDP（马尔可夫决策过程）仍然只让 PPO（近端策略优化）决策 `to_address` 的 shard（分片），不直接移动设备、网关或云端锚点。区别是：`sender_pos`（相关对象分片分布）现在由整个 `anchor_addresses` 集合计算，而不是只看 `from_address` 一个设备锚点。reward（奖励）里的 CSTR（跨分片率）也按多锚点权重加权计算，因此 PPO 学习的是“状态对象靠近整个 anchor set（锚点集合）”和 workload balance（负载均衡）之间的折中。

本阶段优化对象不是每一条 flow（通信流），也不是 27 个物理 IoT 设备本身，而是：

```text
IoT communication state object（物联网通信状态对象）
= device（设备） + dstIp/dstPort/service（目标 IP/端口/服务） + protocol（协议）
```

在当前 full 数据集中，这个对象已经由数据处理脚本显式写入：

```text
state_object_key = device + dstIp + dstPort/service + protocol
to_address       = hash(state_object_key)
from_address     = hash(device_label, device_mac)
```

因此代码里的实际放置对象是 `to_address`，它表示一个长期存在的通信状态对象账户；`state_object_key` 保留为可解释的人类可读标签；`from_address` 是设备锚点账户，用来表示该状态对象依附的 IoT 设备。

一个直观例子是：

```text
AugustDoorBell -> 54.249.82.177:443 -> tls
```

中文含义是：AugustDoorBell 智能门铃访问某个云端服务，并使用 tls（传输层安全协议）。同一个设备、同一个目标服务、同一种协议产生的多条 flow 会归并到同一个通信状态对象。PPO 决定的是这个状态对象账户放到哪个 shard（分片），后续相同对象的 flow 沿用该放置结果。

这样比“每条 flow 单独放置”更接近 SPRING 的 state placement（状态放置）思想：state（状态）是持续存在的账户对象，不是一条通信记录结束就消失。

## 2. State（状态）

基础状态继承 SPRING 的 `11k + 1` 维，其中 `k` 是 shard 数量。

| 状态部分 | 维度 | 中文含义 |
| --- | ---: | --- |
| 最近 5 个窗口的交易负载 | `5k` | 每个 shard 最近处理了多少交易 |
| 最近 5 个窗口的跨片负载 | `5k` | 每个 shard 最近有多少跨分片交易 |
| 相关对象所在 shard 分布 | `k` | 设备锚点账户目前位于哪个 shard |
| 相关对象标记 | `1` | 是否有足够明确的相关对象信息 |

IoT MDP v1 在基础状态后追加 6 个物联网场景特征：

| 特征 | 中文含义 | 来源 |
| --- | --- | --- |
| payload | 载荷强度 | `srcPayloadSize + dstPayloadSize` |
| packet | 包数量 | `srcNumPackets + dstNumPackets` |
| distance | 拓扑距离 | sidecar 的 `distance` |
| link_loss | 链路损耗 | `1 - link_quality` |
| recent_frequency | 近期通信频率 | 已观测到的同设备同协议历史次数 |
| protocol | 协议类型 | `tls/http/dns/udp/...` 映射成数值 |

因此当前 IoT 状态维度是：

```text
state_dim = 11k + 1 + 6
```

如果 `k = 4`，状态维度就是：

```text
11 * 4 + 1 + 6 = 51
```

普通 SPRING baseline（基线）不追加 IoT 特征，4 个分片时仍然是 45 维。

## 3. Action（动作）

本阶段不加入 Top-k candidate shard filtering（候选分片过滤），action（动作）仍然是从所有 shard 中直接选择：

```text
action in {0, 1, ..., k-1}
```

例如 4 个 shard 时：

```text
0 表示放到 shard 0
1 表示放到 shard 1
2 表示放到 shard 2
3 表示放到 shard 3
```

## 4. Transition（状态转移）

PPO 对一个新的通信状态对象做出 action 后，环境会更新：

1. `addr_shard[to_address]`：通信状态对象账户到 shard 的映射。
2. `shard_load`：被选 shard 的对象负载。
3. 最近窗口交易负载。
4. 最近窗口跨片负载。
5. `from_address`：设备锚点账户使用稳定 hash（哈希）分片，不由 PPO 放置。

也就是说，PPO 的任务是决定 `to_address` 这个通信状态对象账户放到哪里；`from_address` 作为设备锚点，用于描述状态对象与设备之间的通信关系。这样任务边界更清楚：减少未来状态对象与设备锚点之间的跨片通信，同时控制负载均衡。

## 5. Reward（奖励）

IoT MDP v1 的 reward（奖励）函数是：

```text
R = w_cstr * (1 - CSTR)
  + w_balance * WLB
  - w_comm * CommunicationCost
  - w_hotspot * HotspotPenalty
```

各项含义：

| 英文 | 中文 | 含义 |
| --- | --- | --- |
| CSTR | 跨分片率 | 跨 shard 交易越少越好 |
| WLB | 负载均衡得分 | 各 shard 负载越均衡越好 |
| CommunicationCost | 通信代价 | 距离远、链路差、跨片时惩罚更高 |
| HotspotPenalty | 热点惩罚 | 单个 shard 过载时惩罚 |

通信代价近似为：

```text
CommunicationCost = CrossShardFlag * normalized_distance * (1 - link_quality) * traffic_weight
```

中文解释：如果通信状态对象账户 `to_address` 和设备锚点账户 `from_address` 落在不同 shard，并且距离远、链路质量差、流量强度高，就给予更高惩罚。

## 6. 主实验参数口径

为了贴近 SPRING 的两阶段实验口径，主实验里的 PPO 只使用标准强化学习目标和 IoT reward，不把工程辅助项写成 proposed method（提出方法）的一部分。

主实验默认关闭：

| 参数 | 中文含义 | 主实验设置 |
| --- | --- | ---: |
| `supervised_coef` | 弱监督辅助损失权重 | `0` |
| `argmax_tie_eps` | 近似平局打破阈值 | `0` |
| `argmax_balance_coef` | 动作分布均衡辅助损失 | `0` |

建议先跑两组 IoT reward 权重：

```text
SPRING-like IoT:
R = 0.70 * (1 - CSTR)
  + 0.25 * WLB
  - 0.05 * CommunicationCost
  - 0.00 * HotspotPenalty

Strong-CSTR IoT:
R = 0.80 * (1 - CSTR)
  + 0.15 * WLB
  - 0.05 * CommunicationCost
  - 0.00 * HotspotPenalty
```

第一组更均衡，第二组更偏向降低 CSTR（跨分片率）。如果后续要讨论 `supervised_coef` 或 tie-break（平局打破），应放到 ablation study（消融实验）中，不放进主方法描述。

## 7. 当前代码入口

普通 SPRING 模式仍然可用：

```powershell
python spring_lite\train_offline.py --csv selectedTxs_300K.csv
```

IoT MDP v1 full 数据集训练示例：

```powershell
python spring_lite\train_offline.py `
  --mdp_mode iot `
  --csv data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --reward_mode iot `
  --batch_size 1024
```

IoT MDP v1 full 数据集离线评估示例：

```powershell
python spring_lite\eval_offline.py `
  --mdp_mode iot `
  --csv data_iot\selectedTxs_iot_multi_anchor_full.csv `
  --sidecar data_iot\iot_flow_sidecar_multi_anchor_full.csv `
  --policy ppo `
  --reward_mode iot
```

## 8. BlockEmulator Evaluation（区块链测试平台评估）

链上评估入口支持 IoT MDP（物联网马尔可夫决策过程）：

1. `paramsConfig.json` 中设置 `SpringIOTMode = 1`。
2. `DatasetFile` 指向 `./data_iot/selectedTxs_iot_multi_anchor_full.csv`。
3. `SpringIOTSidecarFile` 指向 `./data_iot/iot_flow_sidecar_multi_anchor_full.csv`。
4. `SpringModelFile` 指向独立的 51 维 IoT PPO checkpoint（检查点），例如 `spring_lite/checkpoints/spring_iot_ppo.pt`。

链上执行时，Supervisor 会按 `tx_index` 读取 sidecar（旁路特征表），把原始交易映射成：

```text
state object account（状态对象账户，to_address）
-> device anchor account（设备锚点账户，from_address）
```

然后构造：

```text
11 * shardNum + 1 + 6
```

维 state（状态）。4 个 shard 时就是 51 维。

注意：旧 checkpoint（检查点）如果是在旧的 59,535 个重构 state key 口径下训练的，即使模型维度同样是 51，也不建议继续作为主实验 PPO 模型使用。现在的主实验口径已经改成直接使用 full 数据集里的 `to_address` / `state_object_key`，需要重新训练新的 IoT PPO 模型。
