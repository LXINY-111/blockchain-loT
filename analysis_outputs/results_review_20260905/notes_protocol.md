# 训练、选模、候选集消融与负载口径只读复核

审查日期：2026-09-05。只读取已完成历史结果及分析输出；未访问任何进程信息，未读取结果10中 `ppo_top7_w5530_pareto_16s_test2M_i250_seed27_20260905_202840` 目录的任一文件，也未读取与其对应的 `audit/proposed_test2M_seed27_session.json`。没有加载模型、训练或改写原结果/项目源码。下列数值以JSON/CSV文本元数据为证据；本复核没有从模型张量确认训练语义。

## 1. 数据窗口与选模

来源：`E:/project_iot/实验结果9/protocol_freeze_current_top7_w5530_16s.json`、`protocol_freeze_top7_w5530_16s.json`、`offline_training/*/run_context.json`、`offline_training/*/train.jsonl`、`offline_training/*/pareto_manifest.json`、`models/*_selection.json`。

- Train为 `[0,2500000)`；validation为 `[2500000,2944019)`，444,019笔；test为 `[2944019,4944019)`，2M笔。窗口在记录层面不重叠。
- 全部13个历史训练目录均记录相同train/validation窗口；每个train.jsonl有15个 `epoch_done`、15个 `validation` 和1个 `final_validation`，manifest均有16个候选记录（含末次剩余更新后的最终验证，不能说16个独立训练模型）。
- 7月的TopK/权重筛选、8月2日追加seed训练、8月3日hierarchical验证，均使用validation。当前冻结文件日期为8月4日11:05；它显式取代8月2日strict-only冻结，规定不能根据固定test修改超参/选模，且当时 `test_status=not_run_for_current_hierarchical_models`。这是有记录的开发阶段规则修订，应在论文交代，不能称hierarchical容差在全部开发前预注册。
- 本复核支持“记录的训练、验证及测试窗口分开，选模依据验证数据”。由于未审查完整训练程序的数据载入路径、数据预处理和所有历史试验，不能将元数据证据升级为“已彻底证明无泄漏”。

| TopK7权重.55/.30/.10/.05 | epoch/event | selection tier | validation cross | active mean | stage max share | active deficit |
|---|---:|---|---:|---:|---:|---:|
| seed7 | 12 / validation | strict | .6627937093 | 8.116853933 | .3325084788 | 0 |
| seed17 | 15 / final_validation | active_tolerance | .6334233445 | 7.680898876 | .3385244964 | 3.988764% |
| seed27 | 5 / validation | active_tolerance | .6965062306 | 7.829213483 | .3259328192 | 2.134831% |

strict要求active>=8、stage hotspot<=.35，按cross最小排名；strict集合为空才启用最多5% active缺口，仍保持hotspot<=.35，先最小active缺口再最小cross。独立扫描manifest得到三个seed的strict候选数3/0/0，容差范围内候选数5/1/3，与选中记录一致。seed17/27不能写“满足严格活跃分片约束”；三seed均为最终规则可接受是正确表述。

`curation_manifest_20260804.json`及 `summaries/paper_results_index.json` 说明结果从实验结果8整理搬移，原始记录中的旧路径可由搬移记录解释。后者是8月4日快照，其“尚未test”状态不能用于断言9月仍未测试。

## 2. 候选集消融已经存在，但证据层次不同

来源：`E:/project_iot/实验结果9/summaries/python_validation_summary.csv`、`model_selection_summary.csv`、`offline_eval/top7_weight_screen_16s_seed7_20260731_213942.json`。

| seed7验证筛选 | cross | active mean | stage hotspot |
|---|---:|---:|---:|
| K6 | .6787322164 | 8.000000000 | .3293488366 |
| K7 | .6627937093 | 8.116853933 | .3325084788 |
| K8 | .6642283326 | 8.220224719 | .3408462560 |
| K7, weights .50/.35 | .7008754130 | 8.413483146 | .3250598169 |
| K7, weights .60/.25 | .7383130001 | 8.858426966 | .3181536684 |

K7相对K6 cross低1.593851个百分点，相对K8仅低0.143462个百分点。此处是单seed开发集敏感性/调参证据，不能凭此宣称K7普遍最优或显著优于K8。权重.55/.30的结果具有较低cross，但另外权重的active/hotspot更好，不能称所有目标都最好。

真正去候选剪枝的训练消融是 `offline_training/ppo_top0_w5530_pareto_16s_seed{7,17,27}_...`，run_context明确标为 `IoT-PPO without candidate pruning (PPO-TopK0)`，保留IoT reward，state_dim=203，iot_feature_dim=10，K=0、capacity_guard=0。三seed均完成15epochs，但 `selection_failed.json` 记录strict和容差候选数均为0。

| TopK0 seed | 全部候选最大active | 全部候选最小stage hotspot | 最低cross（不代表可行） |
|---|---:|---:|---:|
| 7 | 6.343820225 | .2954519026 | .6478821852 |
| 17 | 5.359550562 | .3486247375 | .8772980436 |
| 27 | 5.352808989 | .3502500340 | .6240341066 |

各列极值未必来自同一checkpoint。此消融支撑“在相同记录的训练预算、IoT状态/奖励及冻结验证准则下，取消候选剪枝未产生可部署checkpoint”。它不提供测试集吞吐/时延，也不能外推为PPO不加剪枝必然失败。不要把selection failure写成0 TPS。

Original SPRING-PPO适配训练同样三seed全部无合格checkpoint；最大active分别2、2、3.047191011，最低stage hotspot分别.5060118082、.5087480293、.4910198904。对应context为mdp=spring、state_dim=177、iot_feature_dim=0、paper reward、K0，而TopK0是IoT状态+IoT奖励。故Original对完整方案是多因素差异的适配基线，不是单独候选剪枝消融，也不能据此否定原论文方法。

Candidate-Only是另一类系统消融：保留候选构造，移除PPO选择器。结果10历史 `audit/candidate_only_validation_seed27_session.json` 与 `_audit/candidate_only_validation_replication_plan.json` 记录validation444k、16分片、250 TPS、seeds7/17/27；9月4日冻结的延续规则明确按完整性通过而非指标有利决定继续。后续已完成数值应由系统结果审查提供，不能再声称“没有Candidate-Only消融”。

结果10 `_audit/candidate_only_shutdown_fix_build.json` 记录13:43旧run异常及PBFT shutdown修复后的重跑，修复binary SHA256为 `5F829BF84368BDF22BFC823B46A09AFA23AC88F7EF58F21DDCF074ED3F11A8D5`。历史proposed seed17/27 audit和Candidate-Only seed27 audit均记录该binary；proposed模型hash与8月4日冻结一致。因此应优先用结果10同binary的三seed配对validation比较，不能把旧异常run算第四个正常复现，也不要把跨版本时延差异解释为模型收益。

## 3. 输入、模型与速率公平性边界

从本次已生成、排除活动run的 `analysis_outputs/results_review_20260905/run_metrics.json` 读取历史配置汇总：所有方法ConsensusMethod=3、身份模式=1、TxBatchSize=1000、BlockSize=1000、Block_Interval=5000、Delay=-1、JitterRange=-1、Bandwidth=10000000；数据窗口、目标速率应按validation/test、200/250/300/350分层比较。SpringIOTMode存在0/1差异，属于模型/基线输入设计差异，不能宣称各方法输入特征完全一致。

同输入交易和相同系统预算允许比较方法系统结果；更丰富状态和奖励的作用需要TopK0等对应消融解释。候选-only无训练模型是消融定义，不能为追求“同模型”给它注入PPO。16分片250TPS对比与300TPS压力测试不能混为同负载排名；多个seed若使用固定交易窗口，是训练随机性/执行复现，不能当成多个独立数据集。

## 4. 对新增负载统计的独立复核

读取：`analysis_outputs/results_review_20260905/build_summary.py`、`review_results.py`、`summary.json`、`run_metrics.json`，以及 `supervisor/measure/measure_ShardLoadVariance_Relay.go`、`consensus_shard/pbft_all/pbftInside_module.go`；没有运行上述扫描脚本，也未重扫原始大文件。

- `measure_ShardLoadVariance_Relay.go:51–59`的原负载为 `normal + (relay1+relay2)/2`，每笔完整跨片交易在两片各计0.5，总量仍为原始交易数。它是折算交易负载。
- `pbftInside_module.go:72–91`将block.Body分别归为normal/relay1/relay2；103–109令Epoch=Block Height；124–143将len(block.Body)写入 `# of all Txs in this block`。因此直接累计CSV得到 `normal+relay1+relay2`，是PBFT已打包交易阶段数，每完整跨片交易占2个阶段。
- `build_summary.py`的16维物理向量以及 `J=(sum L)^2/(16*sum L^2)`、`maxshare=max L/sum L`分母是正确的。summary中22个历史物理统计全部满足physical_stages==expected_physical_stages。
- 物理阶段数不是CPU、字节、密码操作或通信成本测量；各阶段等权是所用计数口径。
- 每epoch统计按同Block Height对齐，不保证跨分片同墙钟时间；只排除sum=0，保留启动与排空期的稀疏高度且每高度等权。因而推荐以完整窗口累计Jain/maxshare为主，每height均值仅作补充，不能称实时同步负载均衡。
- 跨方法物理总阶段数受跨片比例影响，因此比例指标的变化同时包含分子与总工作量变化。不能把更小maxshare自动等同于最忙片绝对工作量下降。test250中PPO三seed最忙片阶段数均值1,276,226.33，NSshard1,274,507，Heuristic1,273,866，Anchor1,280,615；PPO总阶段放大率1.759182，Anchor1.397965。二者maxshare36.2864%与45.8028%的差异远大于最忙片绝对计数差异。

结果10同binary三seed、完整validation窗口的独立复算均值如下：

| 负载口径/统计 | Candidate-Only | PPO TopK7 |
|---|---:|---:|
| 折算交易 aggregate maxshare | .3310087331 | .2850242144 |
| 折算交易 aggregate Jain | .3686516813 | .3701774569 |
| 已打包阶段 aggregate maxshare | .3297879406 | .3321469290 |
| 已打包阶段 aggregate Jain | .3922833220 | .3546876739 |

结论限制：PPO降低了折算交易口径下的累计热点占比，但不能宣称实际打包阶段负载整体更均衡；物理阶段Jain三个seed均低于Candidate-Only，累计maxshare均值也略高。阶段Jain下降不能直接推出时延必然更差，时延和跨片工作量仍须分别报告。全部表格为描述性结果，三个seed不支撑未经检验的显著性措辞。
