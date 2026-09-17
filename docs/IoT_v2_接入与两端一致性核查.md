# 新数据接入、Go/Python 一致性核查与运行命令

更新时间：2026-09-13。本文记录接入修改和验证。用户已完成 seed=7 的正式离线实验及 1 万笔链上交易提交；第 5 节保留首次完整实验命令，已有结果无需重新训练。本次启动脚本修复及清理说明见第 8 节。

## 1. 本次完成了什么，系统设定是否改变

已为 `data_iot_v2` 增加保留真实发送方、接收方和原整数金额的接入分支，并修正核查中发现的两端实现差异。旧多锚点数据仍走旧身份分支。数据目录依然只有构造说明和 7 个 CSV，数据生成规则没有改变。

原 `spring_lite/config.py`、`model.py`、`ppo.py` 没有修改，区块和共识代码也没有修改。沿用以下主线设置：

| 设置 | 本次使用的值与含义 |
|---|---|
| MDP（马尔可夫决策过程）状态结构 | 203 维：最近 5 期交易数和跨片数共 160 维，关系分布 16 维，地址关系标志 1 维，当前账户放置负载 16 维，物联网属性 10 维 |
| 动作 | 16 个分片中选择一个；账户首次出现时放置，后续继续使用已有放置 |
| PPO（近端策略优化） | 原策略网络、优化器、学习率、折扣因子、裁剪、探索熵和更新实现不变 |
| 候选过滤 | Top-K（得分最高的 K 个候选）为 7；容量保护默认关闭；候选负载系数为 1 |
| 关系模式 | 主命令使用模式 1：接收方关联当前批次发送方；先处理发送方，再处理接收方 |
| 全局奖励 | `0.55×跨片改进项 + 0.30×均衡项 − 0.10×通信代价 − 0.05×热点惩罚` |
| 局部奖励 | 保留 Python 原有 IoT 稠密动作奖励；主线训练实际混合比例为局部 0.65、全局 0.35，动作奖励缩放 0.1 |
| 批次及区块 | 每批 1,000 笔；每块最多 1,000 个执行阶段；区块间隔 5,000 毫秒 |

**不能把这次描述成“只改路径，所有状态含义完全没变”。**原数据把设备作为固定锚点，放置合成状态对象；新数据是真实账户之间的交易，双方都必须作为真实账户参与放置。为此调整了输入身份和当前批次场景聚合。203 维的布局、16 个动作及原奖励权重保留，但“锚点数量”等槽位适配为真实通信对端的属性。新模型写入独立的语义版本，禁止因为同为 203 维就混用旧模型。

局部奖励原本就在 Python 的 `iot_dense_balanced`（带均衡项的物联网稠密奖励）中。虽然命令行 `local_reward_weight`（局部奖励权重）默认值显示为 0，实际批次处理会取它与 0.65 的较大值。本次没有去掉它，也没有重新设计一套奖励；Go 原来没有实现同一套 IoT 动作奖励计算，本次补齐相同公式。

本版仍有物联网属性，不能按“已没有 IoT 特征”删除其权重。暂时沿用原权重做新数据首轮测试；尚无正式结果证明应当调高或调低哪个权重。

## 2. 新数据怎样进入系统

1. Python 用 `--tx_identity iot_v2`，Go 用 `SpringIOTIdentityMode=2` 选择新分支。`SpringIOTMode=1` / `--mdp_mode iot` 启用 10 维物联网属性和原 IoT 奖励。
2. 交易主表和 `transaction_scene.csv`（逐交易场景表）逐行配对，核对地址、交易序号和源行号。场景元数据从构造说明读取，也兼容生成器尚未精简的 JSON（结构化数据文本）元数据目录。
3. 保留原交易方向和整数金额。内部统一去掉可选的 `0x` 地址前缀只是字符串规范化，不会生成替代账户。Go 附加场景时不重写交易哈希。
4. 对当前可见批次中的每个账户，汇总它作为发送方和接收方时的场景。发送方使用正向质量，接收方使用反向质量；合法零质量保留，不交换方向。
5. 10 维依次为：不同对端数量/4（上限 1）、归一化平均距离、归一化最小距离、平均有向质量、平均归一化通信强度、设备占比、边缘占比、云端占比、服务占比、控制通信占比。距离沿用原除以 60 并裁剪到 0—1 的公式；均值按交易出现次数计权，最小距离取当前批次最小值。
6. 本版全部逻辑账户为设备，四类占比为 `[1,0,0,0]`。多个账户共用画像或坐标不等于合并账户。画像表、模板表和位置表用于解释、校验与重建；运行时需要的字段已包含在逐交易场景表，不从全量账户未来统计生成决策特征。
7. 每个数据窗口独立冷启动，关系、放置负载和历史由运行时计算。分块生成的全局交易序号偏移与窗口内本地交易编号分别处理。

时长单位仍未解决：保留 `flowDuration`（流持续时长）的原值，沿用原归一化强度公式及零时长回退，**不把该指标声明成字节/秒，也未擅自除以 1,000**。当前场景没有原 IoT 时间序列连续性，也没有完整云边端多角色身份。

## 3. 发现的差异与处理

| 问题 | 修正与作用 |
|---|---|
| 旧身份入口会解释成合成状态对象与固定锚点 | 增加专用真实账户分支，校验双方身份和原金额；旧入口拒绝误接新表 |
| Go 的通信代价强度混入包数和时长，Python 按负载字节量计算 | Go 采用 Python 原公式，使相同场景得到相同通信代价 |
| Go 的局部动作奖励与 Python 的 IoT 稠密奖励不同 | 补齐关系质量、通信代价、过载惩罚和低负载奖励；IoT 在线样本继续采用 0.65 局部混合比例 |
| 非 IoT 的 Go 在线奖励也固定混合 0.65 局部奖励 | 改为非 IoT 主线默认 0，与 Python 原主线一致；不是对所有可选奖励消融参数提供任意映射 |
| 全局奖励数值稳定常量不同 | Go 从 `1e-6` 统一为 Python 的 `1e-8`；对应测试保持严格精度断言 |
| 候选分数极接近时排序不同 | Go 原来将相差不超过 `1e-12` 当成平局，改为与 Python 一样仅完全相等才按分片号破平局；真实数据对照中已复现并验证修正 |
| Go 无法把候选负载系数显式设为 0 | 区分参数缺省和显式零，默认仍为 1；负值与 Python 一样裁剪为 0 |
| 两端启发式基线的负载评分来源不同 | v2 的 Python 使用 Go 的已放置账户数评分和哈希破平局；候选保护规则保持一致 |
| Python 哈希/随机基线可能被候选过滤重选 | 基线关闭过滤，与 Go 对应模式一致；评估日志记录实际使用的候选参数 |
| 同维度旧模型或旧在线经验可能混入新数据训练 | 模型增加 `iot_v2` 身份和独立执行语义版本；推理和在线更新检查元数据；在线经验缓存按模型和配置隔离 |
| 无前缀地址被无条件截掉前两位；异常金额和窗口可能被跳过 | 修正地址规范化；v2 严格检查 18 列、十进制非负整数金额、窗口完整性、场景对应关系，异常明确失败 |
| 文件结束时剩余不足整批交易未提交 | 补齐尾批提交逻辑，避免遗漏最后一部分交易 |
| 新运行误用旧配置和旧输出位置 | 独立目录单独生成配置和端口表，直接引用公共程序、源码和原训练模型，并在启动前核对文件摘要；日志与数据库各自隔离 |

“多数关系加负载”属于 heuristic（启发式）方法；`candidate_only`（仅候选评分）也是无学习基线。二者都不是 PPO，正式对照应分别记录。

一致性有明确范围：核对的是相同输入、相同已有放置和相同历史反馈时的计算。Python 离线历史按输入批次推进；Go 在线历史来自异步提交区块，反馈到达时机和排队可能不同。因此两端在真实异步运行中不保证逐步得到相同观察或分片轨迹。本次没有改造共识或强制同步流水线。模式 2 的时间邻居记忆尚未在 Go v2 实现，两端 v2 入口均明确拒绝；支持模式 0 和模式 1。两端随机数生成器不同，同一随机种子也不表示随机基线逐笔动作相同。

## 4. 已实际执行的验证

- Python 全部 92 项测试通过，包括新数据身份、原大整数金额、分块序号、窗口、双向零链路、旧模型拒绝、运行目录仅生成四个必要文件等。
- 新数据 300,000 笔逐行校验再次通过，7 个 CSV 的内容摘要不变，交易主表与原文件逐字节相同，目录仍为 8 个文件。
- `go test ./...` 全项目测试通过。独立跨语言测试需要下面的专用脚本驱动，普通 Go 测试没有输入时会跳过该项。
- 真实数据模式 1：2,001 笔、3 批、1,851 次新账户放置；逐项比较 203 维状态、关系、候选掩码、动作、局部奖励、全局奖励。
- 真实数据模式 0：跳过前 1,000 笔后取 1,001 笔、2 批、955 次放置，对照通过。
- 实际短训练：2,000 笔训练、后续 300 笔验证、1 轮，发生 1 次 PPO 参数更新，保存最佳及末次模型；另行执行 300 笔模型评估。哈希、启发式和仅候选基线也完成同一 300 笔评估入口检查。
- Go 实际通过常驻 Python 推理接口调用短训练模型，16 个输入的动作、置信度、对数概率及价值估计与 Python 直接调用一致。
- 当时准备的 300 笔链上目录未启动，现已按用户要求删除。用户后来执行的正式 1 万笔链上目录已核实：10,000 个唯一交易均最终提交，7,194 次放置全部来自实际 PPO 推理。

短验证模型在 300 笔窗口的确定性评估跨片率约 13.67%，活跃分片均值 7，最大有效负载份额约 41.5%。这只说明训练、保存和评估能够执行，样本很小且只更新一次，**不是正式性能结论**，不能据此判断 PPO 优于基线或重新调整权重。

首次短验证使用的 `analysis_outputs/iot_v2_integration_20260912/` 已按用户要求清除，不再保留短训练模型和未启动的目录。用户正式结果保存在 `analysis_outputs/iot_v2_seed7_20260913_104049/`。原根目录 `paramsConfig.json`、`ipTable.json`、旧模型和旧实验输出没有作为新实验目标改写。

## 5. 后续正式实验的完整命令

以下在 Anaconda 环境的 PowerShell 中执行。先完成 seed=7（随机种子为 7）的一个完整训练与测试，再决定是否扩展种子。所有数据按**源文件行序**划分，不能未经时间字段核查称为严格时间顺序。

| 段 | 从 0 开始的区间 | 数量 | 用途 |
|---|---|---:|---|
| 训练 | `[0, 200000)` | 200,000 | 学习模型 |
| 验证 | `[200000, 250000)` | 50,000 | 选择模型；沿用原最佳模型评分规则 |
| 测试 | `[250000, 300000)` | 50,000 | 冻结模型后与基线比较 |

先设置目录和线程，使用新输出目录，避免覆盖已有实验。下面使用参数数组，复制时无需处理行尾反引号：

```powershell
Set-Location -LiteralPath 'E:\project_iot\block-emulator-main-iot'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$v2Run = Join-Path (Get-Location) ('analysis_outputs\iot_v2_seed7_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $v2Run -ErrorAction Stop | Out-Null
$v2Model = Join-Path $v2Run 'models\ppo_best.pt'
```

可先复查数据和两端计算：

```powershell
python -B .\spring_lite\validate_iot_v2_dataset.py --dataset-dir .\data_iot_v2 --source-csv .\selectedTxs_300K.csv
if ($LASTEXITCODE -ne 0) { throw 'Data validation failed' }
python -B .\spring_lite\check_iot_v2_alignment.py --start-tx 0 --max-txs 2001 --sender-pos-mode 1
if ($LASTEXITCODE -ne 0) { throw 'Go/Python alignment failed' }
```

执行 20 万笔、15 轮正式离线训练；权重与原主线一致：

```powershell
$trainArgs = @(
  '-B', '.\spring_lite\train_offline.py',
  '--tx_identity', 'iot_v2', '--mdp_mode', 'iot',
  '--start_tx', '0', '--max_txs', '200000',
  '--validation_start_tx', '200000', '--validation_max_txs', '50000',
  '--shards', '16', '--sender_pos_mode', '1',
  '--tx_batch_size', '1000', '--max_block_size', '1000', '--block_interval_ms', '5000',
  '--candidate_top_k', '7', '--capacity_guard', '0', '--candidate_load_weight', '1',
  '--reward_mode', 'iot_dense_balanced',
  '--iot_cstr_weight', '0.55', '--iot_balance_weight', '0.30',
  '--iot_comm_cost_weight', '0.10', '--iot_hotspot_weight', '0.05',
  '--epochs', '15', '--seed', '7', '--device', 'cpu',
  '--model', $v2Model,
  '--last_model', (Join-Path $v2Run 'models\ppo_last.pt'),
  '--epoch_checkpoint_dir', (Join-Path $v2Run 'models\epochs'),
  '--log_jsonl', (Join-Path $v2Run 'train.jsonl'), '--log_interval_batches', '20'
)
python @trainArgs
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
```

在同一个最后 5 万笔窗口比较四种策略。所有策略采用相同冷启动窗口；仅训练好的 PPO 读取模型。哈希关闭候选过滤，另三种按各自方法使用候选规则。

```powershell
foreach ($v2Policy in @('hash', 'heuristic', 'candidate_only', 'ppo')) {
  $evalArgs = @(
    '-B', '.\spring_lite\eval_offline.py',
    '--tx_identity', 'iot_v2', '--mdp_mode', 'iot', '--policy', $v2Policy,
    '--start_tx', '250000', '--max_txs', '50000',
    '--shards', '16', '--sender_pos_mode', '1',
    '--tx_batch_size', '1000', '--max_block_size', '1000', '--block_interval_ms', '5000',
    '--candidate_top_k', '7', '--capacity_guard', '0', '--candidate_load_weight', '1',
    '--reward_mode', 'iot_dense_balanced',
    '--iot_cstr_weight', '0.55', '--iot_balance_weight', '0.30',
    '--iot_comm_cost_weight', '0.10', '--iot_hotspot_weight', '0.05',
    '--seed', '7', '--model', $v2Model,
    '--log_jsonl', (Join-Path $v2Run ('eval_' + $v2Policy + '.jsonl'))
  )
  python @evalArgs
  if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $v2Policy" }
}
```

训练后可再次核对实际模型接口，然后编译并准备链上验收。首轮先取测试段前 1 万笔，在线训练关闭，确定性推理开启，16 分片、每分片 4 个节点、每秒注入 250 笔：

```powershell
python -B .\spring_lite\check_iot_v2_alignment.py --model $v2Model
if ($LASTEXITCODE -ne 0) { throw 'Model interface check failed' }
go build -o .\analysis_outputs\blockEmulator_iot_v2.exe .
if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }
$v2Block = Join-Path $v2Run 'block_ppo_10k'
python -B .\spring_lite\prepare_iot_v2_run.py --policy ppo --model $v2Model --start-tx 250000 --max-txs 10000 --inject-speed 250 --base-port 43000 --seed 7 --output-dir $v2Block
if ($LASTEXITCODE -ne 0) { throw 'Run preparation failed' }
powershell -ExecutionPolicy Bypass -File (Join-Path $v2Block 'run.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Blockchain run failed' }
```

生成器只准备目录；最后一条 `powershell ... run.ps1` 才真正启动节点。需要顺序运行其他链上基线时，修改 `--policy` 为 `hash`、`heuristic` 或 `candidate_only`，去掉 `--model` 并给不同的 `--output-dir`。使用同一批 1 万笔验收窗口。正式链上比较时，各策略再统一为完整测试段的 5 万笔和相同注入率。不要将 1 万笔链上值与 5 万笔离线值当作同窗口直接比较。

运行目录刚准备好时只有 `paramsConfig.json`、`ipTable.json`、`run_context.json`、`run.ps1` 四个文件。程序引用公共 `analysis_outputs/blockEmulator_iot_v2.exe`，Python 引用项目根目录的 `spring_lite`，模型引用原训练模型；数据也读取绝对路径。启动前核对程序、必要推理源码和模型的 SHA-256（文件内容摘要）；准备后如修改这些文件，需重新准备目录。这里记录摘要而不保存旧程序快照，因此不能单凭摘要恢复被删除的旧版本。

端口 43000—43064 若已占用会拒绝启动；已有执行结果的目录也禁止再次启动。脚本只清理自己启动的 Go 进程。输出主要在 `expTest`、`spring_io` 和各节点日志；新增 `run_status.json` 记录开始/完成状态、错误和所有进程退出码。Windows PowerShell 5.1 的进程句柄问题已修正，空退出码会明确报错，绝不当作成功。

观察结果时同时看跨片率、均衡项、活跃分片数、最大负载份额、动作分布、通信代价；链上再看吞吐量和延迟。不能只用跨片率判断 PPO 有效。确认单种子流程完成后，再按相同设置扩展种子和交易规模。

## 6. 本次逐文件修改说明

| 文件 | 修改与作用 |
|---|---|
| `spring_lite/iot_v2.py`（新增） | 真实账户数据契约、行对应检查、分块偏移、双向场景转换和当前批次账户特征聚合 |
| `spring_lite/offline_env.py` | 保留原整数金额；新分支放置双方真实账户，接入原 203 维状态及奖励；禁止误用旧多锚点加载器 |
| `spring_lite/train_offline.py` | 新身份参数、数据路径和模型元数据；避免写入旧默认模型；训练/验证摘要记录新语义 |
| `spring_lite/eval_offline.py` | 新数据评估；对齐启发式，增加仅候选基线；哈希/随机关闭候选过滤并如实记录 |
| `spring_lite/update_online.py` | 检查模型兼容性，按模型与语义隔离在线经验缓存，拒绝静默重置不兼容模型 |
| `spring_lite/prepare_iot_v2_run.py`（新增） | 检查数据窗口和模型后，只生成四个运行文件；引用公共资源并核对摘要，正确收集退出码和执行状态 |
| `spring_lite/check_iot_v2_alignment.py`（新增） | 用真实窗口生成 Python 计算结果，驱动 Go 逐项核对，并可检查实际模型推理接口 |
| `spring_lite/test_iot_v2_integration.py`（新增） | 新身份、大整数、分块/窗口、双方放置、异常输入及旧模型拒绝测试 |
| `params/global_config.go` | 支持真实身份模式 2；候选负载系数支持显式零；拒绝未对齐的时间邻居模式 |
| `supervisor/committee/spring_iot.go` | 读取新场景与索引，核对真实交易而不改写身份；统一通信代价公式 |
| `supervisor/committee/spring_iot_v2.go`（新增） | 读取构造元数据、转换双向特征、按当前批次真实账户放置、复用原基线与 PPO，补齐 IoT 局部奖励 |
| `supervisor/committee/committee_relay.go` | 严格解析真实交易、接入新放置分支，传递通信代价，拒绝 v2 推理失败的静默替代，补齐文件尾批 |
| `supervisor/committee/spring_candidate.go` | 对齐候选排序和平局判定，保留显式零负载系数 |
| `supervisor/committee/spring_call.go` | 向 Python 传递新身份与执行语义兼容条件 |
| `supervisor/committee/spring_train.go` | 对齐稳定常量和主线局部混合比例，在线更新携带模型兼容条件 |
| `supervisor/committee/spring_iot_test.go` | 两项奖励测试采用相同稳定常量，保留原严格断言 |
| `supervisor/committee/spring_iot_v2_test.go`（新增） | 比较真实加载、金额/哈希、状态、候选、奖励和 Go→Python 推理；覆盖构造说明的元数据格式 |
| `data_iot_v2/README_构造说明.md` | 更新接入状态、当前入口和时长解释；保留构造过程、复现参数及 CSV 内容摘要 |
| 本说明文件（新增） | 记录范围、验证证据、局限和完整实验命令 |

工作区中 `prepare_iot_v2_dataset.py`、`validate_iot_v2_dataset.py`、`test_prepare_iot_v2_dataset.py` 以及 10 个辅助 JSON 文件的删除，是此前按要求精简数据目录留下的改动，本次没有再次改写这 3 个数据工具。原代码注释保留，新增分支补充说明；必要的旧说明已按实际含义更新。

## 7. 英文对应中文

| 英文 | 中文 |
|---|---|
| Go / Python | 两种编程语言名称 |
| IoT | 物联网 |
| MDP | 马尔可夫决策过程 |
| PPO | 近端策略优化 |
| CSV / JSON / JSONL | 逗号分隔数据文件 / 结构化数据文本 / 每行一个 JSON 对象的记录文件 |
| heuristic / hash / random | 启发式 / 哈希 / 随机 |
| candidate / candidate_only / Top-K | 候选 / 仅候选评分 / 得分最高的 K 个候选 |
| anchor / sidecar | 锚点 / 附加数据表 |
| state / action / reward | 状态 / 动作 / 奖励 |
| dense / balanced | 稠密的 / 考虑均衡的 |
| local / global | 局部 / 全局 |
| checkpoint / model | 模型检查点，即保存的模型及元数据 / 模型 |
| metadata / load semantics | 元数据，即描述配置的数据 / 负载统计与执行语义 |
| rollout buffer | 训练轨迹经验缓存 |
| seed / epoch / batch / window | 随机种子 / 完整训练轮次 / 批次 / 数据窗口 |
| sender / recipient / peer | 发送方 / 接收方 / 通信对端 |
| sender_pos / action mask | 相关发送方在各分片的分布 / 允许动作的掩码 |
| forward / reverse | 正向 / 反向 |
| feature / profile / template | 特征 / 画像 / 模板 |
| device / edge / cloud / service | 设备 / 边缘 / 云端 / 服务 |
| payload / flowDuration | 通信负载字节量 / 流持续时长 |
| start_tx / max_txs / index_offset | 起始交易序号 / 读取交易数量上限 / 全局序号偏移 |
| online / offline / inference | 在线 / 离线 / 推理，即模型产生动作 |
| deterministic / argmax | 确定性的 / 取最大值对应动作 |
| CSTR / WLB / hotspot | 跨分片交易率 / 工作负载均衡 / 热点 |
| SHA256 / EPS | 文件内容摘要算法 / 防止数值不稳定的小常量 |
| smoke / prepared_not_started | 短流程连通性检查 / 已准备但未启动 |
| prepare / alignment / integration / test | 准备 / 对齐核查 / 接入整合 / 测试 |
| config / env / params / lite / v2 | 配置 / 环境 / 参数 / 轻量版 / 第二版 |
| train / eval / update / check / run | 训练 / 评估 / 更新 / 检查 / 运行 |
| supervisor / committee / relay / call | 监督节点 / 委员会 / 中继 / 调用 |
| README / selectedTxs / data / protocol | 说明文件 / 选中的交易 / 数据 / 协议 |

## 8. 2026-09-13 启动脚本修复与目录清理

### 8.1 本次 1 万笔其实已经提交完成

核查的正式目录为 `analysis_outputs/iot_v2_seed7_20260913_104049/block_ppo_10k`。`Tx_Details.csv` 有 10,000 个唯一交易，全部具有最终提交时间；中继第一、第二阶段各 1,128 笔，跨片率 11.28%。`spring_io/decision_records.jsonl` 有 7,194 次放置，来源全部为 `python_ppo`；`infer_server_stderr.log` 为空。放置次数与交易数不同，因为只在账户首次出现时决定分片。

根因在 Windows PowerShell 5.1 的 `Start-Process -PassThru` 返回对象：原脚本没有提前保留进程句柄，进程结束后 `ExitCode` 可能为空；旧条件将空值判为非零退出。在 Windows PowerShell 5.1 实测：未保留句柄时，退出 0 和退出 7 都可能读到空值；保留句柄后分别准确读到 0 和 7。修复为启动后立即访问并保留 `.Handle`，等待结束后分别检查空值和非零退出码。

原脚本没有保存可靠退出码，因此不能事后声称旧进程退出码全部为 0；这里确认的是原交易全部提交、推理来源正确，以及脚本误报机制已复现。

另外修复了总 TPS（每秒处理交易数）统计：原代码把开头空轮次的零时间当成最早时间，导致旧日志总 TPS 约 `1.0842e-6`，这个数字无效。按原 `Average_TPS.csv` 的有效轮次时间重算：`10000 / 45.504 = 219.7609`。这是根据原日志毫秒时间重算的统计，不是重新运行得到的成绩，也不是按整个脚本耗时计算的吞吐量。旧日志和 CSV 保留原样；新程序跳过空轮次的无效时间，并将空轮次 TPS 写为 0。本次只修正 TPS 模块，其他指标中空轮次的 NaN（未定义数值）占位没有全面改写。

### 8.2 修复后实际完成的复测

- Python 全部 92 项测试通过，`go test ./...` 通过，公共 Go 程序重新编译成功。
- 使用用户正式最佳模型再次核对两端：2,001 笔交易、1,851 次放置、203 维状态、候选掩码和局部/全局奖励通过；16 次实际 Go→Python 推理通过。Python 从公共源码目录运行，工作目录独立。
- 在系统临时目录实际启动新脚本：Windows PowerShell 5.1，16 分片、每片 4 节点，测试段前 1,000 笔，同一最佳模型、种子 7、每秒注入 250 笔。全部 1,000 个唯一交易最终提交；924 次放置全部为 PPO；推理错误日志为空；监督节点加 64 个节点共 65 个进程退出码全部为 0，状态为 `completed`（完成）。新 TPS 总值约 90.4655，开头两个空轮次 TPS 均为 0。小窗口耗时受启动及中继等待影响，不用它与正式 1 万笔比较性能。
- 本次修复后的真实链上复测是 1,000 笔；没有把这次复测说成重新跑完了正式 10,000 笔。复测中间文件放在系统临时目录，没有另建项目内的临时实验目录。

### 8.3 文件夹里保留什么、删除了什么

按用户要求删除 115 个中间文件和副本，共 51,229,462 字节（约 48.9 MiB）。

| 位置 | 处理与用途 |
|---|---|
| `analysis_outputs/iot_v2_integration_20260912/` | 删除整目录：首次小样本调试模型、调试日志和从未启动的 `block_prepared`；不是用户正式实验 |
| `analysis_outputs/iot_v2_quality_20260913/` | 只保留最终 `核查结论与实验步骤.md`；删除临时分析脚本、笔记本、辅助统计和校验输出 6 个文件 |
| 正式 `block_ppo_10k/spring_lite/` | 删除整份重复源码；旧生成器复制了全部 Python 文件，因此把训练、数据生成器和测试也带进去了，并非另一套新数据算法 |
| 正式 `block_ppo_10k/blockEmulator_iot_v2.exe` | 删除旧运行副本；原版本摘要保存在运行记录中，当前统一使用公共程序 |
| 正式 `block_ppo_10k/model.pt` | 与父目录 `models/ppo_best.pt` 逐字节一致，删除副本并把配置改为原模型绝对路径 |
| `analysis_outputs/blockEmulator_iot_v2.exe` | 保留并重新编译，作为此目录下唯一公共可执行程序；支持新旧数据模式，名称中的 v2 不是独立系统 |
| 正式实验 `models/`、`train.jsonl`、`eval_*.jsonl`、`comparison.csv`、`run_settings.json` | 保留真实训练模型、训练/评估记录、对照表和实验设置 |
| 正式链上 `expTest/` | 保留数据库、节点运行记录和指标 CSV；其中 `result/supervisor_measureOutput/` 是汇总结果 |
| 正式链上 `spring_io/` | 保留实际动作、反馈和推理错误日志，供确认模型是否真正参与及诊断使用 |
| 正式链上 `node_*.log`、`supervisor.*.log` | 保留 64 个节点及监督节点的标准输出和错误输出，属于实际运行证据 |
| 旧审计、结果分析、可行性目录及顶层说明 | 已检查，保留既有最终说明文档，没有因日期较早而删除 |

旧正式运行的 `paramsConfig.json` 只更新模型路径；原配置完整保存在 `run_context.json` 的 `resources_before_cleanup`（清理前资源记录）中。该记录还保存旧程序摘要、旧脚本摘要、源码及模型摘要；顶层资源路径指向当前公共文件。旧 `run.ps1` 更新为带结果保护的脚本，不允许在已有数据库和结果上重复运行。旧版本程序副本已删除，文件摘要用于辨认版本，不能替代可恢复的程序快照。

以后项目内只放必要的维护代码、正式数据和实验结果、最终说明。文档制作或排查用的一次性脚本、笔记本和探针放系统临时目录。可重复使用的数据生成器、校验器和回归测试仍属于必要源码，保留在项目根目录的 `spring_lite` 等公共源码目录。

### 8.4 新旧数据本来就有开关

| 数据 | Python 参数 | Go 配置 |
|---|---|---|
| 原多锚点物联网数据 | `--tx_identity iot --mdp_mode iot` | `SpringIOTIdentityMode=1`、`SpringIOTMode=1` |
| 新真实账户配套场景数据 | `--tx_identity iot_v2 --mdp_mode iot` | `SpringIOTIdentityMode=2`、`SpringIOTMode=1` |

还必须匹配相应的数据路径和对应模式训练的模型；不能仅改开关就把旧模型用于新数据。两种模式复用同一套训练、推理、候选和奖励代码中的适用分支。此次修复未调整 MDP（马尔可夫决策过程）、PPO（近端策略优化）、状态维度、动作定义、奖励和权重，也没有改写新数据的 7 个 CSV。

### 8.5 已有模型如何复跑 1 万笔

无需重新训练。若需要自己确认修复后的整套 1 万笔运行，复制下面完整命令；它使用现有正式模型，在同一实验下新建带时间戳的链上目录，保护旧结果。代码和公共程序已由本次修复更新，编译命令保留以方便之后复现。启动前不要并发运行其他占用相同端口的实验。

```powershell
Set-Location -LiteralPath 'E:\project_iot\block-emulator-main-iot'
$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$v2Run = Join-Path (Get-Location) 'analysis_outputs\iot_v2_seed7_20260913_104049'
$v2Model = Join-Path $v2Run 'models\ppo_best.pt'
if (-not (Test-Path -LiteralPath $v2Model -PathType Leaf)) { throw 'Trained model is missing' }
go build -o .\analysis_outputs\blockEmulator_iot_v2.exe .
if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }
$v2Block = Join-Path $v2Run ('block_ppo_10k_fixed_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
python -B .\spring_lite\prepare_iot_v2_run.py --policy ppo --model $v2Model --start-tx 250000 --max-txs 10000 --inject-speed 250 --base-port 43000 --seed 7 --output-dir $v2Block
if ($LASTEXITCODE -ne 0) { throw 'Blockchain run preparation failed' }
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $v2Block 'run.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Blockchain run failed; inspect the new run logs' }
$v2Status = Get-Content -LiteralPath (Join-Path $v2Block 'run_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($v2Status.state -ne 'completed') { throw 'Blockchain run did not complete' }
if (@($v2Status.processes).Count -ne 65) { throw 'Expected 65 recorded processes' }
if (@($v2Status.processes | Where-Object { -not $_.exited -or $null -eq $_.exit_code -or $_.exit_code -ne 0 }).Count -ne 0) { throw 'A process did not exit successfully' }
$v2Details = @(Import-Csv -LiteralPath (Join-Path $v2Block 'expTest\result\supervisor_measureOutput\Tx_Details.csv'))
if ($v2Details.Count -ne 10000) { throw 'Expected 10000 committed transaction records' }
if (@($v2Details.'TxHash (Byte -> Big Int)' | Sort-Object -Unique).Count -ne 10000) { throw 'Duplicate transaction records detected' }
if (@($v2Details | Where-Object { [long]$_.'Tx finally commit timestamp' -le 0 }).Count -ne 0) { throw 'A transaction has no final commit time' }
$v2Status | Select-Object state, started_utc, finished_utc
Write-Output "Verified 10000 committed transactions. Results: $v2Block"
```

验收后可继续相同窗口的链上基线对照，再扩展为完整测试段。一次 PPO 链上验收并不代表已经完成四种策略的链上性能比较。

### 8.6 本次修复逐文件说明

| 文件 | 此次修改及作用 |
|---|---|
| `spring_lite/prepare_iot_v2_run.py` | 改为公共资源路径和摘要；只生成四个准备文件；修复 PowerShell 退出码读取；新增执行状态；保护已有结果 |
| `supervisor/committee/spring_infer_server.go` | 通过 `SPRING_PYTHON_DIR` 定位公共推理源码；未设置时保留原路径行为 |
| `supervisor/committee/spring_call.go` | 在线更新也使用公共源码目录，并复用统一 Python 解释器选择 |
| `supervisor/committee/spring_iot_v2_test.go` | 实际 Go/Python 接口测试改为公共源码路径，从独立工作目录验证推理 |
| `spring_lite/test_iot_v2_integration.py` | 增加四文件准备目录、资源指向、摘要和禁止覆盖的测试 |
| `supervisor/measure/measure_avgTPS_relay.go` | 忽略空轮次无效时间，统一总 TPS 和逐轮 TPS 计算；空轮次写 0 和空时间；修正小数位数参数 |
| `supervisor/measure/measure_avgTPS_relay_test.go` | 新增空轮次、有效区间及全空输入的统计测试 |
| 正式链上 `paramsConfig.json` | 模型路径改为已有最佳模型，模型内容和其他实验参数不变 |
| 正式链上 `run_context.json` | 保存清理前的原配置及摘要；当前资源引用公共路径 |
| 正式链上 `run.ps1` | 更新为修复后的启动脚本；旧目录已有结果，执行时会明确要求新建目录 |
| `analysis_outputs/iot_v2_quality_20260913/核查结论与实验步骤.md` | 更新交付清单、正式实验进度及修复入口，不再依赖已删除的临时工具 |
| `data_iot_v2/README_构造说明.md` | 仅更新已完成正式实验的状态说明，不修改构造规则和数据 |
| 本文件 | 汇总原因、真实复测、清理、数据开关、逐文件修改和完整复跑命令 |

本节术语补充：EXE=可执行程序；TPS=每秒处理交易数；ExitCode=进程退出码；Handle=进程句柄；source=来源；runtime=运行时；stdout/stderr=标准输出/标准错误；run_context=运行上下文；run_status=运行状态；completed/failed=完成/失败；SHA-256=文件内容摘要算法；MiB=1,048,576 字节；regression test=回归测试；NaN=未定义数值。文件名中的其他英文对应第 7 节。
