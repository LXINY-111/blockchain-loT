# 真实链上账户关系 + 物联网场景数据 v2

生成日期：2026-09-12。接入状态更新：2026-09-13。300,000 笔已实际生成，逐行数据校验通过；已增加保留真实账户的专用入口，并完成短训练、离线评估和 Go/Python 对照验证。用户后续已完成 seed=7 的正式离线实验，1 万笔链上交易也已全部提交；启动脚本误报失败已修复，并在临时目录完成修复后的 1,000 笔真实链上复测。接入方式、修复说明和完整命令见 [接入与核查说明](../docs/IoT_v2_接入与两端一致性核查.md)。

## 1. 这次生成了什么

本数据集由“原封不动的链上交易文件 + 与每笔交易对应的场景表 + 稳定的账户映射 + 可追溯的通信样本库”组成。真实交易决定谁与谁交互；物联网原始数据提供通信统计量和空间链路模板。两者没有实测的一一对应关系，组合属于显式仿真场景。

目录仅保留本说明和以下 7 个 CSV（逗号分隔数据文件）。它们共同组成一套 300K（30 万笔）数据，不是七套独立实验数据。除交易主表无表头外，其余 6 表均有表头；下表行数均不计表头。

| 文件 | 内容与行数 | 具体用法 |
|---|---|---|
| selectedTxs_iot_v2.csv | 300,000 笔原交易，无表头、18 列；与项目原 selectedTxs_300K.csv 逐字节相同 | 交易主输入，提供真实发送方、接收方和原金额；与场景表同步读取 |
| transaction_scene.csv | 300,000 行逐交易场景：双方地址、画像、位置、通信模板、包数、字节数、距离和双向质量 | 为对应交易附加物联网属性；第 n 行数据对应主表第 n 笔，tx_index 从 0 开始 |
| account_profiles.csv | 54,403 个账户的固定画像、类型、位置、坐标、首次出现序号 | 以 account_address（账户地址）查询身份属性，供账户初始化和核对；不能按画像或位置合并账户 |
| device_profiles.csv | 27 种设备画像的名称、原设备地址、来源文件及原始/有效/抽样记录数 | 以 profile_id（画像编号）解释账户采用了哪种通信画像；不是 27 个链上账户 |
| flow_templates.csv | 95,055 条通信模板，含协议、双向包数/负载、原始时长、源文件和行号 | 以 template_id（模板编号）追溯场景属性；更大交易数据复用这个固定样本库 |
| sites.csv | 54 个逻辑部署位置及二维米制坐标 | 以 site_id（位置编号）查坐标、计算或核对通信距离；位置编号不代表分片 |
| site_links.csv | 2,916 个有向位置对，含原始质量、有效质量和使用规则 | 以发送位置、接收位置的有序组合查询链路；A→B 与 B→A 分别查，不交换方向 |

### 如何一起使用

1. 顺序读取交易主表和场景表：主表按从 0 开始计列时，第 3 列是发送地址，第 4 列是接收地址，第 8 列是原金额；按日常从 1 开始计列，则分别是第 4、5、9 列。场景表的 from_address、to_address 必须与之相等。
2. 用发送/接收地址连接账户表；用账户的 profile_id 连接设备画像表；用场景的 template_id 连接通信模板表。
3. 用双方 site_id 连接位置表，并以 (from_site_id, to_site_id) 查询正向链路，再交换位置查询反向链路。场景表已经保存算好的距离与双向有效质量，位置/链路表用于核对和后续重建。
4. 训练时按一致规则划分交易主表及场景表的训练段、验证段、测试段；不能单独排序、打乱或过滤其中一张表。当前 7 张表没有预先划分这些数据段。
5. 扩大交易规模时使用第 9 节的 --reuse-library（复用场景库）命令，生成器读取 4 张基础表：设备画像、通信模板、位置、有向链路。新交易会另行生成自己的交易主表、场景表和账户表。

账户表的 first_tx_index（首次出现交易序号）用于审计，不应将全数据的账户出现信息提前喂给策略。模板中的源 IP、端口和时间只表明原通信记录的来源，不是链上账户的实际网络地址和交易时间。

过程状态、逐模板次数、独立校验报告和示例等 10 个辅助 JSON（结构化记录文件）已从本目录移除。必要的种子、版本、统计、来源和文件内容摘要合并在本文末尾；构造示例保留在第 6 节。清理不改变上述 7 个数据文件。

源交易文件和副本的 SHA256 均为：
12297773f5f3e04e4631315d56a86001c43ba3ca001c3c84efcc923f06ca63bf

原始输入目录：E:/project_iot/原始数据集。使用了 flows.zip、mote_locs.txt、connectivity.txt。原输入未修改。

## 2. 第一步：完整保留链上交易骨架

逐笔保留发送方、接收方、金额、18 列格式、所有空字段和文件顺序。不会生成“设备+服务”的私有状态地址，不会把账户替换成所属设备，也不按画像或部署位置合并账户。

本文件有 300,000 笔交易、54,403 个独立地址、64,835 种有向账户对。完全相同的 18 列记录有 5,173 次重复出现，均原样保留。因为原文件没有交易哈希，不能判定这些一定是重复采集，不能擅自去重。

金额是原交易中的整数 wei，与通信字节数无关。本次没有将金额当作网络负载。

原文件缺少交易哈希、区块高度和链上时间，故目前只能称为“按文件顺序的工程样例”。没有把它描述成已核实连续区块的时间数据，也没有用物联网时间替代链上时间。

## 3. 第二步：建立固定的设备通信样本库

完整流式扫描 ZIP 中 27 个设备文件，共读取 4,944,041 条记录，未把整个解压数据放入内存，也没有把旧多锚点数据当成原始来源。

校验包数、负载字节数、时长为合法非负数，时间属于该源数据的 2016–2017 年。排除 8 条年份异常记录，剩余 4,944,033 条可供画像抽样。这里称“可供画像抽样”，不等于都属于业务交易；控制通信也保留并标记。

每种设备使用固定种子的蓄水池抽样，最多保留 4,096 条；少于 4,096 条则全部保留。这是从整个设备文件均匀抽样，不是只取文件开头。27 个文件按文件名排序，得到 profile_00 到 profile_26，最终共 95,055 条模板。

同一条模板中的协议、双向包数、双向负载和时长整体保留，不把不同流的各列分别拼接，避免破坏统计量之间的关系。模板还保存原始文件名、数据行号、时间、IP 和端口用于追溯；这些 IP/端口是模板来源，不是链上账户的真实网络端点。

源文件均衡与设备类别均衡是不同概念：本版账户对 27 种画像近似均匀分配，**没有复制原始 27 台设备的流量份额**；设备原始记录数量很不均衡，该假设已固定记录。

## 4. 第三步：为每个账户固定画像和位置

使用种子 7、版本号和 SHA256 的前 128 位生成稳定编号。简写为：

- 画像编号 = H(版本, 种子, "profile", 账户地址) mod 27。
- 位置编号 = H(版本, 种子, "site", 账户地址) mod 54。

完整序列化规则见生成脚本 stable_int。两类映射使用不同的用途标签。映射只依赖账户自身、种子与冻结的样本库，不读取账户今后的交易次数、未来邻居、模型输出或分片。

54,403 个账户仍是 54,403 个账户。多个账户共用“空气质量监测器”画像，表示多个逻辑实例具有相同的通信特征来源；共用一个坐标表示部署于同一逻辑位置，均不代表合并账户。

**本版是逻辑设备账户的基础场景。全部账户 actor_type=device；没有凭空给地址标注真实网关、云端或服务身份。**因此，原系统四类类型占比中的设备项可为 1，其他三项为 0，不能用这版数据检验完整的多角色异构性。当前保留画像、通信量、位置和链路差异；若研究必须比较云边端多角色，需要另行明确角色映射规则，不能声称本版已包含这种实测身份。

位置编号不是分片编号，未在数据中预先指定账户分片。也没有利用交易图把高频交互账户强行放在同一位置。

## 5. 第四步：给每笔真实交易选取通信属性

对于源文件第 i 笔 A→B：

1. 查 A、B 的固定画像与位置。
2. 在 A 的设备画像样本池中，用 H(版本, 种子, "transaction_template", 全局交易序号, 原交易整行) 取模选择一条模板。
3. 复制这条模板的协议、双向包数、负载字节数、原始时长和控制协议标记。
4. 根据 A、B 的位置，计算欧氏距离，读取 A→B 和 B→A 两个有向链路概率。
5. 以 tx_index 写入场景表，记录模板编号，真实交易 A→B 本身不改变。

这里是“用原通信记录作为交易通信属性模板”，不是声称真实以太坊交易对应了那条 DNS/TLS 流，也不是宣称接收账户确实是原模板中的 DNS/TLS 服务器。

tx_index 默认从 0 开始；source_row 是输入交易 CSV 中从 1 开始的数据行号。追加交易不会重新分配旧账户。分块生成时使用 --index-offset 保持全局序号；分块内 source_row 会从 1 重新计数。

抽样可重复。本次 300,000 笔实际使用 64,698 条不同模板，235,302 次分配复用了已经用过的模板，最大单模板使用 47 次。复用不增加原始测量的独立样本数量。

## 6. 第一笔交易的实际例子

- 原发送方：0x32be343b94f860124dc4fee278fdcbd38c102d88。
- 原接收方：0x104994f45d9d697ca104e5704a7b77d7fec3537c。
- 发送方映射：profile_02，AwairAirQuality（空气质量监测器画像），位置 39，坐标 (30.5, 26)。
- 接收方映射：profile_05，BelkinWemoSwitch（智能开关画像），位置 50，坐标 (38.5, 1)。
- 选中模板：profile_02:4363，即原 Awair 文件的第 4,363 条数据记录。
- 模板协议 DNS，两个方向各 1 个包，负载分别 37 和 85 字节，总计 122 字节。
- 距离约 26.2488095 米。
- 39→50 的概率约 0.0282771，50→39 约 0.0768257；两个方向不同。
- flowDuration 原始值为 184.0；本版保留原值，未将其直接声明为 184 秒。
- 模板时间 2017-03-03 07:17:13.249758 只用于原始数据追溯，不是该链上交易的发生时间。

## 7. 空间、链路与时长的边界

Intel 发布页面明确坐标单位为米，链路记录为整个采集期的平均有向接收成功概率，**不是即时带宽或链上时延**：
[Intel 原始数据说明](https://db.csail.mit.edu/labdata/labdata.html)

本地输入有 54 个坐标位置。无坐标的 0 号节点相关 54 条完整链路和 1 条不完整链路排除；实际使用的 54×54 位置对完整。没有用反向概率替代正向概率，没有根据距离伪造缺失值。不同位置的合法零概率保留，本次 35,390 笔交易对应正向零概率。

同位置的 5,926 笔交易使用“逻辑位置内通信”的模拟约定：距离 0，有效质量 1；site_links.csv 同时保留该位置自环的原始测量值 0，并以 co_located_assumption 区分。这不表示源数据测到了自环质量 1。

**时长单位存在待核实点。**发布方 README 将 flowDuration 标为秒，但本地选中值最高 15,845,942,856，若当秒会超过 500 年，与多月采集跨度明显不符。毫秒解释在数量级上更合理，但本次没有读取原始 PCAP/提取器证明这一点，因此没有擅自除以 1,000。所有原值保留，单位标记 source_raw_no_conversion，未输出“字节/秒”或“毫秒延迟”。

本次有 94,332 条选中模板时长为 0，原样保留，不用极小正数填充来制造巨大速率。接入时沿用原系统的归一化强度公式，零时长沿用原有负载量回退；该特征不解释为已经验证单位的“字节/秒”。若要研究物理速率，仍须核实单位。
[发布方字段说明](https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b)

UNSW 与 Intel 来自不同采集，Ethereum 账户与它们也没有实测匹配。两套背景数据的整个时间范围仅用于独立的场景库校准，没有从链上验证段或测试段学习映射参数。本版不保留原 IoT 时间序列的连续性，也不模拟时变链路。

## 8. 是否足够扩展到更大的区块链数据

**工程生成方面可以扩展；原始场景多样性不会随交易行数自动增长。**

- 100 万、1,000 万笔链上记录都可以使用同一冻结画像库，不要求一笔链上交易消耗一条从未用过的 IoT 记录。
- 追加出现的新账户只计算自己的画像与位置，已有账户映射不变；建议通过 --reuse-library 复用此目录的 95,055 条模板及 54 个位置，保证不同规模使用同一场景定义。
- 更大文件仍要求转换为本生成器明确支持的 18 列格式：小写 0x 地址、非负十进制整数金额、合法标志。尚未接入其他供应商的任意表头或字段格式。
- 生成器逐行读交易，模板池容量有界，但账户表和有向账户对统计会随不同账户/关系数量增长；不是固定内存算法。
- 按当前交易和场景行平均长度，仅这两个 CSV 每百万笔约 370 MB，每千万笔约 3.7 GB，另加账户表、模板库等。这是尺寸估算，不是已执行的百万/千万笔实验。
- 本次实际测量仅为 300K：生成约 47.84 秒，主体数据约 133.8 MB；尚未进行更大数据压力测试。
- 小画像池尤其容易重复：血压计只有 54 条记录、HelloBarbie 150 条。扩大模板上限可增加大设备画像的样本，但不能凭空增加稀少画像的数据或第 55 个真实位置。
- 若论文需要更多设备类型、多角色身份、更大拓扑、动态链路或真实链上时间泛化，还需要补足相应数据/场景假设。重复采样不能作为新增实测设备或独立实验重复。
- 正式大样本链上输入应尽量带交易哈希、区块高度和时间，并保存数据来源；本次 300K 文件缺少这些信息。

因此：这版足以作为完整系统下一步接入与初步实验的数据基础；不能仅凭“能生成更多行”宣称研究数据多样性已足够或算法一定有效。

## 9. 复现和后续扩展命令

在 PowerShell 中执行。现有 data_iot_v2 不会被覆盖。

本目录的校验是只读的，结果显示在终端，不会重新添加辅助文件。本文末尾的参数区供校验和复用读取，请随 7 张表一起保留。原生成命令在新的输出目录仍会产生过程记录，以便排查生成错误；本次仅精简已交付的 data_iot_v2。

重新生成一份同配置数据：
~~~powershell
Set-Location 'E:\project_iot\block-emulator-main-iot'
python -B .\spring_lite\prepare_iot_v2_dataset.py --chain-csv .\selectedTxs_300K.csv --iot-source-dir 'E:\project_iot\原始数据集' --output-dir .\data_iot_v2_rebuild --seed 7 --templates-per-profile 4096
python -B .\spring_lite\validate_iot_v2_dataset.py --dataset-dir .\data_iot_v2_rebuild --source-csv .\selectedTxs_300K.csv
~~~

只复查本次已经生成的数据：
~~~powershell
python -B .\spring_lite\validate_iot_v2_dataset.py --dataset-dir .\data_iot_v2 --source-csv .\selectedTxs_300K.csv
~~~

同时重新核对所有通信样本的原始 ZIP 来源：
~~~powershell
python -B .\spring_lite\validate_iot_v2_dataset.py --dataset-dir .\data_iot_v2 --source-csv .\selectedTxs_300K.csv --verify-template-sources 'E:\project_iot\原始数据集'
~~~

未来已有更大、符合 18 列格式的交易文件时，复用固定场景库。下面 new_chain_1M_18cols.csv 是示例文件名，本次没有下载或创建它：
~~~powershell
python -B .\spring_lite\prepare_iot_v2_dataset.py --chain-csv .\new_chain_1M_18cols.csv --reuse-library .\data_iot_v2 --output-dir .\data_iot_v2_1M --seed 7 --templates-per-profile 4096
python -B .\spring_lite\validate_iot_v2_dataset.py --dataset-dir .\data_iot_v2_1M --source-csv .\new_chain_1M_18cols.csv
~~~

需要增大每种画像的模板上限时，重新从原始 ZIP 建立一个独立版本的库；不要在同组训练/验证/测试之间偷偷更换模板库或种子。

## 10. 数据构造阶段范围与后续接入记录

数据构造阶段仅新增：
- spring_lite/prepare_iot_v2_dataset.py：生成器，含来源校验、固定映射、抽样、复用库和统计。
- spring_lite/validate_iot_v2_dataset.py：独立逐行校验。
- spring_lite/test_prepare_iot_v2_dataset.py：性质测试，覆盖原样复制、追加/分块不重映射、损坏拒绝、有向零链路、不覆盖已有数据，以及精简目录的只读校验和复用。

首次生成时 5 项测试已通过。目录清理后，包含新增精简目录检查在内的 6 项测试全部通过；30 万笔交易重新逐行校验通过，并重新打开原始 ZIP，对全部 95,055 条样本的文件、行号、数值、协议、IP/端口及时间逐条追溯核对，全部一致。7 个 CSV 的内容摘要与清理前完全相同，校验后目录仍只有 8 个文件。

首次生成时已核对 199 个此前的其他已跟踪文件内容和 8 个核心共享文件恢复内容均未改变。目录清理阶段的代码差异也仅涉及下面说明的 3 个数据工具。以上是当时的记录；2026-09-13 的接入改动另见接入与核查说明，不能把构造阶段的范围当成当前整个工作区的范围。

新场景表不是旧“私有状态—固定锚点”协议。Python 训练与评估必须指定 `--tx_identity iot_v2 --mdp_mode iot`；Go 使用 `SpringIOTIdentityMode=2`、`SpringIOTMode=1`。旧 `iot` 身份入口会拒绝新场景表，避免静默改写真实交易身份。建议使用新增的 `prepare_iot_v2_run.py` 生成独立链上运行目录。

数据文件没有提前生成依赖决策状态的 203 维观察。接入代码在当前可见批次汇总真实对端关系和 10 维场景特征，分片负载由运行环境计算；没有把未来全程关系写入静态特征。原关联数量位置用于不同真实对端数，四类占比保留，本版只有设备项非零。

术语：profile=通信画像；template=通信样本模板；site=逻辑部署位置；sidecar=附加数据表；seed=随机种子；reservoir sampling=蓄水池抽样；SHA256=文件内容摘要；actor_type=场景账户类型；MDP=马尔可夫决策过程；PPO=近端策略优化；UTC=协调世界时。

此前目录清理仅适配上述 3 个数据工具：生成器可从说明中读取冻结库参数；校验器可读取说明并保持精简目录只读；测试补充精简目录校验和复用检查。后续接入没有改变构造映射、抽样规则或 7 个 CSV 的内容。

## 11. 随数据保留的复现参数

以下参数区替代独立的库配置和统计文件，校验器与复用命令会读取它。summary 表示本数据统计；library 表示冻结的场景库；seed 是随机种子；index_offset 是全局交易序号偏移；sha256 是文件内容摘要。它们描述构造依据，不代表本次已运行训练。

<details>
<summary>展开查看完整参数与来源摘要</summary>

<!-- iot-v2-metadata:start -->
```json
{
  "summary": {
    "version": "real_accounts_iot_scene_v2_1",
    "seed": 7,
    "index_offset": 0,
    "templates_per_profile": 4096,
    "original_transaction_file": "E:\\project_iot\\block-emulator-main-iot\\selectedTxs_300K.csv",
    "original_transaction_sha256": "12297773f5f3e04e4631315d56a86001c43ba3ca001c3c84efcc923f06ca63bf",
    "transactions": 300000,
    "accounts": 54403,
    "directed_account_pairs": 64835,
    "unique_templates_used": 64698,
    "repeated_template_assignments": 235302,
    "max_template_reuse": 47,
    "zero_directed_quality_rows": 35390,
    "co_located_rows": 5926,
    "zero_duration_rows": 94332,
    "control_template_rows": 20504,
    "dataset_file_sha256": {
      "account_profiles.csv": "04d027d2062efee54b93ac603597f857c687c98aedf8c110d82a0beb1dfae97b",
      "device_profiles.csv": "9f2ed9c144d05d295d9c0a4a00001b26d15a3011e3e5d5a743e106585bf6580c",
      "flow_templates.csv": "bbbfb67c4148e1caf6b9ff8593f7278fd948dbbdd5c0f2ecfd9478bf6410bccd",
      "selectedTxs_iot_v2.csv": "12297773f5f3e04e4631315d56a86001c43ba3ca001c3c84efcc923f06ca63bf",
      "site_links.csv": "fbb0c67fb57c74f5e6ea5154e7cd9eb9ca3e77df2e9bae2c372cc728b2ca66c3",
      "sites.csv": "a75f9f88712ee0a67d25f37f4a038a54c4f612ea5f09d6fef5e37eea438a3a74",
      "transaction_scene.csv": "9e8e4c11f0db512d5d62e049befdb100a41a3c9a92628e68a83857693525415b"
    }
  },
  "library": {
    "version": "real_accounts_iot_scene_v2_1",
    "seed": 7,
    "templates_per_profile": 4096,
    "source_sha256": {
      "flows.zip": "91d99c0f6074df6552db2c224c34023ba6b9bbdb18a3d18e6b156c5e6affce64",
      "mote_locs.txt": "92decbba82c8253636f75e322abdf46400efbfaad765ffcc995a14d30d2f7466",
      "connectivity.txt": "e0e5dfd4dca48819fcbafbcc8c9883fd9b4d24051998e5238bc7d502535322d2"
    },
    "source_paths": {
      "flows.zip": "E:\\project_iot\\原始数据集\\flows.zip",
      "mote_locs.txt": "E:\\project_iot\\原始数据集\\mote_locs.txt",
      "connectivity.txt": "E:\\project_iot\\原始数据集\\connectivity.txt"
    },
    "source_urls": {
      "unsw": "https://iotanalytics.unsw.edu.au/unsw-iotraffic.html",
      "dryad": "https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b",
      "intel": "https://db.csail.mit.edu/labdata/labdata.html"
    },
    "profiles": 27,
    "sites": 54,
    "raw_flows": 4944041,
    "eligible_flows": 4944033,
    "sampled_templates": 95055,
    "rejected_flows": {
      "unexpected_source_year": 8
    },
    "topology_ignored": {
      "links_without_coordinates": 54,
      "incomplete_links_without_coordinates": 1
    },
    "duration_unit": "source_raw_no_conversion",
    "duration_unit_note": "Publisher README labels seconds; local raw values retained. Do not label derived rates as bytes/s before extractor/PCAP verification.",
    "mapping_assumptions": {
      "account_kind": "logical_iot_device",
      "profile_prior": "uniform_across_device_profiles",
      "site_prior": "uniform_across_coordinate_sites",
      "co_located_quality": 1.0,
      "quality_zero": "preserved",
      "edge_cloud_service_roles": "not_invented"
    },
    "library_file_sha256": {
      "flow_templates.csv": "bbbfb67c4148e1caf6b9ff8593f7278fd948dbbdd5c0f2ecfd9478bf6410bccd",
      "device_profiles.csv": "9f2ed9c144d05d295d9c0a4a00001b26d15a3011e3e5d5a743e106585bf6580c",
      "sites.csv": "a75f9f88712ee0a67d25f37f4a038a54c4f612ea5f09d6fef5e37eea438a3a74",
      "site_links.csv": "fbb0c67fb57c74f5e6ea5154e7cd9eb9ca3e77df2e9bae2c372cc728b2ca66c3"
    }
  }
}
```
<!-- iot-v2-metadata:end -->

</details>
