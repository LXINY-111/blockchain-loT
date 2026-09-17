# IoT v2：最后 5 万笔链上四策略对照

日期：2026-09-13。本说明提供由用户执行的正式实验命令，不代表四组链上实验已经完成。使用现有最佳模型，无需重新训练；本次仅新增最终说明文件，没有修改系统代码。

## 1. 本轮固定设置

| 项目 | 设置 |
|---|---|
| 交易窗口 | 从 0 开始的 `[250000,300000)`，即最后 50,000 笔；四组相同 |
| 数据 | `data_iot_v2`，真实账户与逐交易物联网场景 |
| 方法 | hash（哈希）、heuristic（启发式）、candidate_only（仅候选评分）、ppo（近端策略优化） |
| 分片与节点 | 16 分片，每片 4 节点，另有 1 个监督节点 |
| 注入速度 | 四组都为 250 笔/秒，记录实际达到的注入速度 |
| 区块 | 每块最多 1,000 个执行阶段，间隔 5 秒 |
| 模型 | 已有 `iot_v2_seed7_20260913_104049/models/ppo_best.pt`，仅 PPO 使用 |
| 在线训练与采样 | 在线训练关闭，PPO 确定性推理 |
| 其他模型与奖励设置 | 沿用当前准备脚本，不调整原权重 |
| 程序版本 | 只编译一次；先准备全部四组，再顺序启动；启动前自动检查公共程序、推理源码和模型摘要 |
| 隔离 | 每组独立配置、数据库、日志与结果；共用相同端口，必须顺序运行 |

在装有项目依赖的 Anaconda PowerShell 中，按顺序执行下列三个代码块，使用同一个窗口。四组运行期间保持源码、数据、模型和公共程序不变，并让这台机器专用于本轮实验。50,000/250=200 秒仅为每组的目标注入时长；完整运行还包含启动、放置计算、中继提交和自动结束等待，不能按 200 秒判断超时或提前终止。

## 2. 第一步：检查环境、编译一次、准备四组目录

这一代码块只做检查和准备，不启动链上节点。

```powershell
Set-Location -LiteralPath 'E:\project_iot\block-emulator-main-iot'
$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'

$v2Run = Join-Path (Get-Location) 'analysis_outputs\iot_v2_seed7_20260913_104049'
$v2Model = Join-Path $v2Run 'models\ppo_best.pt'
$v2Exe = Join-Path (Get-Location) 'analysis_outputs\blockEmulator_iot_v2.exe'
$v2PS = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$v2Policies = @('hash', 'heuristic', 'candidate_only', 'ppo')
$v2Rate = 250
$v2Suite = Join-Path $v2Run ('block_compare_50k_i{0}_{1}' -f $v2Rate, (Get-Date -Format 'yyyyMMdd_HHmmss'))

if (-not (Test-Path -LiteralPath $v2Model -PathType Leaf)) { throw 'Trained model is missing' }
python -B -c "import sys, torch, numpy; print(sys.executable); print('torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Python dependency check failed' }
python -B .\spring_lite\check_iot_v2_alignment.py --model $v2Model
if ($LASTEXITCODE -ne 0) { throw 'Go/Python model alignment failed' }
go build -o $v2Exe .
if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }

New-Item -ItemType Directory -Path $v2Suite -ErrorAction Stop | Out-Null
foreach ($v2Policy in $v2Policies) {
    $v2Block = Join-Path $v2Suite $v2Policy
    $v2PrepareArgs = @(
        '-B', '.\spring_lite\prepare_iot_v2_run.py',
        '--policy', $v2Policy,
        '--binary', $v2Exe,
        '--start-tx', '250000', '--max-txs', '50000',
        '--inject-speed', "$v2Rate", '--base-port', '43000',
        '--seed', '7', '--output-dir', $v2Block
    )
    if ($v2Policy -eq 'ppo') { $v2PrepareArgs += @('--model', $v2Model) }
    python @v2PrepareArgs
    if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $v2Policy" }
}
Write-Output "Prepared all four policies. Campaign directory: $v2Suite"
```

生成四个子目录：`hash`、`heuristic`、`candidate_only`、`ppo`。每个刚准备好时只有四个文件：`paramsConfig.json`、`ipTable.json`、`run_context.json`、`run.ps1`。不会复制可执行程序、模型或整份 Python 源码。

哈希按原方法不进行候选过滤，其余策略按各自既有规则运行。统一实验条件不意味着把不同方法改成同一个算法。

## 3. 第二步：顺序运行，并核对每组 5 万笔最终提交

这一代码块才真正启动链上实验。顺序为哈希、启发式、仅候选评分、PPO；上一组正常退出后才启动下一组。某组失败会停止，保留日志供检查。

```powershell
foreach ($v2Policy in $v2Policies) {
    $v2Block = Join-Path $v2Suite $v2Policy
    Write-Output "Starting policy: $v2Policy"
    & $v2PS -NoProfile -ExecutionPolicy Bypass -File (Join-Path $v2Block 'run.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Blockchain run failed: $v2Policy. Logs: $v2Block" }

    $v2Status = Get-Content -LiteralPath (Join-Path $v2Block 'run_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($v2Status.state -ne 'completed') { throw "Run is not completed: $v2Policy" }
    if (@($v2Status.processes).Count -ne 65) { throw "Expected 65 process records: $v2Policy" }
    $v2BadProcesses = @($v2Status.processes | Where-Object { -not $_.exited -or $null -eq $_.exit_code -or $_.exit_code -ne 0 })
    if ($v2BadProcesses.Count -ne 0) { throw "A process did not exit successfully: $v2Policy" }

    $v2DetailFile = Join-Path $v2Block 'expTest\result\supervisor_measureOutput\Tx_Details.csv'
    $v2Details = @(Import-Csv -LiteralPath $v2DetailFile)
    if ($v2Details.Count -ne 50000) { throw "Expected 50000 transaction records: $v2Policy" }
    if (@($v2Details.'TxHash (Byte -> Big Int)' | Sort-Object -Unique).Count -ne 50000) { throw "Duplicate transaction records: $v2Policy" }
    if (@($v2Details | Where-Object { [long]$_.'Tx finally commit timestamp' -le 0 }).Count -ne 0) { throw "A transaction has no final commit time: $v2Policy" }
    Write-Output "Verified 50000 committed transactions: $v2Policy"
}
Write-Output "All four blockchain runs completed. Results: $v2Suite"
```

不要在运行中重新编译公共程序或修改公共源码。若端口 43000—43064 已被其他实验占用，启动器会明确报错；应先结束对应实验，而不是按进程名批量终止其他任务。已有执行结果的目录不能再次启动，失败也不应删除日志后强行原地重跑。

## 4. 第三步：使用现有分析器生成最终对照表

每组生成一份必要的 `block_summary.json`（链上结果汇总），总目录生成一份 `comparison_chain_50k.csv`（四策略最终对照表）。这里复用项目现有分析工具，不新增分析脚本。

```powershell
$v2Rows = @()
foreach ($v2Policy in $v2Policies) {
    $v2Block = Join-Path $v2Suite $v2Policy
    $v2Status = Get-Content -LiteralPath (Join-Path $v2Block 'run_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($v2Status.state -ne 'completed') { throw "Cannot summarize an unfinished run: $v2Policy" }
    $v2SummaryFile = Join-Path $v2Block 'block_summary.json'
    $v2AnalyzeArgs = @(
        '-B', '.\spring_lite\analyze_block_eval.py',
        '--result_path', (Join-Path $v2Block 'expTest\result'),
        '--spring_io_path', (Join-Path $v2Block 'spring_io'),
        '--log', (Join-Path $v2Block 'supervisor.out.log'),
        '--shards', '16', '--output_json', $v2SummaryFile
    )
    python @v2AnalyzeArgs | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Result analysis failed: $v2Policy" }
    $v2Report = Get-Content -LiteralPath $v2SummaryFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $v2All = $v2Report.periods.all_active
    $v2Latency = $v2Report.latency_details
    $v2Pace = $v2Report.injection_pace
    if ($v2Latency.unique_tx_hash_count -ne 50000 -or $v2Latency.missing_tx_hash_count -ne 0 -or $v2Latency.duplicate_tx_hash_count -ne 0) { throw "Transaction identity check failed: $v2Policy" }
    if ($v2Latency.valid_latency_count -ne 50000 -or $v2All.effective_total -ne 50000) { throw "Incomplete result statistics: $v2Policy" }
    if ($v2Pace.final_cumulative_tx -ne 50000) { throw "Incomplete injection records: $v2Policy" }
    if ($v2Policy -eq 'ppo') {
        if ($v2Report.decisions.decision_count -le 0 -or $v2Report.decisions.python_ppo_count -ne $v2Report.decisions.decision_count) { throw 'PPO decisions are missing or contain fallback actions' }
    }
    if ($v2Pace.attainment_ratio -lt 0.95) { Write-Warning "Actual injection rate is below 95% of target: $v2Policy. Keep this result for diagnosis." }

    $v2Rows += [pscustomobject]@{
        policy = $v2Policy
        committed_txs = $v2Latency.unique_tx_hash_count
        target_inject_tps = $v2Pace.target_tps
        actual_inject_tps = [math]::Round($v2Pace.actual_offered_tps, 3)
        achieved_ratio = [math]::Round($v2Pace.attainment_ratio, 5)
        cross_ratio = [math]::Round($v2All.weighted_cross_ratio, 6)
        committed_tps = [math]::Round($v2All.wall_tps, 3)
        latency_mean_sec = [math]::Round($v2Latency.mean_sec, 4)
        latency_p95_sec = [math]::Round($v2Latency.p95_sec, 4)
        max_load_share = [math]::Round($v2All.load_balance.aggregate_max_shard_load_share, 6)
        jain_fairness = [math]::Round($v2All.load_balance.aggregate_jain_fairness, 6)
    }
    $v2Rows | Export-Csv -LiteralPath (Join-Path $v2Suite 'comparison_chain_50k.csv') -NoTypeInformation -Encoding UTF8
}
$v2Rows | Format-Table policy,committed_txs,actual_inject_tps,cross_ratio,committed_tps,latency_mean_sec,latency_p95_sec -AutoSize
Write-Output "Final comparison: $(Join-Path $v2Suite 'comparison_chain_50k.csv')"
```

四行全部生成才表示四策略对照完整。表中跨片率和最大负载份额为 0—1 的比例，例如 0.15 表示 15%；吞吐量统一使用 `all_active.wall_tps`，即完整有效提交时间跨度下的交易速率，不用各轮 TPS 的算术平均代替。最大负载份额和 Jain 公平性采用全窗口累计有效负载，公平性越接近 1 越均衡。P95 延迟指 95% 的交易确认延迟不超过这个值。

脚本所报告的 `actual_inject_tps` 是监督节点注入调度记录的速率，不等同于节点实际接收或最终提交速率；其计时从首批首次发送后开始，不包含进程启动和首批放置预处理。达成率低于 95% 的提示是本轮诊断阈值，不是已经验证的系统容量边界。

后续分析提供整个新生成的 `$v2Suite` 目录即可；无需把旧训练目录或模型再复制一份。原始日志、配置、数据库和结果继续留在各策略目录。

如果关闭并重开 PowerShell，只为执行第三步，先恢复变量（将路径换成第一步实际输出的新目录）：

```powershell
Set-Location -LiteralPath 'E:\project_iot\block-emulator-main-iot'
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$v2Policies = @('hash', 'heuristic', 'candidate_only', 'ppo')
$v2Suite = '这里填写第一步实际输出的完整 block_compare_50k_i250_时间戳目录'
if (-not (Test-Path -LiteralPath $v2Suite -PathType Container)) { throw 'Campaign directory does not exist' }
```

## 5. 新数据下的注入速度上限

目前没有链上饱和测试结果，不能给出已经测定的稳定上限。数据集没有自带一个固定的注入速度上限；它与分片放置、交易集中程度、区块参数、共识和机器性能共同有关。

- 参数层面：准备脚本只要求正整数，没有把注入速度限制为 1,000；每批 1,000 不等于每秒最多 1,000。
- 已有实测：1 万笔 PPO 实验设置 250 笔/秒，注入记录最终为 249.995821 笔/秒，全部交易提交；这证明这次有限窗口按目标完成，不证明更高速度不可用或 250 已经达到饱和。
- 区块容量参考：每片每 5 秒最多处理 1,000 个执行阶段，约 200 阶段/秒。跨片交易占用两片各一个阶段，还要考虑最忙分片，不能直接把 16×200=3,200 当作可持续原始交易 TPS。

根据现有 5 万笔离线记录，估计公式为：

`容量参考 = 200 / (最忙分片执行阶段总数 / 50000)`

| 方法 | 最忙分片每笔原始交易对应的阶段数 | 条件性容量估算（原始交易/秒） |
|---|---:|---:|
| 哈希 | 0.28766 | 695.3 |
| 启发式 | 0.24532 | 815.3 |
| 仅候选评分 | 0.24736 | 808.5 |
| PPO | 0.24824 | 805.7 |

这些是**固定当前离线放置分布、理想出块条件下的容量参考**，不是当前电脑的实测极限。它们没有完整计入共识、磁盘、网络、同机 65 个进程和 Python 推理开销；链上放置也可能随反馈和注入速度变化。因此不能宣布“最大注入速度就是 800”，也不能据此保证所有策略在 600 或 700 下不积压。

本轮先统一 250。之后若专门研究容量，可按 250→400→500→600→700→800→1,000 笔/秒逐档比较，各方法在同一档使用相同窗口、重复运行；改变 `$v2Rate` 后从第一步新建整组目录，避免混用结果。超过容量的注入值可以作为压力测试，但不属于已确认的稳定速度。

判断稳定容量，需要同时看实际注入达成率、注入期间的积压变化、提交吞吐量与 P95 延迟。全部交易最终提交并不代表注入期间没有积压；提高目标后只增加排队和排空时间而不增加吞吐，说明已接近或超过瓶颈。5 万笔在高速度下持续时间较短，稳定上限还应使用更长窗口验证，不能仅靠一次短跑确定。

## 6. 命令核查与术语

四种策略的 5 万笔配置已在系统临时目录实际生成并检查，均采用相同二进制和推理源码摘要；没有启动这四组正式实验。现有分析器已对原 1 万笔结果实际调用，确认本说明使用的结果字段存在，原有效吞吐量复算为 219.7609 TPS。PowerShell 命令另行做语法检查。项目内只保留这份最终说明，不保存核查脚本和临时准备目录。

术语：hash=哈希；heuristic=启发式；candidate_only=仅候选评分；PPO=近端策略优化；policy=策略；seed=随机种子；inject=注入；target/actual=目标/实际；committed=已提交；TPS=每秒处理交易数；latency=延迟；P95=第 95 百分位；Jain fairness=Jain 公平性指数；summary/comparison=汇总/对照；fallback=推理失败后的替代动作；completed=完成；prepared_not_started=已准备但未启动。
