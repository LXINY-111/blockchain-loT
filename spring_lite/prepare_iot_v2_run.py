"""为新数据生成独立链上运行目录；不覆盖项目原配置、不启动实验进程。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from config import ROOT_DIR
from iot_v2 import V2_DIRECTORY, V2_LOAD_SEMANTICS, load_rows
from checkpoint_compat import checkpoint_config_mismatches, format_checkpoint_mismatch
from mechanism import add_mechanism_args, settings, metadata, input_dim

POLICIES = {"hash": 0, "heuristic": 1, "ppo": 2, "random": 3, "candidate_only": 7}
RUNTIME_FILES = ("infer_server.py", "update_online.py", "action_mask.py", "action_select.py",
                 "checkpoint_compat.py", "config.py", "heuristic.py", "ppo.py", "model.py", "mechanism.py")


def prepare(args):
    mechanism = settings(args)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite run directory: {output}")
    dataset = args.dataset_dir.resolve()
    chain, scene = dataset / "selectedTxs_iot_v2.csv", dataset / "transaction_scene.csv"
    if args.max_txs <= 0 or args.start_tx < 0 or args.inject_speed <= 0:
        raise ValueError("A positive complete transaction window and injection speed are required")
    if not 1024 <= args.base_port <= 65535 - 64:
        raise ValueError("base-port must leave room for 65 local ports")
    if not args.binary.is_file():
        raise FileNotFoundError("Build the updated Go executable first: " + str(args.binary))
    # 在写运行配置前核对所选窗口，不把不存在的 200 万笔旧窗口沿用到 30 万笔数据。
    count = sum(1 for _ in load_rows(chain, scene, args.start_tx, args.max_txs))
    config = json.loads((ROOT_DIR / "paramsConfig.json").read_text(encoding="utf-8-sig"))
    config.update(ConsensusMethod=3, SpringMode=POLICIES[args.policy], SpringOnlineTrain=0,
                  SpringEvalSample=0, SpringRandomSeed=args.seed, SpringIOTMode=1,
                  SpringIOTIdentityMode=2, SpringIOTFeatureDim=10, SpringIOTSidecarFile=str(scene),
                  SpringModelFile=str(args.model.resolve()) if args.model else "NOT_APPLICABLE",
                  DatasetFile=str(chain), DatasetStartTx=args.start_tx, TotalDataSize=count,
                  SpringSenderPosMode=1, SpringCandidateTopK=7, SpringCapacityGuard=0,
                  SpringCapacityGuardFactor=1.5, SpringCandidateLoadWeight=1.0,
                  SpringIOTCSTRWeight=0.55, SpringIOTBalanceWeight=0.30,
                  SpringIOTCommCostWeight=0.10, SpringIOTHotspotWeight=0.05,
                  SpringIOTHotspotThreshold=0.45, SpringRewardLambda=0.5, SpringRewardBeta=0.1,
                  TxBatchSize=1000, Block_Interval=5000, BlockSize=1000,
                  UseBlocksizeInBytes=0, InjectSpeed=args.inject_speed,
                  ExpDataRootDir=str(output / "expTest"))
    config.update(SpringMechanismVersion=mechanism['mechanism_version'],
                  SpringFeatureMode=mechanism['feature_mode'], SpringSceneCostMode=mechanism['scene_cost_mode'])
    expected = dict(mdp_mode="iot", tx_identity_resolved="iot_v2", shards=16, iot_feature_dim=10,
                    sender_pos_mode=1, reward_mode="iot_dense_balanced", iot_cstr_weight=0.55,
                    iot_balance_weight=0.30, iot_comm_cost_weight=0.10, iot_hotspot_weight=0.05,
                    candidate_top_k=7, capacity_guard=0, capacity_guard_factor=1.5,
                    candidate_load_weight=1.0, max_block_size=1000, block_interval_ms=5000,
                    load_semantics_version=V2_LOAD_SEMANTICS)
    expected.update(metadata(args))
    if args.policy == "ppo":
        if not args.model or not args.model.is_file():
            raise FileNotFoundError("PPO requires a trained v2 --model")
        import torch
        payload = torch.load(args.model, map_location="cpu", weights_only=False)
        dimensions = input_dim(16, 10, mechanism['mechanism_version'])
        if payload.get("state_dim") != dimensions or payload.get("action_dim") != 16:
            raise ValueError(f"PPO model must have {dimensions} input dimensions and 16 actions for this version")
        mismatches = checkpoint_config_mismatches(payload.get("extra", {}), expected)
        if mismatches:
            raise ValueError(format_checkpoint_mismatch(mismatches))
    ports = {str(sid): {str(nid): f"127.0.0.1:{args.base_port + sid*4+nid}"
                        for nid in range(4)} for sid in range(16)}
    ports["2147483647"] = {"0": f"127.0.0.1:{args.base_port+64}"}
    output.mkdir(parents=True)
    # 共享唯一一份程序、源码和训练模型，不把测试/生成工具复制到实验目录。
    # 启动前按摘要检查内容；实验使用的版本依据仍保存在配置中。
    binary = args.binary.resolve()
    python_directory = Path(__file__).resolve().parent
    for name, value in (("paramsConfig.json", config), ("ipTable.json", ports),
                        ("run_context.json", dict(binary=str(binary), python=sys.executable,
                            python_dir=str(python_directory),
                            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                            python_files_sha256={name: hashlib.sha256((python_directory / name).read_bytes()).hexdigest()
                                                 for name in RUNTIME_FILES},
                            model=str(args.model.resolve()) if args.policy == "ppo" else "",
                            model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest() if args.policy == "ppo" else "",
                            shards=16, nodes=4, base_port=args.base_port, policy=args.policy,
                            start_tx=args.start_tx, max_txs=count, expected_checkpoint=expected))):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "run.ps1").write_text(RUN_SCRIPT, encoding="utf-8-sig")
    print(json.dumps(dict(status="prepared_not_started", run_dir=str(output), transactions=count,
                          command=f"powershell -ExecutionPolicy Bypass -File \"{output / 'run.ps1'}\""), ensure_ascii=True))


RUN_SCRIPT = r'''# 共享项目程序和源码；本目录只保存运行配置、日志、数据库和结果。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$run = Get-Content -LiteralPath '.\run_context.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ((Test-Path -LiteralPath '.\run_status.json') -or (Test-Path -LiteralPath '.\supervisor.out.log') -or (Test-Path -LiteralPath '.\expTest')) {
    throw 'This run directory already contains execution results. Prepare a new directory to rerun.'
}
function Assert-ResourceHash($path, $expected) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Run resource is missing: $path" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $expected) {
        throw "Run resource changed after preparation: $path. Prepare a new run directory."
    }
}
Assert-ResourceHash $run.binary $run.binary_sha256
if ($run.model) { Assert-ResourceHash $run.model $run.model_sha256 }
foreach ($file in $run.python_files_sha256.PSObject.Properties) {
    Assert-ResourceHash (Join-Path $run.python_dir $file.Name) $file.Value
}
$env:SPRING_PYTHON = $run.python
$env:SPRING_PYTHON_DIR = $run.python_dir
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Write-Output ("Starting policy={0}; transactions={1}; model={2}; binary={3}" -f $run.policy, $run.max_txs, $run.model, $run.binary)
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$ports = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
foreach ($endpoint in $ports) {
    if ($endpoint.Port -ge $run.base_port -and $endpoint.Port -le ($run.base_port + 64)) {
        throw "Requested run port is already occupied: $($endpoint.Port)"
    }
}
$started = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$outcome = @{state='running'; started_utc=[DateTime]::UtcNow.ToString('o'); error=$null; processes=@()}
$outcome | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath '.\run_status.json' -Encoding UTF8
try {
    for ($sid = 0; $sid -lt $run.shards; $sid++) {
        for ($nid = 0; $nid -lt $run.nodes; $nid++) {
            $process = Start-Process -FilePath $run.binary -ArgumentList @('-S', $run.shards, '-N', $run.nodes, '-s', $sid, '-n', $nid) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "node_${sid}_${nid}.out.log" -RedirectStandardError "node_${sid}_${nid}.err.log"
            $started.Add($process)
            # Windows PowerShell 5.1 必须在退出前保留句柄，否则 ExitCode 可能为空。
            $null = $process.Handle
        }
    }
    $supervisor = Start-Process -FilePath $run.binary -ArgumentList @('-S', $run.shards, '-N', $run.nodes, '-c') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'supervisor.out.log' -RedirectStandardError 'supervisor.err.log'
    $started.Add($supervisor)
    $null = $supervisor.Handle
    $supervisor.WaitForExit()
    if ($null -eq $supervisor.ExitCode) { throw 'Cannot read supervisor exit code' }
    if ($supervisor.ExitCode -ne 0) { throw "Supervisor exit code $($supervisor.ExitCode); read supervisor.err.log" }
    foreach ($process in $started) {
        if (-not $process.WaitForExit(60000)) { throw "Node did not exit after supervisor completion" }
        if ($null -eq $process.ExitCode) { throw "Cannot read process $($process.Id) exit code" }
        if ($process.ExitCode -ne 0) { throw "Node $($process.Id) exit code $($process.ExitCode); read node stderr logs" }
    }
    $outcome.state = 'completed'
    Write-Output 'Run completed. Results are in expTest and spring_io.'
} catch {
    $outcome.state = 'failed'
    $outcome.error = $_.Exception.Message
    throw
} finally {
    # 只清理本脚本启动且仍存活的进程，不按进程名终止其他实验。
    foreach ($process in $started) {
        if (-not $process.HasExited) { Stop-Process -InputObject $process -ErrorAction SilentlyContinue }
    }
    $outcome.finished_utc = [DateTime]::UtcNow.ToString('o')
    $outcome.processes = @($started | ForEach-Object {
        @{pid=$_.Id; exited=$_.HasExited; exit_code=$(if ($_.HasExited) { $_.ExitCode } else { $null })}
    })
    $outcome | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath '.\run_status.json' -Encoding UTF8
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=V2_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT_DIR / "analysis_outputs" / "blockEmulator_iot_v2.exe")
    parser.add_argument("--policy", choices=POLICIES, default="ppo")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--start-tx", type=int, default=250000)
    parser.add_argument("--max-txs", type=int, default=10000)
    parser.add_argument("--inject-speed", type=int, default=250)
    parser.add_argument("--base-port", type=int, default=43000)
    parser.add_argument("--seed", type=int, default=7)
    add_mechanism_args(parser)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
