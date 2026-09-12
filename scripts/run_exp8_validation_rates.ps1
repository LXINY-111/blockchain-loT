param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [int[]]$Rates = @(200, 250, 300, 350),
    [int]$Seed = 7,
    [ValidateRange(0, 16)]
    [int]$CandidateTopK = 7,
    [ValidateRange(0.0, 1.0)]
    [double]$IOTCSTRWeight = 0.55,
    [ValidateRange(0.0, 1.0)]
    [double]$IOTBalanceWeight = 0.30,
    [string]$ExperimentRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PrepareScript = Join-Path $PSScriptRoot "prepare_block_run_16s.ps1.ps1"
$LaunchScript = Join-Path $PSScriptRoot "run_block_eval_16s.ps1.ps1"
$FinalizeScript = Join-Path $ProjectRoot "spring_lite\finalize_exp8_block_run.py"

if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Frozen PPO model not found: $ModelPath"
}
if ($Rates.Count -eq 0 -or ($Rates | Where-Object { $_ -le 0 })) {
    throw "Every validation injection rate must be positive."
}

$completed = New-Object System.Collections.Generic.List[object]
foreach ($rate in $Rates) {
    $run = & $PrepareScript `
        -Window validation `
        -InjectSpeed $rate `
        -Seed $Seed `
        -CandidateTopK $CandidateTopK `
        -IOTCSTRWeight $IOTCSTRWeight `
        -IOTBalanceWeight $IOTBalanceWeight `
        -ModelPath $ModelPath `
        -ExperimentRoot $ExperimentRoot

    Write-Host (
        "[RATE START] scheme=$($run.Scheme) target_tps=$rate " +
        "run=$($run.RunRoot)"
    )
    try {
        $launched = & $LaunchScript -RunRoot $run.RunRoot
        Wait-Process -Id $launched.SupervisorPid

        # Nodes receive the supervisor stop signal independently. Wait until every
        # process has closed its database before running the final state check.
        $deadline = (Get-Date).AddMinutes(10)
        do {
            $running = Get-Process -ErrorAction SilentlyContinue |
                Where-Object { $_.ProcessName -like "blockEmulator*" }
            if (-not $running) {
                break
            }
            if ((Get-Date) -ge $deadline) {
                throw "Timed out waiting for BlockEmulator nodes to stop."
            }
            Start-Sleep -Seconds 2
        } while ($true)

        $completionJson = & python $FinalizeScript --run_root $run.RunRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Finalization failed for target rate $rate."
        }
    }
    catch {
        Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessName -like "blockEmulator*" } |
            Stop-Process -Force -ErrorAction SilentlyContinue
        throw
    }
    $completion = $completionJson | ConvertFrom-Json
    $completed.Add([PSCustomObject]@{
        Scheme = $run.Scheme
        TargetTPS = $rate
        RunRoot = $run.RunRoot
        StateCheck = [string]$completion.state_check
        ActualOfferedTPS = [double]$completion.injection_pace.actual_offered_tps
        WallTPS = [double]$completion.all_active.wall_tps
        CrossRatio = [double]$completion.all_active.weighted_cross_ratio
        AverageLatencySec = [double]$completion.all_active.avg_confirm_latency_sec
    })
    Write-Host ((
            "[RATE DONE] target_tps={0} actual_tps={1:F3} wall_tps={2:F3} " +
            "cross={3:F4} latency={4:F3}s"
        ) -f $rate,
        [double]$completion.injection_pace.actual_offered_tps,
        [double]$completion.all_active.wall_tps,
        [double]$completion.all_active.weighted_cross_ratio,
        [double]$completion.all_active.avg_confirm_latency_sec)
}

$completed
