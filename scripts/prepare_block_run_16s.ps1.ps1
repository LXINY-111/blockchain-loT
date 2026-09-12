param(
    [ValidateSet("validation", "test")]
    [string]$Window = "validation",
    # The paced validation sweep selected 250 TPS as the conservative rate for
    # same-load method comparisons; callers still pass the rate explicitly.
    [int]$InjectSpeed = 250,
    [int]$Seed = 7,
    [ValidateRange(0, 16)]
    [int]$CandidateTopK = 7,
    [ValidateRange(0.0, 1.0)]
    [double]$IOTCSTRWeight = 0.55,
    [ValidateRange(0.0, 1.0)]
    [double]$IOTBalanceWeight = 0.30,
    [string]$ModelPath = "",
    [string]$ExperimentName = "",
    # Optional shared output root, for example E:\project_iot\实验结果10.
    [string]$ExperimentRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PrepareScript = Join-Path $ProjectRoot "spring_lite\prepare_block_run.py"

if ($InjectSpeed -le 0) {
    throw "InjectSpeed must be positive."
}

# Python owns all path construction and writes UTF-8 JSON. This wrapper keeps
# Windows PowerShell 5.1 execution convenient without duplicating the protocol.
$arguments = @(
    $PrepareScript,
    "--window", $Window,
    "--inject_speed", [string]$InjectSpeed,
    "--seed", [string]$Seed,
    "--candidate_top_k", [string]$CandidateTopK,
    "--iot_cstr_weight", [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:R}",
        $IOTCSTRWeight
    ),
    "--iot_balance_weight", [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:R}",
        $IOTBalanceWeight
    )
)
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $arguments += @("--model", $ModelPath)
}
if (-not [string]::IsNullOrWhiteSpace($ExperimentName)) {
    $arguments += @("--experiment", $ExperimentName)
}

$previousExperimentRoot = $env:BLOCKEMULATOR_EXPERIMENT_ROOT
try {
    if (-not [string]::IsNullOrWhiteSpace($ExperimentRoot)) {
        $env:BLOCKEMULATOR_EXPERIMENT_ROOT = [System.IO.Path]::GetFullPath($ExperimentRoot)
    }
    $json = & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment-8 run preparation failed."
    }
}
finally {
    $env:BLOCKEMULATOR_EXPERIMENT_ROOT = $previousExperimentRoot
}
$context = $json | ConvertFrom-Json

[PSCustomObject]@{
    RunRoot = [string]$context.run_root
    ExpTest = [string]$context.exp_test
    ParamsPath = Join-Path $ProjectRoot "paramsConfig.json"
    ModelPath = [string]$context.model
    Scheme = [string]$context.scheme
    Window = [string]$context.phase
    DatasetStartTx = [int64]$context.dataset_start_tx
    TotalDataSize = [int64]$context.total_data_size
    InjectSpeed = [int64]$context.inject_speed
    CandidateTopK = [int]$context.candidate_top_k
    IOTCSTRWeight = [double]$context.iot_reward_weights.cross_shard
    IOTBalanceWeight = [double]$context.iot_reward_weights.balance
    WorkloadProfile = [string]$context.workload_profile
}
