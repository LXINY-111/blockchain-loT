[CmdletBinding()]
param(
    # Seed7 already has a complete failed-selection audit. This entry is
    # intentionally limited to the two remaining preregistered seeds.
    [Parameter(Mandatory = $true)]
    [ValidateSet(17, 27)]
    [int]$Seed,
    # Read-only environment check used before a long training run.
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
# Windows PowerShell 5.1 reads a UTF-8 script without BOM through the active
# legacy code page. Build the Chinese directory name from Unicode code points
# so the canonical Result9 path cannot be corrupted into mojibake.
$resultDirectoryName = -join @(
    [char]0x5B9E,
    [char]0x9A8C,
    [char]0x7ED3,
    [char]0x679C,
    [char]0x0039
)
$resultsRoot = Join-Path (Split-Path $projectRoot -Parent) $resultDirectoryName
$dataset = Join-Path $projectRoot "data_iot\selectedTxs_iot_multi_anchor_full.csv"
$sidecar = Join-Path $projectRoot "data_iot\iot_flow_sidecar_multi_anchor_full.csv"

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Depth = 12
    )
    $Value | ConvertTo-Json -Depth $Depth |
        Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Windows PowerShell 5.1 wraps native stderr as ErrorRecord. Continue
        # mode preserves the full traceback and lets us inspect LASTEXITCODE.
        & python @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath |
            ForEach-Object { Write-Host ([string]$_) }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return [int]$exitCode
}

function Write-SelectionFailureAudit {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$TrainRun,
        [Parameter(Mandatory = $true)][int]$FailedSeed
    )
    $validationRecords = @(Get-Content -LiteralPath (Join-Path $TrainRun "train.jsonl") |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.event -in @("validation", "final_validation") })
    $actionByCandidate = @{}
    foreach ($record in $validationRecords) {
        $key = "$($record.event)|$($record.epoch)"
        $actionByCandidate[$key] = @($record.summary.action_dist)
    }

    $candidateAudit = @($Manifest.all_checkpoints | ForEach-Object {
        $active = [double]$_.active_shards_mean
        $hotspot = [double]$_.stage_max_load_share
        $deficit = [math]::Max([double]0.0, (8.0 - $active) / 8.0)
        [pscustomobject][ordered]@{
            event = $_.event
            epoch = [int]$_.epoch
            cross_ratio = [double]$_.cross_ratio
            active_shards_mean = $active
            active_deficit_ratio = $deficit
            stage_max_load_share = $hotspot
            strict_eligible = ($active -ge 8.0 -and $hotspot -le 0.35)
            tolerance_eligible = ($hotspot -le 0.35 -and $deficit -le 0.05)
            action_dist = $actionByCandidate["$($_.event)|$($_.epoch)"]
            checkpoint_path = $_.checkpoint_path
        }
    })
    $failure = [ordered]@{
        schema_version = 1
        created_at = (Get-Date).ToString("o")
        status = "selection_failed"
        method_label = "Original SPRING-PPO adapted to our IoT BlockEmulator setting"
        seed = $FailedSeed
        selection_policy = "hierarchical_constrained_pareto_v1"
        constraints = [ordered]@{
            min_active_shards = 8.0
            max_stage_hotspot = 0.35
            max_active_deficit_ratio = 0.05
        }
        strict_candidate_count = @($candidateAudit | Where-Object { $_.strict_eligible }).Count
        tolerance_candidate_count = @($candidateAudit | Where-Object { $_.tolerance_eligible }).Count
        reason = "No checkpoint satisfies the frozen strict or active-tolerance layer."
        test_window_accessed = $false
        candidates = $candidateAudit
    }
    $failurePath = Join-Path $TrainRun "selection_failed.json"
    Write-JsonFile -Value $failure -Path $failurePath
    Write-Host ($candidateAudit |
        Select-Object event,epoch,cross_ratio,active_shards_mean,active_deficit_ratio,stage_max_load_share,action_dist |
        Format-Table -Wrap -AutoSize |
        Out-String)
    return $failurePath
}

if (-not (Test-Path -LiteralPath $resultsRoot -PathType Container)) {
    throw "Result9 root is missing: $resultsRoot"
}
if (-not (Test-Path -LiteralPath $dataset -PathType Leaf)) {
    throw "Dataset is missing: $dataset"
}
if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
    throw "IoT sidecar is missing: $sidecar"
}
if ($PreflightOnly) {
    [ordered]@{
        status = "preflight_passed"
        seed = $Seed
        project_root = $projectRoot
        results_root = $resultsRoot
        dataset = $dataset
        sidecar = $sidecar
        writes_performed = $false
    } | ConvertTo-Json -Depth 4
    exit 0
}

$modelPath = Join-Path $resultsRoot "models\original_spring_ppo_16s_seed${Seed}.pt"
$selectionPath = Join-Path $resultsRoot "models\original_spring_ppo_16s_seed${Seed}_selection.json"
if ((Test-Path -LiteralPath $modelPath) -or (Test-Path -LiteralPath $selectionPath)) {
    throw "A frozen model or selection record already exists for seed $Seed; refusing to overwrite it."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$trainRun = Join-Path $resultsRoot "offline_training\original_spring_ppo_pareto_16s_seed${Seed}_$stamp"
New-Item -ItemType Directory -Path $trainRun | Out-Null

$runContext = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToString("o")
    method_label = "Original SPRING-PPO adapted to our IoT BlockEmulator setting"
    seed = $Seed
    shards = 16
    state_dim = 177
    iot_feature_dim = 0
    mdp_mode = "spring"
    tx_identity = "iot"
    reward_mode = "paper"
    lambda_weight = 0.5
    beta = 0.1
    candidate_top_k = 0
    capacity_guard = 0
    capacity_guard_factor = 1.5
    train_window = [ordered]@{ start_tx = 0; max_txs = 2500000 }
    validation_window = [ordered]@{ start_tx = 2500000; max_txs = 444019 }
    test_window_accessed = $false
}
Write-JsonFile -Value $runContext -Path (Join-Path $trainRun "run_context.json")

$trainArguments = @(
    "-u", ".\spring_lite\train_offline.py",
    "--csv", ".\data_iot\selectedTxs_iot_multi_anchor_full.csv",
    "--sidecar", ".\data_iot\iot_flow_sidecar_multi_anchor_full.csv",
    "--mdp_mode", "spring", "--tx_identity", "iot",
    "--model", (Join-Path $trainRun "best_scalar.pt"),
    "--last_model", (Join-Path $trainRun "last.pt"),
    "--epoch_checkpoint_dir", (Join-Path $trainRun "checkpoints"),
    "--pareto_manifest", (Join-Path $trainRun "pareto_manifest.json"),
    "--shards", "16", "--epochs", "15",
    "--start_tx", "0", "--max_txs", "2500000",
    "--validation_start_tx", "2500000", "--validation_max_txs", "444019",
    "--tx_batch_size", "1000", "--max_block_size", "1000",
    "--block_interval_ms", "5000", "--sender_pos_mode", "1",
    "--temporal_top_k", "8", "--reward_mode", "paper",
    "--lambda_weight", "0.5", "--beta", "0.1",
    "--candidate_top_k", "0", "--capacity_guard", "0",
    "--capacity_guard_factor", "1.5", "--candidate_load_weight", "1.0",
    "--eval_every_epochs", "1", "--seed", "$Seed", "--device", "cpu",
    "--log_jsonl", (Join-Path $trainRun "train.jsonl")
)
$trainExitCode = Invoke-PythonLogged `
    -Arguments $trainArguments `
    -LogPath (Join-Path $trainRun "console.log")
if ($trainExitCode -ne 0) {
    throw "Training failed for seed $Seed with exit code $trainExitCode."
}

$epochCheckpoints = @(Get-ChildItem -LiteralPath (Join-Path $trainRun "checkpoints") -Filter "epoch_*.pt" -File)
if ($epochCheckpoints.Count -ne 15) {
    throw "Incomplete audit chain: expected 15 epoch checkpoints, got $($epochCheckpoints.Count)."
}
$manifestPath = Join-Path $trainRun "pareto_manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (@($manifest.all_checkpoints).Count -ne 16) {
    throw "Expected 15 epoch candidates plus one final post-update candidate."
}

$selectionArguments = @(
    ".\spring_lite\select_pareto_checkpoint.py",
    "--manifest", $manifestPath,
    "--output_model", $modelPath,
    "--selection_json", $selectionPath,
    "--min_active_shards", "8",
    "--max_stage_hotspot", "0.35",
    "--max_active_deficit_ratio", "0.05"
)
$selectionExitCode = Invoke-PythonLogged `
    -Arguments $selectionArguments `
    -LogPath (Join-Path $trainRun "hierarchical_selection_stdout.log")
if ($selectionExitCode -ne 0) {
    $failurePath = Write-SelectionFailureAudit `
        -Manifest $manifest `
        -TrainRun $trainRun `
        -FailedSeed $Seed
    Write-Host "Selection failed under the frozen rule; audit saved to $failurePath"
    exit 2
}

$selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$selectedEpoch = [int]$selection.selected.epoch
$selectedEvent = [string]$selection.selected.event
$evalStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalRoot = Join-Path $resultsRoot "offline_eval\original_spring_ppo_${selectedEvent}_epoch${selectedEpoch}_16s_seed${Seed}_validation_$evalStamp"
New-Item -ItemType Directory -Path $evalRoot | Out-Null
Copy-Item -LiteralPath $selectionPath -Destination (Join-Path $evalRoot "selection.json")

$evalArguments = @(
    "-u", ".\spring_lite\eval_offline.py",
    "--csv", ".\data_iot\selectedTxs_iot_multi_anchor_full.csv",
    "--sidecar", ".\data_iot\iot_flow_sidecar_multi_anchor_full.csv",
    "--mdp_mode", "spring", "--tx_identity", "iot",
    "--model", $modelPath, "--policy", "ppo", "--shards", "16",
    "--start_tx", "2500000", "--max_txs", "444019",
    "--tx_batch_size", "1000", "--max_block_size", "1000",
    "--block_interval_ms", "5000", "--sender_pos_mode", "1",
    "--temporal_top_k", "8", "--reward_mode", "paper",
    "--lambda_weight", "0.5", "--beta", "0.1",
    "--candidate_top_k", "0", "--capacity_guard", "0",
    "--capacity_guard_factor", "1.5", "--candidate_load_weight", "1.0",
    "--seed", "$Seed", "--device", "cpu",
    "--log_jsonl", (Join-Path $evalRoot "validation.jsonl")
)
$evalExitCode = Invoke-PythonLogged `
    -Arguments $evalArguments `
    -LogPath (Join-Path $evalRoot "console.log")
if ($evalExitCode -ne 0) {
    throw "Independent validation failed for seed $Seed with exit code $evalExitCode."
}

$validation = Get-Content -LiteralPath (Join-Path $evalRoot "validation.jsonl") -Tail 1 | ConvertFrom-Json
if (
    [int]$validation.loaded_txs -ne 444019 -or
    [int]$validation.dataset_start_tx -ne 2500000 -or
    [int]$validation.dataset_end_tx_exclusive -ne 2944019 -or
    [int]$validation.state_dim -ne 177 -or
    [int]$validation.iot_feature_dim -ne 0 -or
    [int]$validation.candidate_top_k -ne 0 -or
    [int]$validation.capacity_guard -ne 0 -or
    [bool]$validation.sample
) {
    throw "Independent validation protocol audit failed for seed $Seed."
}

$completion = [ordered]@{
    schema_version = 1
    completed_at = (Get-Date).ToString("o")
    status = "validated"
    method_label = "Original SPRING-PPO adapted to our IoT BlockEmulator setting"
    seed = $Seed
    train_run = $trainRun
    model = $modelPath
    model_sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
    selection = $selectionPath
    selection_tier = $selection.selection_tier
    selected_event = $selectedEvent
    selected_epoch = $selectedEpoch
    validation_root = $evalRoot
    validation_summary = $validation.summary
    test_window_accessed = $false
}
Write-JsonFile -Value $completion -Path (Join-Path $evalRoot "run_complete.json")
$completion | ConvertTo-Json -Depth 12
