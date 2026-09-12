param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Binary = Join-Path $ProjectRoot "blockEmulator_Windows_Precompile.exe"

if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
    throw "BlockEmulator binary not found: $Binary"
}
if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "run_context.json") -PathType Leaf)) {
    throw "run_context.json not found under: $RunRoot"
}
$ContextPath = Join-Path $RunRoot "run_context.json"
$context = Get-Content -LiteralPath $ContextPath -Raw -Encoding UTF8 | ConvertFrom-Json

$ParamsPath = Join-Path $ProjectRoot "paramsConfig.json"
$SnapshotPath = Join-Path $RunRoot "paramsConfig.snapshot.json"
if (-not (Test-Path -LiteralPath $SnapshotPath -PathType Leaf)) {
    throw "paramsConfig.snapshot.json not found under: $RunRoot"
}
$currentHash = (Get-FileHash -LiteralPath $ParamsPath -Algorithm SHA256).Hash
$snapshotHash = (Get-FileHash -LiteralPath $SnapshotPath -Algorithm SHA256).Hash
if ($currentHash -ne $snapshotHash) {
    throw "paramsConfig.json does not match this run's snapshot."
}
$config = Get-Content -LiteralPath $ParamsPath -Raw -Encoding UTF8 | ConvertFrom-Json
$expectedExpTest = [System.IO.Path]::GetFullPath((Join-Path $RunRoot "expTest"))
$configuredExpTest = [System.IO.Path]::GetFullPath([string]$config.ExpDataRootDir)
if ($configuredExpTest -ne $expectedExpTest) {
    throw "ExpDataRootDir does not point to this run: $configuredExpTest"
}
if ([int]$context.shards -ne 16) {
    throw "This launcher only supports the fixed 16-shard experiment."
}
if (
    [int64]$config.DatasetStartTx -ne [int64]$context.dataset_start_tx -or
    [int64]$config.TotalDataSize -ne [int64]$context.total_data_size
) {
    throw "paramsConfig.json data window does not match run_context.json."
}
if ([int64]$config.InjectSpeed -ne [int64]$context.inject_speed) {
    throw "paramsConfig.json InjectSpeed does not match run_context.json."
}
if (
    [int64]$config.BlockSize -ne [int64]$context.block_size -or
    [int64]$config.Block_Interval -ne [int64]$context.block_interval_ms
) {
    throw "paramsConfig.json block capacity does not match run_context.json."
}

$running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -like "blockEmulator*" }
if ($running) {
    throw "BlockEmulator processes are already running."
}

$LogRoot = Join-Path $RunRoot "process_logs"
if (Test-Path -LiteralPath $LogRoot) {
    $existing = Get-ChildItem -LiteralPath $LogRoot -Force
    if ($existing.Count -gt 0) {
        throw "Process log directory is not empty: $LogRoot"
    }
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$started = New-Object System.Collections.Generic.List[object]
try {
    for ($shard = 0; $shard -lt 16; $shard++) {
        for ($node = 0; $node -lt 4; $node++) {
            $name = "shard_{0:D2}_node_{1}" -f $shard, $node
            $process = Start-Process `
                -FilePath $Binary `
                -ArgumentList @("-n", $node, "-N", 4, "-s", $shard, "-S", 16) `
                -WorkingDirectory $ProjectRoot `
                -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $LogRoot "$name.stdout.log") `
                -RedirectStandardError (Join-Path $LogRoot "$name.stderr.log") `
                -PassThru
            $started.Add([PSCustomObject]@{
                Role = "shard_node"
                Shard = $shard
                Node = $node
                Pid = $process.Id
            })
        }
    }

    Start-Sleep -Seconds 2
    $supervisor = Start-Process `
        -FilePath $Binary `
        -ArgumentList @("-c", "-N", 4, "-S", 16) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogRoot "supervisor.stdout.log") `
        -RedirectStandardError (Join-Path $LogRoot "supervisor.stderr.log") `
        -PassThru
    $started.Add([PSCustomObject]@{
        Role = "supervisor"
        Shard = $null
        Node = $null
        Pid = $supervisor.Id
    })
}
catch {
    foreach ($record in $started) {
        Stop-Process -Id $record.Pid -Force -ErrorAction SilentlyContinue
    }
    throw
}

$json = $started | ConvertTo-Json -Depth 4
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $RunRoot "processes.json"),
    $json + [Environment]::NewLine,
    $utf8NoBom
)

[PSCustomObject]@{
    RunRoot = $RunRoot
    ProcessCount = $started.Count
    SupervisorPid = $supervisor.Id
    SupervisorLog = Join-Path $LogRoot "supervisor.stdout.log"
}
