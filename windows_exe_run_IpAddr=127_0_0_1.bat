@echo off
REM Explicit legacy opt-in prevents accidentally running the old dataset/model.
if /I not "%~1"=="legacy" (
    echo This is the LEGACY entry. For a prepared IoT run, execute its run.ps1.
    echo To intentionally run this old configuration, pass legacy as the first argument.
    exit /b 2
)
REM Experiment 8: local 16-shard layout, 4 PBFT nodes per shard.
REM Keep -S aligned with params.ShardNum / DEFAULT_SHARD_NUM and ipTable.json.
for /L %%s in (0,1,15) do (
    for /L %%n in (0,1,3) do (
        start cmd /k blockEmulator_Windows_Precompile.exe -n %%n -N 4 -s %%s -S 16
    )
)

start cmd /k blockEmulator_Windows_Precompile.exe -c -N 4 -S 16
