@echo off
REM Experiment 8: local 16-shard layout, 4 PBFT nodes per shard.
REM Keep -S aligned with params.ShardNum / DEFAULT_SHARD_NUM and ipTable.json.
for /L %%s in (0,1,15) do (
    for /L %%n in (0,1,3) do (
        start cmd /k blockEmulator_Windows_Precompile.exe -n %%n -N 4 -s %%s -S 16
    )
)

start cmd /k blockEmulator_Windows_Precompile.exe -c -N 4 -S 16
