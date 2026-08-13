import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from prepare_exp8_baseline import (
    DATASET_PATH,
    PARAMS_PATH,
    RESULTS_ROOT,
    ROOT_DIR,
    SIDECAR_PATH,
    ensure_clean_runtime_directory,
    running_block_emulator_processes,
    write_json,
)
from profile_iot_window import profile_iot_window
from write_experiment_manifest import build_manifest


@dataclass(frozen=True)
class DatasetWindow:
    name: str
    start_tx: int
    max_txs: int
    label: str


DATASET_WINDOWS = {
    "validation": DatasetWindow(
        name="validation",
        start_tx=2_500_000,
        max_txs=444_019,
        label="validation444k",
    ),
    "test": DatasetWindow(
        name="test",
        start_tx=2_944_019,
        max_txs=2_000_000,
        label="test2M",
    ),
}

SHARD_COUNT = 16
DEFAULT_CANDIDATE_TOP_K = 7
DEFAULT_IOT_CSTR_WEIGHT = 0.55
DEFAULT_IOT_BALANCE_WEIGHT = 0.30
# The current small-grid study only changes cross-shard and balance weights.
# Keeping these two terms fixed makes the established wXXYY name unambiguous.
FIXED_IOT_COMM_COST_WEIGHT = 0.10
FIXED_IOT_HOTSPOT_WEIGHT = 0.05


def weight_percent(weight: float, name: str) -> int:
    if weight < 0:
        raise ValueError(f"{name} must be non-negative")
    percent = int(round(weight * 100))
    if not math.isclose(
        weight,
        percent / 100.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} must use increments of 0.01")
    return percent


def proposed_scheme(
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    iot_cstr_weight: float = DEFAULT_IOT_CSTR_WEIGHT,
    iot_balance_weight: float = DEFAULT_IOT_BALANCE_WEIGHT,
) -> str:
    if candidate_top_k < 0 or candidate_top_k > SHARD_COUNT:
        raise ValueError(
            f"candidate_top_k must be in [0, {SHARD_COUNT}]"
        )
    cstr_percent = weight_percent(iot_cstr_weight, "iot_cstr_weight")
    balance_percent = weight_percent(
        iot_balance_weight,
        "iot_balance_weight",
    )
    total_weight = (
        iot_cstr_weight
        + iot_balance_weight
        + FIXED_IOT_COMM_COST_WEIGHT
        + FIXED_IOT_HOTSPOT_WEIGHT
    )
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "IoT reward weights must sum to 1.0; "
            f"got {total_weight:.6f}"
        )
    return (
        f"ppo_top{candidate_top_k}_"
        f"w{cstr_percent:02d}{balance_percent:02d}_pareto"
    )


def default_model(
    seed: int,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    iot_cstr_weight: float = DEFAULT_IOT_CSTR_WEIGHT,
    iot_balance_weight: float = DEFAULT_IOT_BALANCE_WEIGHT,
) -> Path:
    scheme = proposed_scheme(
        candidate_top_k,
        iot_cstr_weight,
        iot_balance_weight,
    )
    return (
        RESULTS_ROOT
        / "models"
        / f"{scheme}_16s_seed{seed}.pt"
    )


def experiment_name(
    scheme: str,
    window: DatasetWindow,
    inject_speed: int,
    seed: int,
    override: str = "",
) -> str:
    canonical = (
        f"{scheme}_16s_{window.label}_"
        f"i{inject_speed}_seed{seed}"
    )
    custom = override.strip()
    if not custom:
        return canonical
    if custom != canonical and not custom.startswith(canonical + "_"):
        raise ValueError(
            "custom experiment name must preserve the canonical "
            f"configuration prefix: {canonical}"
        )
    return custom


def build_proposed_config(
    base_config: Dict[str, object],
    seed: int,
    model_path: Path,
    exp_test: Path,
    window: DatasetWindow,
    inject_speed: int,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    iot_cstr_weight: float = DEFAULT_IOT_CSTR_WEIGHT,
    iot_balance_weight: float = DEFAULT_IOT_BALANCE_WEIGHT,
) -> Dict[str, object]:
    if inject_speed <= 0:
        raise ValueError("inject_speed must be positive")
    proposed_scheme(
        candidate_top_k,
        iot_cstr_weight,
        iot_balance_weight,
    )
    config = copy.deepcopy(base_config)
    config.update(
        {
            "ConsensusMethod": 3,
            "SpringMode": 2,
            "SpringOnlineTrain": 0,
            "SpringEvalSample": 0,
            "SpringRandomSeed": seed,
            "SpringCandidateTopK": candidate_top_k,
            "SpringCapacityGuard": 0,
            "SpringCapacityGuardFactor": 1.5,
            "SpringCandidateLoadWeight": 1.0,
            "SpringIOTMode": 1,
            "SpringIOTIdentityMode": 1,
            "SpringIOTFeatureDim": 10,
            "SpringIOTSidecarFile": (
                "./data_iot/iot_flow_sidecar_multi_anchor_full.csv"
            ),
            "SpringModelFile": str(model_path.resolve()),
            "SpringIOTCSTRWeight": iot_cstr_weight,
            "SpringIOTBalanceWeight": iot_balance_weight,
            "SpringIOTCommCostWeight": FIXED_IOT_COMM_COST_WEIGHT,
            "SpringIOTHotspotWeight": FIXED_IOT_HOTSPOT_WEIGHT,
            "SpringIOTHotspotThreshold": 0.45,
            "SpringRewardLambda": 0.5,
            "SpringRewardBeta": 0.1,
            "SpringSenderPosMode": 1,
            "Block_Interval": 5000,
            "BlockSize": 1000,
            "InjectSpeed": inject_speed,
            "DatasetStartTx": window.start_tx,
            "TotalDataSize": window.max_txs,
            "TxBatchSize": 1000,
            "DatasetFile": "./data_iot/selectedTxs_iot_multi_anchor_full.csv",
            "ExpDataRootDir": str(exp_test.resolve()),
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=sorted(DATASET_WINDOWS), required=True)
    parser.add_argument("--inject_speed", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--candidate_top_k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
    )
    parser.add_argument(
        "--iot_cstr_weight",
        type=float,
        default=DEFAULT_IOT_CSTR_WEIGHT,
    )
    parser.add_argument(
        "--iot_balance_weight",
        type=float,
        default=DEFAULT_IOT_BALANCE_WEIGHT,
    )
    parser.add_argument("--experiment", default="")
    parser.add_argument(
        "--timestamp",
        default="",
        help="optional yyyyMMdd_HHmmss override for reproducible tests",
    )
    parser.add_argument(
        "--skip_large_file_hash",
        action="store_true",
        help="skip the multi-gigabyte dataset hashes in the manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inject_speed <= 0:
        raise ValueError("inject_speed must be positive")
    running = running_block_emulator_processes()
    if running:
        raise RuntimeError(
            "BlockEmulator processes are still running; finalize the current "
            "run before changing paramsConfig.json."
        )
    ensure_clean_runtime_directory()

    window = DATASET_WINDOWS[args.window]
    scheme = proposed_scheme(
        args.candidate_top_k,
        args.iot_cstr_weight,
        args.iot_balance_weight,
    )
    model_path = (
        Path(args.model)
        if args.model
        else default_model(
            args.seed,
            args.candidate_top_k,
            args.iot_cstr_weight,
            args.iot_balance_weight,
        )
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Frozen aligned PPO model not found: {model_path}"
        )

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        raise ValueError("timestamp must use yyyyMMdd_HHmmss")
    experiment = experiment_name(
        scheme=scheme,
        window=window,
        inject_speed=args.inject_speed,
        seed=args.seed,
        override=args.experiment,
    )
    run_root = RESULTS_ROOT / "block_eval" / f"{experiment}_{timestamp}"
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    exp_test = run_root / "expTest"
    exp_test.mkdir(parents=True)

    base_config = json.loads(PARAMS_PATH.read_text(encoding="utf-8-sig"))
    config = build_proposed_config(
        base_config=base_config,
        seed=args.seed,
        model_path=model_path,
        exp_test=exp_test,
        window=window,
        inject_speed=args.inject_speed,
        candidate_top_k=args.candidate_top_k,
        iot_cstr_weight=args.iot_cstr_weight,
        iot_balance_weight=args.iot_balance_weight,
    )
    write_json(PARAMS_PATH, config)
    snapshot_path = run_root / "paramsConfig.snapshot.json"
    write_json(snapshot_path, config)

    workload_profile = profile_iot_window(
        sidecar_path=SIDECAR_PATH,
        start_tx=window.start_tx,
        max_txs=window.max_txs,
        shards=16,
        block_size=1000,
        block_interval_ms=5000,
    )
    workload_profile_path = run_root / "workload_profile.json"
    write_json(workload_profile_path, workload_profile)

    manifest_args = argparse.Namespace(
        experiment=experiment,
        phase=f"block_{window.name}",
        seed=args.seed,
        shards=16,
        dataset=str(DATASET_PATH),
        sidecar=str(SIDECAR_PATH),
        model=str(model_path),
        model_not_applicable=False,
        params=str(snapshot_path),
        start_tx=window.start_tx,
        max_txs=window.max_txs,
        skip_large_file_hash=args.skip_large_file_hash,
    )
    manifest = build_manifest(manifest_args)
    manifest_path = run_root / "experiment_manifest.json"
    write_json(manifest_path, manifest)

    context = {
        "scheme": scheme,
        "role": "proposed",
        "experiment": experiment,
        "phase": window.name,
        "seed": args.seed,
        "shards": 16,
        "dataset": str(DATASET_PATH.resolve()),
        "sidecar": str(SIDECAR_PATH.resolve()),
        "dataset_start_tx": window.start_tx,
        "total_data_size": window.max_txs,
        "inject_speed": args.inject_speed,
        "block_size": 1000,
        "block_interval_ms": 5000,
        "candidate_top_k": args.candidate_top_k,
        "iot_reward_weights": {
            "cross_shard": args.iot_cstr_weight,
            "balance": args.iot_balance_weight,
            "communication_cost": FIXED_IOT_COMM_COST_WEIGHT,
            "hotspot": FIXED_IOT_HOTSPOT_WEIGHT,
        },
        "run_root": str(run_root.resolve()),
        "exp_test": str(exp_test.resolve()),
        "params_snapshot": str(snapshot_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "workload_profile": str(workload_profile_path.resolve()),
        "model": str(model_path.resolve()),
    }
    write_json(run_root / "run_context.json", context)
    print(json.dumps(context, ensure_ascii=True))


if __name__ == "__main__":
    main()
