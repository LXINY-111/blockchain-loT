import argparse
import copy
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import EXPERIMENT_ROOT
from profile_iot_window import profile_iot_window
from write_experiment_manifest import build_manifest


ROOT_DIR = Path(__file__).resolve().parents[1]
# Share the canonical result root with offline training/evaluation so baseline,
# proposed, and Python-only runs cannot silently diverge again.
RESULTS_ROOT = EXPERIMENT_ROOT
PARAMS_PATH = ROOT_DIR / "paramsConfig.json"
DATASET_PATH = ROOT_DIR / "data_iot" / "selectedTxs_iot_multi_anchor_full.csv"
SIDECAR_PATH = ROOT_DIR / "data_iot" / "iot_flow_sidecar_multi_anchor_full.csv"
TEST_START_TX = 2_944_019
TEST_MAX_TXS = 2_000_000
VALIDATION_START_TX = 2_500_000
VALIDATION_MAX_TXS = 444_019
WINDOWS = {
    "validation": (VALIDATION_START_TX, VALIDATION_MAX_TXS, "validation444k"),
    "test": (TEST_START_TX, TEST_MAX_TXS, "test2M"),
}


@dataclass(frozen=True)
class BaselineProfile:
    scheme: str
    experiment_prefix: str
    role: str
    spring_mode: int
    requires_model: bool = False
    nsshard_capacity_factor: float = 1.2


BASELINE_PROFILES: Dict[str, BaselineProfile] = {
    "hash": BaselineProfile("hash", "hash", "baseline", 0),
    "heuristic": BaselineProfile(
        "heuristic",
        "heuristic_pure",
        "baseline",
        1,
    ),
    "original_spring_ppo": BaselineProfile(
        "original_spring_ppo",
        "original_spring_ppo",
        "baseline",
        2,
        requires_model=True,
    ),
    "random": BaselineProfile("random", "random", "baseline", 3),
    "minstate": BaselineProfile("minstate", "minstate", "baseline", 4),
    "anchoronly": BaselineProfile(
        "anchoronly",
        "anchoronly",
        "ablation",
        5,
    ),
    "nsshard_adapted": BaselineProfile(
        "nsshard_adapted",
        "nsshard_adapted_cap12",
        "baseline",
        6,
    ),
}


def running_block_emulator_processes() -> List[str]:
    if os.name != "nt":
        return []
    completed = subprocess.run(
        [
            "tasklist",
            "/FI",
            "IMAGENAME eq blockEmulator_Windows_Precompile.exe",
            "/FO",
            "CSV",
            "/NH",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if "blockemulator_windows_precompile.exe" in line.lower()
    ]


def default_original_model(seed: int) -> Path:
    return RESULTS_ROOT / "models" / f"original_spring_ppo_16s_seed{seed}.pt"


def build_baseline_config(
    base_config: Dict[str, object],
    profile: BaselineProfile,
    seed: int,
    exp_test: Path,
    model_path: Optional[Path],
    start_tx: int = TEST_START_TX,
    max_txs: int = TEST_MAX_TXS,
    inject_speed: int = 1000,
) -> Dict[str, object]:
    if inject_speed <= 0:
        raise ValueError("inject_speed must be positive")
    config = copy.deepcopy(base_config)
    config.update(
        {
            "ConsensusMethod": 3,
            "SpringMode": profile.spring_mode,
            "SpringOnlineTrain": 0,
            "SpringEvalSample": 0,
            "SpringRandomSeed": seed,
            "SpringCandidateTopK": 0,
            "SpringCapacityGuard": 0,
            "SpringCapacityGuardFactor": 1.5,
            "SpringCandidateLoadWeight": 1.0,
            "SpringNSShardCapacityFactor": profile.nsshard_capacity_factor,
            # All comparators use the same IoT state-object/anchor identities.
            # Only the proposed PPO enables the extra 10-dimensional IoT vector.
            "SpringIOTMode": 0,
            "SpringIOTIdentityMode": 1,
            "SpringIOTFeatureDim": 0,
            "SpringIOTSidecarFile": (
                "./data_iot/iot_flow_sidecar_multi_anchor_full.csv"
            ),
            "SpringModelFile": (
                str(model_path.resolve())
                if profile.requires_model and model_path is not None
                else "NOT_APPLICABLE"
            ),
            "SpringIOTCSTRWeight": 0.55,
            "SpringIOTBalanceWeight": 0.30,
            "SpringIOTCommCostWeight": 0.10,
            "SpringIOTHotspotWeight": 0.05,
            "SpringRewardLambda": 0.5,
            "SpringRewardBeta": 0.1,
            "SpringSenderPosMode": 1,
            "DatasetStartTx": start_tx,
            "TotalDataSize": max_txs,
            "TxBatchSize": 1000,
            "BlockSize": 1000,
            "InjectSpeed": inject_speed,
            "DatasetFile": "./data_iot/selectedTxs_iot_multi_anchor_full.csv",
            "ExpDataRootDir": str(exp_test.resolve()),
        }
    )
    return config


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def ensure_clean_runtime_directory() -> None:
    spring_io = ROOT_DIR / "spring_io"
    if spring_io.is_dir() and any(spring_io.iterdir()):
        raise RuntimeError(
            "spring_io is not empty. Finalize/archive the previous run before "
            "preparing another experiment."
        )
    spring_io.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=sorted(BASELINE_PROFILES), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="")
    # Require both values so an omitted argument cannot silently consume the
    # held-out test window or reuse the old 1000 TPS overload setting.
    parser.add_argument("--window", choices=sorted(WINDOWS), required=True)
    parser.add_argument("--inject_speed", type=int, required=True)
    parser.add_argument(
        "--timestamp",
        default="",
        help="optional yyyyMMdd_HHmmss override for reproducible tests",
    )
    parser.add_argument(
        "--skip_large_file_hash",
        action="store_true",
        help="skip the 4 GB dataset/sidecar hashes in the manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inject_speed <= 0:
        raise ValueError("inject_speed must be positive")
    running = running_block_emulator_processes()
    if running:
        raise RuntimeError(
            "BlockEmulator processes are still running; wait for them to exit "
            "before changing paramsConfig.json."
        )
    ensure_clean_runtime_directory()

    profile = BASELINE_PROFILES[args.scheme]
    start_tx, max_txs, window_label = WINDOWS[args.window]
    model_path: Optional[Path] = None
    if profile.requires_model:
        model_path = (
            Path(args.model) if args.model else default_original_model(args.seed)
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                "Original SPRING frozen model not found: "
                f"{model_path}. Train and validate it before this BlockEmulator run."
            )

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        raise ValueError("timestamp must use yyyyMMdd_HHmmss")
    experiment = (
        f"{profile.experiment_prefix}_16s_{window_label}_"
        f"i{args.inject_speed}_seed{args.seed}"
    )
    run_root = RESULTS_ROOT / "block_eval" / f"{experiment}_{timestamp}"
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    exp_test = run_root / "expTest"
    exp_test.mkdir(parents=True)

    base_config = json.loads(PARAMS_PATH.read_text(encoding="utf-8-sig"))
    config = build_baseline_config(
        base_config,
        profile,
        args.seed,
        exp_test,
        model_path,
        start_tx=start_tx,
        max_txs=max_txs,
        inject_speed=args.inject_speed,
    )
    write_json(PARAMS_PATH, config)
    snapshot_path = run_root / "paramsConfig.snapshot.json"
    write_json(snapshot_path, config)

    manifest_args = argparse.Namespace(
        experiment=experiment,
        phase=f"block_{args.window}",
        seed=args.seed,
        shards=16,
        dataset=str(DATASET_PATH),
        sidecar=str(SIDECAR_PATH),
        model=str(model_path) if model_path is not None else "",
        model_not_applicable=not profile.requires_model,
        params=str(snapshot_path),
        start_tx=start_tx,
        max_txs=max_txs,
        skip_large_file_hash=args.skip_large_file_hash,
    )
    manifest = build_manifest(manifest_args)
    manifest_path = run_root / "experiment_manifest.json"
    write_json(manifest_path, manifest)

    workload_profile = profile_iot_window(
        sidecar_path=SIDECAR_PATH,
        start_tx=start_tx,
        max_txs=max_txs,
        shards=16,
        block_size=1000,
        block_interval_ms=5000,
    )
    workload_profile_path = run_root / "workload_profile.json"
    write_json(workload_profile_path, workload_profile)

    context = {
        "scheme": profile.scheme,
        "role": profile.role,
        "experiment": experiment,
        "phase": args.window,
        "seed": args.seed,
        "shards": 16,
        "dataset": str(DATASET_PATH.resolve()),
        "sidecar": str(SIDECAR_PATH.resolve()),
        "dataset_start_tx": start_tx,
        "total_data_size": max_txs,
        "inject_speed": args.inject_speed,
        "block_size": 1000,
        "block_interval_ms": 5000,
        "run_root": str(run_root.resolve()),
        "exp_test": str(exp_test.resolve()),
        "params_snapshot": str(snapshot_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "workload_profile": str(workload_profile_path.resolve()),
        "model": str(model_path.resolve()) if model_path is not None else None,
    }
    write_json(run_root / "run_context.json", context)
    # Escaped output survives Windows PowerShell 5.1 code-page conversion.
    print(json.dumps(context, ensure_ascii=True))


if __name__ == "__main__":
    main()
