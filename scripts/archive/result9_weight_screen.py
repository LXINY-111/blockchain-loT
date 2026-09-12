import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT_DIR / "spring_lite" / "train_offline.py"
SELECT_SCRIPT = ROOT_DIR / "spring_lite" / "select_pareto_checkpoint.py"
EVAL_SCRIPT = ROOT_DIR / "spring_lite" / "eval_offline.py"
CSV_PATH = ROOT_DIR / "data_iot" / "selectedTxs_iot_multi_anchor_full.csv"
SIDECAR_PATH = ROOT_DIR / "data_iot" / "iot_flow_sidecar_multi_anchor_full.csv"

WEIGHT_SPECS: Dict[str, Tuple[float, float]] = {
    "w5035": (0.50, 0.35),
    "w5530": (0.55, 0.30),
    "w6025": (0.60, 0.25),
}

MIN_ACTIVE_SHARDS = 8.0
MAX_STAGE_HOTSPOT = 0.35
MAX_ACTIVE_DEFICIT_RATIO = 0.05
SELECTION_POLICY_NAME = "hierarchical_constrained_pareto_v1"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run_logged(arguments: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline(list(arguments))
    print(f"[COMMAND] {printable}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            list(arguments),
            cwd=ROOT_DIR,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}; see {log_path}"
        )


def common_mdp_arguments(cstr_weight: float, balance_weight: float) -> List[str]:
    return [
        "--csv",
        str(CSV_PATH),
        "--sidecar",
        str(SIDECAR_PATH),
        "--mdp_mode",
        "iot",
        "--tx_identity",
        "iot",
        "--shards",
        "16",
        "--tx_batch_size",
        "1000",
        "--max_block_size",
        "1000",
        "--block_interval_ms",
        "5000",
        "--sender_pos_mode",
        "1",
        "--reward_mode",
        "iot_dense_balanced",
        "--lambda_weight",
        "0.5",
        "--beta",
        "0.1",
        "--iot_cstr_weight",
        format(cstr_weight, ".2f"),
        "--iot_balance_weight",
        format(balance_weight, ".2f"),
        "--iot_comm_cost_weight",
        "0.10",
        "--iot_hotspot_weight",
        "0.05",
        "--candidate_top_k",
        "7",
        "--capacity_guard",
        "0",
        "--capacity_guard_factor",
        "1.5",
        "--candidate_load_weight",
        "1.0",
    ]


def train_and_evaluate(
    results_root: Path,
    weight_name: str,
    seed: int,
    epochs: int,
) -> Dict[str, object]:
    cstr_weight, balance_weight = WEIGHT_SPECS[weight_name]
    scheme = f"ppo_top7_{weight_name}_pareto"
    experiment = f"{scheme}_16s_seed{seed}"
    model_path = results_root / "models" / f"{experiment}.pt"
    selection_path = results_root / "models" / f"{experiment}_selection.json"

    # Never replace a previously archived candidate implicitly.
    if model_path.exists() or selection_path.exists():
        raise FileExistsError(
            f"stable output already exists for {weight_name}: {model_path}"
        )

    train_root = results_root / "offline_training" / f"{experiment}_{timestamp()}"
    train_root.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = train_root / "checkpoints"
    manifest_path = train_root / "pareto_manifest.json"
    best_scalar_path = train_root / "best_scalar.pt"
    last_path = train_root / "last.pt"
    train_log = train_root / "train.jsonl"

    train_args = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *common_mdp_arguments(cstr_weight, balance_weight),
        "--model",
        str(best_scalar_path),
        "--last_model",
        str(last_path),
        "--epoch_checkpoint_dir",
        str(checkpoint_dir),
        "--pareto_manifest",
        str(manifest_path),
        "--epochs",
        str(epochs),
        "--start_tx",
        "0",
        "--max_txs",
        "2500000",
        "--validation_start_tx",
        "2500000",
        "--validation_max_txs",
        "444019",
        "--eval_every_epochs",
        "1",
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--log_jsonl",
        str(train_log),
    ]
    write_json(
        train_root / "run_context.json",
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment": experiment,
            "run_root": str(train_root),
            "train_window": {"start_tx": 0, "max_txs": 2500000},
            "validation_window": {
                "start_tx": 2500000,
                "max_txs": 444019,
            },
            "shards": 16,
            "epochs": epochs,
            "seed": seed,
            "candidate_top_k": 7,
            "weights": {
                "cstr": cstr_weight,
                "balance": balance_weight,
                "comm_cost": 0.10,
                "hotspot": 0.05,
            },
            "load_semantics_version": "primary_owner_execution_v1",
            "arguments": train_args,
        },
    )
    print(f"[WEIGHT START] {weight_name} run={train_root}", flush=True)
    run_logged(train_args, train_root / "stdout.log")

    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(checkpoints) != epochs:
        raise RuntimeError(
            f"expected {epochs} epoch checkpoints, found {len(checkpoints)}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pareto manifest not found: {manifest_path}")

    select_args = [
        sys.executable,
        str(SELECT_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--output_model",
        str(model_path),
        "--selection_json",
        str(selection_path),
        "--min_active_shards",
        format(MIN_ACTIVE_SHARDS, ".1f"),
        "--max_stage_hotspot",
        format(MAX_STAGE_HOTSPOT, ".2f"),
        "--max_active_deficit_ratio",
        format(MAX_ACTIVE_DEFICIT_RATIO, ".2f"),
    ]
    run_logged(select_args, train_root / "selection_stdout.log")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = selection["selected"]
    selected_epoch = int(selected["epoch"])
    write_json(
        train_root / "run_complete.json",
        {
            "completed_at": datetime.now().astimezone().isoformat(),
            "status": "passed",
            "run_root": str(train_root),
            "checkpoint_count": len(checkpoints),
            "best_scalar_sha256": sha256(best_scalar_path),
            "last_sha256": sha256(last_path),
            "pareto_manifest": str(manifest_path),
            "selected_model": str(model_path),
            "selected_model_sha256": sha256(model_path),
            "selection_record": str(selection_path),
            "selected_epoch": selected_epoch,
            "selection_policy": selection.get("selection_policy"),
            "selection_tier": selection.get("selection_tier"),
            "strict_feasible": selection.get("strict_feasible"),
            "constraint_audit": selection.get("constraint_audit"),
            "selection_constraints": {
                "min_active_shards": MIN_ACTIVE_SHARDS,
                "max_stage_hotspot": MAX_STAGE_HOTSPOT,
                "max_active_deficit_ratio": MAX_ACTIVE_DEFICIT_RATIO,
            },
        },
    )

    eval_experiment = (
        f"{scheme}_epoch{selected_epoch}_16s_seed{seed}_validation"
    )
    eval_root = results_root / "offline_eval" / f"{eval_experiment}_{timestamp()}"
    eval_root.mkdir(parents=True, exist_ok=False)
    eval_log = eval_root / "eval.jsonl"
    eval_args = [
        sys.executable,
        str(EVAL_SCRIPT),
        *common_mdp_arguments(cstr_weight, balance_weight),
        "--model",
        str(model_path),
        "--policy",
        "ppo",
        "--start_tx",
        "2500000",
        "--max_txs",
        "444019",
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--log_jsonl",
        str(eval_log),
    ]
    write_json(
        eval_root / "run_context.json",
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment": eval_experiment,
            "run_root": str(eval_root),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "selection_record": str(selection_path),
            "selection_tier": selection.get("selection_tier"),
            "strict_feasible": selection.get("strict_feasible"),
            "window": {"start_tx": 2500000, "max_txs": 444019},
            "arguments": eval_args,
        },
    )
    run_logged(eval_args, eval_root / "stdout.log")
    records = [
        json.loads(line)
        for line in eval_log.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records or not isinstance(records[-1].get("summary"), dict):
        raise RuntimeError(f"evaluation summary not found in {eval_log}")
    summary = records[-1]["summary"]
    write_json(
        eval_root / "run_complete.json",
        {
            "completed_at": datetime.now().astimezone().isoformat(),
            "status": "passed",
            "run_root": str(eval_root),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "summary": summary,
        },
    )
    print(
        "[WEIGHT DONE] "
        f"{weight_name} epoch={selected_epoch} "
        f"cross={float(summary['cross_ratio']):.6f} "
        f"stageHot={float(summary['stage_max_load_share']):.6f} "
        f"active={float(summary['active_shards_mean']):.4f}",
        flush=True,
    )
    return {
        "weight_name": weight_name,
        "selection": selection,
        "eval_root": str(eval_root),
        "summary": summary,
    }


def latest_eval_root(
    results_root: Path,
    weight_name: str,
    seed: int,
    epoch: int,
) -> Path:
    pattern = (
        f"ppo_top7_{weight_name}_pareto_epoch{epoch}_16s_seed{seed}_validation_*"
    )
    candidates = sorted(
        (path for path in (results_root / "offline_eval").glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no validation directory matches {pattern}")
    return candidates[0]


def collect_weight_result(
    results_root: Path,
    weight_name: str,
    seed: int,
) -> Dict[str, object]:
    cstr_weight, balance_weight = WEIGHT_SPECS[weight_name]
    experiment = f"ppo_top7_{weight_name}_pareto_16s_seed{seed}"
    model_path = results_root / "models" / f"{experiment}.pt"
    selection_path = results_root / "models" / f"{experiment}_selection.json"
    if not model_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"selected model is incomplete for {weight_name}")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selected = selection["selected"]
    epoch = int(selected["epoch"])
    eval_root = latest_eval_root(results_root, weight_name, seed, epoch)
    completion_path = eval_root / "run_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
    if completion.get("status") != "passed":
        raise RuntimeError(f"validation did not pass: {completion_path}")
    summary = completion["summary"]
    return {
        "weight_name": weight_name,
        "cstr_weight": cstr_weight,
        "balance_weight": balance_weight,
        "comm_cost_weight": 0.10,
        "hotspot_weight": 0.05,
        "candidate_top_k": 7,
        "seed": seed,
        "selected_epoch": epoch,
        "selection_tier": selection.get("selection_tier", "legacy_strict"),
        "strict_feasible": bool(selection.get("strict_feasible", True)),
        "active_deficit_ratio": float(
            selection.get("constraint_audit", {}).get(
                "active_deficit_ratio",
                max(
                    0.0,
                    (
                        MIN_ACTIVE_SHARDS
                        - float(selected["active_shards_mean"])
                    )
                    / MIN_ACTIVE_SHARDS,
                ),
            )
        ),
        "cross_ratio": float(summary["cross_ratio"]),
        "relation_cross_ratio": float(summary["relation_cross_ratio"]),
        "active_shards_mean": float(summary["active_shards_mean"]),
        "stage_max_load_share": float(summary["stage_max_load_share"]),
        "reward_mean": float(summary["reward_mean"]),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "selection_record": str(selection_path),
        "eval_root": str(eval_root),
    }


def write_screen_summary(
    results_root: Path,
    seed: int,
    rows: Iterable[Dict[str, object]],
) -> Tuple[Path, Path]:
    ordered = sorted(rows, key=lambda row: float(row["cstr_weight"]))
    output_stem = f"top7_weight_screen_16s_seed{seed}_{timestamp()}"
    json_path = results_root / "offline_eval" / f"{output_stem}.json"
    csv_path = results_root / "offline_eval" / f"{output_stem}.csv"
    write_json(
        json_path,
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "selection_protocol": {
                "name": SELECTION_POLICY_NAME,
                "pareto_objectives": {
                    "minimize": ["cross_ratio", "stage_max_load_share"],
                    "maximize": ["active_shards_mean"],
                },
                "strict_constraints": {
                    "min_active_shards": MIN_ACTIVE_SHARDS,
                    "max_stage_hotspot": MAX_STAGE_HOTSPOT,
                },
                "fallback_constraints": {
                    "max_active_deficit_ratio": MAX_ACTIVE_DEFICIT_RATIO,
                    "max_stage_hotspot": MAX_STAGE_HOTSPOT,
                    "hotspot_cap_relaxed": False,
                },
            },
            "rows": ordered,
        },
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and archive the Experiment-8 TopK7 reward-weight screen."
    )
    parser.add_argument("--results_root", required=True)
    parser.add_argument(
        "--weights",
        nargs="+",
        choices=sorted(WEIGHT_SPECS),
        default=["w5035", "w6025"],
        help="Weight variants to train; w5530 is normally reused from the TopK screen.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if not CSV_PATH.is_file() or not SIDECAR_PATH.is_file():
        raise FileNotFoundError("IoT dataset or sidecar is missing")
    for folder in ("models", "offline_training", "offline_eval"):
        (results_root / folder).mkdir(parents=True, exist_ok=True)

    for weight_name in args.weights:
        train_and_evaluate(
            results_root=results_root,
            weight_name=weight_name,
            seed=args.seed,
            epochs=args.epochs,
        )

    rows = [
        collect_weight_result(results_root, weight_name, args.seed)
        for weight_name in ("w5035", "w5530", "w6025")
    ]
    json_path, csv_path = write_screen_summary(results_root, args.seed, rows)
    print("\n[WEIGHT SCREEN COMPLETE]", flush=True)
    for row in rows:
        print(
            f"{row['weight_name']} epoch={row['selected_epoch']} "
            f"cross={100.0 * float(row['cross_ratio']):.4f}% "
            f"stageHot={100.0 * float(row['stage_max_load_share']):.4f}% "
            f"active={float(row['active_shards_mean']):.4f}",
            flush=True,
        )
    print(f"summary_json={json_path}", flush=True)
    print(f"summary_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
