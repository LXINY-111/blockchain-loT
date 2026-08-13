import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from config import (
    DEFAULT_IOT_CSV_PATH,
    DEFAULT_IOT_SIDECAR_PATH,
    DEFAULT_SHARD_NUM,
    DEFAULT_TEST_MAX_TXS,
    DEFAULT_TEST_START_TX,
    MODEL_PATH,
    ROOT_DIR,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> Dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest input not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "sha256": sha256_file(resolved) if include_hash else None,
    }


def git_text(*args: str) -> Optional[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    params_path = Path(args.params)
    params_snapshot = json.loads(params_path.read_text(encoding="utf-8-sig"))
    if args.phase == "block_eval":
        configured_start = int(params_snapshot.get("DatasetStartTx", -1))
        configured_count = int(params_snapshot.get("TotalDataSize", -1))
        if configured_start != args.start_tx or configured_count != args.max_txs:
            raise ValueError(
                "paramsConfig.json data window mismatch: "
                f"configured=({configured_start}, {configured_count}), "
                f"manifest=({args.start_tx}, {args.max_txs})"
            )

    large_hash = not args.skip_large_file_hash
    model_not_applicable = bool(getattr(args, "model_not_applicable", False))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "phase": args.phase,
        "seed": args.seed,
        "shards": args.shards,
        "data_window": {
            "start_tx": args.start_tx,
            "max_txs": args.max_txs,
            "end_tx_exclusive": args.start_tx + args.max_txs,
            "index_space": "valid_transactions",
            "state_history": "window_local_cold_start",
        },
        "files": {
            "dataset": file_record(Path(args.dataset), include_hash=large_hash),
            "sidecar": file_record(Path(args.sidecar), include_hash=large_hash),
            "model": None
            if model_not_applicable
            else file_record(Path(args.model), include_hash=True),
            "params": file_record(params_path, include_hash=True),
        },
        "params_snapshot": params_snapshot,
        "git": {
            "commit": git_text("rev-parse", "HEAD"),
            "status_porcelain": git_text("status", "--porcelain"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--phase",
        choices=["offline_train", "offline_eval", "block_eval"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--dataset", default=str(DEFAULT_IOT_CSV_PATH))
    parser.add_argument("--sidecar", default=str(DEFAULT_IOT_SIDECAR_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument(
        "--model_not_applicable",
        action="store_true",
        help="record a null model for non-learning baselines",
    )
    parser.add_argument("--params", default=str(ROOT_DIR / "paramsConfig.json"))
    parser.add_argument("--start_tx", type=int, default=DEFAULT_TEST_START_TX)
    parser.add_argument("--max_txs", type=int, default=DEFAULT_TEST_MAX_TXS)
    parser.add_argument(
        "--skip_large_file_hash",
        action="store_true",
        help="skip dataset/sidecar SHA-256 when a quick manifest is sufficient",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.start_tx < 0 or args.max_txs <= 0:
        raise ValueError("start_tx must be >= 0 and max_txs must be > 0")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
