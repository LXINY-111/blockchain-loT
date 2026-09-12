import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from prepare_baseline_run import ROOT_DIR, running_block_emulator_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    return parser.parse_args()


def ensure_run_root(path: Path) -> Path:
    resolved = path.resolve()
    context_path = resolved / "run_context.json"
    if not context_path.is_file():
        raise FileNotFoundError(f"run_context.json not found under {resolved}")
    if resolved.parent.name.lower() != "block_eval":
        raise ValueError("run_root must be a direct child of a block_eval directory")

    # The active params file is the authoritative run binding. This supports
    # Result9, Result10, or a later explicitly configured root without removing
    # the path-boundary check.
    config = json.loads(
        (ROOT_DIR / "paramsConfig.json").read_text(encoding="utf-8-sig")
    )
    configured_exp_test = Path(str(config.get("ExpDataRootDir", ""))).resolve()
    expected_exp_test = (resolved / "expTest").resolve()
    if configured_exp_test != expected_exp_test:
        raise ValueError(
            "run_root is not the run selected by paramsConfig.json: "
            f"configured={configured_exp_test}, expected={expected_exp_test}"
        )

    context = json.loads(context_path.read_text(encoding="utf-8-sig"))
    if Path(str(context.get("run_root", ""))).resolve() != resolved:
        raise ValueError("run_context.json run_root does not match the requested run")
    if Path(str(context.get("exp_test", ""))).resolve() != expected_exp_test:
        raise ValueError("run_context.json exp_test does not stay inside run_root")
    return resolved


def require_executed_state_test(output: str) -> None:
    """Require TestFinalResult to execute and pass; a Go test SKIP is failure."""
    actions = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("Test") == "TestFinalResult":
            actions.append(event.get("Action"))
    if "skip" in actions:
        raise RuntimeError("TestFinalResult was skipped; final state is unverified")
    if "fail" in actions or "pass" not in actions:
        raise RuntimeError("TestFinalResult did not execute and pass")


def validate_analysis_integrity(
    analysis: Dict[str, object],
    context: Dict[str, object],
) -> Dict[str, object]:
    expected_total = int(context.get("total_data_size", 0) or 0)
    if expected_total <= 0:
        raise RuntimeError("run_context.json has no positive total_data_size")
    periods = analysis.get("periods", {})
    all_active = periods.get("all_active", {}) if isinstance(periods, dict) else {}
    latency = analysis.get("latency_details", {})
    latency = latency if isinstance(latency, dict) else {}
    effective_total = float(all_active.get("effective_total", 0.0) or 0.0)
    valid_count = int(latency.get("valid_latency_count", 0) or 0)
    row_count = int(latency.get("row_count", 0) or 0)
    unique_count = int(latency.get("unique_tx_hash_count", 0) or 0)
    duplicate_count = int(latency.get("duplicate_tx_hash_count", 0) or 0)
    missing_hash_count = int(latency.get("missing_tx_hash_count", 0) or 0)
    invalid_count = int(latency.get("invalid_latency_count", 0) or 0)

    errors = []
    if not math.isclose(effective_total, expected_total, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"effective_total={effective_total}, expected={expected_total}")
    if row_count != expected_total:
        errors.append(f"latency rows={row_count}, expected={expected_total}")
    if valid_count != expected_total or invalid_count != 0:
        errors.append(
            f"valid/invalid latency={valid_count}/{invalid_count}, "
            f"expected={expected_total}/0"
        )
    if unique_count != expected_total or duplicate_count != 0 or missing_hash_count != 0:
        errors.append(
            f"unique/duplicate/missing hashes={unique_count}/"
            f"{duplicate_count}/{missing_hash_count}, expected={expected_total}/0/0"
        )
    if errors:
        raise RuntimeError("analysis completeness check failed: " + "; ".join(errors))
    return {
        "expected_total": expected_total,
        "effective_total": effective_total,
        "completion_ratio": effective_total / expected_total,
        "latency_coverage_ratio": valid_count / expected_total,
        "unique_transaction_ratio": unique_count / expected_total,
    }


def archive_spring_io(run_root: Path) -> Path:
    source = ROOT_DIR / "spring_io"
    destination = run_root / "spring_io"
    if destination.exists():
        if source.is_dir() and any(source.iterdir()):
            raise FileExistsError(
                "Both archived and live spring_io contain data; refusing to mix runs."
            )
        source.mkdir(parents=True, exist_ok=True)
        return destination
    if source.exists():
        shutil.move(str(source), str(destination))
    else:
        destination.mkdir()
    source.mkdir(parents=True, exist_ok=True)
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_and_record(
    command: List[str],
    output_path: Path,
    env: Dict[str, str],
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    return completed


def main() -> None:
    args = parse_args()
    run_root = ensure_run_root(Path(args.run_root))
    running = running_block_emulator_processes()
    if running:
        raise RuntimeError(
            "BlockEmulator processes are still running; do not finalize a live run."
        )

    context = json.loads(
        (run_root / "run_context.json").read_text(encoding="utf-8-sig")
    )
    snapshot_path = Path(context["params_snapshot"])
    if sha256_file(ROOT_DIR / "paramsConfig.json") != sha256_file(snapshot_path):
        raise RuntimeError(
            "paramsConfig.json no longer matches this run's snapshot; "
            "do not validate results with another experiment's configuration."
        )

    spring_io = archive_spring_io(run_root)
    env = dict(os.environ)
    env["BLOCKEMULATOR_CHECK_FINAL_RESULT"] = "1"
    state_check = run_and_record(
        ["go", "test", "./query", "-run", "^TestFinalResult$", "-count=1", "-json"],
        run_root / "state_check.txt",
        env,
    )
    if state_check.returncode != 0:
        raise RuntimeError(
            "Final chain-state check failed; see "
            f"{run_root / 'state_check.txt'}"
        )
    require_executed_state_test(state_check.stdout)

    exp_test = Path(context["exp_test"])
    analysis_path = run_root / "analysis.json"
    analyzer_command = [
        sys.executable,
        str(ROOT_DIR / "spring_lite" / "analyze_block_eval.py"),
        "--result_path",
        str(exp_test / "result"),
        "--spring_io_path",
        str(spring_io),
        "--shards",
        "16",
        "--output_json",
        str(analysis_path),
    ]
    supervisor_log = run_root / "process_logs" / "supervisor.stdout.log"
    if supervisor_log.is_file():
        analyzer_command.extend(["--log", str(supervisor_log)])
    analyzer = run_and_record(
        analyzer_command,
        run_root / "analyzer_output.txt",
        dict(os.environ),
    )
    if analyzer.returncode != 0:
        raise RuntimeError(
            f"Block evaluation analysis failed; see {run_root / 'analyzer_output.txt'}"
        )

    analysis = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
    integrity = validate_analysis_integrity(analysis, context)
    all_active = analysis.get("periods", {}).get("all_active", {})
    workload_profile_path = Path(
        context.get("workload_profile", run_root / "workload_profile.json")
    )
    workload_profile = (
        json.loads(workload_profile_path.read_text(encoding="utf-8-sig"))
        if workload_profile_path.is_file()
        else None
    )
    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": context["experiment"],
        "run_root": str(run_root),
        "state_check": "passed",
        "protocol_ok": True,
        "integrity": integrity,
        "analysis": str(analysis_path),
        "workload_profile": (
            str(workload_profile_path) if workload_profile is not None else None
        ),
        "owner_anchor_tps_ceiling": (
            workload_profile.get("owner_anchor_tps_ceiling")
            if workload_profile is not None
            else None
        ),
        "injection_pace": analysis.get("injection_pace", {}),
        "latency_details": analysis.get("latency_details", {}),
        "all_active": all_active,
    }
    (run_root / "run_complete.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Escaped output survives Windows PowerShell 5.1 code-page conversion.
    print(json.dumps(completion, ensure_ascii=True))


if __name__ == "__main__":
    main()
