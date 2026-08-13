import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from prepare_exp8_baseline import RESULTS_ROOT, ROOT_DIR, running_block_emulator_processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    return parser.parse_args()


def ensure_results8_run_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (RESULTS_ROOT / "block_eval").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"run_root must stay under {allowed}") from exc
    if not (resolved / "run_context.json").is_file():
        raise FileNotFoundError(f"run_context.json not found under {resolved}")
    return resolved


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
    run_root = ensure_results8_run_root(Path(args.run_root))
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
        ["go", "test", "./query", "-run", "TestFinalResult", "-count=1", "-v"],
        run_root / "state_check.txt",
        env,
    )
    if state_check.returncode != 0:
        raise RuntimeError(
            "Final chain-state check failed; see "
            f"{run_root / 'state_check.txt'}"
        )

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
