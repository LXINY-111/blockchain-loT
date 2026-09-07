"""Recompute Result9 with one auditable multi-objective metric schema.

The script is intentionally read-only with respect to archived experiment runs.
It writes consolidated artifacts under the repository's analysis_outputs folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze_block_eval import (
    analyze_decisions,
    keyed,
    normal_tx,
    read_csv_source,
    summarize_period,
    total_tx,
)


RUN_PATTERN = re.compile(
    r"^(?P<prefix>.+)_16s_"
    r"(?P<window>validation444k|test2M)_i(?P<tps>\d+)_"
    r"seed(?P<seed>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)

METHOD_LABELS = {
    "anchoronly": "AnchorOnly",
    "hash": "Hash",
    "heuristic_pure": "Heuristic",
    "minstate": "MinState",
    "nsshard_adapted_cap12": "NSshard-adapted",
    "random": "Random",
    "ppo_top7_w5530_pareto": "Proposed-TopK7",
    "ppo_top8_w5530_pareto": "Proposed-TopK8",
    "candidate_only_top7_w5530": "Candidate-Only/No-PPO",
}


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_run_name(name: str) -> Optional[Dict[str, object]]:
    match = RUN_PATTERN.match(name)
    if not match:
        return None
    prefix = match.group("prefix")
    return {
        "prefix": prefix,
        "method": METHOD_LABELS.get(prefix, prefix),
        "window": match.group("window"),
        "inject_tps": int(match.group("tps")),
        "seed": int(match.group("seed")),
        "timestamp": match.group("timestamp"),
    }


def nested(mapping: Mapping[str, object], *keys: str, default: object = 0.0) -> object:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def first_mapping_value(
    mappings: Sequence[Mapping[str, object]],
    *keys: str,
    default: object = 0.0,
) -> object:
    for mapping in mappings:
        value = nested(mapping, *keys, default=None)
        if value is not None:
            return value
    return default


def recompute_all_active(run_root: Path, shards: int) -> Dict[str, object]:
    result_path = run_root / "expTest" / "result"
    tx_rows = read_csv_source(result_path, "supervisor_measureOutput/Tx_number.csv")
    cross_rows = read_csv_source(
        result_path,
        "supervisor_measureOutput/CrossTransaction_ratio.csv",
    )
    tps_rows = read_csv_source(result_path, "supervisor_measureOutput/Average_TPS.csv")
    latency_rows = read_csv_source(
        result_path,
        "supervisor_measureOutput/Transaction_Confirm_Latency.csv",
    )
    variance_rows = read_csv_source(
        result_path,
        "supervisor_measureOutput/Shard_Load_Variance.csv",
    )
    active_rows = [row for row in tx_rows if total_tx(row) > 0]
    return summarize_period(
        "all_active",
        active_rows,
        keyed(cross_rows),
        keyed(tps_rows),
        keyed(latency_rows),
        keyed(variance_rows),
        shards,
        5000.0,
    )


def flatten_run(run_root: Path, shards: int = 16) -> Optional[Dict[str, object]]:
    identity = parse_run_name(run_root.name)
    if identity is None:
        return None

    completion = read_json(run_root / "run_complete.json")
    archived = read_json(run_root / "analysis.json")
    config = read_json(run_root / "paramsConfig.snapshot.json")
    all_active = recompute_all_active(run_root, shards)
    load = all_active.get("load_balance", {})
    if not isinstance(load, Mapping):
        load = {}

    spring_io = run_root / "spring_io"
    decisions = analyze_decisions(spring_io, shards) if spring_io.exists() else {}
    latency = first_mapping_value(
        [completion, archived],
        "latency_details",
        default={},
    )
    pace = first_mapping_value(
        [completion, archived],
        "injection_pace",
        default={},
    )
    latency = latency if isinstance(latency, Mapping) else {}
    pace = pace if isinstance(pace, Mapping) else {}

    expected_total = 2_000_000 if identity["window"] == "test2M" else 444_019
    effective_total = float(all_active.get("effective_total", 0.0) or 0.0)
    valid_latency_count = int(latency.get("valid_latency_count", 0) or 0)
    latency_row_count = int(latency.get("row_count", 0) or 0)
    valid_latency_ratio = (
        valid_latency_count / latency_row_count if latency_row_count else 0.0
    )
    actual_offered = float(pace.get("actual_offered_tps", 0.0) or 0.0)
    wall_tps = float(all_active.get("wall_tps", 0.0) or 0.0)

    row: Dict[str, object] = {
        "run_name": run_root.name,
        **identity,
        "state_check": completion.get("state_check", "unknown"),
        "expected_total": expected_total,
        "effective_total": effective_total,
        "completion_ratio": effective_total / expected_total if expected_total else 0.0,
        "valid_latency_count": valid_latency_count,
        "valid_latency_ratio": valid_latency_ratio,
        "target_tps": float(pace.get("target_tps", identity["inject_tps"]) or 0.0),
        "actual_offered_tps": actual_offered,
        "offered_attainment_ratio": float(pace.get("attainment_ratio", 0.0) or 0.0),
        "final_schedule_lag_ms": float(pace.get("final_schedule_lag_ms", 0.0) or 0.0),
        "max_abs_schedule_lag_ms": float(pace.get("max_abs_schedule_lag_ms", 0.0) or 0.0),
        "wall_tps": wall_tps,
        "wall_to_offered_ratio": wall_tps / actual_offered if actual_offered else 0.0,
        "mean_epoch_tps": float(all_active.get("mean_epoch_tps", 0.0) or 0.0),
        "median_epoch_tps": float(all_active.get("median_epoch_tps", 0.0) or 0.0),
        "p05_epoch_tps": float(all_active.get("p05_epoch_tps", 0.0) or 0.0),
        "p95_epoch_tps": float(all_active.get("p95_epoch_tps", 0.0) or 0.0),
        "mean_latency_sec": float(latency.get("mean_sec", 0.0) or 0.0),
        "p50_latency_sec": float(latency.get("p50_sec", 0.0) or 0.0),
        "p95_latency_sec": float(latency.get("p95_sec", 0.0) or 0.0),
        "p99_latency_sec": float(latency.get("p99_sec", 0.0) or 0.0),
        "max_latency_sec": float(latency.get("max_sec", 0.0) or 0.0),
        "cross_total": float(all_active.get("cross_total", 0.0) or 0.0),
        "weighted_cross_ratio": float(all_active.get("weighted_cross_ratio", 0.0) or 0.0),
        "mean_load_variance": float(all_active.get("mean_load_variance", 0.0) or 0.0),
        "p95_load_variance": float(all_active.get("p95_load_variance", 0.0) or 0.0),
        "mean_max_shard_load_share": float(load.get("mean_max_shard_load_share", 0.0) or 0.0),
        "p95_max_shard_load_share": float(load.get("p95_max_shard_load_share", 0.0) or 0.0),
        "mean_active_shards": float(load.get("mean_active_shards", 0.0) or 0.0),
        "min_active_shards": float(load.get("min_active_shards", 0.0) or 0.0),
        "mean_jain_fairness": float(load.get("mean_jain_fairness", 0.0) or 0.0),
        "p05_jain_fairness": float(load.get("p05_jain_fairness", 0.0) or 0.0),
        "mean_load_cv": float(load.get("mean_load_cv", 0.0) or 0.0),
        "p95_load_cv": float(load.get("p95_load_cv", 0.0) or 0.0),
        "aggregate_max_shard_load_share": float(load.get("aggregate_max_shard_load_share", 0.0) or 0.0),
        "aggregate_jain_fairness": float(load.get("aggregate_jain_fairness", 0.0) or 0.0),
        "decision_count": int(decisions.get("decision_count", 0) or 0),
        "python_ppo_ratio": float(decisions.get("python_ppo_ratio", 0.0) or 0.0),
        "fallback_ratio": float(decisions.get("fallback_ratio", 0.0) or 0.0),
        "candidate_only_ratio": float(decisions.get("candidate_only_ratio", 0.0) or 0.0),
        "action_mask_size_mean": float(decisions.get("action_mask_size_mean", 0.0) or 0.0),
        "candidate_major_anchor_coverage_ratio": float(decisions.get("candidate_major_anchor_coverage_ratio", 0.0) or 0.0),
        "candidate_all_related_coverage_ratio": float(decisions.get("candidate_all_related_coverage_ratio", 0.0) or 0.0),
        "candidate_related_mass_coverage_mean": float(decisions.get("candidate_related_mass_coverage_mean", 0.0) or 0.0),
        "major_related_follow_ratio": float(decisions.get("major_related_follow_ratio", 0.0) or 0.0),
        "chosen_any_related_ratio": float(decisions.get("same_as_related_ratio", 0.0) or 0.0),
        "chosen_related_mass_mean": float(decisions.get("chosen_related_mass_mean", 0.0) or 0.0),
        "major_anchor_choice_when_available_ratio": float(decisions.get("major_anchor_choice_when_available_ratio", 0.0) or 0.0),
        "spring_mode": int(config.get("SpringMode", -1) or 0),
        "candidate_top_k": int(config.get("SpringCandidateTopK", 0) or 0),
    }
    row["protocol_ok"] = (
        row["state_check"] == "passed"
        and row["completion_ratio"] >= 0.999999
        and row["valid_latency_ratio"] >= 0.999999
    )
    return row


NUMERIC_GROUP_FIELDS = [
    "effective_total",
    "completion_ratio",
    "valid_latency_ratio",
    "actual_offered_tps",
    "offered_attainment_ratio",
    "wall_tps",
    "wall_to_offered_ratio",
    "mean_latency_sec",
    "p50_latency_sec",
    "p95_latency_sec",
    "p99_latency_sec",
    "max_latency_sec",
    "weighted_cross_ratio",
    "mean_load_variance",
    "mean_max_shard_load_share",
    "p95_max_shard_load_share",
    "mean_active_shards",
    "mean_jain_fairness",
    "mean_load_cv",
    "aggregate_max_shard_load_share",
    "aggregate_jain_fairness",
    "python_ppo_ratio",
    "fallback_ratio",
    "action_mask_size_mean",
    "candidate_major_anchor_coverage_ratio",
    "candidate_all_related_coverage_ratio",
    "candidate_related_mass_coverage_mean",
    "major_related_follow_ratio",
    "chosen_any_related_ratio",
    "chosen_related_mass_mean",
    "major_anchor_choice_when_available_ratio",
]


def group_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["window"]), int(row["inject_tps"]))
        buckets.setdefault(key, []).append(row)

    groups: List[Dict[str, object]] = []
    for key, members in sorted(buckets.items()):
        group: Dict[str, object] = {
            "method": key[0],
            "window": key[1],
            "inject_tps": key[2],
            "n": len(members),
            "seeds": sorted(int(member["seed"]) for member in members),
            "all_protocol_ok": all(bool(member["protocol_ok"]) for member in members),
        }
        for field in NUMERIC_GROUP_FIELDS:
            values = [float(member[field]) for member in members]
            group[f"{field}_mean"] = statistics.fmean(values)
            group[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    return groups


def mean_sd(group: Mapping[str, object], field: str, scale: float = 1.0) -> str:
    mean_value = float(group[f"{field}_mean"]) * scale
    sd_value = group.get(f"{field}_sd")
    if sd_value is None:
        return f"{mean_value:.3f}"
    return f"{mean_value:.3f} ± {float(sd_value) * scale:.3f}"


def build_report(rows: List[Dict[str, object]], groups: List[Dict[str, object]]) -> str:
    formal = [
        group for group in groups
        if group["window"] == "test2M" and group["inject_tps"] == 250
    ]
    validation = [
        group for group in groups
        if group["window"] == "validation444k" and group["inject_tps"] == 250
    ]
    lines = [
        "# Result9 完整多目标复算报告",
        "",
        f"- 已发现并复算 {len(rows)} 个 finalized BlockEmulator 结果目录。",
        f"- 协议完整通过：{sum(bool(row['protocol_ok']) for row in rows)}/{len(rows)}。",
        "- 所有归一化负载指标均由归档的 16 分片逐 epoch 负载重新计算；旧 analysis.json 未被覆盖。",
        "- relation_weighted_cross_ratio 与 weighted_communication_cost 因现有结果未为所有方法保存统一逐交易关系权重，当前标记为不可回算。",
        "",
        "## 正式 test2M / 250 TPS",
        "",
        "| 方法 | n/种子 | 跨片率 % | 最大分片负载占比 | Jain 公平性 | 活跃分片 | 平均延迟 s | P95 s | Wall TPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in formal:
        lines.append(
            "| {method} | {n}/{seeds} | {cross} | {max_share} | {jain} | {active} | {latency} | {p95} | {wall} |".format(
                method=group["method"],
                n=group["n"],
                seeds=",".join(str(seed) for seed in group["seeds"]),
                cross=mean_sd(group, "weighted_cross_ratio", 100.0),
                max_share=mean_sd(group, "mean_max_shard_load_share"),
                jain=mean_sd(group, "mean_jain_fairness"),
                active=mean_sd(group, "mean_active_shards"),
                latency=mean_sd(group, "mean_latency_sec"),
                p95=mean_sd(group, "p95_latency_sec"),
                wall=mean_sd(group, "wall_tps"),
            )
        )

    lines.extend(
        [
            "",
            "## validation444k / 250 TPS",
            "",
            "| 方法 | n/种子 | 跨片率 % | 最大分片负载占比 | Jain 公平性 | 平均延迟 s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in validation:
        lines.append(
            "| {method} | {n}/{seeds} | {cross} | {max_share} | {jain} | {latency} |".format(
                method=group["method"],
                n=group["n"],
                seeds=",".join(str(seed) for seed in group["seeds"]),
                cross=mean_sd(group, "weighted_cross_ratio", 100.0),
                max_share=mean_sd(group, "mean_max_shard_load_share"),
                jain=mean_sd(group, "mean_jain_fairness"),
                latency=mean_sd(group, "mean_latency_sec"),
            )
        )

    reported_keys = {
        (group["method"], group["window"], group["inject_tps"])
        for group in formal + validation
    }
    remaining = [
        group for group in groups
        if (group["method"], group["window"], group["inject_tps"])
        not in reported_keys
    ]
    lines.extend(
        [
            "",
            "## 其余负载与 Top-K 敏感性结果",
            "",
            "| 方法 | 窗口 | TPS | n/种子 | 跨片率 % | 最大分片负载占比 | Jain 公平性 | 平均延迟 s | P95 s | Wall TPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in remaining:
        lines.append(
            "| {method} | {window} | {tps} | {n}/{seeds} | {cross} | {max_share} | {jain} | {latency} | {p95} | {wall} |".format(
                method=group["method"],
                window=group["window"],
                tps=group["inject_tps"],
                n=group["n"],
                seeds=",".join(str(seed) for seed in group["seeds"]),
                cross=mean_sd(group, "weighted_cross_ratio", 100.0),
                max_share=mean_sd(group, "mean_max_shard_load_share"),
                jain=mean_sd(group, "mean_jain_fairness"),
                latency=mean_sd(group, "mean_latency_sec"),
                p95=mean_sd(group, "p95_latency_sec"),
                wall=mean_sd(group, "wall_tps"),
            )
        )

    proposed_formal = [
        row for row in rows
        if row["method"] == "Proposed-TopK7"
        and row["window"] == "test2M"
        and row["inject_tps"] == 250
    ]
    if proposed_formal:
        lines.extend(
            [
                "",
                "## Proposed 候选与动作诊断（正式 3 seeds）",
                "",
            ]
        )
        for field, label in [
            ("candidate_major_anchor_coverage_ratio", "主要关系分片进入 Top-K"),
            ("candidate_all_related_coverage_ratio", "全部关系分片进入 Top-K"),
            ("candidate_related_mass_coverage_mean", "候选关系质量覆盖"),
            ("major_related_follow_ratio", "最终选择主要关系分片"),
            ("chosen_any_related_ratio", "最终选择任一关系分片"),
            ("chosen_related_mass_mean", "最终选择关系质量"),
            ("major_anchor_choice_when_available_ratio", "主要关系分片可用时仍选择它"),
        ]:
            values = [float(row[field]) for row in proposed_formal]
            mean_value = statistics.fmean(values) * 100.0
            sd_value = statistics.stdev(values) * 100.0 if len(values) > 1 else 0.0
            lines.append(f"- {label}: {mean_value:.2f}% ± {sd_value:.2f}%。")

    lines.extend(
        [
            "",
            "## 判读边界",
            "",
            "- 多种子均值只对真正存在多个独立系统运行种子的方法计算；n=1 不报告伪标准差。",
            "- 当前数据可判断跨片、负载、延迟与稳定性折中，不能证明吞吐容量优势；250 TPS 下各方法均受相同注入上限约束。",
            "- Candidate-Only/No-PPO 尚未运行，因此报告中不会伪造该基线结果。",
        ]
    )
    return "\n".join(lines) + "\n"


METRIC_AVAILABILITY = {
    "available_for_all_archived_runs": [
        "state_check",
        "effective_total",
        "completion_ratio",
        "valid_latency_ratio",
        "actual_offered_tps",
        "offered_attainment_ratio",
        "schedule_lag_ms",
        "wall_tps",
        "wall_to_offered_ratio",
        "mean/p50/p95/p99/max_latency_sec",
        "weighted_cross_ratio_by_transaction_count",
        "mean/p95_raw_load_variance",
        "mean/p95_max_shard_load_share",
        "mean/min_active_shards",
        "mean/p05_jain_fairness",
        "mean/p95_load_cv",
        "aggregate_max_shard_load_share",
        "aggregate_jain_fairness",
    ],
    "available_for_runs_with_decision_records": [
        "python_ppo_ratio",
        "fallback_ratio",
        "action_mask_size_mean",
        "candidate_major_anchor_coverage_ratio",
        "candidate_all_related_coverage_ratio",
        "candidate_related_mass_coverage_mean",
        "major_related_follow_ratio",
        "chosen_any_related_ratio",
        "chosen_related_mass_mean",
        "major_anchor_choice_when_available_ratio",
    ],
    "defined_but_not_recoverable_uniformly_from_current_archives": [
        "relation_weighted_cross_ratio",
        "weighted_communication_cost",
        "overload_avoided_ratio_with_counterfactual_placement",
        "backlog_growth_rate",
        "unconfirmed_transaction_count_at_cutoff",
    ],
}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(results_root: Path, output_dir: Path) -> Dict[str, object]:
    run_roots = sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and (path / "run_complete.json").is_file()
    )
    rows = [row for row in (flatten_run(path) for path in run_roots) if row is not None]
    groups = group_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result9_multiobjective_runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "result9_multiobjective_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "result9_metric_availability.json").write_text(
        json.dumps(METRIC_AVAILABILITY, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "result9_multiobjective_report.md").write_text(
        build_report(rows, groups),
        encoding="utf-8",
    )
    write_csv(output_dir / "result9_multiobjective_runs.csv", rows)
    return {
        "run_count": len(rows),
        "group_count": len(groups),
        "protocol_ok_count": sum(bool(row["protocol_ok"]) for row in rows),
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path(r"E:\project_iot\实验结果9\block_eval"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis_outputs"),
    )
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.results_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
