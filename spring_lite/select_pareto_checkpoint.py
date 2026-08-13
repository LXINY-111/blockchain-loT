import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


SELECTION_POLICY_NAME = "hierarchical_constrained_pareto_v1"


def pareto_candidates(manifest: Dict[str, object]) -> List[Dict[str, object]]:
    raw_candidates = manifest.get("pareto_frontier", [])
    if not isinstance(raw_candidates, list):
        return []
    return [
        candidate
        for candidate in raw_candidates
        if isinstance(candidate, dict) and candidate.get("checkpoint_path")
    ]


def active_deficit_ratio(
    candidate: Dict[str, object],
    min_active_shards: float,
) -> float:
    active_shards = float(candidate.get("active_shards_mean", 0.0))
    return max(0.0, (min_active_shards - active_shards) / min_active_shards)


def eligible_candidates(
    manifest: Dict[str, object],
    min_active_shards: float,
    max_stage_hotspot: float,
) -> List[Dict[str, object]]:
    return [
        candidate
        for candidate in pareto_candidates(manifest)
        if float(candidate.get("active_shards_mean", 0.0))
        >= min_active_shards
        and float(candidate.get("stage_max_load_share", 1.0))
        <= max_stage_hotspot
    ]


def fallback_candidates(
    manifest: Dict[str, object],
    min_active_shards: float,
    max_stage_hotspot: float,
    max_active_deficit_ratio: float,
) -> List[Dict[str, object]]:
    # The system hotspot cap remains hard. Only the averaged active-shard
    # threshold receives a bounded tolerance when the strict set is empty.
    return [
        candidate
        for candidate in pareto_candidates(manifest)
        if float(candidate.get("stage_max_load_share", 1.0))
        <= max_stage_hotspot
        and active_deficit_ratio(candidate, min_active_shards)
        <= max_active_deficit_ratio
    ]


def strict_sort_key(candidate: Dict[str, object]) -> tuple:
    return (
        float(candidate.get("cross_ratio", 1.0)),
        float(candidate.get("stage_max_load_share", 1.0)),
        -float(candidate.get("active_shards_mean", 0.0)),
        int(candidate.get("epoch", 0)),
        str(candidate.get("event", "")),
    )


def fallback_sort_key(
    candidate: Dict[str, object],
    min_active_shards: float,
) -> tuple:
    return (
        active_deficit_ratio(candidate, min_active_shards),
        float(candidate.get("cross_ratio", 1.0)),
        float(candidate.get("stage_max_load_share", 1.0)),
        int(candidate.get("epoch", 0)),
        str(candidate.get("event", "")),
    )


def select_candidate_decision(
    manifest: Dict[str, object],
    min_active_shards: float,
    max_stage_hotspot: float,
    max_active_deficit_ratio: float = 0.05,
) -> Dict[str, object]:
    if min_active_shards <= 0.0:
        raise ValueError("min_active_shards must be positive")
    if not 0.0 <= max_stage_hotspot <= 1.0:
        raise ValueError("max_stage_hotspot must be in [0, 1]")
    if not 0.0 <= max_active_deficit_ratio <= 1.0:
        raise ValueError("max_active_deficit_ratio must be in [0, 1]")

    strict_candidates = eligible_candidates(
        manifest,
        min_active_shards=min_active_shards,
        max_stage_hotspot=max_stage_hotspot,
    )
    if strict_candidates:
        selected = min(strict_candidates, key=strict_sort_key)
        selection_tier = "strict"
    else:
        relaxed_candidates = fallback_candidates(
            manifest,
            min_active_shards=min_active_shards,
            max_stage_hotspot=max_stage_hotspot,
            max_active_deficit_ratio=max_active_deficit_ratio,
        )
        if not relaxed_candidates:
            raise RuntimeError(
                "no Pareto checkpoint satisfies the strict guardrails or "
                "the bounded active-shard fallback: "
                f"active_shards >= {min_active_shards}, "
                f"stage_max_load_share <= {max_stage_hotspot}, "
                "or fallback active_deficit_ratio <= "
                f"{max_active_deficit_ratio} with the same hotspot cap"
            )
        selected = min(
            relaxed_candidates,
            key=lambda candidate: fallback_sort_key(
                candidate,
                min_active_shards,
            ),
        )
        selection_tier = "active_tolerance"

    active_shards = float(selected.get("active_shards_mean", 0.0))
    stage_hotspot = float(selected.get("stage_max_load_share", 1.0))
    deficit_ratio = active_deficit_ratio(selected, min_active_shards)
    strict_satisfied = (
        active_shards >= min_active_shards
        and stage_hotspot <= max_stage_hotspot
    )
    return {
        "selection_policy": SELECTION_POLICY_NAME,
        "selection_tier": selection_tier,
        "strict_feasible": bool(strict_candidates),
        "constraint_audit": {
            "active_shards_mean": active_shards,
            "stage_max_load_share": stage_hotspot,
            "active_shards_deficit": max(
                0.0,
                min_active_shards - active_shards,
            ),
            "active_deficit_ratio": deficit_ratio,
            "stage_hotspot_excess": max(
                0.0,
                stage_hotspot - max_stage_hotspot,
            ),
            "strict_constraints_satisfied": strict_satisfied,
            "fallback_constraints_satisfied": (
                stage_hotspot <= max_stage_hotspot
                and deficit_ratio <= max_active_deficit_ratio
            ),
        },
        "selected": selected,
    }


def select_candidate(
    manifest: Dict[str, object],
    min_active_shards: float,
    max_stage_hotspot: float,
    max_active_deficit_ratio: float = 0.05,
) -> Dict[str, object]:
    decision = select_candidate_decision(
        manifest,
        min_active_shards=min_active_shards,
        max_stage_hotspot=max_stage_hotspot,
        max_active_deficit_ratio=max_active_deficit_ratio,
    )
    return decision["selected"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--selection_json", default="")
    parser.add_argument("--min_active_shards", type=float, default=8.0)
    parser.add_argument("--max_stage_hotspot", type=float, default=0.35)
    parser.add_argument(
        "--max_active_deficit_ratio",
        type=float,
        default=0.05,
        help=(
            "fallback tolerance for the active-shard threshold; the hotspot "
            "cap is never relaxed"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    decision = select_candidate_decision(
        manifest,
        min_active_shards=args.min_active_shards,
        max_stage_hotspot=args.max_stage_hotspot,
        max_active_deficit_ratio=args.max_active_deficit_ratio,
    )
    selected = decision["selected"]

    source = Path(str(selected["checkpoint_path"]))
    if not source.is_file():
        raise FileNotFoundError(f"selected checkpoint not found: {source}")
    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)

    selection_path = (
        Path(args.selection_json)
        if args.selection_json
        else output.with_name(f"{output.stem}_selection.json")
    )
    selection = {
        "schema_version": 2,
        "source_manifest": str(manifest_path.resolve()),
        "source_checkpoint": str(source.resolve()),
        "output_model": str(output.resolve()),
        "selection_policy": {
            "name": SELECTION_POLICY_NAME,
            "strict_ranking": [
                "cross_ratio",
                "stage_max_load_share",
                "-active_shards_mean",
                "epoch",
                "event",
            ],
            "fallback_ranking": [
                "active_deficit_ratio",
                "cross_ratio",
                "stage_max_load_share",
                "epoch",
                "event",
            ],
            "hotspot_cap_relaxed_in_fallback": False,
        },
        "constraints": {
            "min_active_shards": args.min_active_shards,
            "max_stage_hotspot": args.max_stage_hotspot,
            "max_active_deficit_ratio": args.max_active_deficit_ratio,
        },
        "selection_tier": decision["selection_tier"],
        "strict_feasible": decision["strict_feasible"],
        "constraint_audit": decision["constraint_audit"],
        "selected": selected,
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=True))


if __name__ == "__main__":
    main()
