import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.distributions import Categorical

from config import (
    ACTION_REWARD_SCALE,
    ACTIVE_SHARD_BONUS_WEIGHT,
    ARGMAX_TIE_BREAK,
    ARGMAX_TIE_EPS,
    BACKLOG_PENALTY_WEIGHT,
    BETA,
    CAPACITY_BACKLOG_MODE,
    DEFAULT_BLOCK_INTERVAL_MS,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_CAPACITY_GUARD,
    DEFAULT_CAPACITY_GUARD_FACTOR,
    DEFAULT_CSV_PATH,
    DEFAULT_EVAL_LOG_JSONL,
    DEFAULT_IOT_REWARD_MODE,
    DEFAULT_IOT_CSV_PATH,
    DEFAULT_IOT_SIDECAR_PATH,
    DEFAULT_SENDER_POS_MODE,
    DEFAULT_SHARD_NUM,
    DEFAULT_TEST_MAX_TXS,
    DEFAULT_TEST_START_TX,
    DEFAULT_TX_BATCH_SIZE,
    HIDDEN_DIM,
    HOTSPOT_PENALTY_WEIGHT,
    HOTSPOT_THRESHOLD,
    IOT_BALANCE_WEIGHT,
    IOT_COMM_COST_WEIGHT,
    IOT_CSTR_WEIGHT,
    IOT_FEATURE_DIM,
    IOT_HOTSPOT_WEIGHT,
    LAMBDA_WEIGHT,
    LOAD_PENALTY_WEIGHT,
    LOAD_SEMANTICS_VERSION,
    LOCAL_REWARD_WEIGHT,
    MIN_ACTIVE_LOAD_SHARE,
    MODEL_PATH,
    REWARD_MODE,
    REWARD_MODE_CHOICES,
    state_dim,
)
from action_mask import mask_logits, normalize_action_mask
from action_select import tie_aware_argmax
from checkpoint_compat import checkpoint_config_mismatches, format_checkpoint_mismatch
from heuristic import addr2shard, heuristic_from_state
from offline_env import (
    BatchMetrics,
    PolicyOutput,
    SenderPosInfo,
    SpringOfflineEnv,
    iter_batches,
    load_iot_transactions,
    load_transactions,
)
from ppo import PPOAgent


def new_summary(
    shards: int,
    max_block_size: int = 1000,
    block_interval_ms: int = DEFAULT_BLOCK_INTERVAL_MS,
) -> Dict[str, object]:
    per_shard_capacity_tps = (
        float(max_block_size) * 1000.0 / float(max(1, block_interval_ms))
    )
    return {
        "batches": 0,
        "tx_count": 0,
        "action_count": 0,
        "effective_tx": 0.0,
        "cross_tx": 0.0,
        "reward_sum": 0.0,
        "norm_var_sum": 0.0,
        "r_cstr_sum": 0.0,
        "r_wlb_sum": 0.0,
        "active_shards_sum": 0.0,
        "active_shard_ratio_sum": 0.0,
        "max_load_share_sum": 0.0,
        "hotspot_penalty_sum": 0.0,
        "load_aware_bonus_sum": 0.0,
        "backlog_penalty_sum": 0.0,
        "communication_cost_sum": 0.0,
        "relation_effective_tx": 0.0,
        "relation_cross_tx": 0.0,
        "owner_loads": [0 for _ in range(shards)],
        "relation_effective_loads": [0.0 for _ in range(shards)],
        "relation_cross_loads": [0.0 for _ in range(shards)],
        "per_shard_capacity_tps": per_shard_capacity_tps,
        "load_semantics_version": LOAD_SEMANTICS_VERSION,
        "related_known_count": 0,
        "same_as_related_count": 0,
        "related_shard_hist": [0 for _ in range(shards)],
        "same_related_by_shard": [0 for _ in range(shards)],
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "action_hist": [0 for _ in range(shards)],
        "loads": [0 for _ in range(shards)],
        "effective_loads": [0.0 for _ in range(shards)],
        "reward_loads": [0.0 for _ in range(shards)],
        "committed_loads": [0.0 for _ in range(shards)],
        "committed_stage_loads": [0.0 for _ in range(shards)],
        "pending_loads": [0.0 for _ in range(shards)],
    }


def update_summary(summary: Dict[str, object], metrics: BatchMetrics) -> None:
    summary["batches"] = int(summary["batches"]) + 1
    summary["tx_count"] = int(summary["tx_count"]) + metrics.tx_count
    summary["action_count"] = int(summary["action_count"]) + metrics.action_count
    summary["effective_tx"] = float(summary["effective_tx"]) + metrics.effective_tx
    summary["cross_tx"] = float(summary["cross_tx"]) + metrics.cross_tx
    summary["reward_sum"] = float(summary["reward_sum"]) + metrics.reward
    summary["norm_var_sum"] = float(summary["norm_var_sum"]) + metrics.normalized_load_variance
    summary["r_cstr_sum"] = float(summary["r_cstr_sum"]) + metrics.r_cstr
    summary["r_wlb_sum"] = float(summary["r_wlb_sum"]) + metrics.r_wlb
    summary["active_shards_sum"] = float(summary["active_shards_sum"]) + metrics.active_shards
    summary["active_shard_ratio_sum"] = (
        float(summary["active_shard_ratio_sum"]) + metrics.active_shard_ratio
    )
    summary["max_load_share_sum"] = float(summary["max_load_share_sum"]) + metrics.max_load_share
    summary["hotspot_penalty_sum"] = (
        float(summary["hotspot_penalty_sum"]) + metrics.hotspot_penalty
    )
    summary["load_aware_bonus_sum"] = (
        float(summary["load_aware_bonus_sum"]) + metrics.load_aware_bonus
    )
    summary["backlog_penalty_sum"] = (
        float(summary["backlog_penalty_sum"]) + metrics.backlog_penalty
    )
    summary["communication_cost_sum"] = (
        float(summary["communication_cost_sum"]) + metrics.communication_cost
    )
    summary["relation_effective_tx"] = (
        float(summary["relation_effective_tx"])
        + float(sum(metrics.relation_effective_loads))
    )
    summary["relation_cross_tx"] = (
        float(summary["relation_cross_tx"])
        + float(sum(metrics.relation_cross_loads))
    )
    summary["related_known_count"] = int(summary["related_known_count"]) + metrics.related_known_count
    summary["same_as_related_count"] = int(summary["same_as_related_count"]) + metrics.same_as_related_count
    related_shard_hist = summary["related_shard_hist"]
    same_related_by_shard = summary["same_related_by_shard"]
    assert isinstance(related_shard_hist, list)
    assert isinstance(same_related_by_shard, list)
    for sid, count in enumerate(metrics.related_shard_hist):
        related_shard_hist[sid] += count
    for sid, count in enumerate(metrics.same_related_by_shard):
        same_related_by_shard[sid] += count
    summary["confidence_sum"] = float(summary["confidence_sum"]) + (
        metrics.confidence_mean * metrics.action_count
    )
    summary["entropy_sum"] = float(summary["entropy_sum"]) + (
        metrics.entropy_mean * metrics.action_count
    )

    action_hist = summary["action_hist"]
    loads = summary["loads"]
    effective_loads = summary["effective_loads"]
    reward_loads = summary["reward_loads"]
    committed_loads = summary["committed_loads"]
    committed_stage_loads = summary["committed_stage_loads"]
    pending_loads = summary["pending_loads"]
    owner_loads = summary["owner_loads"]
    relation_effective_loads = summary["relation_effective_loads"]
    relation_cross_loads = summary["relation_cross_loads"]
    assert isinstance(action_hist, list)
    assert isinstance(loads, list)
    assert isinstance(effective_loads, list)
    assert isinstance(reward_loads, list)
    assert isinstance(committed_loads, list)
    assert isinstance(committed_stage_loads, list)
    assert isinstance(pending_loads, list)
    assert isinstance(owner_loads, list)
    assert isinstance(relation_effective_loads, list)
    assert isinstance(relation_cross_loads, list)

    for sid, count in enumerate(metrics.action_hist):
        action_hist[sid] += count
    for sid, count in enumerate(metrics.loads):
        loads[sid] += count
    for sid, value in enumerate(metrics.effective_loads):
        effective_loads[sid] += value
    for sid, value in enumerate(metrics.reward_loads):
        reward_loads[sid] += value
    for sid, value in enumerate(metrics.committed_loads):
        committed_loads[sid] += value
    for sid, value in enumerate(metrics.committed_stage_loads):
        committed_stage_loads[sid] += value
    for sid, value in enumerate(metrics.pending_loads):
        pending_loads[sid] = value
    for sid, value in enumerate(metrics.owner_loads):
        owner_loads[sid] += value
    for sid, value in enumerate(metrics.relation_effective_loads):
        relation_effective_loads[sid] += value
    for sid, value in enumerate(metrics.relation_cross_loads):
        relation_cross_loads[sid] += value


def summary_view(summary: Dict[str, object]) -> Dict[str, object]:
    batches = max(1, int(summary["batches"]))
    actions = max(1, int(summary["action_count"]))
    effective_tx = float(summary["effective_tx"])
    related_known = int(summary["related_known_count"])

    action_hist = list(summary["action_hist"])
    action_total = sum(action_hist)
    action_dist = [count / action_total if action_total else 0.0 for count in action_hist]
    related_shard_hist = list(summary["related_shard_hist"])
    same_related_by_shard = list(summary["same_related_by_shard"])
    related_follow_by_shard = [
        (
            same_related_by_shard[sid] / related_shard_hist[sid]
            if related_shard_hist[sid]
            else 0.0
        )
        for sid in range(len(related_shard_hist))
    ]
    observed_related_follow = [
        related_follow_by_shard[sid]
        for sid in range(len(related_shard_hist))
        if related_shard_hist[sid] > 0
    ]
    min_related_follow_ratio = (
        min(observed_related_follow) if observed_related_follow else 0.0
    )

    loads = list(summary["loads"])
    effective_loads = list(summary["effective_loads"])
    load_total = sum(effective_loads)
    load_dist = [value / load_total if load_total else 0.0 for value in effective_loads]
    reward_loads = list(summary["reward_loads"])
    reward_load_total = sum(reward_loads)
    reward_load_dist = [
        value / reward_load_total if reward_load_total else 0.0
        for value in reward_loads
    ]
    committed_loads = list(summary["committed_loads"])
    committed_load_total = sum(committed_loads)
    committed_load_dist = [
        value / committed_load_total if committed_load_total else 0.0
        for value in committed_loads
    ]
    committed_stage_loads = list(summary["committed_stage_loads"])
    committed_stage_total = sum(committed_stage_loads)
    committed_stage_dist = [
        value / committed_stage_total if committed_stage_total else 0.0
        for value in committed_stage_loads
    ]
    pending_loads = list(summary["pending_loads"])
    owner_loads = list(summary["owner_loads"])
    owner_total = float(sum(owner_loads))
    owner_load_dist = [
        float(value) / owner_total if owner_total > 0 else 0.0
        for value in owner_loads
    ]
    stage_total = float(sum(loads))
    stage_load_dist = [
        float(value) / stage_total if stage_total > 0 else 0.0
        for value in loads
    ]
    owner_max_load_share = max(owner_load_dist) if owner_load_dist else 0.0
    stage_max_load_share = max(stage_load_dist) if stage_load_dist else 0.0
    tx_count = int(summary["tx_count"])
    stage_max_load_per_tx = (
        max(float(value) for value in loads) / float(tx_count)
        if tx_count > 0 and loads
        else 0.0
    )
    per_shard_capacity_tps = float(summary["per_shard_capacity_tps"])
    owner_anchor_tps_ceiling = (
        per_shard_capacity_tps / owner_max_load_share
        if owner_max_load_share > 0
        else 0.0
    )
    system_stage_tps_ceiling = (
        per_shard_capacity_tps / stage_max_load_per_tx
        if stage_max_load_per_tx > 0
        else 0.0
    )
    relation_effective = float(summary["relation_effective_tx"])
    relation_cross = float(summary["relation_cross_tx"])

    return {
        "batches": int(summary["batches"]),
        "tx_count": int(summary["tx_count"]),
        "action_count": int(summary["action_count"]),
        "effective_tx": effective_tx,
        "cross_tx": float(summary["cross_tx"]),
        "cross_ratio": float(summary["cross_tx"]) / effective_tx if effective_tx > 0 else 0.0,
        "reward_mean": float(summary["reward_sum"]) / batches,
        "norm_var_mean": float(summary["norm_var_sum"]) / batches,
        "r_cstr_mean": float(summary["r_cstr_sum"]) / batches,
        "r_wlb_mean": float(summary["r_wlb_sum"]) / batches,
        "active_shards_mean": float(summary["active_shards_sum"]) / batches,
        "active_shard_ratio_mean": float(summary["active_shard_ratio_sum"]) / batches,
        "max_load_share_mean": float(summary["max_load_share_sum"]) / batches,
        "hotspot_penalty_mean": float(summary["hotspot_penalty_sum"]) / batches,
        "load_aware_bonus_mean": float(summary["load_aware_bonus_sum"]) / batches,
        "backlog_penalty_mean": float(summary["backlog_penalty_sum"]) / batches,
        "communication_cost_mean": float(summary["communication_cost_sum"]) / batches,
        "relation_cross_ratio": (
            relation_cross / relation_effective if relation_effective > 0 else 0.0
        ),
        "owner_loads": owner_loads,
        "owner_load_dist": owner_load_dist,
        "owner_max_load_share": owner_max_load_share,
        "stage_load_dist": stage_load_dist,
        "stage_max_load_share": stage_max_load_share,
        "stage_max_load_per_tx": stage_max_load_per_tx,
        "per_shard_capacity_tps": per_shard_capacity_tps,
        "owner_anchor_tps_ceiling": owner_anchor_tps_ceiling,
        "system_stage_tps_ceiling": system_stage_tps_ceiling,
        "relation_effective_loads": list(summary["relation_effective_loads"]),
        "relation_cross_loads": list(summary["relation_cross_loads"]),
        "load_semantics_version": summary["load_semantics_version"],
        "same_as_related_ratio": (
            int(summary["same_as_related_count"]) / related_known if related_known else 0.0
        ),
        "related_known_count": related_known,
        "same_as_related_count": int(summary["same_as_related_count"]),
        "related_shard_hist": related_shard_hist,
        "same_related_by_shard": same_related_by_shard,
        "related_follow_by_shard": related_follow_by_shard,
        "min_related_follow_ratio": min_related_follow_ratio,
        "confidence_mean": float(summary["confidence_sum"]) / actions,
        "entropy_mean": float(summary["entropy_sum"]) / actions,
        "action_hist": action_hist,
        "action_dist": action_dist,
        "loads": loads,
        "effective_loads": effective_loads,
        "load_dist": load_dist,
        "reward_loads": reward_loads,
        "reward_load_dist": reward_load_dist,
        "committed_loads": committed_loads,
        "committed_load_dist": committed_load_dist,
        "committed_stage_loads": committed_stage_loads,
        "committed_stage_load_dist": committed_stage_dist,
        "pending_loads": pending_loads,
    }


def is_iot_mdp(args: argparse.Namespace) -> bool:
    return str(getattr(args, "mdp_mode", "spring")).strip().lower() == "iot"


def uses_iot_identity(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "tx_identity", "auto")).strip().lower()
    if mode == "iot":
        return True
    if mode == "raw":
        return False
    return is_iot_mdp(args)


def normalize_reward_mode(args: argparse.Namespace) -> None:
    if is_iot_mdp(args) and args.reward_mode == REWARD_MODE:
        args.reward_mode = DEFAULT_IOT_REWARD_MODE


def iot_feature_dim_for_args(args: argparse.Namespace) -> int:
    return IOT_FEATURE_DIM if is_iot_mdp(args) else 0


def eval_state_dim(args: argparse.Namespace, shards: Optional[int] = None) -> int:
    shard_count = int(shards if shards is not None else args.shards)
    return state_dim(shard_count, iot_feature_dim_for_args(args))


def resolve_csv_path(args: argparse.Namespace) -> Path:
    csv_arg = str(getattr(args, "csv", "")).strip()
    if csv_arg:
        return Path(csv_arg)
    return DEFAULT_IOT_CSV_PATH if uses_iot_identity(args) else DEFAULT_CSV_PATH


def resolve_iot_sidecar_path(args: argparse.Namespace) -> Path:
    sidecar = str(getattr(args, "sidecar", "")).strip()
    return Path(sidecar) if sidecar else DEFAULT_IOT_SIDECAR_PATH


def load_eval_transactions(args: argparse.Namespace, csv_path: Path):
    start_tx = int(getattr(args, "start_tx", 0))
    if uses_iot_identity(args):
        sidecar_path = resolve_iot_sidecar_path(args)
        if not sidecar_path.exists():
            raise FileNotFoundError(f"IoT sidecar not found: {sidecar_path}")
        args.sidecar = str(sidecar_path)
        return load_iot_transactions(
            csv_path,
            sidecar_path,
            max_txs=args.max_txs,
            start_tx=start_tx,
        )
    return load_transactions(
        csv_path,
        max_txs=args.max_txs,
        start_tx=start_tx,
    )


def expected_checkpoint_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "mdp_mode": args.mdp_mode,
        "tx_identity_resolved": "iot" if uses_iot_identity(args) else "raw",
        "shards": args.shards,
        "iot_feature_dim": iot_feature_dim_for_args(args),
        "sender_pos_mode": args.sender_pos_mode,
        "reward_mode": args.reward_mode,
        "iot_cstr_weight": args.iot_cstr_weight,
        "iot_balance_weight": args.iot_balance_weight,
        "iot_comm_cost_weight": args.iot_comm_cost_weight,
        "iot_hotspot_weight": args.iot_hotspot_weight,
        "candidate_top_k": args.candidate_top_k,
        "capacity_guard": args.capacity_guard,
        "capacity_guard_factor": args.capacity_guard_factor,
        "candidate_load_weight": args.candidate_load_weight,
        "max_block_size": args.max_block_size,
        "block_interval_ms": args.block_interval_ms,
        "load_semantics_version": LOAD_SEMANTICS_VERSION,
    }


def load_agent(args: argparse.Namespace) -> Optional[PPOAgent]:
    if args.policy != "ppo":
        return None

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    agent = PPOAgent(
        state_dim=eval_state_dim(args),
        action_dim=args.shards,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    payload = agent.load(model_path)
    ckpt_state_dim = int(payload.get("state_dim", -1))
    ckpt_action_dim = int(payload.get("action_dim", -1))
    if ckpt_state_dim != eval_state_dim(args) or ckpt_action_dim != args.shards:
        raise ValueError(
            "checkpoint dimension mismatch: "
            f"state_dim={ckpt_state_dim}, action_dim={ckpt_action_dim}"
        )
    mismatches = checkpoint_config_mismatches(
        payload.get("extra", {}),
        expected_checkpoint_config(args),
    )
    if mismatches and not bool(
        getattr(args, "allow_checkpoint_config_mismatch", False)
    ):
        raise ValueError(format_checkpoint_mismatch(mismatches))
    agent.net.eval()
    return agent


def make_policy(args: argparse.Namespace, agent: Optional[PPOAgent]):
    rng = random.Random(args.seed)

    def ppo_policy(
        state: List[float],
        _address: str,
        _related: str,
        _info: SenderPosInfo,
    ) -> PolicyOutput:
        assert agent is not None
        state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = agent.net(state_t)
            masks = [normalize_action_mask(_info.action_mask, agent.action_dim)]
            masked = mask_logits(logits, masks)
            dist = Categorical(logits=masked)
            probs = torch.softmax(masked, dim=-1)
            if args.sample:
                action = dist.sample()
            else:
                action_id = tie_aware_argmax(
                    probs.squeeze(0).detach().cpu().tolist(),
                    key=f"{_address}|{_related}",
                    tie_eps=args.argmax_tie_eps,
                    enable_tie_break=bool(args.argmax_tie_break),
                )
                action = torch.tensor([action_id], dtype=torch.long, device=agent.device)
            log_prob = dist.log_prob(action)
            confidence = probs.gather(1, action.unsqueeze(1)).squeeze(1)
            entropy = dist.entropy()
        return PolicyOutput(
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            confidence=float(confidence.item()),
            entropy=float(entropy.item()),
            source="python_ppo",
        )

    def heuristic_policy(
        state: List[float],
        _address: str,
        _related: str,
        _info: SenderPosInfo,
    ) -> PolicyOutput:
        return PolicyOutput(
            action=heuristic_from_state(state, args.shards, _info.action_mask),
            source="heuristic",
        )

    def hash_policy(
        _state: List[float],
        address: str,
        _related: str,
        _info: SenderPosInfo,
    ) -> PolicyOutput:
        return PolicyOutput(action=addr2shard(address, args.shards), source="hash")

    def random_policy(
        _state: List[float],
        _address: str,
        _related: str,
        _info: SenderPosInfo,
    ) -> PolicyOutput:
        return PolicyOutput(action=rng.randrange(args.shards), source="random")

    if args.policy == "ppo":
        return ppo_policy
    if args.policy == "heuristic":
        return heuristic_policy
    if args.policy == "hash":
        return hash_policy
    if args.policy == "random":
        return random_policy
    raise ValueError(f"unsupported policy: {args.policy}")


def write_jsonl(path: str, record: Dict[str, object]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    csv_path = resolve_csv_path(args)
    args.csv = str(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    txs = load_eval_transactions(args, csv_path)
    if not txs:
        raise RuntimeError(f"no valid transactions loaded from {csv_path}")
    if args.max_txs > 0 and len(txs) != args.max_txs:
        raise RuntimeError(
            f"evaluation window is incomplete: loaded={len(txs)}, "
            f"requested={args.max_txs}"
        )

    agent = load_agent(args)
    policy = make_policy(args, agent)
    env = SpringOfflineEnv(
        shards=args.shards,
        tx_batch_size=args.tx_batch_size,
        max_block_size=args.max_block_size,
        block_interval_ms=args.block_interval_ms,
        lambda_weight=args.lambda_weight,
        beta=args.beta,
        sender_pos_mode=args.sender_pos_mode,
        temporal_top_k=args.temporal_top_k,
        local_reward_weight=args.local_reward_weight,
        action_reward_scale=args.action_reward_scale,
        load_penalty_weight=args.load_penalty_weight,
        active_shard_bonus_weight=args.active_shard_bonus_weight,
        hotspot_penalty_weight=args.hotspot_penalty_weight,
        hotspot_threshold=args.hotspot_threshold,
        min_active_load_share=args.min_active_load_share,
        capacity_backlog_mode=args.capacity_backlog_mode,
        backlog_penalty_weight=args.backlog_penalty_weight,
        reward_mode=args.reward_mode,
        iot_feature_dim=iot_feature_dim_for_args(args),
        iot_cstr_weight=args.iot_cstr_weight,
        iot_balance_weight=args.iot_balance_weight,
        iot_comm_cost_weight=args.iot_comm_cost_weight,
        iot_hotspot_weight=args.iot_hotspot_weight,
        candidate_top_k=args.candidate_top_k,
        capacity_guard=args.capacity_guard,
        capacity_guard_factor=args.capacity_guard_factor,
        candidate_load_weight=args.candidate_load_weight,
    )

    summary = new_summary(
        args.shards,
        args.max_block_size,
        args.block_interval_ms,
    )
    for batch_idx, batch in enumerate(iter_batches(txs, args.tx_batch_size), start=1):
        _actions, metrics = env.run_batch(batch, policy)
        update_summary(summary, metrics)

        if args.log_interval_batches > 0 and batch_idx % args.log_interval_batches == 0:
            view = summary_view(summary)
            print(
                "[EVAL] "
                f"batch={batch_idx} tx={view['tx_count']} "
                f"cross={view['cross_ratio']:.4f} "
                f"relCross={view['relation_cross_ratio']:.4f} "
                f"normVar={view['norm_var_mean']:.4f} "
                f"hotspot={view['max_load_share_mean']:.4f} "
                f"ownerHot={view['owner_max_load_share']:.4f} "
                f"stageHot={view['stage_max_load_share']:.4f} "
                f"ownerCap={view['owner_anchor_tps_ceiling']:.2f} "
                f"stageCap={view['system_stage_tps_ceiling']:.2f} "
                f"active={view['active_shards_mean']:.2f} "
                f"backlog={view['backlog_penalty_mean']:.4f} "
                f"conf={view['confidence_mean']:.4f} "
                f"polEnt={view['entropy_mean']:.4f} "
                f"same_related={view['same_as_related_ratio']:.4f} "
                f"relMin={view['min_related_follow_ratio']:.4f}"
            )

    result = {
        "policy": args.policy,
        "sample": bool(args.sample),
        "model": str(args.model) if args.policy == "ppo" else "",
        "csv": str(csv_path),
        "sidecar": args.sidecar,
        "dataset_start_tx": args.start_tx,
        "dataset_end_tx_exclusive": args.start_tx + len(txs),
        "dataset_window_state": "window_local_cold_start",
        "mdp_mode": args.mdp_mode,
        "tx_identity": args.tx_identity,
        "loaded_txs": len(txs),
        "shards": args.shards,
        "state_dim": eval_state_dim(args),
        "iot_feature_dim": iot_feature_dim_for_args(args),
        "tx_batch_size": args.tx_batch_size,
        "max_block_size": args.max_block_size,
        "block_interval_ms": args.block_interval_ms,
        "load_semantics_version": LOAD_SEMANTICS_VERSION,
        "sender_pos_mode": args.sender_pos_mode,
        "reward_mode": args.reward_mode,
        "lambda_weight": args.lambda_weight,
        "beta": args.beta,
        "load_penalty_weight": args.load_penalty_weight,
        "local_reward_weight": args.local_reward_weight,
        "action_reward_scale": args.action_reward_scale,
        "active_shard_bonus_weight": args.active_shard_bonus_weight,
        "hotspot_penalty_weight": args.hotspot_penalty_weight,
        "hotspot_threshold": args.hotspot_threshold,
        "min_active_load_share": args.min_active_load_share,
        "capacity_backlog_mode": args.capacity_backlog_mode,
        "backlog_penalty_weight": args.backlog_penalty_weight,
        "iot_cstr_weight": args.iot_cstr_weight,
        "iot_balance_weight": args.iot_balance_weight,
        "iot_comm_cost_weight": args.iot_comm_cost_weight,
        "iot_hotspot_weight": args.iot_hotspot_weight,
        "candidate_top_k": args.candidate_top_k,
        "capacity_guard": args.capacity_guard,
        "capacity_guard_factor": args.capacity_guard_factor,
        "candidate_load_weight": args.candidate_load_weight,
        "argmax_tie_break": args.argmax_tie_break,
        "argmax_tie_eps": args.argmax_tie_eps,
        "summary": summary_view(summary),
    }
    write_jsonl(args.log_jsonl, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--sidecar", type=str, default="")
    parser.add_argument("--mdp_mode", choices=["spring", "iot"], default="iot")
    parser.add_argument("--tx_identity", choices=["auto", "raw", "iot"], default="auto")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--policy", choices=["ppo", "heuristic", "hash", "random"], default="ppo")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--start_tx", type=int, default=DEFAULT_TEST_START_TX)
    parser.add_argument("--max_txs", type=int, default=DEFAULT_TEST_MAX_TXS)
    parser.add_argument("--tx_batch_size", type=int, default=DEFAULT_TX_BATCH_SIZE)
    parser.add_argument("--max_block_size", type=int, default=1000)
    parser.add_argument(
        "--block_interval_ms",
        type=int,
        default=DEFAULT_BLOCK_INTERVAL_MS,
    )
    parser.add_argument("--sender_pos_mode", type=int, default=DEFAULT_SENDER_POS_MODE)
    parser.add_argument("--temporal_top_k", type=int, default=8)
    parser.add_argument(
        "--reward_mode",
        choices=list(REWARD_MODE_CHOICES),
        default=REWARD_MODE,
    )
    parser.add_argument("--lambda_weight", type=float, default=LAMBDA_WEIGHT)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--load_penalty_weight", type=float, default=LOAD_PENALTY_WEIGHT)
    parser.add_argument("--local_reward_weight", type=float, default=LOCAL_REWARD_WEIGHT)
    parser.add_argument("--action_reward_scale", type=float, default=ACTION_REWARD_SCALE)
    parser.add_argument(
        "--active_shard_bonus_weight",
        type=float,
        default=ACTIVE_SHARD_BONUS_WEIGHT,
    )
    parser.add_argument(
        "--hotspot_penalty_weight",
        type=float,
        default=HOTSPOT_PENALTY_WEIGHT,
    )
    parser.add_argument("--hotspot_threshold", type=float, default=HOTSPOT_THRESHOLD)
    parser.add_argument("--min_active_load_share", type=float, default=MIN_ACTIVE_LOAD_SHARE)
    parser.add_argument("--capacity_backlog_mode", type=int, default=CAPACITY_BACKLOG_MODE)
    parser.add_argument("--backlog_penalty_weight", type=float, default=BACKLOG_PENALTY_WEIGHT)
    parser.add_argument("--iot_cstr_weight", type=float, default=IOT_CSTR_WEIGHT)
    parser.add_argument("--iot_balance_weight", type=float, default=IOT_BALANCE_WEIGHT)
    parser.add_argument("--iot_comm_cost_weight", type=float, default=IOT_COMM_COST_WEIGHT)
    parser.add_argument("--iot_hotspot_weight", type=float, default=IOT_HOTSPOT_WEIGHT)
    parser.add_argument("--candidate_top_k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--capacity_guard", type=int, default=DEFAULT_CAPACITY_GUARD)
    parser.add_argument("--capacity_guard_factor", type=float, default=DEFAULT_CAPACITY_GUARD_FACTOR)
    parser.add_argument("--candidate_load_weight", type=float, default=1.0)
    parser.add_argument("--argmax_tie_break", type=int, default=ARGMAX_TIE_BREAK)
    parser.add_argument("--argmax_tie_eps", type=float, default=ARGMAX_TIE_EPS)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow_checkpoint_config_mismatch",
        action="store_true",
        help="diagnostic only: allow a checkpoint whose saved experiment metadata differs",
    )
    parser.add_argument("--log_interval_batches", type=int, default=0)
    parser.add_argument("--log_jsonl", type=str, default=str(DEFAULT_EVAL_LOG_JSONL))

    args = parser.parse_args()
    normalize_reward_mode(args)
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
