import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from config import (
    ACTION_REWARD_SCALE,
    ACTIVE_SHARD_BONUS_WEIGHT,
    ARGMAX_BALANCE_COEF,
    ARGMAX_BALANCE_TEMPERATURE,
    ARGMAX_TIE_BREAK,
    ARGMAX_TIE_EPS,
    BACKLOG_PENALTY_WEIGHT,
    BATCH_SIZE,
    BETA,
    CAPACITY_BACKLOG_MODE,
    CHECKPOINT_DIR,
    CLIP_EPS,
    DEFAULT_CSV_PATH,
    DEFAULT_IOT_CSV_PATH,
    DEFAULT_IOT_SIDECAR_PATH,
    DEFAULT_SENDER_POS_MODE,
    DEFAULT_SHARD_NUM,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    HIDDEN_DIM,
    HOTSPOT_PENALTY_WEIGHT,
    HOTSPOT_THRESHOLD,
    IOT_BALANCE_WEIGHT,
    IOT_COMM_COST_WEIGHT,
    IOT_CSTR_WEIGHT,
    IOT_FEATURE_DIM,
    IOT_HOTSPOT_WEIGHT,
    LAMBDA_WEIGHT,
    LEARNING_RATE,
    LOAD_PENALTY_WEIGHT,
    LOCAL_REWARD_WEIGHT,
    MIN_ACTIVE_LOAD_SHARE,
    MODEL_PATH,
    PPO_EPOCHS,
    REWARD_MODE,
    REWARD_MODE_CHOICES,
    SUPERVISED_COEF,
    VALUE_CLIP,
    state_dim,
)
from action_mask import mask_logits, normalize_action_mask
from action_select import distribution_floor_shortfall, tie_aware_argmax
from offline_env import (
    BatchMetrics,
    PlacementAction,
    PolicyOutput,
    SenderPosInfo,
    SpringOfflineEnv,
    iter_batches,
    load_iot_transactions,
    load_transactions,
)
from ppo import PPOAgent, RolloutBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_iot_mdp(args: argparse.Namespace) -> bool:
    return str(getattr(args, "mdp_mode", "spring")).strip().lower() == "iot"


def iot_feature_dim_for_args(args: argparse.Namespace) -> int:
    return IOT_FEATURE_DIM if is_iot_mdp(args) else 0


def agent_state_dim(args: argparse.Namespace, shards: Optional[int] = None) -> int:
    shard_count = int(shards if shards is not None else args.shards)
    return state_dim(shard_count, iot_feature_dim_for_args(args))


def resolve_csv_path(args: argparse.Namespace) -> Path:
    csv_arg = str(getattr(args, "csv", "")).strip()
    if csv_arg:
        return Path(csv_arg)
    return DEFAULT_IOT_CSV_PATH if is_iot_mdp(args) else DEFAULT_CSV_PATH


def resolve_iot_sidecar_path(args: argparse.Namespace) -> Path:
    sidecar = str(getattr(args, "sidecar", "")).strip()
    return Path(sidecar) if sidecar else DEFAULT_IOT_SIDECAR_PATH


def load_training_transactions(args: argparse.Namespace, csv_path: Path):
    if is_iot_mdp(args):
        sidecar_path = resolve_iot_sidecar_path(args)
        if not sidecar_path.exists():
            raise FileNotFoundError(f"IoT sidecar not found: {sidecar_path}")
        args.sidecar = str(sidecar_path)
        return load_iot_transactions(
            csv_path,
            sidecar_path,
            max_txs=args.max_txs,
        )
    return load_transactions(csv_path, max_txs=args.max_txs)


def make_agent(args: argparse.Namespace) -> PPOAgent:
    agent = PPOAgent(
        state_dim=agent_state_dim(args),
        action_dim=args.shards,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip,
        ppo_epochs=args.ppo_epochs,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_clip=args.value_clip,
        supervised_coef=args.supervised_coef,
        argmax_balance_coef=args.argmax_balance_coef,
        argmax_balance_temperature=args.argmax_balance_temperature,
        device=args.device,
    )
    agent.argmax_tie_eps = float(args.argmax_tie_eps)
    agent.argmax_tie_break = int(args.argmax_tie_break)

    model_path = Path(args.model)
    if args.resume and model_path.exists():
        payload = agent.load(model_path)
        ckpt_state_dim = int(payload.get("state_dim", -1))
        ckpt_action_dim = int(payload.get("action_dim", -1))
        if ckpt_state_dim != agent_state_dim(args) or ckpt_action_dim != args.shards:
            raise ValueError(
                "checkpoint dimension mismatch: "
                f"state_dim={ckpt_state_dim}, action_dim={ckpt_action_dim}"
            )
        print(f"[RESUME] loaded checkpoint: {model_path}")

    return agent


def sample_agent_policy(
    agent: PPOAgent,
    state: List[float],
    _address: str,
    _related: str,
    _info: SenderPosInfo,
) -> PolicyOutput:
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.net(state_t)
        masks = [normalize_action_mask(_info.action_mask, agent.action_dim)]
        masked = mask_logits(logits, masks)
        dist = Categorical(logits=masked)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        probs = torch.softmax(masked, dim=-1)
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


def deterministic_agent_policy(
    agent: PPOAgent,
    state: List[float],
    address: str,
    related: str,
    _info: SenderPosInfo,
) -> PolicyOutput:
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.net(state_t)
        masks = [normalize_action_mask(_info.action_mask, agent.action_dim)]
        masked = mask_logits(logits, masks)
        dist = Categorical(logits=masked)
        probs = torch.softmax(masked, dim=-1)
        action_id = tie_aware_argmax(
            probs.squeeze(0).detach().cpu().tolist(),
            key=f"{address}|{related}",
            tie_eps=args_tie_eps(agent, fallback=ARGMAX_TIE_EPS),
            enable_tie_break=args_tie_break(agent, fallback=ARGMAX_TIE_BREAK),
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
        source="python_ppo_argmax",
    )


def args_tie_eps(agent: PPOAgent, fallback: float) -> float:
    return float(getattr(agent, "argmax_tie_eps", fallback))


def args_tie_break(agent: PPOAgent, fallback: int) -> bool:
    return bool(int(getattr(agent, "argmax_tie_break", fallback)))


def add_actions_to_buffer(buffer: RolloutBuffer, actions: List[PlacementAction]) -> None:
    for idx, action in enumerate(actions):
        if not str(action.source).startswith("python_ppo"):
            continue
        if idx + 1 < len(actions):
            next_state = actions[idx + 1].state
        else:
            next_state = action.next_state

        target_action = -1
        target_weight = 0.0
        if action.related_known and 0 <= action.related_shard:
            target_action = action.related_shard
            target_weight = max(0.05, min(1.0, float(action.related_weight)))

        buffer.add(
            state=action.state,
            next_state=next_state,
            action=action.action,
            log_prob=action.log_prob,
            reward=action.reward,
            done=action.done,
            value=action.value,
            target_action=target_action,
            target_weight=target_weight,
            action_mask=action.action_mask,
        )


def new_summary(shards: int) -> Dict[str, object]:
    return {
        "batches": 0,
        "tx_count": 0,
        "action_count": 0,
        "effective_tx": 0.0,
        "cross_tx": 0.0,
        "reward_sum": 0.0,
        "norm_var_sum": 0.0,
        "r_wlb_sum": 0.0,
        "active_shards_sum": 0.0,
        "active_shard_ratio_sum": 0.0,
        "max_load_share_sum": 0.0,
        "hotspot_penalty_sum": 0.0,
        "load_aware_bonus_sum": 0.0,
        "backlog_penalty_sum": 0.0,
        "communication_cost_sum": 0.0,
        "related_known_count": 0,
        "same_as_related_count": 0,
        "related_shard_hist": [0 for _ in range(shards)],
        "same_related_by_shard": [0 for _ in range(shards)],
        "local_reward_sum": 0.0,
        "action_reward_sum": 0.0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "action_hist": [0 for _ in range(shards)],
    }


def update_summary(summary: Dict[str, object], metrics: BatchMetrics) -> None:
    summary["batches"] = int(summary["batches"]) + 1
    summary["tx_count"] = int(summary["tx_count"]) + metrics.tx_count
    summary["action_count"] = int(summary["action_count"]) + metrics.action_count
    summary["effective_tx"] = float(summary["effective_tx"]) + metrics.effective_tx
    summary["cross_tx"] = float(summary["cross_tx"]) + metrics.cross_tx
    summary["reward_sum"] = float(summary["reward_sum"]) + metrics.reward
    summary["norm_var_sum"] = float(summary["norm_var_sum"]) + metrics.normalized_load_variance
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
    summary["local_reward_sum"] = float(summary["local_reward_sum"]) + (
        metrics.local_reward_mean * metrics.action_count
    )
    summary["action_reward_sum"] = float(summary["action_reward_sum"]) + (
        metrics.action_reward_mean * metrics.action_count
    )
    summary["confidence_sum"] = float(summary["confidence_sum"]) + (
        metrics.confidence_mean * metrics.action_count
    )
    summary["entropy_sum"] = float(summary["entropy_sum"]) + (
        metrics.entropy_mean * metrics.action_count
    )

    hist = summary["action_hist"]
    assert isinstance(hist, list)
    for sid, count in enumerate(metrics.action_hist):
        hist[sid] += count


def summary_view(summary: Dict[str, object]) -> Dict[str, object]:
    batches = max(1, int(summary["batches"]))
    actions = max(1, int(summary["action_count"]))
    effective = float(summary["effective_tx"])
    related_known = int(summary["related_known_count"])

    hist = list(summary["action_hist"])
    hist_total = sum(hist)
    action_dist = [count / hist_total if hist_total else 0.0 for count in hist]
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

    return {
        "batches": int(summary["batches"]),
        "tx_count": int(summary["tx_count"]),
        "action_count": int(summary["action_count"]),
        "cross_ratio": float(summary["cross_tx"]) / effective if effective > 0 else 0.0,
        "reward_mean": float(summary["reward_sum"]) / batches,
        "norm_var_mean": float(summary["norm_var_sum"]) / batches,
        "r_wlb_mean": float(summary["r_wlb_sum"]) / batches,
        "active_shards_mean": float(summary["active_shards_sum"]) / batches,
        "active_shard_ratio_mean": float(summary["active_shard_ratio_sum"]) / batches,
        "max_load_share_mean": float(summary["max_load_share_sum"]) / batches,
        "hotspot_penalty_mean": float(summary["hotspot_penalty_sum"]) / batches,
        "load_aware_bonus_mean": float(summary["load_aware_bonus_sum"]) / batches,
        "backlog_penalty_mean": float(summary["backlog_penalty_sum"]) / batches,
        "communication_cost_mean": float(summary["communication_cost_sum"]) / batches,
        "same_as_related_ratio": (
            int(summary["same_as_related_count"]) / related_known if related_known else 0.0
        ),
        "related_known_count": related_known,
        "same_as_related_count": int(summary["same_as_related_count"]),
        "local_reward_mean": float(summary["local_reward_sum"]) / actions,
        "action_reward_mean": float(summary["action_reward_sum"]) / actions,
        "confidence_mean": float(summary["confidence_sum"]) / actions,
        "entropy_mean": float(summary["entropy_sum"]) / actions,
        "action_hist": hist,
        "action_dist": action_dist,
        "related_shard_hist": related_shard_hist,
        "same_related_by_shard": same_related_by_shard,
        "related_follow_by_shard": related_follow_by_shard,
        "min_related_follow_ratio": min_related_follow_ratio,
    }


def write_jsonl(path: str, record: Dict[str, object]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_env(args: argparse.Namespace) -> SpringOfflineEnv:
    return SpringOfflineEnv(
        shards=args.shards,
        tx_batch_size=args.tx_batch_size,
        max_block_size=args.max_block_size,
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


def evaluate_agent_argmax(
    agent: PPOAgent,
    args: argparse.Namespace,
    txs: Sequence,
) -> Dict[str, object]:
    env = make_env(args)
    summary = new_summary(args.shards)
    was_training = agent.net.training
    agent.net.eval()

    for batch in iter_batches(txs, args.tx_batch_size):
        _actions, metrics = env.run_batch(
            batch,
            lambda state, address, related, info: deterministic_agent_policy(
                agent, state, address, related, info
            ),
        )
        update_summary(summary, metrics)

    if was_training:
        agent.net.train()

    return summary_view(summary)


def validation_score(
    view: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float]]:
    action_dist = [float(x) for x in view.get("action_dist", [])]
    underuse = distribution_floor_shortfall(action_dist, args.best_min_action_share)
    active_shortfall = max(
        0.0,
        float(args.best_min_active_shards) - float(view["active_shards_mean"]),
    ) / float(max(1, args.shards))
    hotspot_excess = max(
        0.0,
        float(view["max_load_share_mean"]) - float(args.best_max_hotspot),
    )
    cross_excess = max(
        0.0,
        float(view["cross_ratio"]) - float(args.best_max_cross),
    )
    related_follow_shortfall = max(
        0.0,
        float(args.best_min_related_follow)
        - float(view.get("min_related_follow_ratio", 0.0)),
    )
    same_related_shortfall = max(
        0.0,
        float(args.best_min_same_related)
        - float(view.get("same_as_related_ratio", 0.0)),
    )

    score = float(view["reward_mean"])
    score -= float(args.best_hotspot_penalty) * hotspot_excess
    score -= float(args.best_active_penalty) * active_shortfall
    score -= float(args.best_action_floor_penalty) * underuse
    score -= float(args.best_cross_penalty) * cross_excess
    score -= float(args.best_related_follow_penalty) * related_follow_shortfall
    score -= float(args.best_same_related_penalty) * same_related_shortfall

    return score, {
        "score": score,
        "underuse": underuse,
        "active_shortfall": active_shortfall,
        "hotspot_excess": hotspot_excess,
        "cross_excess": cross_excess,
        "related_follow_shortfall": related_follow_shortfall,
        "same_related_shortfall": same_related_shortfall,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    csv_path = resolve_csv_path(args)
    args.csv = str(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    txs = load_training_transactions(args, csv_path)
    if not txs:
        raise RuntimeError(f"no valid transactions loaded from {csv_path}")

    print(
        "[DATA] "
        f"txs={len(txs)} csv={csv_path} shards={args.shards} "
        f"tx_batch_size={args.tx_batch_size} sender_pos_mode={args.sender_pos_mode} "
        f"mdp_mode={args.mdp_mode} reward_mode={args.reward_mode} "
        f"entropy_coef={args.entropy_coef} "
        f"supervised_coef={args.supervised_coef}"
    )

    agent = make_agent(args)
    buffer = RolloutBuffer()
    env = make_env(args)

    update_count = 0
    start_time = time.time()
    overall = new_summary(args.shards)
    validation_txs = txs[: args.eval_max_txs] if args.eval_max_txs > 0 else txs
    last_model_path = (
        Path(args.last_model)
        if args.last_model
        else model_path.with_name(f"{model_path.stem}_last{model_path.suffix}")
    )
    best_score: Optional[float] = None
    best_record: Optional[Dict[str, object]] = None

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or not args.keep_placement:
            env.reset()

        epoch_summary = new_summary(args.shards)
        print(f"\n[TRAIN] epoch={epoch}/{args.epochs}")

        for batch_idx, batch in enumerate(iter_batches(txs, args.tx_batch_size), start=1):
            actions, metrics = env.run_batch(
                batch,
                lambda state, address, related, info: sample_agent_policy(
                    agent, state, address, related, info
                ),
            )
            add_actions_to_buffer(buffer, actions)
            update_summary(epoch_summary, metrics)
            update_summary(overall, metrics)

            if len(buffer) >= args.batch_size:
                loss_info = agent.update(buffer)
                buffer.clear()
                update_count += 1

                record = {
                    "event": "update",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "update_count": update_count,
                    "elapsed_sec": time.time() - start_time,
                    "loss_info": loss_info,
                    "epoch_summary": summary_view(epoch_summary),
                }
                write_jsonl(args.log_jsonl, record)
                update_view = summary_view(epoch_summary)

                print(
                    "[UPDATE] "
                    f"epoch={epoch} batch={batch_idx} updates={update_count} "
                    f"samples={loss_info.get('num_samples', 0)} "
                    f"loss={loss_info.get('loss', 0.0):.6f} "
                    f"pi={loss_info.get('policy_loss', 0.0):.6f} "
                    f"entropy={loss_info.get('entropy', 0.0):.4f} "
                    f"sup={loss_info.get('supervised_loss', 0.0):.4f} "
                    f"supW={loss_info.get('supervised_loss_weighted', 0.0):.4f} "
                    f"supN={loss_info.get('supervised_target_count', 0)} "
                    f"kl={loss_info.get('approx_kl', 0.0):.6f} "
                    f"argBal={loss_info.get('argmax_balance_loss', 0.0):.6f} "
                    f"cross={update_view['cross_ratio']:.4f} "
                    f"hotspot={update_view['max_load_share_mean']:.4f} "
                    f"active={update_view['active_shards_mean']:.2f} "
                    f"backlog={update_view['backlog_penalty_mean']:.4f} "
                    f"same_related={update_view['same_as_related_ratio']:.4f} "
                    f"relMin={update_view['min_related_follow_ratio']:.4f}"
                )

                if args.save_every_updates > 0 and update_count % args.save_every_updates == 0:
                    periodic_path = model_path if args.disable_best_checkpoint else last_model_path
                    agent.save(
                        periodic_path,
                        extra=build_extra(
                            args,
                            len(txs),
                            update_count,
                            summary_view(overall),
                            checkpoint_kind="periodic",
                            best_record=best_record,
                        ),
                    )

            if args.log_interval_batches > 0 and batch_idx % args.log_interval_batches == 0:
                view = summary_view(epoch_summary)
                print(
                    "[PROGRESS] "
                    f"epoch={epoch} batch={batch_idx} tx={view['tx_count']} "
                    f"actions={view['action_count']} cross={view['cross_ratio']:.4f} "
                    f"normVar={view['norm_var_mean']:.4f} "
                    f"hotspot={view['max_load_share_mean']:.4f} "
                    f"active={view['active_shards_mean']:.2f} "
                    f"backlog={view['backlog_penalty_mean']:.4f} "
                    f"conf={view['confidence_mean']:.4f} "
                    f"polEnt={view['entropy_mean']:.4f} "
                    f"same_related={view['same_as_related_ratio']:.4f} "
                    f"relMin={view['min_related_follow_ratio']:.4f} "
                    f"action_dist={[round(x, 3) for x in view['action_dist']]}"
                )

        epoch_view = summary_view(epoch_summary)
        write_jsonl(
            args.log_jsonl,
            {
                "event": "epoch_done",
                "epoch": epoch,
                "elapsed_sec": time.time() - start_time,
                "summary": epoch_view,
            },
        )
        print(
            "[EPOCH DONE] "
            f"epoch={epoch} tx={epoch_view['tx_count']} actions={epoch_view['action_count']} "
            f"cross={epoch_view['cross_ratio']:.4f} "
            f"normVar={epoch_view['norm_var_mean']:.4f} "
            f"hotspot={epoch_view['max_load_share_mean']:.4f} "
                f"active={epoch_view['active_shards_mean']:.2f} "
                f"backlog={epoch_view['backlog_penalty_mean']:.4f} "
                f"conf={epoch_view['confidence_mean']:.4f} "
                f"polEnt={epoch_view['entropy_mean']:.4f} "
                f"reward={epoch_view['reward_mean']:.4f} "
            f"same_related={epoch_view['same_as_related_ratio']:.4f} "
            f"relMin={epoch_view['min_related_follow_ratio']:.4f}"
        )

        if (
            not args.disable_best_checkpoint
            and args.eval_every_epochs > 0
            and epoch % args.eval_every_epochs == 0
        ):
            validation_view = evaluate_agent_argmax(agent, args, validation_txs)
            score, score_parts = validation_score(validation_view, args)
            validation_record = {
                "event": "validation",
                "epoch": epoch,
                "update_count": update_count,
                "elapsed_sec": time.time() - start_time,
                "score": score,
                "score_parts": score_parts,
                "summary": validation_view,
            }
            write_jsonl(args.log_jsonl, validation_record)
            print(
                "[VALID] "
                f"epoch={epoch} score={score:.6f} "
                f"cross={validation_view['cross_ratio']:.4f} "
                f"normVar={validation_view['norm_var_mean']:.4f} "
                f"hotspot={validation_view['max_load_share_mean']:.4f} "
                f"active={validation_view['active_shards_mean']:.2f} "
                f"backlog={validation_view['backlog_penalty_mean']:.4f} "
                f"conf={validation_view['confidence_mean']:.4f} "
                f"polEnt={validation_view['entropy_mean']:.4f} "
                f"same_related={validation_view['same_as_related_ratio']:.4f} "
                f"relMin={validation_view['min_related_follow_ratio']:.4f} "
                f"action_dist={[round(x, 3) for x in validation_view['action_dist']]}"
            )

            if best_score is None or score > best_score:
                best_score = score
                best_record = validation_record
                agent.save(
                    model_path,
                    extra=build_extra(
                        args,
                        len(txs),
                        update_count,
                        summary_view(overall),
                        validation_summary=validation_view,
                        validation_score_value=score,
                        validation_score_parts=score_parts,
                        checkpoint_kind="best",
                        best_record=best_record,
                    ),
                )
                print(f"[BEST] epoch={epoch} score={score:.6f} saved={model_path}")

    if len(buffer) > 0:
        loss_info = agent.update(buffer)
        buffer.clear()
        update_count += 1
        print(f"[FINAL UPDATE] updates={update_count} loss={loss_info}")

    overall_view = summary_view(overall)

    if args.disable_best_checkpoint:
        agent.save(
            model_path,
            extra=build_extra(
                args,
                len(txs),
                update_count,
                overall_view,
                checkpoint_kind="final",
            ),
        )
        final_model_path = model_path
    else:
        final_validation_view = evaluate_agent_argmax(agent, args, validation_txs)
        final_score, final_score_parts = validation_score(final_validation_view, args)
        final_record = {
            "event": "final_validation",
            "update_count": update_count,
            "elapsed_sec": time.time() - start_time,
            "score": final_score,
            "score_parts": final_score_parts,
            "summary": final_validation_view,
        }
        write_jsonl(args.log_jsonl, final_record)

        agent.save(
            last_model_path,
            extra=build_extra(
                args,
                len(txs),
                update_count,
                overall_view,
                validation_summary=final_validation_view,
                validation_score_value=final_score,
                validation_score_parts=final_score_parts,
                checkpoint_kind="last",
                best_record=best_record,
            ),
        )
        print(f"[LAST] score={final_score:.6f} saved={last_model_path}")

        if best_score is None or final_score > best_score:
            best_score = final_score
            best_record = final_record
            agent.save(
                model_path,
                extra=build_extra(
                    args,
                    len(txs),
                    update_count,
                    overall_view,
                    validation_summary=final_validation_view,
                    validation_score_value=final_score,
                    validation_score_parts=final_score_parts,
                    checkpoint_kind="best",
                    best_record=best_record,
                ),
            )
            print(f"[BEST] final score={final_score:.6f} saved={model_path}")

        final_model_path = model_path

    print("\n[DONE]")
    print(f"model saved to: {final_model_path}")
    if not args.disable_best_checkpoint:
        print(f"last model saved to: {last_model_path}")
        if best_score is not None:
            print(f"best validation score: {best_score:.6f}")
    print(json.dumps(overall_view, ensure_ascii=False, indent=2))


def build_extra(
    args: argparse.Namespace,
    loaded_txs: int,
    update_count: int,
    summary: Dict[str, object],
    validation_summary: Optional[Dict[str, object]] = None,
    validation_score_value: Optional[float] = None,
    validation_score_parts: Optional[Dict[str, float]] = None,
    checkpoint_kind: str = "unknown",
    best_record: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "model_source": "train_offline",
        "checkpoint_kind": checkpoint_kind,
        "loaded_txs": loaded_txs,
        "offline_update_count": update_count,
        "mdp_mode": args.mdp_mode,
        "sidecar": args.sidecar,
        "shards": args.shards,
        "state_dim": agent_state_dim(args),
        "iot_feature_dim": iot_feature_dim_for_args(args),
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "clip": args.clip,
        "ppo_epochs": args.ppo_epochs,
        "gae_lambda": args.gae_lambda,
        "entropy_coef": args.entropy_coef,
        "value_clip": args.value_clip,
        "supervised_coef": args.supervised_coef,
        "batch_size": args.batch_size,
        "tx_batch_size": args.tx_batch_size,
        "max_block_size": args.max_block_size,
        "sender_pos_mode": args.sender_pos_mode,
        "temporal_top_k": args.temporal_top_k,
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
        "argmax_balance_coef": args.argmax_balance_coef,
        "argmax_balance_temperature": args.argmax_balance_temperature,
        "eval_every_epochs": args.eval_every_epochs,
        "eval_max_txs": args.eval_max_txs,
        "best_min_active_shards": args.best_min_active_shards,
        "best_max_hotspot": args.best_max_hotspot,
        "best_max_cross": args.best_max_cross,
        "best_min_action_share": args.best_min_action_share,
        "best_min_related_follow": args.best_min_related_follow,
        "best_min_same_related": args.best_min_same_related,
        "best_hotspot_penalty": args.best_hotspot_penalty,
        "best_active_penalty": args.best_active_penalty,
        "best_action_floor_penalty": args.best_action_floor_penalty,
        "best_cross_penalty": args.best_cross_penalty,
        "best_related_follow_penalty": args.best_related_follow_penalty,
        "best_same_related_penalty": args.best_same_related_penalty,
        "epochs": args.epochs,
        "seed": args.seed,
        "summary": summary,
        "validation_summary": validation_summary,
        "validation_score": validation_score_value,
        "validation_score_parts": validation_score_parts,
        "best_record": best_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--sidecar", type=str, default="")
    parser.add_argument("--mdp_mode", choices=["spring", "iot"], default="spring")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_NUM)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_txs", type=int, default=300000)
    parser.add_argument("--tx_batch_size", type=int, default=1000)
    parser.add_argument("--max_block_size", type=int, default=1000)
    parser.add_argument("--sender_pos_mode", type=int, default=DEFAULT_SENDER_POS_MODE)
    parser.add_argument("--temporal_top_k", type=int, default=8)
    parser.add_argument("--keep_placement", action="store_true")
    parser.add_argument(
        "--reward_mode",
        choices=list(REWARD_MODE_CHOICES),
        default=REWARD_MODE,
    )

    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--clip", type=float, default=CLIP_EPS)
    parser.add_argument("--ppo_epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--gae_lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--value_clip", type=float, default=VALUE_CLIP)
    parser.add_argument("--supervised_coef", type=float, default=SUPERVISED_COEF)
    parser.add_argument("--argmax_balance_coef", type=float, default=ARGMAX_BALANCE_COEF)
    parser.add_argument(
        "--argmax_balance_temperature",
        type=float,
        default=ARGMAX_BALANCE_TEMPERATURE,
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
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
    parser.add_argument("--candidate_top_k", type=int, default=0)
    parser.add_argument("--capacity_guard", type=int, default=0)
    parser.add_argument("--capacity_guard_factor", type=float, default=1.2)
    parser.add_argument("--candidate_load_weight", type=float, default=1.0)
    parser.add_argument("--argmax_tie_break", type=int, default=ARGMAX_TIE_BREAK)
    parser.add_argument("--argmax_tie_eps", type=float, default=ARGMAX_TIE_EPS)

    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--eval_max_txs", type=int, default=300000)
    parser.add_argument("--disable_best_checkpoint", action="store_true")
    parser.add_argument("--last_model", type=str, default="")
    parser.add_argument("--best_min_active_shards", type=float, default=3.6)
    parser.add_argument("--best_max_hotspot", type=float, default=0.50)
    parser.add_argument("--best_max_cross", type=float, default=0.35)
    parser.add_argument("--best_min_action_share", type=float, default=0.05)
    parser.add_argument("--best_min_related_follow", type=float, default=0.30)
    parser.add_argument("--best_min_same_related", type=float, default=0.40)
    parser.add_argument("--best_hotspot_penalty", type=float, default=2.0)
    parser.add_argument("--best_active_penalty", type=float, default=1.0)
    parser.add_argument("--best_action_floor_penalty", type=float, default=1.0)
    parser.add_argument("--best_cross_penalty", type=float, default=0.5)
    parser.add_argument("--best_related_follow_penalty", type=float, default=0.5)
    parser.add_argument("--best_same_related_penalty", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_every_updates", type=int, default=0)
    parser.add_argument("--log_interval_batches", type=int, default=50)
    parser.add_argument("--log_jsonl", type=str, default="")

    args = parser.parse_args()
    if args.mdp_mode == "iot" and args.reward_mode == REWARD_MODE:
        args.reward_mode = "iot"
    train(args)


if __name__ == "__main__":
    main()
