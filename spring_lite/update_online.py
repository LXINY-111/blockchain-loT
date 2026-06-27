import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLIP_EPS,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    HIDDEN_DIM,
    LEARNING_RATE,
    MIN_FLUSH_BATCH_SIZE,
    MODEL_PATH,
    PPO_EPOCHS,
    REWARD_CLIP,
    SUPERVISED_COEF,
    VALUE_CLIP,
    state_dim,
)
from ppo import PPOAgent, RolloutBuffer


ROLLOUT_BUFFER_PATH = CHECKPOINT_DIR / "online_rollout_buffer.json"


def now_ns() -> int:
    return time.time_ns()


def safe_float(x: Any, default: Any = 0.0) -> Any:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def update_iot_feature_dim(data: Dict[str, Any]) -> int:
    raw_value = data.get("iot_feature_dim", data.get("IOTFeatureDim", 0))
    return max(0, safe_int(raw_value, 0))


def update_state_dim(shards: int, iot_feature_dim: int = 0) -> int:
    return state_dim(shards, max(0, int(iot_feature_dim)))


def clip_value(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-limit, min(limit, value))


def load_update_input(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("online update input must be a JSON object")

    return data


def build_agent(
    shards: int,
    model_path: Path,
    iot_feature_dim: int = 0,
) -> Tuple[PPOAgent, Dict[str, Any], str]:
    expected_state_dim = update_state_dim(shards, iot_feature_dim)

    agent = PPOAgent(
        state_dim=expected_state_dim,
        action_dim=shards,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        clip_eps=CLIP_EPS,
        ppo_epochs=PPO_EPOCHS,
        gae_lambda=GAE_LAMBDA,
        entropy_coef=ENTROPY_COEF,
        value_clip=VALUE_CLIP,
        supervised_coef=SUPERVISED_COEF,
        device="cpu",
    )

    old_payload: Dict[str, Any] = {}
    source = "new_model"

    if model_path.exists():
        try:
            payload = agent.load(model_path)
            ckpt_state_dim = int(payload.get("state_dim", -1))
            ckpt_action_dim = int(payload.get("action_dim", -1))

            if ckpt_state_dim == expected_state_dim and ckpt_action_dim == shards:
                old_payload = payload
                source = "loaded_model"
            else:
                old_payload = {}
                source = "reset_dim_mismatch"

        except Exception as e:
            old_payload = {}
            source = f"reset_load_failed_{type(e).__name__}"

    return agent, old_payload, source


def build_buffer_from_update(data: Dict[str, Any]) -> Tuple[RolloutBuffer, Dict[str, Any]]:
    shards = safe_int(data.get("shards", 4), 4)
    iot_feature_dim = update_iot_feature_dim(data)
    expected_state_dim = update_state_dim(shards, iot_feature_dim)

    batch_reward = safe_float(data.get("reward", 0.0), 0.0)

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")

    next_states = data.get("next_states", [])
    if not isinstance(next_states, list):
        next_states = []

    buffer = RolloutBuffer()
    skipped = 0
    action_hist = [0 for _ in range(shards)]

    related_known_count = 0
    same_as_related_count = 0
    confidence_sum = 0.0
    entropy_sum = 0.0
    log_prob_sum = 0.0
    local_reward_sum = 0.0
    supervised_target_count = 0
    supervised_weight_sum = 0.0

    for idx, item in enumerate(actions):
        if not isinstance(item, dict):
            skipped += 1
            continue

        state = item.get("state", [])
        if not isinstance(state, list) or len(state) != expected_state_dim:
            skipped += 1
            continue

        action = safe_int(item.get("action", -1), -1)
        if action < 0 or action >= shards:
            skipped += 1
            continue

        log_prob = safe_float(item.get("log_prob", None), None)
        value = safe_float(item.get("value", None), None)

        # PPO 更新必须依赖旧策略 log_prob 和旧 value。
        if log_prob is None or value is None:
            skipped += 1
            continue

        clean_state = [safe_float(x, 0.0) for x in state]

        raw_next_state = item.get("next_state", None)
        if not isinstance(raw_next_state, list) and idx < len(next_states):
            raw_next_state = next_states[idx]

        if isinstance(raw_next_state, list) and len(raw_next_state) == expected_state_dim:
            clean_next_state = [safe_float(x, 0.0) for x in raw_next_state]
        else:
            clean_next_state = clean_state

        # 关键：
        # 这里不再把 top-level batch_reward 自动塞给每个 action。
        # Go 侧已经负责设置：
        #   前面 action.reward = 0
        #   最后 action.reward = block_reward
        #   最后 action.done = true
        item_reward = clip_value(
            safe_float(item.get("reward", 0.0), 0.0),
            REWARD_CLIP,
        )
        item_done = bool(item.get("done", False))
        item_confidence = safe_float(item.get("confidence", 0.0), 0.0)
        item_entropy = safe_float(item.get("entropy", 0.0), 0.0)
        item_local_reward = safe_float(item.get("local_reward", item_reward), item_reward)
        item_related_known = bool(item.get("related_known", False))
        item_related_shard = safe_int(item.get("related_shard", -1), -1)
        item_related_weight = safe_float(item.get("related_weight", 0.0), 0.0)

        target_action = -1
        target_weight = 0.0
        if item_related_known and 0 <= item_related_shard < shards:
            target_action = item_related_shard
            target_weight = max(0.05, min(1.0, item_related_weight))
            supervised_target_count += 1
            supervised_weight_sum += target_weight

        buffer.add(
            state=clean_state,
            action=action,
            log_prob=log_prob,
            reward=item_reward,
            done=item_done,
            value=value,
            next_state=clean_next_state,
            target_action=target_action,
            target_weight=target_weight,
            action_mask=item.get("action_mask", []),
        )

        action_hist[action] += 1
        confidence_sum += item_confidence
        entropy_sum += item_entropy
        log_prob_sum += log_prob
        local_reward_sum += item_local_reward

        if item_related_known:
            related_known_count += 1
            if bool(item.get("same_as_related", False)):
                same_as_related_count += 1

    # 防御：如果 Go 侧忘了设置 done=true，就把最后一个有效 action 设为轨迹结束。
    if len(buffer) > 0 and not any(buffer.dones):
        buffer.dones[-1] = True

    meta = {
        "shards": shards,
        "iot_feature_dim": iot_feature_dim,
        "state_dim": expected_state_dim,
        "batch_reward": batch_reward,
        "skipped": skipped,
        "action_hist": action_hist,
        "related_known_count": related_known_count,
        "same_as_related_count": same_as_related_count,
        "action_confidence_mean": confidence_sum / float(len(buffer)) if len(buffer) else 0.0,
        "action_entropy_mean": entropy_sum / float(len(buffer)) if len(buffer) else 0.0,
        "action_log_prob_mean": log_prob_sum / float(len(buffer)) if len(buffer) else 0.0,
        "local_reward_mean": local_reward_sum / float(len(buffer)) if len(buffer) else 0.0,
        "supervised_target_count": supervised_target_count,
        "supervised_weight_mean": (
            supervised_weight_sum / float(supervised_target_count)
            if supervised_target_count
            else 0.0
        ),
    }

    return buffer, meta


def buffer_to_records(buffer: RolloutBuffer) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for i in range(len(buffer)):
        records.append(
            {
                "state": buffer.states[i].tolist(),
                "next_state": buffer.next_states[i].tolist(),
                "action": int(buffer.actions[i]),
                "log_prob": float(buffer.log_probs[i]),
                "reward": float(buffer.rewards[i]),
                "done": bool(buffer.dones[i]),
                "value": float(buffer.values[i]),
                "target_action": int(buffer.target_actions[i]),
                "target_weight": float(buffer.target_weights[i]),
                "action_mask": list(buffer.action_masks[i]),
            }
        )

    return records


def records_to_buffer(
    records: List[Dict[str, Any]],
    shards: int,
    iot_feature_dim: int = 0,
) -> RolloutBuffer:
    expected_state_dim = update_state_dim(shards, iot_feature_dim)
    buffer = RolloutBuffer()

    for item in records:
        if not isinstance(item, dict):
            continue

        state = item.get("state", [])
        if not isinstance(state, list) or len(state) != expected_state_dim:
            continue
        next_state = item.get("next_state", state)
        if not isinstance(next_state, list) or len(next_state) != expected_state_dim:
            next_state = state

        action = safe_int(item.get("action", -1), -1)
        if action < 0 or action >= shards:
            continue

        log_prob = safe_float(item.get("log_prob", None), None)
        value = safe_float(item.get("value", None), None)

        if log_prob is None or value is None:
            continue

        buffer.add(
            state=[safe_float(x, 0.0) for x in state],
            next_state=[safe_float(x, 0.0) for x in next_state],
            action=action,
            log_prob=log_prob,
            reward=safe_float(item.get("reward", 0.0), 0.0),
            done=bool(item.get("done", False)),
            value=value,
            target_action=safe_int(item.get("target_action", -1), -1),
            target_weight=safe_float(item.get("target_weight", 0.0), 0.0),
            action_mask=item.get("action_mask", []),
        )

    return buffer


def load_pending_buffer(
    path: Path,
    shards: int,
    iot_feature_dim: int = 0,
) -> Tuple[RolloutBuffer, Dict[str, Any]]:
    if not path.exists():
        return RolloutBuffer(), {"status": "new"}

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            return RolloutBuffer(), {"status": "bad_payload_reset"}

        old_shards = safe_int(payload.get("shards", shards), shards)
        if old_shards != shards:
            return RolloutBuffer(), {
                "status": "reset_shards_mismatch",
                "old_shards": old_shards,
                "new_shards": shards,
            }
        old_iot_feature_dim = safe_int(
            payload.get("iot_feature_dim", iot_feature_dim),
            iot_feature_dim,
        )
        if old_iot_feature_dim != iot_feature_dim:
            return RolloutBuffer(), {
                "status": "reset_iot_feature_dim_mismatch",
                "old_iot_feature_dim": old_iot_feature_dim,
                "new_iot_feature_dim": iot_feature_dim,
            }

        records = payload.get("records", [])
        if not isinstance(records, list):
            return RolloutBuffer(), {"status": "bad_records_reset"}

        buffer = records_to_buffer(records, shards, iot_feature_dim)

        meta = {
            "status": "loaded",
            "old_size": len(buffer),
            "old_num_trajectories": int(sum(1 for d in buffer.dones if d)),
        }

        return buffer, meta

    except Exception as e:
        return RolloutBuffer(), {"status": f"load_failed_{type(e).__name__}"}


def save_pending_buffer(
    path: Path,
    buffer: RolloutBuffer,
    shards: int,
    iot_feature_dim: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(buffer) == 0:
        if path.exists():
            path.unlink()
        return

    payload = {
        "time_unix_nano": now_ns(),
        "shards": shards,
        "iot_feature_dim": iot_feature_dim,
        "size": len(buffer),
        "num_trajectories": int(sum(1 for d in buffer.dones if d)),
        "records": buffer_to_records(buffer),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def append_buffer(dst: RolloutBuffer, src: RolloutBuffer) -> None:
    for i in range(len(src)):
        dst.add(
            state=src.states[i],
            next_state=src.next_states[i],
            action=src.actions[i],
            log_prob=src.log_probs[i],
            reward=src.rewards[i],
            done=src.dones[i],
            value=src.values[i],
            target_action=src.target_actions[i],
            target_weight=src.target_weights[i],
            action_mask=src.action_masks[i],
        )


def rollout_stats(buffer: RolloutBuffer) -> Dict[str, Any]:
    if len(buffer) == 0:
        return {
            "size": 0,
            "num_trajectories": 0,
            "reward_sum": 0.0,
            "reward_mean": 0.0,
            "supervised_target_count": 0,
            "supervised_weight_mean": 0.0,
        }

    reward_sum = float(sum(buffer.rewards))
    supervised_weights = [
        float(w)
        for target, w in zip(buffer.target_actions, buffer.target_weights)
        if target >= 0 and w > 0
    ]

    return {
        "size": len(buffer),
        "num_trajectories": int(sum(1 for d in buffer.dones if d)),
        "reward_sum": reward_sum,
        "reward_mean": reward_sum / float(len(buffer)),
        "supervised_target_count": len(supervised_weights),
        "supervised_weight_mean": (
            float(sum(supervised_weights)) / float(len(supervised_weights))
            if supervised_weights
            else 0.0
        ),
    }


def append_train_log(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_update(input_path: Path, model_path: Path, log_path: Path) -> Dict[str, Any]:
    data = load_update_input(input_path)

    batch_id = safe_int(data.get("batch_id", 0), 0)
    feedback_epoch = safe_int(data.get("feedback_epoch", -1), -1)
    shards = safe_int(data.get("shards", 4), 4)
    iot_feature_dim = update_iot_feature_dim(data)
    flush_update = bool(data.get("flush_update", False))

    new_buffer, meta = build_buffer_from_update(data)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    pending_buffer, pending_meta = load_pending_buffer(
        ROLLOUT_BUFFER_PATH,
        shards,
        iot_feature_dim,
    )
    old_pending_size = len(pending_buffer)

    append_buffer(pending_buffer, new_buffer)

    pending_stats = rollout_stats(pending_buffer)

    result: Dict[str, Any] = {
        "ok": False,
        "updated": False,
        "time_unix_nano": now_ns(),
        "batch_id": batch_id,
        "feedback_epoch": feedback_epoch,
        "shards": shards,
        "iot_feature_dim": iot_feature_dim,
        "state_dim": meta["state_dim"],
        "input_path": str(input_path),
        "model_path": str(model_path),
        "log_path": str(log_path),
        "rollout_buffer_path": str(ROLLOUT_BUFFER_PATH),
        "new_actions": len(new_buffer),
        "skipped": meta["skipped"],
        "batch_reward": meta["batch_reward"],
        "new_action_hist": meta["action_hist"],
        "related_known_count": meta["related_known_count"],
        "same_as_related_count": meta["same_as_related_count"],
        "same_as_related_ratio": (
            float(meta["same_as_related_count"]) / float(meta["related_known_count"])
            if meta["related_known_count"] > 0
            else 0.0
        ),
        "action_confidence_mean": meta["action_confidence_mean"],
        "action_entropy_mean": meta["action_entropy_mean"],
        "action_log_prob_mean": meta["action_log_prob_mean"],
        "local_reward_mean": meta["local_reward_mean"],
        "pending_meta": pending_meta,
        "old_pending_size": old_pending_size,
        "pending_size": pending_stats["size"],
        "pending_num_trajectories": pending_stats["num_trajectories"],
        "pending_reward_sum": pending_stats["reward_sum"],
        "rollout_threshold": BATCH_SIZE,
        "min_flush_threshold": MIN_FLUSH_BATCH_SIZE,
        "entropy_coef": ENTROPY_COEF,
        "reward_clip": REWARD_CLIP,
        "value_clip": VALUE_CLIP,
        "flush_update": flush_update,
        "feedback_aggregate_count": safe_int(data.get("feedback_aggregate_count", 0), 0),
        "feedback_first_epoch": safe_int(data.get("feedback_first_epoch", 0), 0),
        "feedback_last_epoch": safe_int(data.get("feedback_last_epoch", 0), 0),
        "feedback_weight_sum": safe_float(data.get("feedback_weight_sum", 0.0), 0.0),
        "feedback_aggregate_window": safe_int(data.get("feedback_aggregate_window", 0), 0),
        "feedback_max_matches": safe_int(data.get("feedback_max_matches", 0), 0),
        "cross_rate": safe_float(data.get("cross_rate", 0.0), 0.0),
        "effective_tx": safe_float(data.get("effective_tx", 0.0), 0.0),
        "cross_tx": safe_float(data.get("cross_tx", 0.0), 0.0),
        "normalized_load_variance": safe_float(
            data.get("normalized_load_variance", 0.0),
            0.0,
        ),
        "running_avg_cross_rate": safe_float(data.get("running_avg_cross_rate", 0.0), 0.0),
        "running_avg_reward": safe_float(data.get("running_avg_reward", 0.0), 0.0),
        "running_avg_norm_var": safe_float(data.get("running_avg_norm_var", 0.0), 0.0),
        "rewarded_epoch_count": safe_int(data.get("rewarded_epoch_count", 0), 0),
        "total_tx": safe_int(data.get("total_tx", 0), 0),
        "total_inner": safe_int(data.get("total_inner", 0), 0),
        "total_relay1": safe_int(data.get("total_relay1", 0), 0),
        "total_relay2": safe_int(data.get("total_relay2", 0), 0),
        "supervised_target_count": pending_stats["supervised_target_count"],
        "supervised_target_ratio": (
            float(pending_stats["supervised_target_count"]) / float(pending_stats["size"])
            if pending_stats["size"]
            else 0.0
        ),
        "supervised_weight_mean": pending_stats["supervised_weight_mean"],
        "loss_info": {},
        "model_source": "",
        "message": "",
    }

    if len(new_buffer) == 0:
        save_pending_buffer(ROLLOUT_BUFFER_PATH, pending_buffer, shards, iot_feature_dim)
        result["ok"] = True
        result["message"] = "skip_no_valid_new_actions"
        append_train_log(log_path, result)
        return result

    # 关键：不再每个 online_update 文件都更新。
    # 累计到 BATCH_SIZE=2048 条 action 后，再执行一次 PPO 更新。
    update_threshold = BATCH_SIZE
    if flush_update:
        update_threshold = MIN_FLUSH_BATCH_SIZE

    if len(pending_buffer) < update_threshold:
        save_pending_buffer(ROLLOUT_BUFFER_PATH, pending_buffer, shards, iot_feature_dim)

        result["ok"] = True
        result["updated"] = False
        if flush_update:
            result["message"] = "flush_waiting_min_samples"
        else:
            result["message"] = "buffering_not_enough_samples"

        append_train_log(log_path, result)
        return result

    agent, old_payload, model_source = build_agent(shards, model_path, iot_feature_dim)
    result["model_source"] = model_source

    loss_info = agent.update(pending_buffer)

    old_extra = old_payload.get("extra", {}) if isinstance(old_payload, dict) else {}
    if not isinstance(old_extra, dict):
        old_extra = {}

    old_update_count = safe_int(old_extra.get("online_update_count", 0), 0)
    new_update_count = old_update_count + 1

    extra = {
        "online_update_count": new_update_count,
        "last_batch_id": batch_id,
        "last_feedback_epoch": feedback_epoch,
        "last_batch_reward": meta["batch_reward"],
        "last_cross_rate": result["cross_rate"],
        "last_normalized_load_variance": result["normalized_load_variance"],
        "last_effective_tx": result["effective_tx"],
        "last_cross_tx": result["cross_tx"],
        "last_running_avg_cross_rate": result["running_avg_cross_rate"],
        "last_action_confidence_mean": result["action_confidence_mean"],
        "last_action_entropy_mean": result["action_entropy_mean"],
        "last_local_reward_mean": result["local_reward_mean"],
        "last_supervised_target_count": result["supervised_target_count"],
        "last_supervised_weight_mean": result["supervised_weight_mean"],
        "last_new_actions": len(new_buffer),
        "last_pending_size_used_for_update": len(pending_buffer),
        "last_pending_num_trajectories": pending_stats["num_trajectories"],
        "last_update_time_unix_nano": result["time_unix_nano"],
        "iot_feature_dim": iot_feature_dim,
        "state_dim": meta["state_dim"],
    }

    agent.save(model_path, extra=extra)

    # 更新成功后清空已用 rollout buffer。
    save_pending_buffer(ROLLOUT_BUFFER_PATH, RolloutBuffer(), shards, iot_feature_dim)

    result["ok"] = True
    result["updated"] = True
    result["loss_info"] = loss_info
    result["online_update_count"] = new_update_count
    if flush_update and len(pending_buffer) < BATCH_SIZE:
        result["message"] = "updated_with_flush_rollout"
    else:
        result["message"] = "updated_with_trajectory_rollout"

    append_train_log(log_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument(
        "--log",
        default=str(Path("spring_io") / "online_train_log.jsonl"),
    )

    args = parser.parse_args()

    try:
        result = run_update(
            input_path=Path(args.input),
            model_path=Path(args.model),
            log_path=Path(args.log),
        )

    except Exception as e:
        result = {
            "ok": False,
            "updated": False,
            "time_unix_nano": now_ns(),
            "message": f"exception_{type(e).__name__}: {e}",
            "input_path": args.input,
            "model_path": args.model,
            "log_path": args.log,
            "loss_info": {},
        }

        try:
            append_train_log(Path(args.log), result)
        except Exception:
            pass

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
