# spring_lite/infer_server.py
# 常驻 PPO 推理进程：
# Go 通过 stdin 发送一行 JSON 请求；
# Python 通过 stdout 返回一行 JSON 结果；
# 避免每个新地址都重新启动 Python、重新加载模型。

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributions import Categorical

from action_mask import mask_logits, normalize_action_mask
from action_select import tie_aware_argmax
from checkpoint_compat import checkpoint_config_mismatches, format_checkpoint_mismatch
from config import (
    ARGMAX_TIE_BREAK,
    ARGMAX_TIE_EPS,
    HIDDEN_DIM,
    MODEL_PATH,
    state_dim,
)
from heuristic import heuristic_from_state
from ppo import PPOAgent
from mechanism import input_dim


class AgentCache:
    def __init__(self) -> None:
        self.agent: Optional[PPOAgent] = None
        self.shards: int = -1
        self.state_dim: int = -1
        self.model_path: str = ""
        self.model_mtime_ns: int = -1
        self.checkpoint_config_signature: str = ""

    def _create_fresh_agent(self, shards: int, iot_feature_dim: int = 0, version='legacy', layout=None) -> PPOAgent:
        return PPOAgent(
            state_dim=input_dim(shards, iot_feature_dim, version, layout),
            action_dim=shards,
            hidden_dim=HIDDEN_DIM,
            device="cpu",
        )

    def _model_mtime(self, model_path: Path) -> int:
        if not model_path.exists():
            return -1
        return int(model_path.stat().st_mtime_ns)

    def get_agent(
        self,
        shards: int,
        model_path: Path,
        iot_feature_dim: int = 0,
        allow_model_init: bool = True,
        expected_checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[PPOAgent], str]:
        version = (expected_checkpoint or {}).get('mechanism_version', 'legacy')
        layout = (expected_checkpoint or {}).get('state_layout')
        expected_dim = input_dim(shards, iot_feature_dim, version, layout)
        current_mtime = self._model_mtime(model_path)
        model_path_key = str(model_path)
        checkpoint_config_signature = json.dumps(
            expected_checkpoint or {},
            ensure_ascii=True,
            sort_keys=True,
        )

        # 如果模型已经加载，且模型文件没有变化，就直接复用内存里的 agent。
        if (
            self.agent is not None
            and self.shards == shards
            and self.state_dim == expected_dim
            and self.model_path == model_path_key
            and self.model_mtime_ns == current_mtime
            and self.checkpoint_config_signature == checkpoint_config_signature
        ):
            self.agent.net.eval()
            return self.agent, "python_ppo"

        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果模型不存在，创建一个随机初始化的 PPO 模型，保证不会回退 heuristic。
        if not model_path.exists():
            if not allow_model_init:
                return None, "no_model"

            agent = self._create_fresh_agent(shards, iot_feature_dim, version, layout)
            agent.save(
                model_path,
                extra={
                    **(expected_checkpoint or {}),
                    "model_source": "init_by_infer_server",
                    "online_update_count": 0,
                    "iot_feature_dim": int(iot_feature_dim),
                },
            )
            self.agent = agent
            self.shards = shards
            self.state_dim = expected_dim
            self.model_path = model_path_key
            self.model_mtime_ns = self._model_mtime(model_path)
            self.checkpoint_config_signature = checkpoint_config_signature
            agent.net.eval()
            return agent, "python_ppo"

        # 如果模型存在，加载模型。
        agent = self._create_fresh_agent(shards, iot_feature_dim, version, layout)

        try:
            payload = torch.load(model_path, map_location=agent.device)
            ckpt_state_dim = int(payload.get("state_dim", -1))
            ckpt_action_dim = int(payload.get("action_dim", -1))

            if ckpt_state_dim != expected_dim or ckpt_action_dim != shards:
                return None, (
                    f"model_dim_mismatch: ckpt_state_dim={ckpt_state_dim}, "
                    f"ckpt_action_dim={ckpt_action_dim}, expected_dim={expected_dim}, shards={shards}"
                )

            mismatches = checkpoint_config_mismatches(
                payload.get("extra", {}),
                expected_checkpoint or {},
            )
            if mismatches:
                return None, format_checkpoint_mismatch(mismatches)

            agent.net.load_state_dict(payload["model_state_dict"])

        except Exception as exc:
            # 加载失败只返回原因；冻结实验不得重置训练模型或静默换策略。
            return None, f"load_failed:{exc}"

        self.agent = agent
        self.shards = shards
        self.state_dim = expected_dim
        self.model_path = model_path_key
        self.model_mtime_ns = self._model_mtime(model_path)
        self.checkpoint_config_signature = checkpoint_config_signature
        agent.net.eval()
        return agent, "python_ppo"


CACHE = AgentCache()


def safe_heuristic(item: Dict[str, Any], shards: int, reason: str, request_id: int) -> Dict[str, Any]:
    state = item.get("state", [])
    shard = heuristic_from_state(state, shards, item.get("action_mask", []))

    if shard < 0 or shard >= shards:
        shard = 0
        reason = "safe_default"

    return {
        "address": str(item.get("address", "")),
        "related": str(item.get("related", "")),
        "shard": int(shard),
        "source": f"python_heuristic_{reason}",
        "confidence": 0.0,
        "entropy": 0.0,
        "batch_id": int(request_id),
        "log_prob": 0.0,
        "value": 0.0,
    }


def normalize_items(raw_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    items: List[Dict[str, Any]] = []

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        state = raw.get("state", [])
        if not isinstance(state, list):
            state = []
        action_mask = raw.get("action_mask", [])
        if not isinstance(action_mask, list):
            action_mask = []

        items.append(
            {
                "address": str(raw.get("address", "")),
                "related": str(raw.get("related", "")),
                "state": [float(x) for x in state],
                "action_mask": action_mask,
            }
        )

    return items


def infer_items(
    items: List[Dict[str, Any]],
    shards: int,
    sample: bool,
    request_id: int,
    model_path: Path,
    iot_feature_dim: int = 0,
    allow_model_init: bool = True,
    expected_checkpoint: Optional[Dict[str, Any]] = None,
    cache: Optional[AgentCache] = None,
) -> List[Dict[str, Any]]:
    expected_dim = input_dim(shards, iot_feature_dim, (expected_checkpoint or {}).get('mechanism_version', 'legacy'),
                             (expected_checkpoint or {}).get('state_layout'))
    agent_cache = cache or CACHE
    agent, model_status = agent_cache.get_agent(
        shards,
        model_path,
        iot_feature_dim,
        allow_model_init,
        expected_checkpoint,
    )

    if agent is None:
        if expected_checkpoint:
            raise RuntimeError(f"frozen model unavailable: {model_status}")
        return [safe_heuristic(item, shards, model_status, request_id) for item in items]

    outputs: List[Optional[Dict[str, Any]]] = [None for _ in items]
    valid_indices: List[int] = []
    valid_states: List[List[float]] = []
    valid_masks: List[List[int]] = []

    for idx, item in enumerate(items):
        state = item.get("state", [])
        if len(state) != expected_dim:
            if expected_checkpoint:
                raise RuntimeError(f'frozen model input dimension mismatch: got {len(state)}, expected {expected_dim}')
            outputs[idx] = safe_heuristic(item, shards, "dim_mismatch", request_id)
            continue

        valid_indices.append(idx)
        valid_states.append(state)
        valid_masks.append(normalize_action_mask(item.get("action_mask", []), shards))

    if valid_states:
        with torch.no_grad():
            states_t = torch.tensor(
                valid_states,
                dtype=torch.float32,
                device=agent.device,
            )

            logits, values = agent.net(states_t)
            masked = mask_logits(logits, valid_masks)
            dist = Categorical(logits=masked)
            probs = torch.softmax(masked, dim=-1)

            if sample:
                actions = dist.sample()
            else:
                action_ids = []
                for local_idx, item_idx in enumerate(valid_indices):
                    item = items[item_idx]
                    action_ids.append(
                        tie_aware_argmax(
                            probs[local_idx].detach().cpu().tolist(),
                            key=f"{item.get('address', '')}|{item.get('related', '')}",
                            tie_eps=ARGMAX_TIE_EPS,
                            enable_tie_break=bool(ARGMAX_TIE_BREAK),
                        )
                    )
                actions = torch.tensor(
                    action_ids,
                    dtype=torch.long,
                    device=agent.device,
                )

            log_probs = dist.log_prob(actions)
            confidences = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            entropies = dist.entropy()

            for local_idx, item_idx in enumerate(valid_indices):
                item = items[item_idx]

                shard = int(actions[local_idx].item())
                confidence = float(confidences[local_idx].item())
                entropy = float(entropies[local_idx].item())
                log_prob = float(log_probs[local_idx].item())
                value = float(values[local_idx].item())

                if shard < 0 or shard >= shards:
                    outputs[item_idx] = safe_heuristic(item, shards, "bad_action", request_id)
                    continue

                outputs[item_idx] = {
                    "address": item.get("address", ""),
                    "related": item.get("related", ""),
                    "shard": shard,
                    "source": "python_ppo",
                    "confidence": confidence,
                    "entropy": entropy,
                    "batch_id": int(request_id),
                    "log_prob": log_prob,
                    "value": value,
                }

    final_outputs: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if outputs[idx] is None:
            outputs[idx] = safe_heuristic(item, shards, "unknown", request_id)
        final_outputs.append(outputs[idx])

    return final_outputs


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    request_id = int(req.get("request_id", 0))
    shards = int(req.get("shards", 4))
    iot_feature_dim = max(0, int(req.get("iot_feature_dim", 0)))
    allow_model_init = bool(req.get("allow_model_init", True))
    sample = bool(req.get("sample", False))
    model_path = Path(str(req.get("model", MODEL_PATH)))
    expected_checkpoint = req.get("expected_checkpoint", {})
    if not isinstance(expected_checkpoint, dict):
        expected_checkpoint = {}

    if shards <= 0:
        return {
            "request_id": request_id,
            "items": [],
            "error": f"invalid shards: {shards}",
        }

    items = normalize_items(req.get("items", []))
    outputs = infer_items(
        items=items,
        shards=shards,
        sample=sample,
        request_id=request_id,
        model_path=model_path,
        iot_feature_dim=iot_feature_dim,
        allow_model_init=allow_model_init,
        expected_checkpoint=expected_checkpoint,
    )

    return {
        "request_id": request_id,
        "items": outputs,
    }


def main() -> None:
    for raw_line in sys.stdin.buffer:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            resp = handle_request(req)

        except Exception as exc:
            resp = {
                "request_id": 0,
                "items": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        sys.stdout.buffer.write(
            (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
        )
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
