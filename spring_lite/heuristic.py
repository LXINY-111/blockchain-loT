import hashlib
from typing import List, Optional, Sequence

from action_mask import best_allowed_action, normalize_action_mask


def addr2shard(addr: str, shards: int) -> int:
    addr = addr.strip()
    if addr.startswith("0x") or addr.startswith("0X"):
        addr = addr[2:]

    if len(addr) > 8:
        addr = addr[-8:]

    try:
        return int(addr, 16) % shards
    except ValueError:
        # IoT MDP 中的 iot_state/iot_peer 不是十六进制地址。
        # Python 内置 hash() 会按进程随机化，论文实验需要稳定哈希。
        digest = hashlib.sha256(addr.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % shards


def heuristic_scores_from_state(state: List[float], shards: int) -> List[float]:
    """
    Python 推理兜底策略：
    1. 如果 sender_pos 显示交易相关地址已经在某个分片，优先靠近该分片；
    2. 同时使用最近 5 个窗口交易量作为负载惩罚；
    3. 如果没有相关地址，则选择近期负载最小分片。
    """
    if len(state) < 11 * shards + 1:
        return [0.0 for _ in range(shards)]

    recent_num = state[: 5 * shards]
    sender_pos = state[10 * shards : 11 * shards]

    loads = [0.0 for _ in range(shards)]
    for w in range(5):
        for sid in range(shards):
            loads[sid] += float(recent_num[w * shards + sid])

    scores: List[float] = []
    for sid in range(shards):
        score = -loads[sid]

        # multi-anchor（多锚点）场景下，相关锚点可能分散在多个 shard（分片）。
        # 用连续 sender_pos 加分，而不是只在 >0.5 时触发，避免退回纯负载规则。
        if sid < len(sender_pos):
            score += 1000.0 * float(sender_pos[sid])

        scores.append(score)

    return scores


def heuristic_from_state(
    state: List[float],
    shards: int,
    action_mask: Optional[Sequence[int]] = None,
) -> int:
    scores = heuristic_scores_from_state(state, shards)
    mask = normalize_action_mask(action_mask, shards)
    return best_allowed_action(scores, mask, shards)
