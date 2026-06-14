from typing import Any, List, Optional, Sequence

import torch


def normalize_action_mask(mask: Optional[Sequence[Any]], shards: int) -> List[int]:
    """Return a safe 0/1 action mask; fall back to all shards if empty."""
    shard_count = max(0, int(shards))
    if shard_count == 0:
        return []

    values = list(mask or [])
    normalized = []
    for sid in range(shard_count):
        value = values[sid] if sid < len(values) else 1
        normalized.append(1 if float(value) > 0.0 else 0)

    if not any(normalized):
        return [1 for _ in range(shard_count)]
    return normalized


def action_allowed(mask: Optional[Sequence[Any]], action: int, shards: int) -> bool:
    normalized = normalize_action_mask(mask, shards)
    return 0 <= int(action) < len(normalized) and normalized[int(action)] > 0


def best_allowed_action(scores: Sequence[float], mask: Optional[Sequence[Any]], shards: int) -> int:
    normalized = normalize_action_mask(mask, shards)
    best_sid = 0
    best_score = float("-inf")
    for sid in range(max(0, int(shards))):
        if sid >= len(normalized) or normalized[sid] <= 0:
            continue
        score = float(scores[sid]) if sid < len(scores) else 0.0
        if score > best_score:
            best_score = score
            best_sid = sid
    return int(best_sid)


def mask_logits(logits: torch.Tensor, masks: Optional[Sequence[Sequence[Any]]]) -> torch.Tensor:
    """Set invalid action logits to a very negative value before Categorical."""
    if masks is None:
        return logits

    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=logits.device)
    if mask_t.ndim == 1:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.shape != logits.shape:
        return logits

    has_any = mask_t.any(dim=-1, keepdim=True)
    safe_mask = torch.where(has_any, mask_t, torch.ones_like(mask_t))
    return logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min / 2)
