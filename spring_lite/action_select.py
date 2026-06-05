import hashlib
from typing import Iterable, List, Sequence


def stable_hash_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo


def tie_aware_argmax(
    probabilities: Sequence[float],
    key: str,
    tie_eps: float,
    enable_tie_break: bool = True,
) -> int:
    if not probabilities:
        return 0

    probs = [float(x) for x in probabilities]
    best = max(range(len(probs)), key=lambda idx: probs[idx])

    if not enable_tie_break or tie_eps <= 0:
        return best

    best_prob = probs[best]
    candidates: List[int] = [
        idx for idx, prob in enumerate(probs) if best_prob - prob <= float(tie_eps)
    ]
    if len(candidates) <= 1:
        return best

    chosen = stable_hash_index(key, len(candidates))
    return candidates[chosen]


def distribution_floor_shortfall(values: Iterable[float], floor: float) -> float:
    return sum(max(0.0, float(floor) - float(value)) for value in values)
