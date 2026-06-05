import csv
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import (
    ACTION_REWARD_SCALE,
    ACTIVE_SHARD_BONUS_WEIGHT,
    BACKLOG_PENALTY_WEIGHT,
    BETA,
    CAPACITY_BACKLOG_MODE,
    DEFAULT_SENDER_POS_MODE,
    EPS,
    HOTSPOT_PENALTY_WEIGHT,
    HOTSPOT_THRESHOLD,
    LAMBDA_WEIGHT,
    LOAD_PENALTY_WEIGHT,
    LOCAL_REWARD_WEIGHT,
    MIN_ACTIVE_LOAD_SHARE,
    REWARD_MODE,
    state_dim,
)
from heuristic import addr2shard


@dataclass(frozen=True)
class Tx:
    sender: str
    recipient: str


@dataclass
class SenderPosInfo:
    sender_pos: List[float]
    related_summary: str
    related_known: bool
    related_shard: int
    related_weight: float
    related_in_current_batch: bool
    related_count: int


@dataclass
class PolicyOutput:
    action: int
    log_prob: float = 0.0
    value: float = 0.0
    confidence: float = 0.0
    entropy: float = 0.0
    source: str = "policy"


@dataclass
class PlacementAction:
    batch_id: int
    address: str
    related: str
    state: List[float]
    next_state: List[float]
    action: int
    log_prob: float
    value: float
    confidence: float
    entropy: float
    local_reward: float
    reward: float = 0.0
    done: bool = False
    related_known: bool = False
    related_shard: int = -1
    related_weight: float = 0.0
    related_count: int = 0
    sender_pos: List[float] = field(default_factory=list)
    same_as_related: bool = False
    related_in_current_batch: bool = False
    shard_load_before: int = 0
    shard_load_mean_before: float = 0.0
    load_penalty: float = 0.0


@dataclass
class BatchMetrics:
    batch_id: int
    tx_count: int
    action_count: int
    total_tx: int
    total_inner: int
    total_relay1: int
    total_relay2: int
    effective_tx: float
    cross_tx: float
    cross_rate: float
    raw_load_variance: float
    normalized_load_variance: float
    r_cstr: float
    r_wlb: float
    abs_load_diff: float
    active_shards: int
    active_shard_ratio: float
    max_load_share: float
    hotspot_penalty: float
    load_aware_bonus: float
    backlog_penalty: float
    reward: float
    loads: List[int]
    effective_loads: List[float]
    reward_loads: List[float]
    committed_loads: List[float]
    pending_loads: List[float]
    cross_loads: List[float]
    action_hist: List[int]
    related_known_count: int
    same_as_related_count: int
    local_reward_mean: float
    action_reward_mean: float
    confidence_mean: float
    entropy_mean: float
    reward_mode: str
    related_shard_hist: List[int]
    same_related_by_shard: List[int]
    related_follow_by_shard: List[float]
    min_related_follow_ratio: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "tx_count": self.tx_count,
            "action_count": self.action_count,
            "total_tx": self.total_tx,
            "total_inner": self.total_inner,
            "total_relay1": self.total_relay1,
            "total_relay2": self.total_relay2,
            "effective_tx": self.effective_tx,
            "cross_tx": self.cross_tx,
            "cross_rate": self.cross_rate,
            "raw_load_variance": self.raw_load_variance,
            "normalized_load_variance": self.normalized_load_variance,
            "r_cstr": self.r_cstr,
            "r_wlb": self.r_wlb,
            "abs_load_diff": self.abs_load_diff,
            "active_shards": self.active_shards,
            "active_shard_ratio": self.active_shard_ratio,
            "max_load_share": self.max_load_share,
            "hotspot_penalty": self.hotspot_penalty,
            "load_aware_bonus": self.load_aware_bonus,
            "backlog_penalty": self.backlog_penalty,
            "reward": self.reward,
            "loads": self.loads,
            "effective_loads": self.effective_loads,
            "reward_loads": self.reward_loads,
            "committed_loads": self.committed_loads,
            "pending_loads": self.pending_loads,
            "cross_loads": self.cross_loads,
            "action_hist": self.action_hist,
            "related_known_count": self.related_known_count,
            "same_as_related_count": self.same_as_related_count,
            "same_as_related_ratio": (
                self.same_as_related_count / self.related_known_count
                if self.related_known_count
                else 0.0
            ),
            "local_reward_mean": self.local_reward_mean,
            "action_reward_mean": self.action_reward_mean,
            "confidence_mean": self.confidence_mean,
            "entropy_mean": self.entropy_mean,
            "reward_mode": self.reward_mode,
            "related_shard_hist": self.related_shard_hist,
            "same_related_by_shard": self.same_related_by_shard,
            "related_follow_by_shard": self.related_follow_by_shard,
            "min_related_follow_ratio": self.min_related_follow_ratio,
        }


PolicyFn = Callable[[List[float], str, str, SenderPosInfo], PolicyOutput]


def normalize_address(addr: str) -> str:
    addr = str(addr).strip()
    if addr.startswith("0x") or addr.startswith("0X"):
        addr = addr[2:]
    return addr


def parse_tx_row(row: Sequence[str]) -> Optional[Tx]:
    if len(row) < 9:
        return None

    sender = str(row[3]).strip()
    recipient = str(row[4]).strip()

    if str(row[6]) != "0" or str(row[7]) != "0":
        return None
    if len(sender) <= 16 or len(recipient) <= 16:
        return None
    if sender == recipient:
        return None

    sender = normalize_address(sender)
    recipient = normalize_address(recipient)
    if not sender or not recipient or sender == recipient:
        return None

    return Tx(sender=sender, recipient=recipient)


def load_transactions(csv_path: Path, max_txs: int = 0) -> List[Tx]:
    txs: List[Tx] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            tx = parse_tx_row(row)
            if tx is None:
                continue
            txs.append(tx)
            if max_txs > 0 and len(txs) >= max_txs:
                break
    return txs


def iter_batches(txs: Sequence[Tx], batch_size: int) -> Iterable[List[Tx]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(txs), batch_size):
        yield list(txs[start : start + batch_size])


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def normalize_state_value(value: float, max_block_size: int, tx_batch_size: int) -> float:
    denom = float(max(max_block_size, tx_batch_size))
    if denom <= 0:
        denom = 100.0

    x = math.log1p(max(0.0, float(value))) / math.log1p(denom * 4.0)
    return clamp(x, 0.0, 1.0)


def address_flag_from_related_count(related_count: int) -> float:
    return 1.0 if related_count >= 4 else 0.0


def local_action_reward(
    chosen_shard: int,
    related_known: bool,
    related_shard: int,
    related_weight: float,
    shard_load_before: int,
    load_mean_before: float,
) -> Tuple[float, float]:
    reward = 0.0

    if related_known:
        if chosen_shard == related_shard:
            reward += 1.2 * related_weight
        else:
            reward -= 1.0 * related_weight

    load_penalty = 0.0
    if load_mean_before > 1e-9:
        load_penalty = float(shard_load_before) / load_mean_before
        if load_penalty > 1.0:
            overload = load_penalty - 1.0
            reward -= 0.70 * overload
            if load_penalty > 1.25:
                reward -= 0.35 * (load_penalty - 1.25)
        else:
            reward += 0.08 * (1.0 - load_penalty)

    return reward, load_penalty


def spring_reward(
    effective_loads: Sequence[float],
    cross_loads: Sequence[float],
    balance_loads: Optional[Sequence[float]] = None,
    lambda_weight: float = LAMBDA_WEIGHT,
    beta: float = BETA,
    load_penalty_weight: float = LOAD_PENALTY_WEIGHT,
    active_shard_bonus_weight: float = ACTIVE_SHARD_BONUS_WEIGHT,
    hotspot_penalty_weight: float = HOTSPOT_PENALTY_WEIGHT,
    hotspot_threshold: float = HOTSPOT_THRESHOLD,
    min_active_load_share: float = MIN_ACTIVE_LOAD_SHARE,
    backlog_penalty: float = 0.0,
    backlog_penalty_weight: float = BACKLOG_PENALTY_WEIGHT,
    reward_mode: str = REWARD_MODE,
) -> Tuple[float, Dict[str, float]]:
    effective_tx = float(sum(effective_loads))
    cross_tx = float(sum(cross_loads))
    shard_count = max(1, len(effective_loads))
    load_basis = list(balance_loads) if balance_loads is not None else list(effective_loads)
    if len(load_basis) != shard_count:
        load_basis = list(effective_loads)
    load_total = float(sum(load_basis))

    if effective_tx <= EPS:
        return 0.0, {
            "effective_tx": 0.0,
            "cross_tx": 0.0,
            "cross_rate": 0.0,
            "raw_load_variance": 0.0,
            "normalized_load_variance": 0.0,
            "r_cstr": 0.0,
            "r_wlb": 0.0,
            "abs_load_diff": 0.0,
            "active_shards": 0.0,
            "active_shard_ratio": 0.0,
            "max_load_share": 0.0,
            "hotspot_penalty": 0.0,
            "load_aware_bonus": 0.0,
            "backlog_penalty": 0.0,
            "reward_load_total": 0.0,
            "reward_mode": str(reward_mode),
        }

    cross_rate = clamp(cross_tx / effective_tx, 0.0, 1.0)
    r_cstr = clamp(1.0 - cross_rate, 0.0, 1.0)

    if load_total <= EPS:
        load_basis = list(effective_loads)
        load_total = effective_tx

    avg_load = load_total / float(shard_count)
    raw_abs_diff = 0.0
    raw_var = 0.0

    for load in load_basis:
        diff = float(load) - avg_load
        raw_abs_diff += abs(diff)
        raw_var += diff * diff

    raw_var /= float(shard_count)
    norm_var = raw_var / (avg_load * avg_load + EPS)
    norm_var = norm_var / (1.0 + norm_var)

    normalized_abs_diff = 0.0
    if avg_load > EPS:
        normalized_abs_diff = raw_abs_diff / (avg_load + EPS)

    balance_score = clamp(1.0 - norm_var, 0.0, 1.0)
    r_wlb = math.exp(-beta * normalized_abs_diff) * balance_score
    r_wlb = clamp(r_wlb, 0.0, 1.0)

    load_shares = [
        clamp(float(load) / (load_total + EPS), 0.0, 1.0)
        for load in load_basis
    ]
    active_threshold = clamp(float(min_active_load_share), 0.0, 1.0)
    active_shards = sum(1 for share in load_shares if share >= active_threshold)
    active_shard_ratio = float(active_shards) / float(shard_count)

    max_load_share = max(load_shares) if load_shares else 0.0
    hotspot_threshold = clamp(float(hotspot_threshold), 0.0, 1.0 - EPS)
    hotspot_penalty = clamp(
        (max_load_share - hotspot_threshold) / (1.0 - hotspot_threshold + EPS),
        0.0,
        1.0,
    )
    load_aware_bonus = (
        float(active_shard_bonus_weight) * active_shard_ratio
        - float(hotspot_penalty_weight) * hotspot_penalty
    )

    base_reward = lambda_weight * r_cstr + (1.0 - lambda_weight) * r_wlb
    mode = str(reward_mode).strip().lower()
    if mode == "paper":
        # Paper-aligned baseline: keep the MDP reward focused on cross-shard
        # transaction reduction and workload balance. The extra terms are still
        # reported as diagnostics, but they do not affect the PPO target here.
        reward = base_reward
    elif mode == "enhanced":
        # Enhanced mode keeps the stabilizers/penalties introduced for later
        # ablation and innovation experiments.
        reward = base_reward
        reward -= load_penalty_weight * norm_var
        reward += load_aware_bonus
        reward -= float(backlog_penalty_weight) * clamp(float(backlog_penalty), 0.0, 1.0)
    else:
        raise ValueError(f"unknown reward_mode: {reward_mode}")

    if math.isnan(reward) or math.isinf(reward):
        reward = 0.0

    return reward, {
        "effective_tx": effective_tx,
        "cross_tx": cross_tx,
        "cross_rate": cross_rate,
        "raw_load_variance": raw_var,
        "normalized_load_variance": norm_var,
        "r_cstr": r_cstr,
        "r_wlb": r_wlb,
        "abs_load_diff": normalized_abs_diff,
        "active_shards": float(active_shards),
        "active_shard_ratio": active_shard_ratio,
        "max_load_share": max_load_share,
        "hotspot_penalty": hotspot_penalty,
        "load_aware_bonus": load_aware_bonus,
        "backlog_penalty": clamp(float(backlog_penalty), 0.0, 1.0),
        "reward_load_total": load_total,
        "reward_mode": mode,
    }


class SpringOfflineEnv:
    def __init__(
        self,
        shards: int = 4,
        tx_batch_size: int = 1000,
        max_block_size: int = 1000,
        lambda_weight: float = LAMBDA_WEIGHT,
        beta: float = BETA,
        sender_pos_mode: int = DEFAULT_SENDER_POS_MODE,
        temporal_top_k: int = 8,
        local_reward_weight: float = LOCAL_REWARD_WEIGHT,
        action_reward_scale: float = ACTION_REWARD_SCALE,
        load_penalty_weight: float = LOAD_PENALTY_WEIGHT,
        active_shard_bonus_weight: float = ACTIVE_SHARD_BONUS_WEIGHT,
        hotspot_penalty_weight: float = HOTSPOT_PENALTY_WEIGHT,
        hotspot_threshold: float = HOTSPOT_THRESHOLD,
        min_active_load_share: float = MIN_ACTIVE_LOAD_SHARE,
        capacity_backlog_mode: int = CAPACITY_BACKLOG_MODE,
        backlog_penalty_weight: float = BACKLOG_PENALTY_WEIGHT,
        reward_mode: str = REWARD_MODE,
    ) -> None:
        if shards <= 0:
            raise ValueError("shards must be positive")
        if tx_batch_size <= 0:
            raise ValueError("tx_batch_size must be positive")

        self.shards = int(shards)
        self.tx_batch_size = int(tx_batch_size)
        self.max_block_size = int(max_block_size)
        self.lambda_weight = float(lambda_weight)
        self.beta = float(beta)
        self.sender_pos_mode = int(sender_pos_mode)
        self.temporal_top_k = int(temporal_top_k)
        self.local_reward_weight = float(local_reward_weight)
        self.action_reward_scale = float(action_reward_scale)
        self.load_penalty_weight = float(load_penalty_weight)
        self.active_shard_bonus_weight = float(active_shard_bonus_weight)
        self.hotspot_penalty_weight = float(hotspot_penalty_weight)
        self.hotspot_threshold = float(hotspot_threshold)
        self.min_active_load_share = float(min_active_load_share)
        self.capacity_backlog_mode = int(capacity_backlog_mode)
        self.backlog_penalty_weight = float(backlog_penalty_weight)
        self.reward_mode = str(reward_mode).strip().lower()

        self.addr_shard: Dict[str, int] = {}
        self.shard_load: List[int] = []
        self.pending_loads: List[float] = []
        self.pending_cross_loads: List[float] = []
        self.recent_stats: Deque[Dict[str, List[float]]] = deque(maxlen=5)
        self.temporal_neighbors: Dict[str, Counter] = defaultdict(Counter)
        self.batch_id = 0
        self.reset()

    def reset(self) -> None:
        self.addr_shard = {}
        self.shard_load = [0 for _ in range(self.shards)]
        self.pending_loads = [0.0 for _ in range(self.shards)]
        self.pending_cross_loads = [0.0 for _ in range(self.shards)]
        self.recent_stats = deque(maxlen=5)
        for _ in range(5):
            self.recent_stats.append(
                {
                    "num": [0.0 for _ in range(self.shards)],
                    "cross": [0.0 for _ in range(self.shards)],
                }
            )
        self.temporal_neighbors = defaultdict(Counter)
        self.batch_id = 0

    def normalize_count(self, value: float) -> float:
        return normalize_state_value(value, self.max_block_size, self.tx_batch_size)

    def build_state_from_sender_pos(self, sender_pos: Sequence[float], flag: float) -> List[float]:
        state: List[float] = []

        for stat in self.recent_stats:
            for sid in range(self.shards):
                state.append(self.normalize_count(stat["num"][sid]))

        for stat in self.recent_stats:
            for sid in range(self.shards):
                state.append(self.normalize_count(stat["cross"][sid]))

        for sid in range(self.shards):
            value = 0.0
            if sid < len(sender_pos):
                value = float(sender_pos[sid])
            state.append(clamp(value, 0.0, 1.0))

        state.append(1.0 if flag != 0 else 0.0)

        expected_dim = state_dim(self.shards)
        if len(state) != expected_dim:
            raise RuntimeError(f"state dim mismatch: got {len(state)}, expected {expected_dim}")
        return state

    def build_batch_related(self, txs: Sequence[Tx]) -> Dict[str, Set[str]]:
        related: Dict[str, Set[str]] = defaultdict(set)

        def add(a: str, b: str) -> None:
            if a and b and a != b:
                related[a].add(b)

        for tx in txs:
            if self.sender_pos_mode == 1:
                add(tx.recipient, tx.sender)
            else:
                add(tx.sender, tx.recipient)
                add(tx.recipient, tx.sender)

        return related

    def _temporal_related(self, addr: str) -> List[str]:
        if self.sender_pos_mode != 2 or self.temporal_top_k <= 0:
            return []
        return [peer for peer, _ in self.temporal_neighbors.get(addr, Counter()).most_common(self.temporal_top_k)]

    def build_sender_pos(
        self,
        addr: str,
        fallback_related: Optional[str],
        batch_placement: Dict[str, int],
        batch_related: Dict[str, Set[str]],
    ) -> SenderPosInfo:
        related_set: Set[str] = set(batch_related.get(addr, set()))

        if self.sender_pos_mode == 2:
            related_set.update(self._temporal_related(addr))

        if fallback_related:
            if self.sender_pos_mode != 1 or not related_set:
                related_set.add(fallback_related)

        related_keys = sorted(peer for peer in related_set if peer and peer != addr)
        shard_counts = [0 for _ in range(self.shards)]
        known_count = 0
        related_in_current_batch = False

        for peer in related_keys:
            if peer in batch_placement:
                sid = int(batch_placement[peer])
                if 0 <= sid < self.shards:
                    shard_counts[sid] += 1
                    known_count += 1
                    related_in_current_batch = True
                    continue

            sid = self.addr_shard.get(peer)
            if sid is not None and 0 <= sid < self.shards:
                shard_counts[sid] += 1
                known_count += 1

        major_shard = -1
        major_count = 0
        for sid, count in enumerate(shard_counts):
            if count > major_count:
                major_count = count
                major_shard = sid

        sender_pos = [0.0 for _ in range(self.shards)]
        related_weight = 0.0
        if known_count > 0:
            for sid, count in enumerate(shard_counts):
                sender_pos[sid] = float(count) / float(known_count)
            related_weight = float(major_count) / float(known_count)

        summary_keys = related_keys[:8]
        related_summary = ",".join(summary_keys)
        if len(related_keys) > len(summary_keys):
            related_summary = f"{related_summary},+{len(related_keys) - len(summary_keys)}"

        return SenderPosInfo(
            sender_pos=sender_pos,
            related_summary=related_summary,
            related_known=known_count > 0 and major_shard >= 0,
            related_shard=major_shard,
            related_weight=related_weight,
            related_in_current_batch=related_in_current_batch,
            related_count=len(related_keys),
        )

    def _safe_policy_output(
        self,
        output: PolicyOutput,
        address: str,
    ) -> PolicyOutput:
        action = int(output.action)
        if action < 0 or action >= self.shards:
            action = addr2shard(address, self.shards)
            return PolicyOutput(
                action=action,
                log_prob=0.0,
                value=0.0,
                confidence=0.0,
                entropy=0.0,
                source="safe_hash_fallback",
            )
        output.action = action
        return output

    def _place_address(
        self,
        addr: str,
        fallback_related: Optional[str],
        batch_placement: Dict[str, int],
        batch_related: Dict[str, Set[str]],
        policy: PolicyFn,
        batch_id: int,
    ) -> Optional[PlacementAction]:
        if not addr or addr in self.addr_shard:
            return None

        info = self.build_sender_pos(
            addr=addr,
            fallback_related=fallback_related,
            batch_placement=batch_placement,
            batch_related=batch_related,
        )
        state = self.build_state_from_sender_pos(
            info.sender_pos,
            address_flag_from_related_count(info.related_count),
        )

        output = self._safe_policy_output(
            policy(state, addr, info.related_summary, info),
            addr,
        )

        chosen_shard = int(output.action)
        shard_load_before = self.shard_load[chosen_shard]
        load_mean_before = (
            float(sum(self.shard_load)) / float(len(self.shard_load))
            if self.shard_load
            else 0.0
        )
        local_reward, load_penalty = local_action_reward(
            chosen_shard=chosen_shard,
            related_known=info.related_known,
            related_shard=info.related_shard,
            related_weight=info.related_weight,
            shard_load_before=shard_load_before,
            load_mean_before=load_mean_before,
        )

        self.addr_shard[addr] = chosen_shard
        self.shard_load[chosen_shard] += 1
        batch_placement[addr] = chosen_shard

        next_info = self.build_sender_pos(
            addr=addr,
            fallback_related=fallback_related,
            batch_placement=batch_placement,
            batch_related=batch_related,
        )
        next_state = self.build_state_from_sender_pos(
            next_info.sender_pos,
            address_flag_from_related_count(next_info.related_count),
        )

        same_as_related = info.related_known and chosen_shard == info.related_shard

        return PlacementAction(
            batch_id=batch_id,
            address=addr,
            related=info.related_summary,
            state=state,
            next_state=next_state,
            action=chosen_shard,
            log_prob=float(output.log_prob),
            value=float(output.value),
            confidence=float(output.confidence),
            entropy=float(output.entropy),
            local_reward=local_reward,
            related_known=info.related_known,
            related_shard=info.related_shard,
            related_weight=info.related_weight,
            related_count=info.related_count,
            sender_pos=list(info.sender_pos),
            same_as_related=same_as_related,
            related_in_current_batch=info.related_in_current_batch,
            shard_load_before=shard_load_before,
            shard_load_mean_before=load_mean_before,
            load_penalty=load_penalty,
        )

    def _update_temporal_neighbors(self, txs: Sequence[Tx]) -> None:
        if self.sender_pos_mode != 2:
            return
        for tx in txs:
            self.temporal_neighbors[tx.sender][tx.recipient] += 1
            self.temporal_neighbors[tx.recipient][tx.sender] += 1

    def _simulate_batch_loads(
        self,
        txs: Sequence[Tx],
    ) -> Tuple[List[int], List[float], List[float], int, int, int]:
        loads = [0 for _ in range(self.shards)]
        effective_loads = [0.0 for _ in range(self.shards)]
        cross_loads = [0.0 for _ in range(self.shards)]
        total_inner = 0
        total_relay1 = 0
        total_relay2 = 0

        for tx in txs:
            sender_shard = self.addr_shard.get(tx.sender)
            recipient_shard = self.addr_shard.get(tx.recipient)

            if sender_shard is None:
                sender_shard = addr2shard(tx.sender, self.shards)
                self.addr_shard[tx.sender] = sender_shard
            if recipient_shard is None:
                recipient_shard = addr2shard(tx.recipient, self.shards)
                self.addr_shard[tx.recipient] = recipient_shard

            if sender_shard == recipient_shard:
                loads[sender_shard] += 1
                effective_loads[sender_shard] += 1.0
                total_inner += 1
            else:
                loads[sender_shard] += 1
                loads[recipient_shard] += 1
                effective_loads[sender_shard] += 0.5
                effective_loads[recipient_shard] += 0.5
                cross_loads[sender_shard] += 0.5
                cross_loads[recipient_shard] += 0.5
                total_relay1 += 1
                total_relay2 += 1

        return loads, effective_loads, cross_loads, total_inner, total_relay1, total_relay2

    def _apply_capacity_backlog(
        self,
        effective_loads: Sequence[float],
        cross_loads: Sequence[float],
    ) -> Tuple[List[float], List[float], List[float], List[float], float]:
        # This is a light-weight stand-in for BlockEmulator's block capacity and
        # relay backlog. It keeps the SPRING state dimension unchanged, but makes
        # repeated overload on one shard visible through both reward and recent
        # block statistics.
        if self.capacity_backlog_mode == 0:
            zeros = [0.0 for _ in range(self.shards)]
            return (
                [float(v) for v in effective_loads],
                [float(v) for v in effective_loads],
                [float(v) for v in cross_loads],
                zeros,
                0.0,
            )

        capacity = float(max(1, self.max_block_size))
        reward_loads = [0.0 for _ in range(self.shards)]
        committed_loads = [0.0 for _ in range(self.shards)]
        committed_cross_loads = [0.0 for _ in range(self.shards)]
        next_pending = [0.0 for _ in range(self.shards)]
        next_pending_cross = [0.0 for _ in range(self.shards)]

        for sid in range(self.shards):
            incoming = float(effective_loads[sid]) if sid < len(effective_loads) else 0.0
            incoming_cross = float(cross_loads[sid]) if sid < len(cross_loads) else 0.0
            pending = self.pending_loads[sid] if sid < len(self.pending_loads) else 0.0
            pending_cross = (
                self.pending_cross_loads[sid]
                if sid < len(self.pending_cross_loads)
                else 0.0
            )

            pressure = max(0.0, pending + incoming)
            cross_pressure = max(0.0, pending_cross + incoming_cross)
            committed = min(pressure, capacity)

            commit_ratio = committed / (pressure + EPS) if pressure > EPS else 0.0
            committed_cross = min(cross_pressure * commit_ratio, committed)

            reward_loads[sid] = pressure
            committed_loads[sid] = committed
            committed_cross_loads[sid] = committed_cross
            next_pending[sid] = max(0.0, pressure - committed)
            next_pending_cross[sid] = max(0.0, cross_pressure - committed_cross)

        self.pending_loads = next_pending
        self.pending_cross_loads = next_pending_cross

        pending_total = float(sum(next_pending))
        incoming_total = float(sum(effective_loads))
        capacity_total = capacity * float(self.shards)
        backlog_penalty = clamp(
            pending_total / (incoming_total + capacity_total + EPS),
            0.0,
            1.0,
        )

        return (
            reward_loads,
            committed_loads,
            committed_cross_loads,
            list(next_pending),
            backlog_penalty,
        )

    def run_batch(
        self,
        txs: Sequence[Tx],
        policy: PolicyFn,
    ) -> Tuple[List[PlacementAction], BatchMetrics]:
        self.batch_id += 1
        batch_id = self.batch_id
        batch_related = self.build_batch_related(txs)
        batch_placement: Dict[str, int] = {}
        actions: List[PlacementAction] = []

        if self.sender_pos_mode == 1:
            for tx in txs:
                action = self._place_address(
                    tx.sender,
                    None,
                    batch_placement,
                    batch_related,
                    policy,
                    batch_id,
                )
                if action is not None:
                    actions.append(action)

            for tx in txs:
                action = self._place_address(
                    tx.recipient,
                    tx.sender,
                    batch_placement,
                    batch_related,
                    policy,
                    batch_id,
                )
                if action is not None:
                    actions.append(action)
        else:
            for tx in txs:
                sender_action = self._place_address(
                    tx.sender,
                    tx.recipient,
                    batch_placement,
                    batch_related,
                    policy,
                    batch_id,
                )
                if sender_action is not None:
                    actions.append(sender_action)

                recipient_action = self._place_address(
                    tx.recipient,
                    tx.sender,
                    batch_placement,
                    batch_related,
                    policy,
                    batch_id,
                )
                if recipient_action is not None:
                    actions.append(recipient_action)

        loads, effective_loads, cross_loads, total_inner, total_relay1, total_relay2 = (
            self._simulate_batch_loads(txs)
        )
        (
            reward_loads,
            committed_loads,
            committed_cross_loads,
            pending_loads,
            backlog_penalty,
        ) = self._apply_capacity_backlog(effective_loads, cross_loads)
        reward, reward_parts = spring_reward(
            effective_loads=effective_loads,
            cross_loads=cross_loads,
            balance_loads=reward_loads,
            lambda_weight=self.lambda_weight,
            beta=self.beta,
            load_penalty_weight=self.load_penalty_weight,
            active_shard_bonus_weight=self.active_shard_bonus_weight,
            hotspot_penalty_weight=self.hotspot_penalty_weight,
            hotspot_threshold=self.hotspot_threshold,
            min_active_load_share=self.min_active_load_share,
            backlog_penalty=backlog_penalty,
            backlog_penalty_weight=self.backlog_penalty_weight,
            reward_mode=self.reward_mode,
        )

        self.recent_stats.append(
            {
                "num": [float(v) for v in committed_loads],
                "cross": [float(v) for v in committed_cross_loads],
            }
        )
        self._update_temporal_neighbors(txs)

        block_signal = clamp(reward, -1.0, 1.0)
        action_hist = [0 for _ in range(self.shards)]
        related_known_count = 0
        same_as_related_count = 0
        local_reward_sum = 0.0
        action_reward_sum = 0.0
        confidence_sum = 0.0
        entropy_sum = 0.0
        related_shard_hist = [0 for _ in range(self.shards)]
        same_related_by_shard = [0 for _ in range(self.shards)]

        for action in actions:
            local_signal = clamp(action.local_reward, -1.0, 1.0)
            shaped = (
                self.local_reward_weight * local_signal
                + (1.0 - self.local_reward_weight) * block_signal
            )
            action.reward = self.action_reward_scale * clamp(shaped, -1.0, 1.0)
            action.done = False

            action_hist[action.action] += 1
            local_reward_sum += action.local_reward
            action_reward_sum += action.reward
            confidence_sum += action.confidence
            entropy_sum += action.entropy
            if action.related_known:
                related_known_count += 1
                if 0 <= action.related_shard < self.shards:
                    related_shard_hist[action.related_shard] += 1
            if action.same_as_related:
                same_as_related_count += 1
                if 0 <= action.related_shard < self.shards:
                    same_related_by_shard[action.related_shard] += 1

        if actions:
            actions[-1].done = True

        related_follow_by_shard = [
            (
                float(same_related_by_shard[sid]) / float(related_shard_hist[sid])
                if related_shard_hist[sid]
                else 0.0
            )
            for sid in range(self.shards)
        ]
        observed_follow = [
            related_follow_by_shard[sid]
            for sid in range(self.shards)
            if related_shard_hist[sid] > 0
        ]
        min_related_follow_ratio = min(observed_follow) if observed_follow else 0.0

        total_tx = total_inner + total_relay1 + total_relay2
        metrics = BatchMetrics(
            batch_id=batch_id,
            tx_count=len(txs),
            action_count=len(actions),
            total_tx=total_tx,
            total_inner=total_inner,
            total_relay1=total_relay1,
            total_relay2=total_relay2,
            effective_tx=float(reward_parts["effective_tx"]),
            cross_tx=float(reward_parts["cross_tx"]),
            cross_rate=float(reward_parts["cross_rate"]),
            raw_load_variance=float(reward_parts["raw_load_variance"]),
            normalized_load_variance=float(reward_parts["normalized_load_variance"]),
            r_cstr=float(reward_parts["r_cstr"]),
            r_wlb=float(reward_parts["r_wlb"]),
            abs_load_diff=float(reward_parts["abs_load_diff"]),
            active_shards=int(reward_parts["active_shards"]),
            active_shard_ratio=float(reward_parts["active_shard_ratio"]),
            max_load_share=float(reward_parts["max_load_share"]),
            hotspot_penalty=float(reward_parts["hotspot_penalty"]),
            load_aware_bonus=float(reward_parts["load_aware_bonus"]),
            backlog_penalty=float(reward_parts["backlog_penalty"]),
            reward=reward,
            loads=loads,
            effective_loads=effective_loads,
            reward_loads=reward_loads,
            committed_loads=committed_loads,
            pending_loads=pending_loads,
            cross_loads=cross_loads,
            action_hist=action_hist,
            related_known_count=related_known_count,
            same_as_related_count=same_as_related_count,
            local_reward_mean=local_reward_sum / float(len(actions)) if actions else 0.0,
            action_reward_mean=action_reward_sum / float(len(actions)) if actions else 0.0,
            confidence_mean=confidence_sum / float(len(actions)) if actions else 0.0,
            entropy_mean=entropy_sum / float(len(actions)) if actions else 0.0,
            reward_mode=str(reward_parts["reward_mode"]),
            related_shard_hist=related_shard_hist,
            same_related_by_shard=same_related_by_shard,
            related_follow_by_shard=related_follow_by_shard,
            min_related_follow_ratio=min_related_follow_ratio,
        )

        return actions, metrics
