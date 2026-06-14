from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV_PATH = ROOT_DIR / "selectedTxs_300K.csv"
DEFAULT_IOT_CSV_PATH = ROOT_DIR / "data_iot" / "selectedTxs_iot_multi_anchor_full.csv"
DEFAULT_IOT_SIDECAR_PATH = ROOT_DIR / "data_iot" / "iot_flow_sidecar_multi_anchor_full.csv"
CHECKPOINT_DIR = ROOT_DIR / "spring_lite" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "spring_ppo.pt"

DEFAULT_SHARD_NUM = 4

# SPRING paper state dimension: 11k + 1.
# It contains five recent windows of num_tx, five recent windows of cross_tx,
# k sender_pos entries, and one address-type flag.
def state_dim(shards: int, iot_feature_dim: int = 0) -> int:
    base_dim = 11 * shards + 1
    # IoT-MDP-B-lite appends current_load[k] before the compact IoT features.
    if int(iot_feature_dim) > 0:
        base_dim += shards
    return base_dim + max(0, int(iot_feature_dim))


# IoT-MDP-B-lite appends 10 compact scene features:
# anchor count, distance/link-quality stats, traffic rate, anchor-type shares,
# and a control protocol flag. For four shards this makes the PPO input
# dimension 59.
IOT_FEATURE_DIM = 10

# IoT MDP v1 奖励权重。第一阶段先保证能跑通和收敛，所以仍然以
# CSTR（跨分片率）和 balance（负载均衡）为主，通信代价和热点为辅。
IOT_CSTR_WEIGHT = 0.55
IOT_BALANCE_WEIGHT = 0.30
IOT_COMM_COST_WEIGHT = 0.10
IOT_HOTSPOT_WEIGHT = 0.05


HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
PPO_EPOCHS = 4
# Paper-aligned PPO mini-batch trigger. With tx_batch_size=1000 this usually
# aggregates several placement batches before one PPO update.
BATCH_SIZE = 1024
MIN_FLUSH_BATCH_SIZE = 256
GAE_LAMBDA = 0.95
# Mainline PPO exploration pressure. Keep this as a standard PPO parameter;
# auxiliary losses below are disabled by default for a cleaner paper-aligned
# baseline.
ENTROPY_COEF = 0.01
VALUE_CLIP = 0.2
REWARD_CLIP = 0.2
# Optional ablation only: weak supervised auxiliary loss. The main IoT PPO
# method should keep this disabled so the policy is learned from reward.
SUPERVISED_COEF = 0.0  # 弱监督辅助损失，主实验默认关闭

# 对齐 SPRING 表 1 的默认设置：lambda=0.5, beta=0.1
LAMBDA_WEIGHT = 0.5
BETA = 0.1

# PPO-v1.6 starts from a paper-aligned MDP baseline. The extra load/hotspot/
# backlog terms are kept for the later enhanced method, but the default reward
# path below only uses lambda * r_cstr + (1 - lambda) * r_wlb.
IOT_DENSE_BALANCED_REWARD_MODE = "iot_dense_balanced"
REWARD_MODE_CHOICES = (
    "paper",
    "enhanced",
    "iot",
    "iot_dense",
    IOT_DENSE_BALANCED_REWARD_MODE,
)

REWARD_MODE = "paper"  # choices: see REWARD_MODE_CHOICES
LOAD_PENALTY_WEIGHT = 0.0
LOCAL_REWARD_WEIGHT = 0.0
ACTION_REWARD_SCALE = 0.1
ACTIVE_SHARD_BONUS_WEIGHT = 0.0
HOTSPOT_PENALTY_WEIGHT = 0.0
HOTSPOT_THRESHOLD = 0.45
MIN_ACTIVE_LOAD_SHARE = 0.03
CAPACITY_BACKLOG_MODE = 0
BACKLOG_PENALTY_WEIGHT = 0.0

# C-lite+ keeps the 59-dim state unchanged, but gives under-used shards a
# bounded action-level dense reward（动作级稠密奖励）bonus（奖励项）.
IOT_DENSE_BALANCED_LOW_LOAD_BONUS_WEIGHT = 0.25
IOT_DENSE_BALANCED_LOW_LOAD_BONUS_CAP = 0.18

# Enhanced-mode defaults used for ablation/innovation experiments after the
# paper-aligned baseline is stable.
ENHANCED_LOAD_PENALTY_WEIGHT = 2.25
ENHANCED_LOCAL_REWARD_WEIGHT = 0.10
ENHANCED_ACTION_REWARD_SCALE = 0.1
ENHANCED_ACTIVE_SHARD_BONUS_WEIGHT = 0.20
ENHANCED_HOTSPOT_PENALTY_WEIGHT = 0.80
ENHANCED_CAPACITY_BACKLOG_MODE = 1
ENHANCED_BACKLOG_PENALTY_WEIGHT = 0.35

# Optional engineering guards for ablation/diagnostics, not part of the
# main paper method. With ARGMAX_TIE_EPS=0, deterministic inference uses the
# plain PPO argmax except for exact ties. With ARGMAX_BALANCE_COEF=0, PPO loss
# has no extra action-distribution regularizer.
ARGMAX_TIE_BREAK = 1
ARGMAX_TIE_EPS = 0.0
ARGMAX_BALANCE_COEF = 0.0
ARGMAX_BALANCE_TEMPERATURE = 0.20

# 0 = current bidirectional TxBatch relation graph
# 1 = paper-like recipient -> senders sender_pos
# 2 = bidirectional TxBatch graph plus temporal-neighbor memory
DEFAULT_SENDER_POS_MODE = 1

EPS = 1e-8
