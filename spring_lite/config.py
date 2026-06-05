from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CSV_PATH = ROOT_DIR / "selectedTxs_300K.csv"
CHECKPOINT_DIR = ROOT_DIR / "spring_lite" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "spring_ppo.pt"

DEFAULT_SHARD_NUM = 4

# SPRING 论文状态维度：11k + 1
# 5k 个最近总交易数 + 5k 个最近跨片交易数 + k 个 sender_pos + 1 个地址类型标记
def state_dim(shards: int) -> int:
    return 11 * shards + 1


HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
PPO_EPOCHS = 4
# Paper-aligned PPO mini-batch trigger. With tx_batch_size=1000 this usually
# aggregates several placement batches before one PPO update.
BATCH_SIZE = 2048
MIN_FLUSH_BATCH_SIZE = 256
GAE_LAMBDA = 0.95
# PPO-v1.6.1: lower entropy pressure so the actor can move away from the
# uniform random policy after the weak related-shard supervision starts.
ENTROPY_COEF = 0.01
VALUE_CLIP = 0.2
REWARD_CLIP = 0.2
# PPO-v1.6.1: weak supervised auxiliary loss. It nudges a new address toward
# the shard where its known related addresses already live, while the paper
# reward still remains the main reinforcement-learning objective.
SUPERVISED_COEF = 0.03 #弱监督辅助损失

# 对齐 SPRING 表 1 的默认设置：lambda=0.5, beta=0.1
LAMBDA_WEIGHT = 0.5
BETA = 0.1

# PPO-v1.6 starts from a paper-aligned MDP baseline. The extra load/hotspot/
# backlog terms are kept for the later enhanced method, but the default reward
# path below only uses lambda * r_cstr + (1 - lambda) * r_wlb.
REWARD_MODE = "paper"  # choices: paper, enhanced
LOAD_PENALTY_WEIGHT = 0.0
LOCAL_REWARD_WEIGHT = 0.0
ACTION_REWARD_SCALE = 0.1
ACTIVE_SHARD_BONUS_WEIGHT = 0.0
HOTSPOT_PENALTY_WEIGHT = 0.0
HOTSPOT_THRESHOLD = 0.45
MIN_ACTIVE_LOAD_SHARE = 0.03
CAPACITY_BACKLOG_MODE = 0
BACKLOG_PENALTY_WEIGHT = 0.0

# Enhanced-mode defaults used for ablation/innovation experiments after the
# paper-aligned baseline is stable.
ENHANCED_LOAD_PENALTY_WEIGHT = 2.25
ENHANCED_LOCAL_REWARD_WEIGHT = 0.10
ENHANCED_ACTION_REWARD_SCALE = 0.1
ENHANCED_ACTIVE_SHARD_BONUS_WEIGHT = 0.20
ENHANCED_HOTSPOT_PENALTY_WEIGHT = 0.80
ENHANCED_CAPACITY_BACKLOG_MODE = 1
ENHANCED_BACKLOG_PENALTY_WEIGHT = 0.35

# Argmax collapse guard:
# ARGMAX_TIE_EPS: if action probabilities are within this gap from the best
# one, treat them as a near-tie and break it deterministically by address hash.
# ARGMAX_BALANCE_*: during PPO updates, softly penalize a batch whose sharpened
# policy distribution puts most near-argmax mass on only a few shard labels.
ARGMAX_TIE_BREAK = 1
ARGMAX_TIE_EPS = 0.05
ARGMAX_BALANCE_COEF = 0.05
ARGMAX_BALANCE_TEMPERATURE = 0.20

# 0 = current bidirectional TxBatch relation graph
# 1 = paper-like recipient -> senders sender_pos
# 2 = bidirectional TxBatch graph plus temporal-neighbor memory
DEFAULT_SENDER_POS_MODE = 1

EPS = 1e-8
