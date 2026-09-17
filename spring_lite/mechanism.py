"""版本化的机制配置。旧模型保持旧语义，新实验必须显式选择 v3。"""
import math
from config import state_dim

# v3.1 仅移除账户 flag，控制协议占比仍属于十维 IoT 场景输入。
# 旧布局只供历史记录/冻结模型重放；新训练与链上配置统一使用当前布局。
V3_STATE_LAYOUT = 'iot_cost_by_shard_no_address_flag_v3_1'
V3_PREVIOUS_STATE_LAYOUT = 'iot_cost_by_shard_v3'


def add_mechanism_args(parser):
    parser.add_argument('--mechanism_version', '--mechanism-version', choices=['legacy', 'v3'], default='legacy')
    parser.add_argument('--feature_mode', '--feature-mode', choices=['full', 'no_iot'], default='full')
    parser.add_argument('--scene_cost_mode', '--scene-cost-mode', choices=['auto', 'off', 'cross'], default='auto')
    parser.add_argument('--rollout_batches', type=int, default=8,
                        help='v3: minimum complete transaction batches before a PPO update')


def settings(args):
    version = getattr(args, 'mechanism_version', 'legacy')
    features = getattr(args, 'feature_mode', 'full')
    cost = getattr(args, 'scene_cost_mode', 'auto')
    if version not in ('legacy', 'v3') or features not in ('full', 'no_iot'):
        raise ValueError('Invalid mechanism/feature mode')
    if cost == 'auto':
        cost = 'cross' if version == 'v3' else 'legacy'
    if version == 'legacy' and (features != 'full' or cost != 'legacy'):
        raise ValueError('Independent ablation switches require --mechanism_version v3')
    if version == 'v3' and cost not in ('cross', 'off'):
        raise ValueError('v3 scene_cost_mode must be cross or off')
    return dict(mechanism_version=version, feature_mode=features, scene_cost_mode=cost)


def metadata(args):
    result = settings(args)
    # Missing version in a historical checkpoint means legacy, not v3.
    if result['mechanism_version'] == 'legacy':
        return {}
    return dict(result, trajectory_version='batch_clock_continuous_v1',
                reward_load_basis='stage', state_layout=V3_STATE_LAYOUT)


def address_flag_dim(version='legacy', layout=None):
    if version == 'legacy':
        return 1
    if version != 'v3' or layout not in (None, V3_STATE_LAYOUT, V3_PREVIOUS_STATE_LAYOUT):
        raise ValueError('Unknown mechanism/state layout')
    return int(layout == V3_PREVIOUS_STATE_LAYOUT)


def input_dim(shards, iot_feature_dim, version='legacy', layout=None):
    flag_dim = address_flag_dim(version, layout)
    return state_dim(shards, iot_feature_dim) + (shards - 1 + flag_dim if version == 'v3' else 0)


def validate_args(args):
    config = settings(args)
    if config['mechanism_version'] == 'v3':
        if getattr(args, 'tx_identity', None) != 'iot_v2' or getattr(args, 'mdp_mode', None) != 'iot':
            raise ValueError('v3 requires explicit --tx_identity iot_v2 --mdp_mode iot')
        if getattr(args, 'capacity_backlog_mode', 0) != 0:
            raise ValueError('v3 does not treat the legacy batch-capacity approximation as a timed queue')
        if getattr(args, 'keep_placement', False):
            raise ValueError('v3 finite training episodes reset placement between epochs')
        if getattr(args, 'rollout_batches', 8) < 1:
            raise ValueError('rollout_batches must be positive')
        if getattr(args, 'reward_mode', 'iot_dense_balanced') != 'iot_dense_balanced':
            raise ValueError('v3 requires reward_mode iot_dense_balanced')
        if not 0 <= getattr(args, 'local_reward_weight', .65) <= 1:
            raise ValueError('local_reward_weight must be in [0, 1]')
        scale = getattr(args, 'action_reward_scale', .1)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('action_reward_scale must be finite and positive')
        if getattr(args, 'supervised_coef', 0) != 0:
            raise ValueError('v3 continuous PPO currently requires supervised_coef=0')
    return config


def load_statistics(loads):
    """完整窗口的阶段负载统计；空窗口不伪装成完美公平。"""
    values = [float(v) for v in loads]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Load vector must be nonempty, finite and nonnegative')
    total = sum(values)
    mean = total / len(values)
    variance = sum((v-mean)**2 for v in values) / len(values)
    return dict(loads=values, total=total, variance=variance,
                normalized_variance=variance/(mean*mean) if mean else None,
                jain_fairness=total*total/(len(values)*sum(v*v for v in values)) if total else None,
                max_share=max(values)/total if total else None)
