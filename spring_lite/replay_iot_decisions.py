"""只读重放链上记录的实际状态/掩码，区分决策函数差异与历史反馈差异。

不生成交易、不启动链、不改历史结果。旧记录缺少原始压力时，只在计数
未被归一化裁剪且能够精确还原为整数时使用它；无法还原就明确报错。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from mechanism import input_dim, address_flag_dim
from offline_env import SpringOfflineEnv


def restore_candidate_env(config, state, shards, layout=None):
    version = config.get('SpringMechanismVersion', 'legacy')
    if len(state) != input_dim(shards, 10, version, layout) or not all(math.isfinite(v) and 0 <= v <= 1 for v in state):
        raise ValueError('Invalid recorded state/layout')
    block = int(config['BlockSize'])
    batch = int(config['TxBatchSize'])
    denom = max(block, batch)
    denom = denom if denom > 0 else 100
    counts = []
    for value in state[:5*shards]:
        if value >= 1:
            raise ValueError('Saturated historical count: raw pressure is required for exact replay')
        count = math.expm1(value * math.log1p(denom*4))
        if abs(count-round(count)) > 1e-6:
            raise ValueError('Historical state count cannot be uniquely reconstructed')
        counts.append(round(count))
    env = SpringOfflineEnv(shards=shards, max_block_size=block, tx_batch_size=batch,
        iot_feature_dim=10, candidate_top_k=config['SpringCandidateTopK'],
        capacity_guard=config['SpringCapacityGuard'],
        capacity_guard_factor=config['SpringCapacityGuardFactor'],
        candidate_load_weight=config['SpringCandidateLoadWeight'])
    # 这里仅调用候选函数；归一化的账户占比足以恢复放置压力，无需猜总账户数。
    offset = 11*shards + address_flag_dim(version, layout)
    env.shard_load = list(state[offset:offset+shards])
    env.recent_stats = [{'num':counts[i*shards:(i+1)*shards]} for i in range(5)]
    return env


def replay(run_dir, model=None):
    config = json.loads((run_dir/'paramsConfig.json').read_text(encoding='utf-8-sig'))
    context = json.loads((run_dir/'run_context.json').read_text(encoding='utf-8-sig'))
    if config.get('SpringIOTIdentityMode') != 2 or config.get('SpringIOTMode') != 1:
        raise ValueError('Replay supports explicit real-account IoT runs')
    policy = context['policy']
    if policy not in ('candidate_only', 'ppo') or config.get('SpringEvalSample', 0) != 0:
        raise ValueError('Replay requires a deterministic candidate_only or PPO run')
    shards = context['shards']
    records_path = run_dir/'spring_io'/'decision_records.jsonl'
    with records_path.open(encoding='utf-8-sig') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError('No recorded placement decisions')
    masks, decisions, examples = 0, 0, []
    for index, record in enumerate(records):
        state, mask = record['state'], record['action_mask']
        if len(mask) != shards or not any(mask) or any(v not in (0,1) for v in mask):
            raise ValueError('Invalid recorded action mask')
        expected_source = 'go_candidate_only' if policy == 'candidate_only' else 'python_ppo'
        if not record['source'].startswith(expected_source):
            raise ValueError('Recorded run contains policy fallback: '+record['source'])
        env = restore_candidate_env(config, state, shards, context.get('expected_checkpoint', {}).get('state_layout'))
        related = state[10*shards:11*shards]
        actual_mask = env.build_candidate_action_mask(record['address'], related)
        masks += int(actual_mask != mask)
        if policy == 'candidate_only':
            action = env._best_candidate_action(record['address'], related, mask)
            if action != record['shard']:
                decisions += 1
                if len(examples) < 8:
                    examples.append(dict(record=index, address=record['address'],
                                         recorded=record['shard'], replayed=action))
    if policy == 'ppo':
        from infer_server import AgentCache, infer_items
        model = model or Path(context['model'])
        if not model.is_file():
            raise FileNotFoundError('Moved model: pass --model with its current path')
        if hashlib.sha256(model.read_bytes()).hexdigest() != context['model_sha256']:
            raise ValueError('Replay model hash differs from the model used by the recorded run')
        cache = AgentCache()
        for offset in range(0, len(records), 256):
            part = records[offset:offset+256]
            outputs = infer_items(part, shards, False, 0, model, 10, False,
                                  context['expected_checkpoint'], cache)
            for local, (record, output) in enumerate(zip(part, outputs)):
                if not output['source'].startswith('python_ppo'):
                    raise ValueError('Inference replay fell back to another policy')
                if record['shard'] != output['shard']:
                    decisions += 1
                    if len(examples) < 8:
                        examples.append(dict(record=offset+local, address=record['address'],
                                             recorded=record['shard'], replayed=output['shard']))
    return dict(status='passed' if masks == decisions == 0 else 'mismatch',
        policy=policy, decisions=len(records), mask_mismatches=masks, action_mismatches=decisions,
        examples=examples, recorded_input_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
        scope='same recorded states and masks; does not equate asynchronous chain history with offline history')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir, args.model)
    document = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        # 结果只写用户指定位置；不覆盖旧的核查凭据。
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(document+'\n')
    print(document)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
