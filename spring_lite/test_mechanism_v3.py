"""机制方向与回报边界测试；不把旧结果当成新版训练效果。"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from offline_env import SpringOfflineEnv, Tx, PolicyOutput
from ppo import PPOAgent
from continuous_rollout import ContinuousRollout
from checkpoint_compat import checkpoint_config_mismatches, resolve_checkpoint_path
from mechanism import (validate_args, load_statistics, input_dim, metadata,
                       V3_STATE_LAYOUT, V3_PREVIOUS_STATE_LAYOUT)
from replay_iot_decisions import restore_candidate_env


def action(state=1):
    return SimpleNamespace(state=[state], next_state=[state+1], action=0, log_prob=0.,
                           local_reward=0., value=0., action_mask=[1], source='python_ppo')


class FakeAgent:
    gamma = .99
    gae_lambda = .95
    def __init__(self):
        self.saved = []
    def update(self, buffer):
        self.saved.append(dict(rewards=list(buffer.rewards), dones=list(buffer.dones),
                               next_states=[list(s) for s in buffer.next_states],
                               discounts=list(buffer.discounts)))
        return {'num_samples':len(buffer)}


class MechanismTests(unittest.TestCase):
    def test_only_address_flag_is_removed_and_control_feature_is_retained(self):
        legacy = SpringOfflineEnv(shards=16, iot_feature_dim=10)
        current = SpringOfflineEnv(shards=16, iot_feature_dim=10, mechanism_version='v3',
                                   scene_cost_mode='cross', reward_mode='iot_dense_balanced')
        features = [.25,.2,.1,.8,.3,1,0,0,0,.75]
        current.current_cost_vector = [.4,.6] + [0.]*14
        old_state = legacy.build_state_from_sender_pos([1.]+[0.]*15, 1, features) + current.current_cost_vector
        new_state = current.build_state_from_sender_pos([1.]+[0.]*15, 1, features)
        self.assertEqual(len(old_state), 219)
        self.assertEqual(len(new_state), 218)
        self.assertEqual(new_state, old_state[:176]+old_state[177:])
        self.assertEqual(new_state, current.build_state_from_sender_pos([1.]+[0.]*15, 0, features))
        self.assertEqual(new_state[201], .75)  # 第 202 维控制协议占比没有被删除。
        self.assertEqual(input_dim(16,10,'v3',V3_PREVIOUS_STATE_LAYOUT),219)
        self.assertEqual(input_dim(16,10,'v3'),218)
        self.assertEqual(input_dim(16,10),203)
        expected = metadata(SimpleNamespace(mechanism_version='v3'))
        self.assertEqual(expected['state_layout'], V3_STATE_LAYOUT)
        old_metadata = dict(expected, state_layout=V3_PREVIOUS_STATE_LAYOUT)
        self.assertTrue(checkpoint_config_mismatches(old_metadata, expected))

    def test_replay_uses_layout_to_find_current_load_columns(self):
        cfg = dict(SpringMechanismVersion='v3', BlockSize=1000, TxBatchSize=1000,
                   SpringCandidateTopK=1, SpringCapacityGuard=0,
                   SpringCapacityGuardFactor=1.5, SpringCandidateLoadWeight=1)
        state = [0.]*218
        state[176:192] = [.25,.75]+[0.]*14
        old = state[:176]+[1.]+state[176:]
        new_env = restore_candidate_env(cfg,state,16,V3_STATE_LAYOUT)
        old_env = restore_candidate_env(cfg,old,16,V3_PREVIOUS_STATE_LAYOUT)
        self.assertEqual(new_env._candidate_load_pressures(), old_env._candidate_load_pressures())
        with self.assertRaisesRegex(ValueError, 'state/layout'):
            restore_candidate_env(cfg,old,16,V3_STATE_LAYOUT)

    def scenario(self, selected, feature='full', cost='cross', version='v3'):
        env = SpringOfflineEnv(shards=2, iot_feature_dim=10, sender_pos_mode=1,
                               mechanism_version=version, feature_mode=feature,
                               scene_cost_mode=cost, reward_mode='iot_dense_balanced')
        env.addr_shard.update(b=0,c=1)
        env.shard_load[:] = [1,1]
        txs = [Tx('a', peer, communication_cost_weight=value, real_accounts=True,
                  iot_features=(.25,.5,.5,.2,.7,1,0,0,0,0),
                  recipient_iot_features=(.25,.5,.5,.2,.7,1,0,0,0,0))
               for peer,value in [('b',.9),('c',.1)]]
        return env.run_batch(txs, lambda *unused: PolicyOutput(action=selected, source='python_ppo'))

    def test_high_cost_peer_direction_and_stage_work(self):
        a,m = self.scenario(0)
        b,n = self.scenario(1)
        self.assertGreater(a[0].local_reward,b[0].local_reward)
        self.assertAlmostEqual(m.communication_cost,.05)
        self.assertAlmostEqual(n.communication_cost,.45)
        self.assertEqual(m.reward_loads,[2.,1.])
        self.assertEqual(m.effective_loads,[1.5,.5])
        self.assertEqual(a[0].cost_by_shard,[.9,.1])
        self.assertFalse(a[0].done)

    def test_independent_ablation_preserves_layout_and_reward(self):
        full,m = self.scenario(0)
        masked,n = self.scenario(0,'no_iot')
        off,o = self.scenario(0,cost='off')
        self.assertEqual(len(full[0].state),len(masked[0].state))
        self.assertEqual(full[0].state[:-11],masked[0].state[:-11])
        self.assertEqual(masked[0].state[-11:],[0.]*11)
        self.assertAlmostEqual(m.reward,n.reward)
        self.assertAlmostEqual(full[0].local_reward,masked[0].local_reward)
        self.assertEqual(off[0].state,full[0].state)
        self.assertEqual(o.communication_cost,0)

    def test_legacy_cost_and_terminal_unchanged(self):
        a,m = self.scenario(0,cost='legacy',version='legacy')
        self.assertTrue(a[0].done)
        self.assertAlmostEqual(m.communication_cost,.5)
        self.assertEqual(m.reward_loads,m.effective_loads)

    def test_later_batch_changes_prior_return(self):
        agent=object.__new__(PPOAgent); agent.gamma=.99;agent.gae_lambda=.95
        before=agent._gae_returns([.1,.2],[False,True],[0,0],[0,0])[0]
        after=agent._gae_returns([.1,.8],[False,True],[0,0],[0,0])[0]
        self.assertGreater(after[0],before[0])
        terminated=agent._gae_returns([.1,.8],[True,True],[0,0],[0,0])[0]
        self.assertAlmostEqual(terminated[0],.1)

    def test_revisit_reward_and_real_bootstrap_before_update(self):
        agent=FakeAgent(); collector=ContinuousRollout(agent,1,2)
        collector.add_batch([action()],.2)
        collector.add_batch([],.4)
        self.assertEqual(len(agent.saved),0)
        collector.before_decision([9])
        record=agent.saved[0]
        self.assertAlmostEqual(record['rewards'][0],.035*(.2+.99*.4))
        self.assertAlmostEqual(record['discounts'][0],.99**2)
        self.assertEqual(record['dones'],[False])
        self.assertEqual(record['next_states'],[[9]])
        collector.add_batch([action(9)],.3)
        collector.finish()
        self.assertEqual(agent.saved[1]['dones'],[True])

    def test_batch_global_reward_not_multiplied_by_new_accounts(self):
        for count in (1,7):
            agent=FakeAgent();collector=ContinuousRollout(agent)
            collector.add_batch([action(i) for i in range(count)],.8)
            collector.finish()
            self.assertAlmostEqual(sum(agent.saved[0]['rewards']),.035*.8)

    def test_logged_action_reward_matches_collector_and_revisits_are_audited(self):
        actions, metrics = self.scenario(0)
        collector = ContinuousRollout(FakeAgent())
        collector.add_batch(actions, metrics.reward)
        self.assertAlmostEqual(sum(a.reward for a in actions), sum(collector.buffer.rewards))
        collector.add_batch([], .5)
        self.assertEqual(collector.summary()['revisit_only_batches'], 1)
        self.assertAlmostEqual(collector.summary()['revisit_global_reward_sum'], .0175)

    def test_invalid_v3_configuration_fails_and_stage_fairness_is_explicit(self):
        args = dict(mechanism_version='v3', tx_identity='iot_v2', mdp_mode='iot')
        for change in ({'capacity_backlog_mode':1}, {'supervised_coef':.1},
                       {'reward_mode':'paper'}, {'action_reward_scale':float('nan')}):
            with self.assertRaises(ValueError):
                validate_args(SimpleNamespace(**dict(args, **change)))
        self.assertAlmostEqual(load_statistics([2,1])['jain_fairness'], .9)
        self.assertIsNone(load_statistics([0,0])['jain_fairness'])

    def test_checkpoint_version_and_moved_manifest(self):
        self.assertTrue(checkpoint_config_mismatches({'mechanism_version':'v3'},{}))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'epoch.pt').write_bytes(b'fixture')
            self.assertEqual(resolve_checkpoint_path(root/'manifest.json','epoch.pt'),root/'epoch.pt')
            self.assertEqual(resolve_checkpoint_path(root/'manifest.json',str(root/'gone'/'epoch.pt')),root/'epoch.pt')

    def test_replay_restores_pressure_and_rejects_clipped_counts(self):
        import math
        cfg = dict(BlockSize=1000, TxBatchSize=1000, SpringCandidateTopK=1,
                   SpringCapacityGuard=0, SpringCapacityGuardFactor=1.5,
                   SpringCandidateLoadWeight=1)
        state = [0.] * 35  # 两分片旧 IoT 状态：12*S + 1 + 10。
        state[0] = math.log1p(10)/math.log1p(4000)
        state[23:25] = [.25,.75]
        restored = restore_candidate_env(cfg, state, 2)
        self.assertEqual(restored._candidate_load_pressures(), [252.,750.])
        state[0] = 1
        with self.assertRaisesRegex(ValueError, 'Saturated'):
            restore_candidate_env(cfg, state, 2)

    def test_online_entry_cannot_reset_v3_even_when_version_is_omitted(self):
        import torch
        from update_online import run_update
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/'v3.pt'
            torch.save({'extra':{'mechanism_version':'v3'}}, model)
            original = model.read_bytes()
            data = root/'update.json'
            data.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'v3 checkpoint'):
                run_update(data, model, root/'log.jsonl')
            self.assertEqual(model.read_bytes(), original)
            self.assertFalse((root/'log.jsonl').exists())
            from infer_batch import load_agent
            with self.assertRaisesRegex(ValueError, 'must not reset'):
                load_agent(16, model)
            self.assertEqual(model.read_bytes(), original)

    def test_frozen_wrong_state_cannot_fall_back(self):
        from infer_server import infer_items
        cache = SimpleNamespace(get_agent=lambda *args: (FakeAgent(), 'python_ppo'))
        with self.assertRaisesRegex(RuntimeError, 'input dimension mismatch'):
            infer_items([{'state':[0.]*203}], 16, False, 0, Path('unused'), 10, False,
                        {'mechanism_version':'v3'}, cache)

    def test_v3_model_selection_retains_cost_tradeoff(self):
        from train_offline import checkpoint_dominates
        from select_pareto_checkpoint import select_candidate_decision
        a = dict(mechanism_version='v3', summary=dict(cross_ratio=.1,
            stage_max_load_share=.2, active_shards_mean=12, communication_cost_mean=.2))
        b = dict(mechanism_version='v3', summary=dict(cross_ratio=.11,
            stage_max_load_share=.2, active_shards_mean=12, communication_cost_mean=.1))
        self.assertFalse(checkpoint_dominates(a,b))
        self.assertFalse(checkpoint_dominates(b,a))
        candidates = [dict(x['summary'], checkpoint_path=f'{i}.pt', validation_score=score)
                      for i,(x,score) in enumerate([(a,.4),(b,.5)])]
        new = select_candidate_decision(dict(mechanism_version='v3',pareto_frontier=candidates),8,.35)
        old = select_candidate_decision(dict(pareto_frontier=candidates),8,.35)
        self.assertEqual(new['selected']['checkpoint_path'],'1.pt')
        self.assertEqual(old['selected']['checkpoint_path'],'0.pt')


if __name__ == '__main__':
    unittest.main()
