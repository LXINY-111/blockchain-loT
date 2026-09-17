"""诊断必须保持原优化结果与随机数序列，并正确分解跨批奖励。"""
import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from continuous_rollout import ContinuousRollout
from mechanism import input_dim
from offline_env import SpringOfflineEnv, PolicyOutput, Tx
from ppo import PPOAgent
from train_offline import evaluate_agent_sampled
from training_diagnostics import placement_diagnostics, value_stats


class TrainingDiagnosticsTests(unittest.TestCase):
    def fixture(self, feature='full', cost='cross', terminal=True):
        torch.manual_seed(7)
        agent = PPOAgent(input_dim(2,10,'v3'),2,hidden_dim=8,diagnostics=True)
        env = SpringOfflineEnv(shards=2,tx_batch_size=4,iot_feature_dim=10,
                               mechanism_version='v3',feature_mode=feature,scene_cost_mode=cost,
                               sender_pos_mode=1,reward_mode='iot_dense_balanced',candidate_top_k=2)
        collector = ContinuousRollout(agent,10000,8,scene_cost_mode=cost)
        def policy(state, address, related, info):
            collector.before_decision(state)
            a, lp, value = agent.select_action(state,info.action_mask)
            return PolicyOutput(a,lp,value,source='python_ppo')
        for b in range(3):
            txs = [Tx(f'{b*4+i+10:040x}',f'{i+1:040x}',real_accounts=True,
                      communication_cost_weight=.2+.1*i,
                      iot_features=(.2,.3,.4,.5,.6,1,0,0,0,0),
                      recipient_iot_features=(.3,.4,.5,.6,.7,1,0,0,0,0)) for i in range(4)]
            actions, metrics = env.run_batch(txs,policy)
            collector.add_batch(actions,metrics.reward)
        if terminal:
            collector.buffer.dones[-1] = True
            collector.buffer.discounts[-1] = 0.
            collector.buffer.trace_discounts[-1] = 0.
        return agent, collector, actions

    def test_diagnostics_do_not_change_parameters_optimizer_or_rng(self):
        enabled, collector, _ = self.fixture()
        disabled = copy.deepcopy(enabled)
        disabled.diagnostics = False
        before = torch.get_rng_state().clone()
        normal = disabled.update(copy.deepcopy(collector.buffer))
        observed = enabled.update(copy.deepcopy(collector.buffer))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        report = observed.pop('diagnostics')
        self.assertEqual(normal,observed)
        for left,right in zip(disabled.net.parameters(),enabled.net.parameters()):
            self.assertTrue(torch.equal(left,right))
        for left,right in zip(disabled.optimizer.state.values(),enabled.optimizer.state.values()):
            for key in left:
                self.assertTrue(torch.equal(left[key],right[key]))
        self.assertLess(report['advantage']['reconstruction_max_error'],2e-6)
        self.assertLess(report['reward']['accounting_max_error'],1e-7)
        self.assertLess(report['reward']['raw_reconstruction_max_error'],1e-6)
        self.assertLess(report['gradient_steps'][0]['unclipped_component_gradient_sum_error'],2e-6)
        self.assertTrue(report['terminal'])
        self.assertEqual(len(report['gradient_steps']),4)
        self.assertEqual(report['advantage']['boundary_bootstrap_subset_of_value']['max'],0.)
        json.dumps(report,allow_nan=False)

    def test_candidate_rewards_reconstruct_full_no_iot_and_cost_off(self):
        for feature,cost in [('full','cross'),('no_iot','cross'),('full','off')]:
            with self.subTest(feature=feature,cost=cost):
                _,_,actions = self.fixture(feature,cost)
                records = placement_diagnostics(actions,.65,.1,cost)
                self.assertLess(max(x['raw_reward_reconstruction_error'] for x in records),1e-6)
                if cost=='off':
                    self.assertTrue(all(v==0 for x in records for v in x['candidate_parts']['iot']))

    def test_revisit_discount_is_counted_in_global_component(self):
        _,collector,_ = self.fixture(terminal=False)
        before = collector.buffer.diagnostic_steps[-1].copy()
        collector.add_batch([],.2)
        collector.add_batch([],-.3)
        last = collector.buffer.diagnostic_steps[-1]
        self.assertEqual(before['local_reward'],last['local_reward'])
        self.assertAlmostEqual(last['global_reward']-before['global_reward'],.035*(.99*.2-.99**2*.3))
        self.assertAlmostEqual(last['local_reward']+last['global_reward'],collector.buffer.rewards[-1])
        collector.before_decision(collector.buffer.states[0])
        collector.finish()
        report = collector.updates[0]['diagnostics']
        self.assertLess(report['reward']['accounting_max_error'],1e-7)
        self.assertLess(report['advantage']['reconstruction_max_error'],2e-6)
        self.assertEqual(len(collector.buffer.diagnostic_steps),0)

    def test_sampled_validation_restores_all_cpu_rngs(self):
        random.seed(6); np.random.seed(6); torch.manual_seed(6)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        def consume(*args,**kwargs):
            self.assertTrue(kwargs['sample'])
            return dict(python=random.random(),numpy=float(np.random.random()),torch=float(torch.rand(())))
        with patch('train_offline.evaluate_agent_argmax',side_effect=consume):
            first = evaluate_agent_sampled(None,SimpleNamespace(),[],1007)
            second = evaluate_agent_sampled(None,SimpleNamespace(),[],1007)
        self.assertEqual(first,second)
        self.assertEqual(python_before,random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1],np.random.get_state()[1]))
        self.assertEqual(numpy_before[2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))

    def test_explained_variance_undefined_for_constant_targets(self):
        self.assertIsNone(value_stats(np.array([1.,2.]),np.array([3.,3.]))['explained_variance'])

    def test_nonterminal_bootstrap_and_advantage_accounting(self):
        agent,collector,_ = self.fixture(terminal=False)
        collector.before_decision(collector.buffer.states[0])
        report = agent.update(collector.buffer)['diagnostics']
        self.assertFalse(report['terminal'])
        self.assertLess(report['advantage']['reconstruction_max_error'],2e-6)
        self.assertGreater(abs(report['advantage']['boundary_bootstrap_subset_of_value']['mean']),1e-8)

    def test_cli_records_diagnostics_and_isolated_sampled_validation(self):
        from test_prepare_iot_v2_dataset import generate, sources
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset,csv = generate(root,sources(root),'dataset',count=16)
            log = root/'train.jsonl'
            command = [sys.executable,'-B',str(Path(__file__).with_name('train_offline.py')),
                       '--csv',str(csv),'--sidecar',str(dataset/'transaction_scene.csv'),
                       '--model',str(root/'best.pt'),'--last_model',str(root/'last.pt'),
                       '--log_jsonl',str(log),'--mechanism_version','v3','--mdp_mode','iot',
                       '--tx_identity','iot_v2','--reward_mode','iot_dense_balanced',
                       '--shards','2','--candidate_top_k','2','--epochs','1',
                       '--max_txs','8','--validation_start_tx','8','--validation_max_txs','8',
                       '--tx_batch_size','4','--rollout_batches','2','--batch_size','1',
                       '--hidden_dim','8','--diagnostics','--diagnostic_eval_seeds','1007']
            result = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records[0]['event'],'diagnostic_start')
            self.assertEqual(len(records[0]['initial_parameters_sha256']),64)
            updates = [r for r in records if r['event']=='continuous_update']
            self.assertTrue(updates[-1]['loss_info']['diagnostics']['terminal'])
            self.assertEqual(len([r for r in records if r['event']=='validation_sampled']),1)


if __name__ == '__main__':
    unittest.main()
