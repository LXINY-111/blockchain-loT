"""连续放置轨迹：按交易批次折扣，纯回访批的执行奖励也进入回报。

同一批内部的放置不推进执行时钟。每批全局奖励只记一次，局部奖励取
动作均值。只有 epoch 的有限数据窗口结束才终止；更新片段使用下一次
真实决策状态 bootstrap（后续价值估计），更新后才采样下一策略动作。
"""
import numpy as np
from ppo import RolloutBuffer


class ContinuousRollout:
    def __init__(self, agent, min_actions=1024, min_batches=8, local_weight=.65, scale=.1,
                 scene_cost_mode='cross'):
        self.agent = agent
        self.buffer = RolloutBuffer()
        self.min_actions, self.min_batches = min_actions, min_batches
        self.local_weight, self.scale = local_weight, scale
        self.diagnostics = getattr(agent, 'diagnostics', False)
        self.scene_cost_mode = scene_cost_mode
        self.batches = 0
        self.pending = False
        self.elapsed = 0
        self.updates = []
        self.executed_batches = 0
        self.revisit_only_batches = 0
        self.global_reward_sum = 0.
        self.revisit_global_reward_sum = 0.

    def before_decision(self, state):
        if not self.pending:
            return
        self.buffer.next_states[-1] = np.asarray(state, dtype=np.float32)
        self.buffer.discounts[-1] = self.agent.gamma ** self.elapsed
        self.buffer.trace_discounts[-1] = (self.agent.gamma * self.agent.gae_lambda) ** self.elapsed
        self.pending = False
        if len(self.buffer) >= self.min_actions and self.batches >= self.min_batches:
            self._update()

    def add_batch(self, actions, batch_reward):
        self.executed_batches += 1
        self.batches += 1
        global_reward = self.scale * (1-self.local_weight) * float(np.clip(batch_reward, -1, 1))
        self.global_reward_sum += global_reward
        if not actions:
            self.revisit_only_batches += 1
            self.revisit_global_reward_sum += global_reward
            if not self.pending:
                raise ValueError('Execution reward has no preceding placement transition')
            self.buffer.rewards[-1] += self.agent.gamma ** self.elapsed * global_reward
            if self.diagnostics:
                self.buffer.diagnostic_steps[-1]['global_reward'] += self.agent.gamma ** self.elapsed * global_reward
            self.elapsed += 1
            return
        if self.pending:
            raise ValueError('Call before_decision with the real next state before sampling actions')
        if self.diagnostics:
            from training_diagnostics import placement_diagnostics
            records = placement_diagnostics(actions, self.local_weight, self.scale,
                                            self.scene_cost_mode)
        for i, action in enumerate(actions):
            if not action.source.startswith('python_ppo'):
                raise ValueError('Continuous PPO cannot silently skip non-PPO actions')
            reward = self.scale*self.local_weight*float(np.clip(action.local_reward, -1, 1))/len(actions)
            if i == len(actions)-1:
                reward += global_reward
            self.buffer.add(state=action.state, next_state=actions[i+1].state if i+1<len(actions) else action.next_state,
                            action=action.action, log_prob=action.log_prob, reward=reward,
                            done=False, value=action.value, action_mask=action.action_mask,
                            discount=1., trace_discount=1.)
            if self.diagnostics:
                records[i]['global_reward'] = global_reward if i == len(actions)-1 else 0.
                self.buffer.diagnostic_steps.append(records[i])
        self.pending = True
        self.elapsed = 1

    def _update(self):
        self.updates.append(self.agent.update(self.buffer))
        self.buffer.clear()
        self.batches = 0

    def finish(self):
        if len(self.buffer):
            self.buffer.dones[-1] = True
            self.buffer.discounts[-1] = 0.
            self.buffer.trace_discounts[-1] = 0.
            self._update()
        self.pending = False

    def summary(self):
        return dict(executed_batches=self.executed_batches,
                    revisit_only_batches=self.revisit_only_batches,
                    global_reward_sum=self.global_reward_sum,
                    revisit_global_reward_sum=self.revisit_global_reward_sum,
                    effective_local_weight=self.local_weight, reward_scale=self.scale,
                    discount_clock='transaction_batch', global_reward_multiplicity=1)
