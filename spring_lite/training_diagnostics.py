"""v3 离线训练的观测量：只读计算，不重写奖励、不消耗随机数。

反事实仅指同一已访问状态的即时局部奖励，不能当作其他动作的完整回报。
GAE 分项在标准化前相加；梯度分项使用总优势的共同标准差。
"""
import hashlib

import numpy as np
import torch

from action_mask import mask_logits
from config import (EPS, IOT_DENSE_BALANCED_LOW_LOAD_BONUS_WEIGHT,
                    IOT_DENSE_BALANCED_LOW_LOAD_BONUS_CAP)
from offline_env import iot_dense_action_reward


def state_dict_sha256(net):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return dict(count=0, mean=None, std=None, min=None, p50=None, max=None)
    return dict(count=int(a.size), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), p50=float(np.median(a)), max=float(a.max()))


def value_stats(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    residual = target - prediction
    variance = float(np.var(target))
    return dict(prediction_mean=float(prediction.mean()), target_mean=float(target.mean()),
                target_std=float(np.std(target)), residual_mean=float(residual.mean()),
                rmse=float(np.sqrt(np.mean(residual**2))),
                explained_variance=(1-float(np.var(residual))/variance if variance > 1e-12 else None))


def placement_diagnostics(actions, local_weight, scale, scene_cost_mode):
    records = []
    for position, action in enumerate(actions):
        shards = len(action.action_mask)
        # v3 的账户占比保留在 full/no_iot 两种状态中。比例即可恢复 n_s / mean(n)。
        shares = action.state[11*shards:12*shards]
        mean = sum(shares)/shards
        peers = action.sender_pos
        costs = action.cost_by_shard  # no_iot 输入虽置零，真实奖励成本仍保留。
        candidates = [i for i, allowed in enumerate(action.action_mask) if allowed]
        parts = dict(related=[], cross=[], load=[], iot=[])
        raw = []
        for sid in candidates:
            base, _ = iot_dense_action_reward(
                sid, peers, 0., shares[sid], mean,
                IOT_DENSE_BALANCED_LOW_LOAD_BONUS_WEIGHT,
                IOT_DENSE_BALANCED_LOW_LOAD_BONUS_CAP)
            anchor = 1.4*max(0., min(1., peers[sid]))
            cross = -.8*(1-max(0., min(1., peers[sid]))) if sum(peers) > EPS else 0.
            cost = -.45*max(0., sum(costs)-costs[sid]) if scene_cost_mode == 'cross' else 0.
            for name, val in [('related', anchor), ('cross', cross),
                              ('load', base-anchor-cross), ('iot', cost)]:
                parts[name].append(val)
            raw.append(base+cost)
        clipped = np.clip(raw, -1., 1.)
        selected = candidates.index(action.action)
        records.append(dict(
            batch_id=action.batch_id, batch_size=len(actions), position=position,
            local_reward=scale*local_weight*float(np.clip(action.local_reward, -1., 1.))/len(actions),
            local_scale=scale*local_weight/len(actions), global_reward=0.,
            candidate_raw=raw, candidate_parts=parts,
            raw_reward_reconstruction_error=abs(raw[selected]-action.local_reward),
            candidate_clip_spread=float(np.ptp(clipped)),
            chosen_local_regret=float(max(clipped)-clipped[selected]),
            chosen_local_optimal=bool(clipped[selected] >= max(clipped)-1e-8)))
    return records


def _gradient(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                      for p, g in zip(parameters, grads)])


def _parameters(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters]).clone()


def _grad_norm(parameters):
    return float(torch.linalg.vector_norm(torch.cat([
        (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in parameters])).item())


def _cosine(a, b):
    denom = float(a.norm()*b.norm())
    return float(torch.dot(a, b))/denom if denom > 1e-20 else None


class UpdateDiagnostics:
    def __init__(self, agent, buffer, returns, advantages, next_values):
        if len(buffer.diagnostic_steps) != len(buffer):
            raise ValueError('v3 diagnostics require one accounting record per action')
        self.agent, self.buffer = agent, buffer
        self.actor, self.critic = list(agent.net.actor.parameters()), list(agent.net.critic.parameters())
        self.returns = returns
        self.steps = []
        meta = buffer.diagnostic_steps
        self.local = np.asarray([x['local_reward'] for x in meta], dtype=np.float32)
        self.global_reward = np.asarray([x['global_reward'] for x in meta], dtype=np.float32)
        zeros = np.zeros(len(buffer), dtype=np.float32)
        def gae(rewards, values, next_v):
            return agent._gae_returns(rewards, buffer.dones, values, next_v,
                                      buffer.discounts, buffer.trace_discounts)[1]
        self.components = dict(local=gae(self.local, zeros, zeros),
                               global_reward=gae(self.global_reward, zeros, zeros),
                               value=gae(zeros, np.asarray(buffer.values, dtype=np.float32), next_values))
        local_current = np.zeros_like(zeros)
        running = 0.
        for i in reversed(range(len(buffer))):
            if i == len(buffer)-1 or meta[i]['batch_id'] != meta[i+1]['batch_id']:
                running = 0.
            running += float(self.local[i])
            local_current[i] = running
        # 更新片段末端 bootstrap 的贡献是 value 分项的子集，不再重复计入总和。
        boundary = np.zeros_like(zeros)
        discount = buffer.discounts[-1] if buffer.discounts[-1] is not None else agent.gamma
        boundary[-1] = 0. if buffer.dones[-1] else discount*next_values[-1]
        for i in reversed(range(len(buffer)-1)):
            trace = buffer.trace_discounts[i]
            trace = agent.gamma*agent.gae_lambda if trace is None else trace
            boundary[i] = trace*(not buffer.dones[i])*boundary[i+1]
        raw = np.concatenate([x['candidate_raw'] for x in meta])
        position = np.asarray([(x['position']+.5)/x['batch_size'] for x in meta])
        names = list(self.components)
        stacked = np.stack(list(self.components.values()))
        centered = stacked.astype(np.float64)-stacked.mean(axis=1, keepdims=True, dtype=np.float64)
        # 仅 3 个分项，用逐元素乘积计算协方差，避免额外启动 NumPy BLAS 线程池。
        covariance = np.mean(centered[:,None,:]*centered[None,:,:], axis=2)
        self.report = dict(
            schema='v3_training_diagnostics_v1', samples=len(buffer),
            terminal=bool(buffer.dones[-1]),
            batch_ids=list(dict.fromkeys(x['batch_id'] for x in meta)),
            batch_sizes=list(dict.fromkeys((x['batch_id'], x['batch_size']) for x in meta)),
            reward=dict(local_sum=float(self.local.sum(dtype=np.float64)),
                        global_sum=float(self.global_reward.sum(dtype=np.float64)),
                        total_sum=float(sum(buffer.rewards)),
                        accounting_max_error=float(np.max(np.abs(
                            self.local+self.global_reward-np.asarray(buffer.rewards)))),
                        local_scale=stats([x['local_scale'] for x in meta]),
                        raw_candidate=stats(raw),
                        raw_candidate_parts={name: stats([v for x in meta for v in x['candidate_parts'][name]])
                                             for name in ('related', 'cross', 'load', 'iot')},
                        candidate_clip_low_fraction=float(np.mean(raw < -1)),
                        candidate_clip_high_fraction=float(np.mean(raw > 1)),
                        candidate_clip_spread=stats([x['candidate_clip_spread'] for x in meta]),
                        chosen_local_regret=stats([x['chosen_local_regret'] for x in meta]),
                        chosen_local_optimal_fraction=float(np.mean([x['chosen_local_optimal'] for x in meta])),
                        raw_reconstruction_max_error=max(x['raw_reward_reconstruction_error'] for x in meta)),
            value_before=value_stats(np.asarray(buffer.values), returns),
            advantage=dict(raw=stats(advantages), components={k:stats(v) for k,v in self.components.items()},
                           reconstruction_max_error=float(np.max(np.abs(stacked.sum(0)-advantages))),
                           covariance_names=names,
                           covariance=covariance.tolist(),
                           current_batch_local=stats(local_current),
                           later_batch_local=stats(self.components['local']-local_current),
                           boundary_bootstrap_subset_of_value=stats(boundary)))
        self.report['position_groups'] = []
        for lo, hi in [(0.,.25),(.25,.5),(.5,.75),(.75,1.)]:
            take = (position >= lo) & (position < hi)
            self.report['position_groups'].append(dict(
                interval=[lo,hi], advantage=stats(advantages[take]),
                components={k:stats(v[take]) for k,v in self.components.items()}))
        optimal = np.asarray([x['chosen_local_optimal'] for x in meta])
        self.report['local_reward_groups'] = {
            name:dict(count=int(take.sum()), advantage=stats(advantages[take]))
            for name,take in [('locally_optimal',optimal),('other',~optimal)]}
        self.report['local_reward_groups']['scope'] = 'observational across visited states; not counterfactual advantages'
        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=agent.device)

    def before_step(self, policy_loss, entropy_loss, value_loss, ratio, probs, values):
        gp = _gradient(policy_loss, self.actor)
        ge = _gradient(entropy_loss, self.actor)
        gv = _gradient(value_loss, self.critic)
        entry = dict(iteration=len(self.steps)+1, policy_grad_norm=float(gp.norm()),
                     entropy_weighted_grad_norm=float(ge.norm()),
                     value_weighted_grad_norm=float(gv.norm()),
                     policy_entropy_cosine=_cosine(gp,ge),
                     actor_combined_grad_norm=float((gp+ge).norm()),
                     value=value_stats(values.detach().cpu().numpy(), self.returns))
        if not self.steps:
            self.old_probs = probs.detach().clone()
            entry['initial_ratio_max_error'] = float((ratio.detach()-1).abs().max())
            # 分解的是未裁剪 surrogate 的一阶梯度；与真实 policy 梯度的残差同时报告。
            # 所有分项共享总优势标准差，绝不各自标准化后比较。
            denominator = self.advantages.std()+1e-8 if len(ratio)>1 else 1.
            gradients = {}
            for name, component in self.components.items():
                a = torch.tensor(component, dtype=torch.float32, device=ratio.device)
                if len(ratio)>1:
                    a = (a-a.mean())/denominator
                gradients[name] = _gradient(-(ratio*a).mean(), self.actor)
            entry['advantage_component_gradients'] = {
                name:dict(norm=float(g.norm()), cosine_with_policy=_cosine(g,gp))
                for name,g in gradients.items()}
            entry['unclipped_component_gradient_sum_error'] = float((sum(gradients.values())-gp).norm())
        self.old_actor, self.old_critic = _parameters(self.actor), _parameters(self.critic)
        self.steps.append(entry)

    def before_clip(self):
        self.steps[-1].update(actor_pre_clip_norm=_grad_norm(self.actor),
                              critic_pre_clip_norm=_grad_norm(self.critic),
                              total_pre_clip_norm=_grad_norm(self.actor+self.critic))

    def after_step(self):
        entry = self.steps[-1]
        post = _grad_norm(self.actor+self.critic)
        pre = entry['total_pre_clip_norm']
        entry.update(actor_post_clip_norm=_grad_norm(self.actor),
                     critic_post_clip_norm=_grad_norm(self.critic), total_post_clip_norm=post,
                     measured_clip_scale=post/pre if pre > 0 else 1.,
                     actor_parameter_delta_norm=float((_parameters(self.actor)-self.old_actor).norm()),
                     critic_parameter_delta_norm=float((_parameters(self.critic)-self.old_critic).norm()))

    def finish(self, states, masks):
        with torch.no_grad():
            logits, values = self.agent.net(states)
            current = torch.softmax(mask_logits(logits,masks),dim=-1)
            tiny = torch.finfo(current.dtype).tiny
            kl = (self.old_probs*(self.old_probs.clamp_min(tiny).log()-current.clamp_min(tiny).log())).sum(-1)
        self.report['value_after'] = value_stats(values.cpu().numpy(), self.returns)
        self.report['gradient_steps'] = self.steps
        self.report['policy_full_kl_after'] = float(kl.mean())
        self.report['policy_before'] = self.policy_stats(self.old_probs, states, masks)
        self.report['policy_after'] = self.policy_stats(current, states, masks)
        return self.report

    def policy_stats(self, probs, states, masks):
        p, s, m = probs.detach().cpu().numpy(), states.detach().cpu().numpy(), masks.cpu().numpy()>0
        related = s[:,10*self.agent.action_dim:11*self.agent.action_dim]>EPS
        known = related.any(1)
        sorted_p = np.sort(p,axis=1)
        related_prob = (p*related).sum(1)
        uniform = (related*m).sum(1)/m.sum(1)
        return dict(max_probability=stats(p.max(1)),
                    top_two_gap=stats(sorted_p[:,-1]-sorted_p[:,-2]) if p.shape[1]>1 else stats([]),
                    entropy_mean=float(-(p*np.log(np.maximum(p,1e-30))).sum(1).mean()),
                    known_count=int(known.sum()), related_probability=stats(related_prob[known]),
                    uniform_candidate_related_probability=stats(uniform[known]))
