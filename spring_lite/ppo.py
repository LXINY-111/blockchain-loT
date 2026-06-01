from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import ActorCritic


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    target_actions: List[int] = field(default_factory=list)
    target_weights: List[float] = field(default_factory=list)

    def add(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        next_state=None,
        target_action=-1,
        target_weight=0.0,
    ):
        clean_state = np.asarray(state, dtype=np.float32)
        self.states.append(clean_state)
        if next_state is None:
            self.next_states.append(clean_state.copy())
        else:
            self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.target_actions.append(int(target_action))
        self.target_weights.append(float(target_weight))

    def clear(self):
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.target_actions.clear()
        self.target_weights.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        value_clip: float = 0.2,
        supervised_coef: float = 0.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.value_clip = value_clip
        self.supervised_coef = supervised_coef
        self.device = torch.device(device)

        self.net = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, state) -> Tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action(state_t)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def deterministic_action(self, state) -> Tuple[int, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, confidence, _ = self.net.act_deterministic(state_t)

        return int(action.item()), float(confidence.item())

    def _gae_returns(self, rewards, dones, values, next_values):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for idx in reversed(range(len(rewards))):
            non_terminal = 0.0 if dones[idx] else 1.0
            delta = rewards[idx] + self.gamma * next_values[idx] * non_terminal - values[idx]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[idx] = gae

        returns = advantages + np.asarray(values, dtype=np.float32)
        return returns.astype(np.float32), advantages.astype(np.float32)

    def update(self, buffer: RolloutBuffer):
        if len(buffer) == 0:
            return {"num_samples": 0, "message": "empty_buffer"}

        states = torch.tensor(
            np.asarray(buffer.states),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            buffer.actions,
            dtype=torch.long,
            device=self.device,
        )
        next_states = torch.tensor(
            np.asarray(buffer.next_states),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            buffer.log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            buffer.values,
            dtype=torch.float32,
            device=self.device,
        )
        target_actions = torch.tensor(
            buffer.target_actions,
            dtype=torch.long,
            device=self.device,
        )
        target_weights = torch.tensor(
            buffer.target_weights,
            dtype=torch.float32,
            device=self.device,
        )
        supervised_mask = (target_actions >= 0) & (target_weights > 0)
        supervised_target_count = int(supervised_mask.sum().item())
        supervised_weight_mean = (
            float(target_weights[supervised_mask].mean().item())
            if supervised_target_count > 0
            else 0.0
        )

        # GAE keeps the block reward propagation while using next_state values,
        # which gives a lower-variance advantage estimate for PPO.
        with torch.no_grad():
            _, bootstrap_next_values = self.net(next_states)

        returns_np, advantages_np = self._gae_returns(
            rewards=np.asarray(buffer.rewards, dtype=np.float32),
            dones=np.asarray(buffer.dones, dtype=bool),
            values=np.asarray(buffer.values, dtype=np.float32),
            next_values=bootstrap_next_values.detach().cpu().numpy(),
        )
        returns = torch.tensor(
            returns_np,
            dtype=torch.float32,
            device=self.device,
        )

        # advantage = GAE(delta_t), using next_state value estimates.
        advantages_raw = torch.tensor(
            advantages_np,
            dtype=torch.float32,
            device=self.device,
        )

        adv_mean_before_norm = float(advantages_raw.mean().item())
        adv_std_before_norm = float(advantages_raw.std().item()) if len(buffer) > 1 else 0.0
        return_mean = float(returns.mean().item())
        return_std = float(returns.std().item()) if len(buffer) > 1 else 0.0

        if len(buffer) > 1:
            advantages = (advantages_raw - advantages_raw.mean()) / (
                advantages_raw.std() + 1e-8
            )
        else:
            advantages = advantages_raw

        last_loss = {}
        last_approx_kl = 0.0
        last_clip_fraction = 0.0
        last_supervised_loss = 0.0

        for _ in range(self.ppo_epochs):
            log_probs, entropy, values = self.net.evaluate_actions(states, actions)
            logits, _ = self.net(states)

            ratio = torch.exp(log_probs - old_log_probs)
            with torch.no_grad():
                last_approx_kl = float((old_log_probs - log_probs).mean().item())
                last_clip_fraction = float(
                    ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                )

            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            ) * advantages

            policy_loss = -torch.min(unclipped, clipped).mean()
            if self.value_clip > 0:
                values_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss_unclipped = (values - returns).pow(2)
                value_loss_clipped = (values_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
            else:
                value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            supervised_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if supervised_target_count > 0 and self.supervised_coef > 0:
                supervised_ce = F.cross_entropy(
                    logits[supervised_mask],
                    target_actions[supervised_mask],
                    reduction="none",
                )
                supervised_loss = (
                    supervised_ce * target_weights[supervised_mask]
                ).sum() / (target_weights[supervised_mask].sum() + 1e-8)

            loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
                + self.supervised_coef * supervised_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            last_supervised_loss = float(supervised_loss.item())

            last_loss = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "value_clip": float(self.value_clip),
                "entropy": float(entropy.mean().item()),
                "entropy_coef": float(self.entropy_coef),
                "approx_kl": last_approx_kl,
                "clip_fraction": last_clip_fraction,
                "num_samples": len(buffer),
                "num_trajectories": int(sum(1 for d in buffer.dones if d)),
                "return_mean": return_mean,
                "return_std": return_std,
                "adv_mean_before_norm": adv_mean_before_norm,
                "adv_std_before_norm": adv_std_before_norm,
                "reward_sum": float(sum(buffer.rewards)),
                "supervised_loss": last_supervised_loss,
                "supervised_target_count": supervised_target_count,
                "supervised_weight_mean": supervised_weight_mean,
            }

        return last_loss
    
    def save(self, path, extra=None):
        payload = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "model_state_dict": self.net.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["model_state_dict"])
        return payload
