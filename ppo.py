"""PPO adapted from https://github.com/nikhilbarhate99/PPO-PyTorch (continuous action only).

Differences from the original:
  * Vectorized: every quantity in the rollout buffer has a leading (num_envs,)
    dimension, so a single rollout collects T*N transitions.
  * Discounted returns are computed per-env using the per-step terminal flag
    (so envs that reset mid-rollout don't bleed return across episodes).
  * No gym dependency; the env passes torch tensors directly on the GPU.

Algorithm itself (Gaussian policy with tanh-mean, MC returns w/ normalization,
clipped surrogate objective, K epochs per update) is unchanged from the
reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class RolloutBuffer:
    """Stores trajectories from N parallel envs as lists of per-step tensors."""

    def __init__(self):
        self.actions: list[torch.Tensor] = []      # each: (N, action_dim)
        self.states: list[torch.Tensor] = []       # each: (N, state_dim)
        self.logprobs: list[torch.Tensor] = []     # each: (N,)
        self.rewards: list[torch.Tensor] = []      # each: (N,)
        self.state_values: list[torch.Tensor] = [] # each: (N,)
        self.is_terminals: list[torch.Tensor] = [] # each: (N,) bool

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    """Gaussian policy + value head. Mean is tanh-bounded to (-1, 1)."""

    def __init__(self, state_dim: int, action_dim: int, action_std_init: float, device: torch.device):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init, device=device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std: float):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std, device=self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor):
        """state: (N, state_dim). Returns sampled action (N, A), logprob (N,), value (N,)."""
        action_mean = self.actor(state)
        cov_mat = torch.diag_embed(self.action_var.expand_as(action_mean))
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(state).squeeze(-1)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """state: (B, state_dim), action: (B, A). Returns logprob (B,), value (B,), entropy (B,)."""
        action_mean = self.actor(state)
        cov_mat = torch.diag_embed(self.action_var.expand_as(action_mean))
        dist = MultivariateNormal(action_mean, cov_mat)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)
        return logprob, value, entropy


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        action_std_init: float,
        device: torch.device,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_std = action_std_init
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std: float):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, decay_rate: float, min_action_std: float):
        self.action_std = max(round(self.action_std - decay_rate, 4), min_action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """state: (N, state_dim) on self.device. Pushes to buffer, returns action (N, A)."""
        with torch.no_grad():
            action, logprob, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(state_val)
        return action

    def update(self):
        # Stack per-step lists -> (T, N, ...)
        rewards_buf = torch.stack(self.buffer.rewards)        # (T, N)
        terminals_buf = torch.stack(self.buffer.is_terminals)  # (T, N) bool
        T, N = rewards_buf.shape

        # Per-env discounted Monte Carlo returns, reset at episode boundaries.
        returns = torch.zeros_like(rewards_buf)
        discounted = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            # If env terminated at step t, the return at t is just r_t (chain breaks).
            discounted = rewards_buf[t] + self.gamma * discounted * (~terminals_buf[t]).float()
            returns[t] = discounted

        # Flatten (T, N, ...) -> (T*N, ...)
        returns = returns.reshape(-1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states = torch.stack(self.buffer.states).reshape(T * N, -1)
        old_actions = torch.stack(self.buffer.actions).reshape(T * N, -1)
        old_logprobs = torch.stack(self.buffer.logprobs).reshape(T * N).detach()
        old_state_values = torch.stack(self.buffer.state_values).reshape(T * N).detach()

        advantages = returns.detach() - old_state_values

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path: str):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str):
        sd = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(sd)
        self.policy.load_state_dict(sd)
