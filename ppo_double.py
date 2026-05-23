"""PPO for the cart double-pendulum env.

Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch (continuous action).

Differences from the original:
  * Vectorized: every quantity in the rollout buffer has a leading (num_envs,)
    dimension, so a single rollout collects T*N transitions.
  * Discounted returns are computed per-env using the per-step terminal flag
    (so envs that reset mid-rollout don't bleed return across episodes).
  * No gym dependency; the env passes torch tensors directly on the GPU.

Algorithm itself (Gaussian policy with tanh-mean, MC returns w/ normalization,
clipped surrogate objective, K epochs per update) is unchanged from the
reference.

This file is the double-pendulum-only copy of the PPO code. ``ppo_cartpole.py``
is the separate copy for the single cartpole env — edit each independently.
Underactuated double-pendulum swing-up is hard; adding GAE here is the most
likely useful divergence from the cartpole copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


# =====================================================================
# Per-env PPO hyperparameters for the CART DOUBLE PENDULUM.
# Edit these to tune double-pendulum training without affecting cartpole.
# =====================================================================
HYPERPARAMS = {
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "gamma": 0.995,               # extended horizon (~3.3s @ 60Hz) to match swing-up duration
    "gae_lambda": 0.95,           # GAE bias-variance tradeoff (1.0 = MC, 0.0 = TD)
    "K_epochs": 10,
    "eps_clip": 0.2,
    "action_std_init": 0.6,
    "action_std_decay_rate": 0.025,
    "min_action_std": 0.20,
    "action_std_decay_freq": 20,  # iters between action_std decays
    "save_freq": 25,              # iters between checkpoint saves
}


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

        # Bigger network (128x128) than the cartpole copy — double-pendulum
        # dynamics are harder to fit. BREAKING for pre-existing checkpoints.
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
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
        gae_lambda: float = 0.95,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
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

    def update(self, next_state: torch.Tensor | None = None):
        """PPO update with GAE(λ) advantages.

        ``next_state`` is the env's observation right AFTER the last buffered
        step; its value bootstraps the GAE recursion at the rollout boundary.
        If omitted, the bootstrap value defaults to 0 (higher variance for
        rollouts that don't reach a terminal).
        """
        # Stack per-step lists -> (T, N, ...)
        rewards_buf = torch.stack(self.buffer.rewards)        # (T, N)
        terminals_buf = torch.stack(self.buffer.is_terminals)  # (T, N) bool
        values_buf = torch.stack(self.buffer.state_values)    # (T, N)
        T, N = rewards_buf.shape

        # Bootstrap value at the rollout boundary: V(s_T).
        if next_state is not None:
            with torch.no_grad():
                next_value = self.policy_old.critic(next_state).squeeze(-1)
        else:
            next_value = torch.zeros(N, device=self.device)

        # GAE(λ) backwards pass:
        #   δ_t = r_t + γ V(s_{t+1}) (1 - d_t) - V(s_t)
        #   A_t = δ_t + γ λ (1 - d_t) A_{t+1}
        advantages = torch.zeros_like(rewards_buf)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            nonterminal = (~terminals_buf[t]).float()
            v_next = next_value if t == T - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + self.gamma * v_next * nonterminal - values_buf[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae

        # Bootstrap targets for the value head.
        returns = advantages + values_buf

        # Flatten (T, N, ...) -> (T*N, ...). Whiten advantages ONLY (returns
        # are bootstrap targets with a meaningful scale — don't normalize them).
        advantages_flat = advantages.reshape(-1)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-7)
        returns_flat = returns.reshape(-1)

        old_states = torch.stack(self.buffer.states).reshape(T * N, -1)
        old_actions = torch.stack(self.buffer.actions).reshape(T * N, -1)
        old_logprobs = torch.stack(self.buffer.logprobs).reshape(T * N).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages_flat
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_flat
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns_flat) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())  # update policy weight
        self.buffer.clear()

    def save(self, checkpoint_path: str):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str):
        sd = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(sd)
        self.policy.load_state_dict(sd)
