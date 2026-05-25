"""Load a trained PPO checkpoint and run the policy on the cartpole env.

Run with:
    cd ~/Desktop/cartpole_ws
    ~/IsaacLab/isaaclab.sh -p play.py --headless --checkpoint runs/<run_name>/policy_final.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained PPO policy on the single cartpole.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs to roll out.")
parser.add_argument("--num_steps", type=int, default=1000, help="Total env steps to run.")
parser.add_argument("--deterministic", action="store_true", help="Use action mean instead of sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import ppo as ppo_mod  # noqa: E402
from env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402


def main():
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = CartpoleEnv(cfg=cfg, render_mode=None)
    device = env.device

    # Throwaway hyperparameters — play only loads weights, never trains.
    ppo = ppo_mod.PPO(
        state_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        lr_actor=0.0, lr_critic=0.0,
        gamma=0.99, K_epochs=1, eps_clip=0.2,
        action_std_init=0.10,  # nearly deterministic
        device=device,
    )
    ppo.load(args_cli.checkpoint)
    print(f"[play] loaded {args_cli.checkpoint}")

    obs_dict, _ = env.reset()
    state = obs_dict["policy"]

    ep_rewards = torch.zeros(args_cli.num_envs, device=device)
    ep_lengths = torch.zeros(args_cli.num_envs, device=device)
    returns_log: list[float] = []
    lengths_log: list[int] = []

    with torch.no_grad():
        for step in range(args_cli.num_steps):
            if args_cli.deterministic:
                # Use the mean of the Gaussian (no sampling).
                action = ppo.policy_old.actor(state)
            else:
                action, _, _ = ppo.policy_old.act(state)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            ep_rewards += reward
            ep_lengths += 1
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                returns_log.extend(ep_rewards[idx].cpu().tolist())
                lengths_log.extend(ep_lengths[idx].cpu().tolist())
                ep_rewards[idx] = 0.0
                ep_lengths[idx] = 0.0

            state = obs_dict["policy"]

    if returns_log:
        mean_ret = sum(returns_log) / len(returns_log)
        mean_len = sum(lengths_log) / len(lengths_log)
        print(f"[play] {len(returns_log)} episodes  ep_return={mean_ret:.2f}  ep_length={mean_len:.1f}")
    else:
        print("[play] no episodes completed in the given step budget")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
