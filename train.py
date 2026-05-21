"""Train PPO on the Isaac Lab cartpole env.

Run with:
    cd ~/IsaacLab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
    cd ~/Desktop/cartpole_ws
    ~/IsaacLab/isaaclab.sh -p train.py --headless

Edit hyperparameters in the HYPERPARAMS block below.
"""

from __future__ import annotations

import argparse

# ---- AppLauncher must be set up BEFORE any other isaaclab/isaacsim import ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on Isaac Lab cartpole.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel envs.")
parser.add_argument("--max_iters", type=int, default=200, help="Number of PPO updates.")
parser.add_argument("--rollout_steps", type=int, default=128, help="Env steps per update (per env).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default=None, help="Subdir under runs/. Defaults to timestamp.")
parser.add_argument("--play_after", action="store_true",
                    help="After training, roll out the freshly trained policy in the same process (no app restart).")
parser.add_argument("--play_steps", type=int, default=1500,
                    help="Env steps to roll out when --play_after is set.")
parser.add_argument("--play_deterministic", action="store_true",
                    help="When playing after training, use the policy mean instead of sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Now safe to import the env and torch-heavy things ----
import os  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from cartpole_env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402
from ppo import PPO  # noqa: E402


# =====================================================================
# ============================ HYPERPARAMS ============================
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
K_EPOCHS = 40
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.6
ACTION_STD_DECAY_RATE = 0.05
MIN_ACTION_STD = 0.10
ACTION_STD_DECAY_FREQ = 20   # iters between std decays
SAVE_FREQ = 25               # iters between checkpoint saves
LOG_FREQ = 1                 # iters between stdout / tensorboard lines
# =====================================================================


def main():
    torch.manual_seed(args_cli.seed)

    # ---- env ----
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = CartpoleEnv(cfg=cfg, render_mode=None)
    device = env.device
    print(f"[train] num_envs={args_cli.num_envs}  device={device}  "
          f"obs={cfg.observation_space}  act={cfg.action_space}", flush=True)

    # ---- PPO agent ----
    ppo = PPO(
        state_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        K_epochs=K_EPOCHS,
        eps_clip=EPS_CLIP,
        action_std_init=ACTION_STD_INIT,
        device=device,
    )

    # ---- logging ----
    run_name = args_cli.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    print(f"[train] logging to {run_dir}", flush=True)

    # ---- rollout ----
    obs_dict, _ = env.reset()
    state = obs_dict["policy"]

    ep_rewards = torch.zeros(args_cli.num_envs, device=device)
    ep_lengths = torch.zeros(args_cli.num_envs, device=device)

    t_start = time.time()
    total_env_steps = 0

    for it in range(args_cli.max_iters):
        ep_returns_iter: list[float] = []
        ep_lengths_iter: list[int] = []

        for _ in range(args_cli.rollout_steps):
            action = ppo.select_action(state)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)

            ep_rewards += reward
            ep_lengths += 1
            if done.any():
                done_idx = done.nonzero(as_tuple=True)[0]
                ep_returns_iter.extend(ep_rewards[done_idx].cpu().tolist())
                ep_lengths_iter.extend(ep_lengths[done_idx].cpu().tolist())
                ep_rewards[done_idx] = 0.0
                ep_lengths[done_idx] = 0.0

            state = obs_dict["policy"]
            total_env_steps += args_cli.num_envs

        ppo.update()

        if (it + 1) % ACTION_STD_DECAY_FREQ == 0:
            ppo.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

        if (it + 1) % LOG_FREQ == 0:
            mean_ret = sum(ep_returns_iter) / max(len(ep_returns_iter), 1)
            mean_len = sum(ep_lengths_iter) / max(len(ep_lengths_iter), 1)
            sps = total_env_steps / (time.time() - t_start + 1e-9)
            print(
                f"iter {it+1:4d}/{args_cli.max_iters}  "
                f"ep_ret={mean_ret:7.2f}  ep_len={mean_len:6.1f}  "
                f"n_eps={len(ep_returns_iter):4d}  "
                f"std={ppo.action_std:.3f}  "
                f"sps={sps:7.0f}",
                flush=True,
            )
            writer.add_scalar("rollout/ep_return_mean", mean_ret, it)
            writer.add_scalar("rollout/ep_length_mean", mean_len, it)
            writer.add_scalar("rollout/n_episodes", len(ep_returns_iter), it)
            writer.add_scalar("ppo/action_std", ppo.action_std, it)
            writer.add_scalar("perf/steps_per_sec", sps, it)

        if (it + 1) % SAVE_FREQ == 0:
            ckpt = os.path.join(run_dir, f"policy_{it+1:04d}.pt")
            ppo.save(ckpt)
            print(f"[train] saved {ckpt}")

    final = os.path.join(run_dir, "policy_final.pt")
    ppo.save(final)
    print(f"[train] saved {final}", flush=True)
    writer.close()

    # --- optional in-process playback (no second Isaac Sim startup) ----
    if args_cli.play_after:
        print(f"[train] playing trained policy for {args_cli.play_steps} steps "
              f"(deterministic={args_cli.play_deterministic})", flush=True)
        obs_dict, _ = env.reset()
        state = obs_dict["policy"]
        ep_returns = []
        ep_lengths = []
        ep_r = torch.zeros(args_cli.num_envs, device=device)
        ep_l = torch.zeros(args_cli.num_envs, device=device)
        with torch.no_grad():
            for _ in range(args_cli.play_steps):
                if args_cli.play_deterministic:
                    action = ppo.policy_old.actor(state)
                else:
                    action, _, _ = ppo.policy_old.act(state)
                obs_dict, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                ep_r += reward
                ep_l += 1
                if done.any():
                    idx = done.nonzero(as_tuple=True)[0]
                    ep_returns.extend(ep_r[idx].cpu().tolist())
                    ep_lengths.extend(ep_l[idx].cpu().tolist())
                    ep_r[idx] = 0.0
                    ep_l[idx] = 0.0
                state = obs_dict["policy"]
        if ep_returns:
            print(f"[play] {len(ep_returns)} eps  ep_return={sum(ep_returns)/len(ep_returns):.2f}  "
                  f"ep_length={sum(ep_lengths)/len(ep_lengths):.1f}", flush=True)
        else:
            print("[play] no eps completed in the given step budget", flush=True)

    env.close()


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
