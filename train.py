"""Train PPO on the Isaac Lab cartpole env.

Run with:
    cd ~/IsaacLab && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
    cd ~/Desktop/cartpole_ws
    ~/IsaacLab/isaaclab.sh -p train.py --headless

PPO hyperparameters are per-env: edit the HYPERPARAMS dict in ppo_cartpole.py
or ppo_double.py.
"""

from __future__ import annotations

import argparse

# ---- AppLauncher must be set up BEFORE any other isaaclab/isaacsim import ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on Isaac Lab single cartpole.")
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

import ppo as ppo_mod  # noqa: E402  -- exposes PPO, HYPERPARAMS
from env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402


# PPO hyperparameters live in ppo.py's HYPERPARAMS dict. LOG_FREQ is
# display-only and not really a hyperparameter.
LOG_FREQ = 1  # iters between stdout / tensorboard lines


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
    hp = ppo_mod.HYPERPARAMS
    ppo = ppo_mod.PPO(
        state_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        lr_actor=hp["lr_actor"],
        lr_critic=hp["lr_critic"],
        gamma=hp["gamma"],
        K_epochs=hp["K_epochs"],
        eps_clip=hp["eps_clip"],
        action_std_init=hp["action_std_init"],
        device=device,
        gae_lambda=hp["gae_lambda"],
    )

    # ---- logging ----
    run_name = args_cli.run_name or time.strftime("%Y%m%d-%H%M%S")
    runs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_dir = os.path.join(runs_root, run_name)
    # If the target dir already has event files or checkpoints from a previous
    # run, append a HHMMSS suffix so each training run lands in its own dir.
    if os.path.isdir(run_dir) and any(
        f.startswith("events.out") or f.endswith(".pt") for f in os.listdir(run_dir)
    ):
        run_name = f"{run_name}_{time.strftime('%H%M%S')}"
        run_dir = os.path.join(runs_root, run_name)
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

        # `state` here is the env's obs AFTER the last buffered step — used by
        # GAE in ppo_double.PPO.update() to bootstrap V(s_T); ignored by ppo_cartpole.
        ppo.update(next_state=state)

        if (it + 1) % hp["action_std_decay_freq"] == 0:
            ppo.decay_action_std(hp["action_std_decay_rate"], hp["min_action_std"])

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

        if (it + 1) % hp["save_freq"] == 0:
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
