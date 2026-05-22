# Isaac Lab Cartpole + PPO

A minimal cartpole RL playground built on **NVIDIA Isaac Lab**, with a small **PPO** implementation adapted from [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch) and vectorized to use Isaac Lab's parallel envs.

Two environments are included: a single cartpole (`cartpole`) and an underactuated cart + double pendulum (`double`), both configured for **swing-up**. Pick one at server-startup time with `ENV=...`.

A **persistent-server** workflow lets you boot Isaac Sim once and run many `train` / `play` commands against the same loaded simulator — Ctrl+C interrupts a single command without tearing the simulator down.

## Files

| File | Purpose |
|---|---|
| `cartpole_env.py` | `DirectRLEnv` subclass with banner-marked **STATE** and **REWARD** sections |
| `cart_double_pendulum_env.py` | Underactuated cart + double-pendulum `DirectRLEnv` (swing-up) |
| `env_registry.py` | Maps env names (`cartpole`, `double`) to their classes |
| `ppo.py` | PPO algorithm (ActorCritic + RolloutBuffer, vectorized for N parallel envs) |
| `server.py` | Long-running process: boots Isaac Sim once, listens on a Unix socket |
| `client.py` | Thin stdlib-only client that sends JSON commands to the server |
| `train.py` / `play.py` | Standalone runners (boot Isaac Sim per call) — kept for one-shot use |
| `Makefile` | All entry points |

## Requirements

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU + driver (validated on driver 550–580 with RTX 3090)
- Conda env with **Isaac Lab 2.3+** installed (the `isaaclab` env in this repo's setup is at `~/IsaacLab`)
- PyTorch with CUDA

This repo does *not* ship Isaac Lab itself; install it separately following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

## Quick start (recommended workflow)

### 1. Boot the persistent server (terminal A)
```bash
make env                  # single cartpole, GUI viewport
make env ENV=double       # cart + double pendulum
make env HEADLESS=1       # no GUI (faster startup, headless training)
make env NUM_ENVS=512     # more parallel envs
```

The server loads ONE environment for its lifetime. To switch envs, stop the
server (Ctrl+C or `make kill-server`) and boot it again with a different `ENV`.
Wait until you see `[server] listening on .server.sock`.

### 2. Train (terminal B)
```bash
make train                                     # default: 200 iters
make train MAX_ITERS=400 RUN_NAME=swingup_v1   # named run
make train RESUME=runs/swingup_v1/policy_final.pt MAX_ITERS=200
```

Ctrl+C here kills the client; the server stays alive for the next command.

### 3. Play a trained policy (terminal B)
```bash
make play                                                     # newest checkpoint
make play CHECKPOINT=runs/swingup_v1/policy_final.pt          # specific
make play PLAY_DET=1                                          # deterministic
```

### 4. Watch metrics
```bash
make tensorboard      # opens on http://localhost:6006
```

### 5. Shut the server down
- Ctrl+C in terminal A, or
- `make kill-server` from anywhere (uses SIGKILL because Isaac Sim absorbs softer signals)

## Tweaking the environment

Open `cartpole_env.py` and look for the banner comments:

```python
# ============================== STATE =================================
def _get_observations(self):
    ...
```

```python
# ============================== REWARD ================================
def _get_rewards(self):
    ...
```

Each section ships with the swing-up default plus a couple of commented alternatives (5-D state with sin/cos, sparse reward, balance variant, etc.).

**Important:** if you change the state dim, update `observation_space` in `CartpoleEnvCfg` at the top of the same file.

After editing, restart the server (Ctrl+C → `make env`) to pick up the change. Hot reload is not supported.

## Standalone scripts (no server)

For one-shot or CI use you can bypass the server entirely:

```bash
make smoke         # 5-iter sanity check
make train-once    # equivalent to train.py
make play-once     # equivalent to play.py
```

Each call boots Isaac Sim from scratch (~30 s).

## Algorithm notes

- **PPO** uses Monte-Carlo returns (no GAE) to stay close to the reference implementation. Cartpole converges fine without GAE; longer-horizon tasks would benefit from adding it.
- **Continuous action only** (1-D force on the cart slider). The policy outputs a `tanh`-bounded mean and adds Gaussian noise with a decaying std (`action_std_decay_*` in `server.py`).
- **Vectorized rollout buffer** stores tensors of shape `(T, N, …)` where T is the rollout horizon and N is `num_envs`. Discounted returns are computed per-env so envs that reset mid-rollout don't bleed across episodes.
- **No `gymnasium` install needed** — Isaac Lab's `DirectRLEnv` satisfies the interface internally; observations and rewards are torch tensors on the GPU.

## Layout of `runs/`

Each training run creates `runs/<run_name>/` containing:
- `policy_NNNN.pt` — periodic checkpoints (every 25 iters by default)
- `policy_final.pt` — final weights
- TensorBoard event files

Pass any of these `.pt` files to `make play CHECKPOINT=...` to roll out that snapshot.

## Acknowledgements

- PPO algorithm adapted from [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch).
- Cartpole asset and base `DirectRLEnv` structure from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab).
