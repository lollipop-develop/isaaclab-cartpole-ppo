# Design: Cart + Double-Pendulum Environment

Date: 2026-05-22
Status: approved (design), pending spec review

## Goal

Add a second RL environment to `cartpole_ws`: an underactuated **cart +
double pendulum** (cart with two series-connected pole links). It must run
with the existing PPO unchanged, and live alongside the existing single
cartpole, selectable at server-startup time.

## Scope

In scope:
- New env file `cart_double_pendulum_env.py` (swing-up task).
- A small `env_registry.py` mapping env names to classes.
- `--env` flag plumbed into `server.py`, `train.py`, `play.py`.
- Makefile `ENV` variable.

Explicitly NOT in scope:
- No changes to `ppo.py` — it is already dimension-agnostic
  (`state_dim` / `action_dim` are constructor args).
- No changes to `client.py` — the server picks the env at boot; clients
  send the same train/play commands regardless of env.
- No multi-task / shared-policy training. Each env is trained separately.

## System being modeled

NVIDIA Isaac Lab ships `CART_DOUBLE_PENDULUM_CFG`
(`isaaclab_assets.robots.cart_double_pendulum`) with three joints:

| Joint | Meaning | Actuated? |
|---|---|---|
| `slider_to_cart` | cart translation | YES (effort applied) |
| `cart_to_pole`   | link-1 angle (θ₁) | no — free joint |
| `pole_to_pendulum` | link-2 angle relative to link-1 (θ₂) | no — free joint |

Underactuated: the policy commands only a 1-D force on the cart slider.
The two pole joints swing freely under physics. This is the classic
"double inverted pendulum on a cart" — `action_space = 1`, identical in
shape to the existing single cartpole, so PPO needs no change.

## Approach (chosen: Approach 1 — registry + `--env` flag)

Rejected alternatives:
- Duplicate scripts (`server_double.py` …) — pure duplication, every fix
  made twice.
- One mega env file with an internal `system` switch — fills the file with
  conditionals and wrecks the clean banner-marked STATE/REWARD sections.

## Components

### 1. `cart_double_pendulum_env.py`

A `DirectRLEnv` mirroring `cartpole_env.py`'s structure, including the
banner-marked **STATE** and **REWARD** sections so it is easy to edit.

- Asset: `CART_DOUBLE_PENDULUM_CFG` cloned to `/World/envs/env_.*/Robot`.
- Joint name fields: `cart_dof_name = "slider_to_cart"`,
  `pole_dof_name = "cart_to_pole"`, `pendulum_dof_name = "pole_to_pendulum"`.
- Action: effort applied to `slider_to_cart` only;
  `action_scale` config field, `action_space = 1`.
- Task: **swing-up**.
  - Reset: links hang down — θ₁ sampled near π, θ₂ sampled near 0
    (link-2 aligned with link-1), both with small uniform noise; cart near 0.
  - `episode_length_s` ≈ 8–10 s.
  - No angle-based termination (links sweep the full range). Termination
    only on cart out-of-bounds (`max_cart_pos`) or timeout. Controlled by a
    `fail_*` config field set huge, same idiom as the single cartpole.
- State (default, 6-D): `[cart_pos, cart_vel, θ₁, θ̇₁, θ₂, θ̇₂]`.
  `observation_space = 6`. A commented sin/cos variant is offered (8-D:
  `sin θ₁, cos θ₁, sin(θ₁+θ₂), cos(θ₁+θ₂), θ̇₁, θ̇₂, cart_pos, cart_vel`)
  with a reminder to update `observation_space`.
- Reward (default): cos-based, both links upright →
  `r_upright = cos(θ₁) + cos(θ₁ + θ₂)` (max +2 fully up, −2 fully down)
  plus cart-center penalty, cart/joint velocity penalties, and a
  termination penalty. Commented alternatives provided (sparse, tip-height).
- `observation_space`, `action_space`, `action_scale`, `max_cart_pos`,
  episode length, and initial-angle ranges are all `CartDoublePendulumEnvCfg`
  fields, mirroring the single cartpole.

### 2. `env_registry.py`

```python
from cartpole_env import CartpoleEnv, CartpoleEnvCfg
from cart_double_pendulum_env import CartDoublePendulumEnv, CartDoublePendulumEnvCfg

ENVS = {
    "cartpole": (CartpoleEnv, CartpoleEnvCfg),
    "double":   (CartDoublePendulumEnv, CartDoublePendulumEnvCfg),
}
```

Both env modules import `isaaclab.*` at module load, so `env_registry.py`
must be imported **after** `AppLauncher` boots. `server.py`/`train.py`/
`play.py` already defer their env imports, so they import the registry in
the same deferred block.

### 3. Plumbing

- `server.py`: add `--env` arg (choices = registry keys, default
  `cartpole`). After AppLauncher boot, look up `(EnvCls, CfgCls)` from the
  registry, build the cfg, set `num_envs`, construct the env. Everything
  downstream (`make_ppo` reading `cfg.observation_space`, train/play
  handlers) is unchanged.
- `train.py`, `play.py`: same `--env` arg for the standalone path.
- `client.py`: unchanged. The env is fixed when the server boots; clients
  are env-agnostic.

### 4. Makefile

- New variable `ENV ?= cartpole`.
- `env` target passes `--env $(ENV)` to `server.py`.
- `train-once` / `play-once` / `smoke` pass `--env $(ENV)` to the
  standalone scripts.
- Help text documents `make env ENV=double`.

## Data flow (unchanged from current design)

`make env ENV=double` → `server.py` builds `CartDoublePendulumEnv` once →
listens on socket. `make train` / `make play` → `client.py` sends JSON →
server runs PPO against the loaded env → streams output. PPO sizes its
networks from `cfg.observation_space` (6) and `cfg.action_space` (1).

## Testing / verification

- Headless smoke test: `make smoke ENV=double` (or a short
  `server.py --env double` + a 3-iter client train) must run without error
  and produce decreasing-then-improving `ep_ret`.
- Confirm `--env cartpole` still works (no regression).
- Confirm PPO auto-sizes: server log prints `obs=6 act=1` for `double`.

## Known risks

- **Swing-up of an underactuated double pendulum is genuinely hard** — one
  of the hardest classic control benchmarks. Vanilla PPO with Monte-Carlo
  returns may need tuning (`action_scale`, episode length, `action_std`
  schedule) and may only partially solve the task. The environment will be
  *correct*; convergence is an open empirical question, out of scope for
  this design. Adding GAE later is the most likely needed improvement.
- Initial-angle convention: θ₂ is link-2 relative to link-1. "Both hanging"
  is θ₁≈π, θ₂≈0. Implementation must verify joint sign/zero conventions
  against the actual asset during the build.
