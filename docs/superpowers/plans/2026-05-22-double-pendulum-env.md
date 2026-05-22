# Cart + Double-Pendulum Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an underactuated cart + double-pendulum swing-up environment to `cartpole_ws`, selectable alongside the existing single cartpole via an `--env` flag, with the existing PPO unchanged.

**Architecture:** A new `DirectRLEnv` subclass (`cart_double_pendulum_env.py`) modeled on the existing `cartpole_env.py`. A tiny `env_registry.py` maps env names to `(EnvClass, CfgClass)` pairs. `server.py`/`train.py`/`play.py` gain an `--env` argument and look the env up in the registry. `ppo.py` and `client.py` are untouched — PPO already sizes its networks from `cfg.observation_space`/`cfg.action_space`.

**Tech Stack:** Python 3.10, NVIDIA Isaac Lab 2.3 (`DirectRLEnv`), PyTorch, conda env `isaaclab`, launched via `~/IsaacLab/isaaclab.sh -p`.

**Note on testing:** This project has no `pytest` suite and cannot have one — every code path requires a live Isaac Sim instance plus a GPU (importing the env modules fails without a booted `SimulationApp`). "Tests" in this plan are therefore **syntax checks** (`python -m py_compile`, no sim needed) plus **smoke runs** (short headless training through `isaaclab.sh`, checking for specific stdout and absence of tracebacks). This matches how the rest of the project is verified.

**Conventions used throughout:**
- `ACT` below is shorthand for: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab`
- All commands run from `/home/shigeki-u/Desktop/cartpole_ws`.
- Isaac Sim cold-boots in ~30 s; smoke runs take ~1 min each.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `cart_double_pendulum_env.py` | create | Underactuated cart + double-pendulum `DirectRLEnv`, swing-up task |
| `env_registry.py` | create | Maps env name → `(EnvClass, CfgClass)` |
| `train.py` | modify | Add `--env`; build env via registry |
| `play.py` | modify | Add `--env`; build env via registry |
| `server.py` | modify | Add `--env`; build env via registry |
| `Makefile` | modify | Add `ENV` variable, pass `--env` to all scripts |
| `README.md` | modify | Document `make env ENV=double` |

`ppo.py` and `client.py` are intentionally NOT modified.

---

## Task 1: Create the double-pendulum environment

**Files:**
- Create: `cart_double_pendulum_env.py`

- [ ] **Step 1: Write the full environment file**

Create `cart_double_pendulum_env.py` with exactly this content:

```python
"""Cart + double-pendulum DirectRLEnv (underactuated, swing-up).

Single-agent: the policy commands ONLY a 1-D force on the cart slider.
The two pole joints (cart_to_pole, pole_to_pendulum) are free — they swing
under physics. This is the classic underactuated "double inverted pendulum
on a cart".

Same banner-marked STATE and REWARD sections as cartpole_env.py — edit those
to experiment. If you change the state dimension, also update
``observation_space`` in CartDoublePendulumEnvCfg below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle (radians) into [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@configclass
class CartDoublePendulumEnvCfg(DirectRLEnvCfg):
    # ---- episode / sim ----
    decimation = 2
    episode_length_s = 8.0  # swing-up needs more time than balance
    action_scale = 100.0  # [N] applied to the cart slider

    # ---- spaces ----
    # observation_space MUST match the dim returned by _get_observations().
    action_space = 1
    observation_space = 6
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ---- robot ----
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    # ---- scene (num_envs overridden from CLI) ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=4.0, replicate_physics=True
    )

    # ---- reset / termination ----
    max_cart_pos = 3.0  # cart out-of-bounds threshold [m]

    # Initial joint angles, multiplied by pi internally. At joint angle 0 the
    # links point UP (the asset is an *inverted* double pendulum).
    # SWING-UP DEFAULT: link-1 hangs down (theta1 ~ pi); link-2 aligned with
    # link-1 (theta2 ~ 0, i.e. both hanging straight down).
    initial_pole_angle_range = [0.95, 1.05]       # link-1 ~ pi +/- 0.05*pi
    initial_pendulum_angle_range = [-0.05, 0.05]  # link-2 ~ 0 +/- 0.05*pi

    # Link-1 termination angle. SWING-UP: huge so it never triggers (the links
    # sweep the full range). For a balance task, set this to math.pi / 2.
    fail_pole_angle = 1e9


class CartDoublePendulumEnv(DirectRLEnv):
    cfg: CartDoublePendulumEnvCfg

    def __init__(self, cfg: CartDoublePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    # ----------------------------------------------------------------------
    # Scene setup (rarely edited)
    # ----------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------------------------------------------------------------
    # Action application
    # ----------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # Underactuated: effort applied to the cart slider ONLY.
        # The two pole joints are free and swing under physics.
        self.robot.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    # ======================================================================
    # ============================== STATE =================================
    # Edit here to change what the policy sees.
    # Keep CartDoublePendulumEnvCfg.observation_space in sync with the dim.
    # ======================================================================
    def _get_observations(self) -> dict:
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(-1)
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(-1)
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(-1)
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(-1)
        pend_pos = self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(-1)
        pend_vel = self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(-1)

        # --- DEFAULT: 6-D state ----------------------------------------
        # Angles wrapped to [-pi, pi]. theta2 here is link-2 RELATIVE to link-1.
        obs = torch.cat([
            cart_pos, cart_vel,
            wrap_to_pi(pole_pos), pole_vel,
            wrap_to_pi(pend_pos), pend_vel,
        ], dim=-1)
        # ---------------------------------------------------------------

        # --- VARIANT: 8-D sin/cos (smooth, no wrap discontinuity) ------
        # theta1 = pole_pos
        # theta2_abs = pole_pos + pend_pos          # link-2 absolute angle
        # obs = torch.cat([
        #     cart_pos, cart_vel,
        #     torch.sin(theta1), torch.cos(theta1), pole_vel,
        #     torch.sin(theta2_abs), torch.cos(theta2_abs), pend_vel,
        # ], dim=-1)
        # ---> set observation_space = 8 in CartDoublePendulumEnvCfg above

        return {"policy": obs}

    # ======================================================================
    # ============================== REWARD ================================
    # Edit here to change the reward function. Each term is a tensor of shape
    # (num_envs,); the final reward is their sum.
    # ======================================================================
    def _get_rewards(self) -> torch.Tensor:
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
        pend_pos = self.joint_pos[:, self._pendulum_dof_idx[0]]
        pend_vel = self.joint_vel[:, self._pendulum_dof_idx[0]]
        terminated = self.reset_terminated.float()

        # Absolute angle of link-2 from vertical = theta1 + theta2.
        theta2_abs = pole_pos + pend_pos

        # --- SWING-UP DEFAULT --------------------------------------------
        # cos(angle) is +1 when a link points up, -1 when it hangs down.
        # Both links up -> r_upright = +2.
        r_upright = torch.cos(pole_pos) + torch.cos(theta2_abs)
        both_up = (torch.cos(pole_pos) > 0.95) & (torch.cos(theta2_abs) > 0.95)
        r_at_top_slow = both_up.float() * (-0.1 * (pole_vel.pow(2) + pend_vel.pow(2)))
        r_cart_center = -0.1 * cart_pos.pow(2)
        r_cart_quiet = -0.005 * cart_vel.abs()
        r_terminate = -10.0 * terminated
        reward = r_upright + r_at_top_slow + r_cart_center + r_cart_quiet + r_terminate
        # -----------------------------------------------------------------

        # --- VARIANT: sparse (reward only when both links are near upright)
        # reward = both_up.float() - 10.0 * terminated

        return reward

    # ----------------------------------------------------------------------
    # Termination
    # ----------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(
            torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
        )
        # SWING-UP: fail_pole_angle is huge, so this check never trips.
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > self.cfg.fail_pole_angle, dim=1
        )
        return out_of_bounds, time_out

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            self.cfg.initial_pendulum_angle_range[0] * math.pi,
            self.cfg.initial_pendulum_angle_range[1] * math.pi,
            joint_pos[:, self._pendulum_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

- [ ] **Step 2: Syntax check**

Run: `cd ~/Desktop/cartpole_ws && python -m py_compile cart_double_pendulum_env.py && echo OK`
Expected: prints `OK` with no `SyntaxError`. (This only checks syntax; it does not import Isaac Lab.)

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add cart_double_pendulum_env.py
git commit -m "Add underactuated cart + double-pendulum swing-up env"
```

---

## Task 2: Create the environment registry

**Files:**
- Create: `env_registry.py`

- [ ] **Step 1: Write the registry file**

Create `env_registry.py` with exactly this content:

```python
"""Registry mapping env names to (EnvClass, CfgClass) pairs.

IMPORTANT: importing this module imports the env modules, which import
``isaaclab.*`` / ``isaaclab_assets.*``. Those imports only succeed AFTER
``AppLauncher`` has booted the Isaac Sim runtime. So import this module
inside the post-AppLauncher section of server.py / train.py / play.py,
never at the top of a script.
"""

from cartpole_env import CartpoleEnv, CartpoleEnvCfg
from cart_double_pendulum_env import CartDoublePendulumEnv, CartDoublePendulumEnvCfg

# Keys here MUST match the argparse choices in server.py / train.py / play.py.
ENVS = {
    "cartpole": (CartpoleEnv, CartpoleEnvCfg),
    "double": (CartDoublePendulumEnv, CartDoublePendulumEnvCfg),
}
```

- [ ] **Step 2: Syntax check**

Run: `cd ~/Desktop/cartpole_ws && python -m py_compile env_registry.py && echo OK`
Expected: prints `OK` with no `SyntaxError`.

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add env_registry.py
git commit -m "Add env registry mapping names to env classes"
```

---

## Task 3: Add `--env` to train.py (first runtime verification)

**Files:**
- Modify: `train.py`

This task wires `train.py` to the registry. It is the first task whose
verification actually boots Isaac Sim, so it verifies Task 1, Task 2, and
Task 3 together.

- [ ] **Step 1: Add the `--env` argument**

In `train.py`, find this line:

```python
    parser.add_argument("--run_name", type=str, default=None, help="Subdir under runs/. Defaults to timestamp.")
```

Immediately AFTER it, add:

```python
    parser.add_argument("--env", default="cartpole", choices=["cartpole", "double"],
                        help="Which environment to train on.")
```

- [ ] **Step 2: Replace the env import**

In `train.py`, find this line:

```python
from cartpole_env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402
```

Replace it with:

```python
from env_registry import ENVS  # noqa: E402
```

- [ ] **Step 3: Replace env construction**

In `train.py`, inside `main()`, find these three lines:

```python
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = CartpoleEnv(cfg=cfg, render_mode=None)
```

Replace them with:

```python
    env_cls, cfg_cls = ENVS[args_cli.env]
    cfg = cfg_cls()
    cfg.scene.num_envs = args_cli.num_envs
    env = env_cls(cfg=cfg, render_mode=None)
```

- [ ] **Step 4: Show the env name in the startup log**

In `train.py`, find this line:

```python
    print(f"[train] num_envs={args_cli.num_envs}  device={device}  "
          f"obs={cfg.observation_space}  act={cfg.action_space}", flush=True)
```

Replace it with:

```python
    print(f"[train] env={args_cli.env}  num_envs={args_cli.num_envs}  device={device}  "
          f"obs={cfg.observation_space}  act={cfg.action_space}", flush=True)
```

- [ ] **Step 5: Syntax check**

Run: `cd ~/Desktop/cartpole_ws && python -m py_compile train.py && echo OK`
Expected: prints `OK`.

- [ ] **Step 6: Runtime smoke test — double pendulum**

Run:
```bash
cd ~/Desktop/cartpole_ws && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && ~/IsaacLab/isaaclab.sh -p train.py --env double --headless --num_envs 64 --max_iters 3 --rollout_steps 64 2>&1 | grep -E "^iter|\[train\]|Error|Traceback"
```
Expected output contains:
- `[train] env=double  num_envs=64  device=cuda:0  obs=6  act=1`
- three lines `iter 1/3 ...`, `iter 2/3 ...`, `iter 3/3 ...`
- NO `Error` / `Traceback` lines

If you see `Traceback`, the most likely causes (in order): a joint name
typo in `cart_double_pendulum_env.py`; the asset not cloning (try removing
`replicate_physics=True` is NOT the fix — instead check the Kit log named in
the error); `observation_space` not matching the obs tensor width.

- [ ] **Step 7: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add train.py
git commit -m "Wire --env flag into train.py via registry"
```

---

## Task 4: Add `--env` to play.py

**Files:**
- Modify: `play.py`

- [ ] **Step 1: Add the `--env` argument**

In `play.py`, find this line:

```python
parser.add_argument("--deterministic", action="store_true", help="Use action mean instead of sampling.")
```

Immediately AFTER it, add:

```python
parser.add_argument("--env", default="cartpole", choices=["cartpole", "double"],
                    help="Which environment to run.")
```

- [ ] **Step 2: Replace the env import**

In `play.py`, find this line:

```python
from cartpole_env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402
```

Replace it with:

```python
from env_registry import ENVS  # noqa: E402
```

- [ ] **Step 3: Replace env construction**

In `play.py`, inside `main()`, find these three lines:

```python
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = CartpoleEnv(cfg=cfg, render_mode=None)
```

Replace them with:

```python
    env_cls, cfg_cls = ENVS[args_cli.env]
    cfg = cfg_cls()
    cfg.scene.num_envs = args_cli.num_envs
    env = env_cls(cfg=cfg, render_mode=None)
```

- [ ] **Step 4: Syntax check**

Run: `cd ~/Desktop/cartpole_ws && python -m py_compile play.py && echo OK`
Expected: prints `OK`.

- [ ] **Step 5: Runtime smoke test — play double-pendulum policy**

This uses the checkpoint produced by Task 3's smoke run. First find it:
```bash
cd ~/Desktop/cartpole_ws && ls -t runs/*/policy_final.pt | head -1
```
Then run play with that path:
```bash
cd ~/Desktop/cartpole_ws && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && ~/IsaacLab/isaaclab.sh -p play.py --env double --headless --checkpoint "$(ls -t runs/*/policy_final.pt | head -1)" --num_envs 16 --num_steps 200 2>&1 | grep -E "\[play\]|Error|Traceback"
```
Expected: a `[play] loaded ...` line and a final `[play] N eps ...` (or `[play] no eps completed ...`) line, NO `Traceback`.

- [ ] **Step 6: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add play.py
git commit -m "Wire --env flag into play.py via registry"
```

---

## Task 5: Add `--env` to server.py

**Files:**
- Modify: `server.py`

- [ ] **Step 1: Add the `--env` argument**

In `server.py`, find this line:

```python
parser.add_argument("--seed", type=int, default=42)
```

Immediately AFTER it, add:

```python
parser.add_argument("--env", default="cartpole", choices=["cartpole", "double"],
                    help="Which environment to load for this server session.")
```

- [ ] **Step 2: Replace the env import**

In `server.py`, find this line:

```python
from cartpole_env import CartpoleEnv, CartpoleEnvCfg  # noqa: E402
```

Replace it with:

```python
from env_registry import ENVS  # noqa: E402
```

- [ ] **Step 3: Replace env construction**

In `server.py`, find these three lines:

```python
cfg = CartpoleEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = CartpoleEnv(cfg=cfg, render_mode=None)
```

Replace them with:

```python
_env_cls, _cfg_cls = ENVS[args_cli.env]
cfg = _cfg_cls()
cfg.scene.num_envs = args_cli.num_envs
env = _env_cls(cfg=cfg, render_mode=None)
```

- [ ] **Step 4: Show the env name in the server log**

In `server.py`, find this block:

```python
print(
    f"[server] env ready  num_envs={args_cli.num_envs}  device={device}  "
    f"obs={cfg.observation_space}  act={cfg.action_space}  "
    f"episode_s={cfg.episode_length_s}",
    flush=True,
)
```

Replace it with:

```python
print(
    f"[server] env={args_cli.env} ready  num_envs={args_cli.num_envs}  device={device}  "
    f"obs={cfg.observation_space}  act={cfg.action_space}  "
    f"episode_s={cfg.episode_length_s}",
    flush=True,
)
```

- [ ] **Step 5: Syntax check**

Run: `cd ~/Desktop/cartpole_ws && python -m py_compile server.py && echo OK`
Expected: prints `OK`.

- [ ] **Step 6: Runtime smoke test — server + client with double pendulum**

Start the server in the background:
```bash
cd ~/Desktop/cartpole_ws && rm -f .server.sock && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && nohup ~/IsaacLab/isaaclab.sh -p server.py --headless --socket .server.sock --num_envs 64 --env double > /tmp/dp_srv.log 2>&1 &
```
Wait for the socket to appear:
```bash
cd ~/Desktop/cartpole_ws && until [ -S .server.sock ] || grep -qE "Error|Traceback" /tmp/dp_srv.log; do sleep 1; done; grep "\[server\]" /tmp/dp_srv.log
```
Expected: a line `[server] env=double ready  num_envs=64  ...  obs=6  act=1 ...` and `[server] listening on .server.sock`.

Send a short training command via the client:
```bash
cd ~/Desktop/cartpole_ws && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python client.py train --max_iters 3 --rollout_steps 64 --run_name dp_srv_smoke 2>&1 | grep -E "^iter|\[server\]"
```
Expected: `[server] training ...` then `iter 1/3 ... iter 3/3 ...` then `[server] saved runs/dp_srv_smoke/policy_final.pt`.

Shut the server down:
```bash
pkill -9 -f "server.py --socket .server.sock" ; rm -f ~/Desktop/cartpole_ws/.server.sock ; echo "server stopped"
```

- [ ] **Step 7: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add server.py
git commit -m "Wire --env flag into server.py via registry"
```

---

## Task 6: Add `ENV` variable to the Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the `ENV` variable**

In `Makefile`, find this block:

```make
# --- server / training defaults ----------------------------------------------
NUM_ENVS       ?= 256
MAX_ITERS      ?= 200
ROLLOUT_STEPS  ?= 128
SEED           ?= 42
RUN_NAME       ?=
RESUME         ?=
```

Replace it with:

```make
# --- server / training defaults ----------------------------------------------
ENV            ?= cartpole       # cartpole | double
NUM_ENVS       ?= 256
MAX_ITERS      ?= 200
ROLLOUT_STEPS  ?= 128
SEED           ?= 42
RUN_NAME       ?=
RESUME         ?=
```

- [ ] **Step 2: Pass `--env` to the `env` target**

In `Makefile`, find this block:

```make
	$(ACTIVATE) && $(ISAACLAB) -p server.py $(HEADLESS_FLAG) \
	    --socket $(SOCKET) --num_envs $(NUM_ENVS) --seed $(SEED)
```

Replace it with:

```make
	$(ACTIVATE) && $(ISAACLAB) -p server.py $(HEADLESS_FLAG) \
	    --socket $(SOCKET) --num_envs $(NUM_ENVS) --seed $(SEED) --env $(ENV)
```

- [ ] **Step 3: Pass `--env` to the `smoke` target**

In `Makefile`, find this block:

```make
smoke:
	$(ACTIVATE) && $(ISAACLAB) -p train.py --headless \
	    --num_envs 64 --max_iters 5 --rollout_steps 64
```

Replace it with:

```make
smoke:
	$(ACTIVATE) && $(ISAACLAB) -p train.py --headless \
	    --num_envs 64 --max_iters 5 --rollout_steps 64 --env $(ENV)
```

- [ ] **Step 4: Pass `--env` to the `train-once` target**

In `Makefile`, find this block:

```make
train-once:
	$(ACTIVATE) && $(ISAACLAB) -p train.py $(HEADLESS_FLAG) \
	    --num_envs $(NUM_ENVS) --max_iters $(MAX_ITERS) \
	    --rollout_steps $(ROLLOUT_STEPS) --seed $(SEED) \
	    $(if $(RUN_NAME),--run_name $(RUN_NAME),)
```

Replace it with:

```make
train-once:
	$(ACTIVATE) && $(ISAACLAB) -p train.py $(HEADLESS_FLAG) \
	    --num_envs $(NUM_ENVS) --max_iters $(MAX_ITERS) \
	    --rollout_steps $(ROLLOUT_STEPS) --seed $(SEED) --env $(ENV) \
	    $(if $(RUN_NAME),--run_name $(RUN_NAME),)
```

- [ ] **Step 5: Pass `--env` to the `play-once` target**

In `Makefile`, find this block:

```make
	$(ACTIVATE) && $(ISAACLAB) -p play.py $(HEADLESS_FLAG) \
	    --checkpoint "$$CKPT" --num_envs $(NUM_ENVS) --num_steps $(PLAY_STEPS) \
	    $(if $(PLAY_DET),--deterministic,)
```

Replace it with:

```make
	$(ACTIVATE) && $(ISAACLAB) -p play.py $(HEADLESS_FLAG) \
	    --checkpoint "$$CKPT" --num_envs $(NUM_ENVS) --num_steps $(PLAY_STEPS) \
	    --env $(ENV) $(if $(PLAY_DET),--deterministic,)
```

- [ ] **Step 6: Document `ENV` in the help text**

In `Makefile`, find this line:

```make
	@echo "  make env              - boot Isaac Sim + server (foreground, GUI on)"
```

Replace it with:

```make
	@echo "  make env              - boot Isaac Sim + server (foreground, GUI on)"
	@echo "  make env ENV=double   - boot the server with the double-pendulum env"
```

Then find this line:

```make
	@echo "Vars: NUM_ENVS MAX_ITERS ROLLOUT_STEPS SEED RUN_NAME RESUME"
```

Replace it with:

```make
	@echo "Vars: ENV NUM_ENVS MAX_ITERS ROLLOUT_STEPS SEED RUN_NAME RESUME"
```

- [ ] **Step 7: Verify the Makefile expands correctly**

Run: `cd ~/Desktop/cartpole_ws && make -n env ENV=double 2>&1 | grep -- "--env"`
Expected: a line containing `server.py` ... `--env double`.

Run: `cd ~/Desktop/cartpole_ws && make -n smoke 2>&1 | grep -- "--env"`
Expected: a line containing `train.py` ... `--env cartpole` (the default).

- [ ] **Step 8: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add Makefile
git commit -m "Add ENV variable to Makefile, plumb --env to all targets"
```

---

## Task 7: Update the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Mention both environments in the intro**

In `README.md`, find this line:

```markdown
The environment is configured for **swing-up** by default (pole starts hanging) but can be switched to a balance task in a few lines.
```

Replace it with:

```markdown
Two environments are included: a single cartpole (`cartpole`) and an underactuated cart + double pendulum (`double`), both configured for **swing-up**. Pick one at server-startup time with `ENV=...`.
```

- [ ] **Step 2: Document the `cart_double_pendulum_env.py` file**

In `README.md`, find this table row:

```markdown
| `cartpole_env.py` | `DirectRLEnv` subclass with banner-marked **STATE** and **REWARD** sections |
```

Immediately AFTER it, add this row:

```markdown
| `cart_double_pendulum_env.py` | Underactuated cart + double-pendulum `DirectRLEnv` (swing-up) |
| `env_registry.py` | Maps env names (`cartpole`, `double`) to their classes |
```

- [ ] **Step 3: Document the `ENV` flag in the quick-start**

In `README.md`, find this block:

```markdown
### 1. Boot the persistent server (terminal A)
```bash
make env                  # GUI viewport
make env HEADLESS=1       # no GUI (faster startup, headless training)
make env NUM_ENVS=512     # more parallel envs
```
```

Replace it with:

```markdown
### 1. Boot the persistent server (terminal A)
```bash
make env                  # single cartpole, GUI viewport
make env ENV=double       # cart + double pendulum
make env HEADLESS=1       # no GUI (faster startup, headless training)
make env NUM_ENVS=512     # more parallel envs
```

The server loads ONE environment for its lifetime. To switch envs, stop the
server (Ctrl+C or `make kill-server`) and boot it again with a different `ENV`.
```

- [ ] **Step 4: Commit**

```bash
cd ~/Desktop/cartpole_ws
git add README.md
git commit -m "Document the double-pendulum env and ENV flag in README"
```

---

## Task 8: Regression check — single cartpole still works

**Files:** none (verification only)

- [ ] **Step 1: Smoke test the cartpole path**

Run:
```bash
cd ~/Desktop/cartpole_ws && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && ~/IsaacLab/isaaclab.sh -p train.py --env cartpole --headless --num_envs 64 --max_iters 3 --rollout_steps 64 2>&1 | grep -E "^iter|\[train\]|Error|Traceback"
```
Expected:
- `[train] env=cartpole  num_envs=64  device=cuda:0  obs=4  act=1`
- three `iter` lines
- NO `Traceback`

This confirms the registry refactor did not break the original env.

- [ ] **Step 2: Final cleanup commit (if anything is uncommitted)**

```bash
cd ~/Desktop/cartpole_ws
git status --short
```
Expected: empty output (everything already committed). If anything shows,
investigate before committing — nothing in Task 8 changes files.

---

## Self-Review Notes

- **Spec coverage:** new env file (Task 1), `env_registry.py` (Task 2),
  `--env` in server/train/play (Tasks 5/3/4), Makefile `ENV` (Task 6),
  README (Task 7). PPO and client untouched, as the spec requires.
- **Type consistency:** the registry tuple order is `(EnvClass, CfgClass)`
  in Task 2; Tasks 3/4/5 all unpack it as `env_cls, cfg_cls = ENVS[...]`
  (server uses `_env_cls, _cfg_cls`). Config classes: `CartpoleEnvCfg`,
  `CartDoublePendulumEnvCfg`. Env classes: `CartpoleEnv`,
  `CartDoublePendulumEnv`. `observation_space` is 4 for cartpole, 6 for
  double — matched to the obs tensors.
- **Known risk (from the spec):** double-pendulum swing-up may not fully
  converge with vanilla PPO. The smoke tests in this plan only verify the
  env *runs and trains without error* — they do NOT assert the task is
  solved. Tuning for convergence is deliberately out of scope.
