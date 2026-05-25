"""Cartpole DirectRLEnv with clearly marked STATE and REWARD sections.

This is the file you'll edit when experimenting. Two places to change:

  1. ``_get_observations`` (the STATE block) — what the policy sees.
     If you change the dimension, also update ``observation_space`` in
     ``CartpoleEnvCfg`` below.

  2. ``_get_rewards`` (the REWARD block) — the reward function.
     Add/remove/scale individual terms freely.

Action is a 1-D continuous force on the cart slider (scaled by ``action_scale``).
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

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # ---- episode / sim ----
    decimation = 2
    episode_length_s = 8.0  # swing-up needs more time than balance
    action_scale = 100.0  # [N] applied to the cart slider

    # ---- spaces ----
    # NOTE: observation_space MUST match the dim of the tensor returned by
    # _get_observations(). If you change the state, change this number too.
    action_space = 1
    observation_space = 4
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ---- robot ----
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # ---- scene (num_envs overridden from CLI in train.py) ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # ---- reset / termination ----
    max_cart_pos = 3.0  # cart out-of-bounds threshold [m]

    # Initial pole angle, multiplied by pi internally. Pole hangs at +/- pi.
    # SWING-UP DEFAULT: pole hangs down with small noise (angle near pi).
    initial_pole_angle_range = [0.95, 1.05]   # ~ pi +/- 0.05*pi (pole hanging)
    # BALANCE VARIANT: pole upright with small perturbation.
    # initial_pole_angle_range = [-0.25, 0.25]  # ~ 0 +/- pi/4

    # Pole termination angle. SWING-UP: disabled (pole sweeps full range).
    # Set to a large number so it never triggers. For balance, use math.pi / 2.
    fail_pole_angle = 1e9


class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    # ----------------------------------------------------------------------
    # Scene setup (rarely edited)
    # ----------------------------------------------------------------------
    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------------------------------------------------------------
    # Action application
    # ----------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Policy outputs values in roughly [-1, 1] (tanh + small Gaussian noise);
        # scale to force in Newtons.
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    # ======================================================================
    # ============================== STATE =================================
    # Edit here to change what the policy sees.
    # Keep ``CartpoleEnvCfg.observation_space`` in sync with the output dim.
    # ======================================================================
    def _get_observations(self) -> dict:
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(-1)
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(-1)
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(-1)
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(-1)

        # --- DEFAULT: 4-D state ----------------------------------------
        obs = torch.cat([pole_pos, pole_vel, cart_pos, cart_vel], dim=-1)
        # ---------------------------------------------------------------

        # --- VARIANT A: 5-D with sin/cos pole angle (handles wrap-around)
        # obs = torch.cat([
        #     torch.sin(pole_pos), torch.cos(pole_pos),
        #     pole_vel, cart_pos, cart_vel,
        # ], dim=-1)
        # ---> set observation_space = 5 in CartpoleEnvCfg above

        # --- VARIANT B: 3-D minimal (no cart info; harder task)
        # obs = torch.cat([pole_pos, pole_vel, cart_vel], dim=-1)
        # ---> set observation_space = 3

        return {"policy": obs}

    # ======================================================================
    # ============================== REWARD ================================
    # Edit here to change the reward function. Each term is computed as a
    # tensor of shape (num_envs,) and the final reward is their sum.
    # ======================================================================
    def _get_rewards(self) -> torch.Tensor:
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        terminated = self.reset_terminated.float()  # 1 if env just hit a failure termination

        # --- SWING-UP DEFAULT --------------------------------------------
        # cos(angle) is +1 at upright, -1 at hanging: dense and smooth.
        # Bonus for being near upright AND moving slowly (encourages stopping at top).
        r_upright = torch.cos(pole_pos)                          # in [-1, 1]
        r_at_top_slow = (torch.cos(pole_pos) > 0.95).float() * (-0.1 * pole_vel.pow(2))
        r_cart_center = -0.01 * cart_pos.pow(2)                  # keep cart near 0
        r_cart_quiet = -0.005 * cart_vel.abs()
        r_terminate = -10.0 * terminated                         # hit cart bound = big penalty
        reward = r_upright + r_at_top_slow + r_cart_center + r_cart_quiet + r_terminate
        # -----------------------------------------------------------------

        # --- BALANCE VARIANT (the original, for an upright start) --------
        # r_alive = 1.0 * (1.0 - terminated)
        # r_terminate = -2.0 * terminated
        # r_pole_upright = -1.0 * pole_pos.pow(2)
        # r_cart_quiet = -0.01 * cart_vel.abs()
        # r_pole_quiet = -0.005 * pole_vel.abs()
        # reward = r_alive + r_terminate + r_pole_upright + r_cart_quiet + r_pole_quiet

        # --- VARIANT: sparse swing-up (only reward at the top, no shaping)
        # reward = (torch.cos(pole_pos) > 0.95).float() - 10.0 * terminated

        return reward

    # ----------------------------------------------------------------------
    # Termination
    # ----------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # SWING-UP: cfg.fail_pole_angle is huge, so this check never trips.
        # BALANCE: set cfg.fail_pole_angle = math.pi / 2 to re-enable.
        out_of_bounds = out_of_bounds | torch.any(
            torch.abs(self.joint_pos[:, self._pole_dof_idx]) > self.cfg.fail_pole_angle, dim=1
        )
        return out_of_bounds, time_out

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
