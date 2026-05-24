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
    action_scale = 250.0  # [N] applied to the cart slider (well under the 400 N actuator limit)

    # ---- spaces ----
    # observation_space MUST match the dim returned by _get_observations().
    action_space = 1
    observation_space = 8
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # ---- robot ----
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    # ---- scene (num_envs overridden from CLI) ----
    # NOTE: no clone_in_fabric=True here (unlike cartpole_env.py). Isaac Lab's
    # reference cart_double_pendulum env omits it; we match the reference to
    # avoid fabric-cloning issues with the double-pendulum asset.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=4.0, replicate_physics=True
    )

    # ---- reset / termination ----
    # Widened 3.0 -> 5.0 -> 7.0. The policy now swings the cart aggressively
    # enough that 5 m was too tight; cart was hitting the boundary mid-episode.
    max_cart_pos = 7.0  # cart out-of-bounds threshold [m]

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

        # --- DEFAULT: 8-D sin/cos state (smooth, no wrap discontinuity) ---
        # Pendulum spins through ±π often during swing-up; wrap_to_pi creates
        # a sharp input jump at ±π that confuses the policy. sin/cos is C-infinity.
        theta1 = pole_pos                          # link-1 angle
        theta2_abs = pole_pos + pend_pos           # link-2 absolute angle
        obs = torch.cat([
            cart_pos, cart_vel,
            torch.sin(theta1), torch.cos(theta1), pole_vel,
            torch.sin(theta2_abs), torch.cos(theta2_abs), pend_vel,
        ], dim=-1)
        # -------------------------------------------------------------------

        # --- VARIANT: 6-D raw state (wrapped to [-π, π]) -------------------
        # obs = torch.cat([
        #     cart_pos, cart_vel,
        #     wrap_to_pi(pole_pos), pole_vel,
        #     wrap_to_pi(pend_pos), pend_vel,
        # ], dim=-1)
        # ---> set observation_space = 6 in CartDoublePendulumEnvCfg above

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
        # NOTE: keep this coefficient small (-0.1). Larger values (e.g. -0.5)
        # cause the policy to "crawl" — moving fast through upright costs
        # hundreds of points (v² scales quickly), so the policy gives up on
        # real swing-up and tries to quasi-statically align the pendulum.
        r_at_top_slow = both_up.float() * (-0.1 * (pole_vel.pow(2) + pend_vel.pow(2)))
        # Explicit +1/step bonus for being tightly upright. Without this the
        # cos shaping flattens near the top and the policy has no strong
        # gradient pulling it to "stay" once it gets there.
        r_stay_at_top = both_up.float() * 2.0
        r_cart_center = -0.01 * cart_pos.pow(2)
        # Smooth boundary repulsion: ~0 in the middle, ramps up sharply near
        # ±max_cart_pos. 4th power keeps it tiny for moderate swings (e.g. at
        # cart=±3.5 of 7 it's just -0.03) but lethal near the bound (-0.5
        # at exactly the wall). Pulls the cart back BEFORE termination triggers.
        r_cart_bound_proximity = -0.5 * (cart_pos / self.cfg.max_cart_pos).pow(4)
        # r_cart_quiet = -0.005 * cart_vel.abs()
        # Was -10.0; reduced so the policy stops fearing the cart bound
        # and can explore swinging the cart through ±max_cart_pos for energy pumping.
        r_terminate = -1.0 * terminated
        reward = (
            r_upright + r_at_top_slow + r_stay_at_top
            + r_cart_center + r_cart_bound_proximity + r_terminate
        )
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
