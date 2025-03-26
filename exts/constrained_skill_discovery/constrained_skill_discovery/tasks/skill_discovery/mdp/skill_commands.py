# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .skill_commands_cfg import SkillCommandActionCfg, SkillCommandCfg


class SkillCommandAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: SkillCommandActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: SkillCommandActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )


class SkillCommands(CommandTerm):
    """Command generator that generates position commands based on the terrain.

    The position commands are sampled from the terrain mesh and the heading commands are either set
    to point towards the target or are sampled uniformly.
    """

    cfg: SkillCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: SkillCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.skills = torch.zeros(self.num_envs, cfg.skill_dim, device=self.device)

        self.skill_dim = cfg.skill_dim
        self.discrete_skills = cfg.discrete_skills

        # set the maximum magnitude of the skills
        self.max_magnitude = cfg.max_magnitude
        self.normalise = cfg.normalise

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.skills

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new skills
        # randomize the skills
        if self.discrete_skills:
            skills = torch.randint(
                0,
                self.skill_dim,
                (len(env_ids),),
                dtype=torch.float,
                device=self.device,
            )
            # Get one hot encoding:
            self.skills[env_ids] = torch.nn.functional.one_hot(
                skills.to(torch.int64), self.skill_dim
            ).to(torch.float32)
        else:
            # skills = torch.randn((len(env_ids),self.skill_dim), dtype=torch.float, device=self.device)
            # mean = torch.tensor([0.0, 0.0])  # Mean vector
            # covariance_matrix = 0.3*torch.tensor([[1.0, 0.], [0., 1.0]])  # Covariance matrix

            # distro = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix)
            # skills = distro.sample([len(env_ids)]).clamp(min=-1.0, max=1.0).to(self.device)#*15
            # skills = torch.stack((torch.distributions.beta.Beta(1.5,1. ).sample([len(env_ids)]),torch.distributions.beta.Beta(1.5,1. ).sample([len(env_ids)])), dim=1).to(self.device)
            # Normalize skills:

            # Sample with Box-Mueller algorithm (from inside an N-dimensional sphere of max radius max_magnitude)
            X = torch.randn((len(env_ids), self.skill_dim), device=self.device)
            U = torch.rand((len(env_ids), 1), device=self.device)

            skills = (
                self.max_magnitude
                * U ** (1 / self.skill_dim)
                / torch.sqrt(torch.sum(torch.square(X), dim=-1, keepdim=True))
                * X
            )

            # skills = (
            #     torch.distributions.Normal(0, self.max_magnitude / 3)
            #     .sample((len(env_ids), self.skill_dim))
            #     .to(self.device)
            # )

            # theta = torch.distributions.uniform.Uniform(0,2*torch.pi).sample([len(env_ids)])
            # radius = torch.distributions.uniform.Uniform(0.0**2,self.max_magnitude**2).sample([len(env_ids)]) ** 0.5

            # skills = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1).to(self.device)

            # radius[radius<1.] = 0.0
            # skills = torch.rand((len(env_ids),self.skill_dim), dtype=torch.float, device=self.device) * 2*max_magnitude - max_magnitude

            # skills = torch.stack((radius*torch.cos(theta), radius*torch.sin(theta)), dim=1).to(self.device)

            # radius = torch.distributions.uniform.Uniform(0.0,self.max_magnitude).sample([len(env_ids),1]).to(self.device)
            # Magnitude is either 0.5 or 1:
            # radius_binary = radius.clone()
            # radius_binary[radius>=self.max_magnitude/2] = self.max_magnitude
            # radius_binary[radius<self.max_magnitude/2] = self.max_magnitude/2

            # skills = radius_binary * skills / torch.norm(skills, dim=1).unsqueeze(1)

            if self.normalise:
                skills = skills / torch.norm(skills, dim=1).unsqueeze(1)

            # self.skills[env_ids] = radius * skills / torch.norm(skills, dim=1).unsqueeze(1)
            self.skills[env_ids] = skills

            # skills_magnitude = torch.rand((len(env_ids),1), dtype=torch.float, device=self.device)

            # self.skills[env_ids] = self.skills[env_ids] * skills_magnitude
            # skills = torch.rand((len(env_ids),self.skill_dim), dtype=torch.float, device=self.device)
            # Scale between -50 and 50
            # print('-')
            # self.skills[env_ids] = skills * 1000 - 500

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0 and self.cfg.resampling_time_range[1] > 0.0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command()

    def _evaluation_commands(self, env_ids: Sequence[int]):
        """Generate the evaluation commands. Uses the angular pattern for 2D skills and random for 3D skills.
        Args:
            env_ids: The environment IDs to generate the commands for.
        """
        if self.skill_dim == 2:
            angles = torch.linspace(0, 2 * np.pi, len(env_ids), device=self.device)
            skills = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
            self.skills[env_ids] = skills * torch.linspace(
                0, self.max_magnitude, len(env_ids) // 10, device=self.device
            ).repeat(10).unsqueeze(1)
        else:
            self._resample_command(env_ids) 

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "skill_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/commanded_skill"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.skill_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.skill_visualizer.set_visibility(True)
        else:
            if hasattr(self, "skill_visualizer"):
                self.skill_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        skill_arrow_scale, skill_arrow_quat = self._resolve_skill_to_arrow(
            self.skills[:, :2]
        )
        # display markers
        self.skill_visualizer.visualize(base_pos_w, skill_arrow_quat, skill_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_skill_to_arrow(
        self, skill: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the unit skill vector to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.skill_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            skill.shape[0], 1
        )
        arrow_scale[:, 0] *= 2.0
        # arrow-direction
        heading_angle = torch.atan2(skill[:, 1], skill[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        # base_quat_w = self.robot.data.root_quat_w
        # arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
