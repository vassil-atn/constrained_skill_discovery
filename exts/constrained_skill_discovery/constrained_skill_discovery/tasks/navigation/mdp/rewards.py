# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity towards the goal."""
    asset: RigidObject = env.scene[asset_cfg.name]

    des_pos = (
        env.command_manager.get_term(command_name).pos_command_w[:, :2]
        - asset.data.root_pos_w[:, :2]
    )
    des_pos_unit = des_pos / (torch.norm(des_pos, dim=-1, keepdim=True) + 1e-6)

    lin_vel = asset.data.root_lin_vel_w[:, :2]
    # compute the reward
    reward = torch.sum(lin_vel * des_pos_unit, dim=-1)

    return reward


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str
) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()
