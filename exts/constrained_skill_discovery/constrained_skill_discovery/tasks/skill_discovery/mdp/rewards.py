# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


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
    lin_vel_unit = lin_vel / (torch.norm(lin_vel, dim=-1, keepdim=True) + 1e-6)
    # compute the reward
    reward = torch.sum(lin_vel_unit * des_pos_unit, dim=-1)

    return reward


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_rudin(
    env: ManagerBasedRLEnv,
    command_name: str,
    reward_duration_s: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    t_steps = env.episode_length_buf
    reward_dur = reward_duration_s / (env.step_dt)
    ep_len = env.max_episode_length
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    reward = torch.zeros_like(distance)
    idx_reward = t_steps >= (ep_len - reward_dur)

    reward[idx_reward] = (1 / reward_dur) * (1 / (1 + distance[idx_reward] ** 2))

    return reward


def end_effector_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The end-effector positions of the asset.

    NOTE: Only the end-effectors configured in :attr:`asset_cfg.end_effector_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.data.body_pos_w[:, asset_cfg.body_ids].reshape(-1, 3)
        - env.scene.env_origins
    )


def energy_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The instantenous energy used by the robot."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    dof_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    dof_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    return torch.sum(torch.abs(dof_vel) * torch.abs(dof_torque), dim=1)


def sd_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    max_magnitude = env.command_manager._terms["skills"].max_magnitude
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > (
        max_magnitude / 10
    )
    return reward


def vel_to_target_commands(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for moving towards the goal."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    target_rel_pos = env.command_manager.get_command("position_commands")[:, :2]

    base_lin_vel = asset.data.root_lin_vel_w[:, :2]

    # compute the reward
    reward = torch.sum(base_lin_vel * target_rel_pos, dim=-1) / (
        torch.norm(base_lin_vel, dim=-1) * torch.norm(target_rel_pos, dim=-1) + 1e-6
    )

    return reward


def position_commands(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for reaching a target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    target_positions = env.command_manager.get_command("position_commands")[:, :2]
    # current_pos = asset.data.root_pos_w[:, :2]
    # current_pos_local = current_pos - env.scene.env_origins[:, :2]
    # at_end_of_episode = env.episode_length_buf >= 2*env.max_episode_length/3
    # compute the reward
    reward = 1 / (1 + torch.norm(target_positions, dim=-1))
    return reward


def body_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    reward = (
        ~torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
            )[0]
            > threshold,
            dim=-1,
        )
    ).float()

    return reward


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)
