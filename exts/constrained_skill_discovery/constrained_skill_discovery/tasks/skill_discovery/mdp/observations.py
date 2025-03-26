from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    # envTypeCheck = ManagerBasedRLEnv


def end_effector_velocity(
    env: ManagerBasedRLEnv,
    transform_cfg: SceneEntityCfg = SceneEntityCfg("end_effector_frame_transformer"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the velocity of the robot arm end effector in the world frame."""
    # extract the used quantities (to enable type-hinting)
    frame_transformer: FrameTransformer = env.scene[transform_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.body_state_w[:, -1, 7:10]


def end_effector_position(
    env: ManagerBasedRLEnv,
    transform_cfg: SceneEntityCfg = SceneEntityCfg("end_effector_frame_transformer"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the position of the robot arm end effector in the world frame."""
    # extract the used quantities (to enable type-hinting)
    frame_transformer: FrameTransformer = env.scene[transform_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    # base_pos = asset.data.root_pos_w
    # base_pos_local = base_pos.clone()

    # base_pos_local[:,:2] = base_pos[:,:2] - env.scene.env_origins[:,:2]

    # end_effector_pos_b = frame_transformer.data.target_pos_source.view(env.num_envs, -1).clone()

    # # end_effector_pos_b[:,:3] = end_effector_pos_b[:,:3] + env.scene.env_origins[:,:3]

    # end_effector_pos = torch.zeros_like(end_effector_pos_b)

    # end_effector_pos[:,:3] = end_effector_pos_b[:,:3] + base_pos_local

    end_effector_pos = (
        asset.data.body_state_w[:, -1, 0:3] - env.scene.env_origins[:, :3]
    )

    return end_effector_pos


def base_yaw(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The yaw of the robot base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return euler_xyz_from_quat(asset.data.root_quat_w)[2].reshape(-1, 1)


# def joint_pos(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """The joint positions of the asset.

#     NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return asset.data.joint_pos[:, asset_cfg.joint_ids]


# def joint_vel(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ):
#     """The joint velocities of the asset .


#     NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return asset.data.joint_vel[:, asset_cfg.joint_ids]
def base_roll_pitch_yaw(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat(
        (roll.unsqueeze(-1), pitch.unsqueeze(-1), yaw.unsqueeze(-1)), dim=-1
    )


def base_lin_vel_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    return torch.zeros((env.num_envs, 3), device=env.device)


def base_quaternion(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Get the quaternion of the robot base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w


def base_position(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Get the position of the robot base in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    base_pos = asset.data.root_pos_w
    base_pos_local = base_pos.clone()
    # base_pos_local[:, :2] = (
    #     base_pos[:, :2] - env.initial_root_states[: env.num_envs, :2]
    # )  # - env.scene.env_origins[:,:2]
    base_pos_local[:, :2] = base_pos[:, :2] - env.scene.env_origins[:, :2]
    return base_pos_local


def foot_position_b(
    env: ManagerBasedRLEnv,
    transform_cfg: SceneEntityCfg = SceneEntityCfg("foot_frame_transformer"),
) -> torch.Tensor:
    """Get the position of the robot feet in the robot base frame."""
    # extract the used quantities (to enable type-hinting)
    frame_transformer: FrameTransformer = env.scene[transform_cfg.name]

    return frame_transformer.data.target_pos_source.view(env.num_envs, -1)


def energy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The instantenous energy used by the robot."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    dof_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    dof_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    return torch.sum(torch.abs(dof_vel) * torch.abs(dof_torque), dim=1, keepdim=True)
