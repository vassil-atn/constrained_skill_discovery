# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    FrameTransformerCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import constrained_skill_discovery.tasks.skill_discovery.mdp as mdp
from constrained_skill_discovery.lab.terrains.config.rough import ROUGH_TERRAINS_CFG

# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
##
# Pre-defined configs
##
# from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip
from constrained_skill_discovery.lab_assets.anymal import ANYMAL_D_CFG # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG # isort: skip


@configclass
class SkillsCfg:
    """Command specifications for the MDP."""

    skills = mdp.SkillCommandCfg(
        asset_name="robot",
        discrete_skills=False,
        skill_dim=3,
        max_magnitude=1.0,
        normalise=False,
        resampling_time_range=(0, 0),
        debug_vis=False,
    )

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0), heading=(-3.14, 3.14)
        ),
        simple_heading=False,
        debug_vis=True,
        resampling_time_range=(12, 12),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """This is the low level policy that takes some observations and returns an action."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0
        )
        base_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25
        )
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # yaw = ObsTerm(func=mdp.base_yaw, noise=Unoise(n_min=-0.1, n_max=0.1),scale=1.)
        base_quat = ObsTerm(
            func=mdp.base_quaternion, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # velocity_commands = ObsTerm(func=mdp.base_lin_vel_cmd, scale=0.0)

        joint_pos = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01), scale=1.0
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     scale=0.0,
        # )
        # base_pos = ObsTerm(
        #     func=mdp.base_position, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.0
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class SkillConditioningCfg(ObsGroup):
        selected_skills = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "skills"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SkillDiscoveryObsCfg(ObsGroup):
        """Observations for skill discovery group."""

        base_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0
        )
        base_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25
        )
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # yaw = ObsTerm(func=mdp.base_yaw, noise=Unoise(n_min=-0.1, n_max=0.1),scale=1.)
        base_quat = ObsTerm(
            func=mdp.base_quaternion, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=0.0)
        # velocity_commands = ObsTerm(func=mdp.base_lin_vel_cmd, noise=Unoise(n_min=-0.1, n_max=0.1),scale=0.0)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
            # clip=(-3.0, 3.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05
        )
        # actions = ObsTerm(func=mdp.last_action,scale=0.0)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     scale=0.0
        # )
        base_pos = ObsTerm(
            func=mdp.base_position, noise=Unoise(n_min=-0.1, n_max=0.1), scale=1.0
        )
        # foot_pos_b = ObsTerm(func=mdp.foot_position_b, noise=Unoise(n_min=-0.1, n_max=0.1), params={"transform_cfg": SceneEntityCfg("foot_frame_transformer")})
        # energy = ObsTerm(func=mdp.energy, noise=Unoise(n_min=-0.1, n_max=0.1),scale=0.1)
        # end_effector_position = ObsTerm(func=mdp.end_effector_position, noise=Unoise(n_min=-0.1, n_max=0.1), params={"transform_cfg": SceneEntityCfg("end_effector_frame_transformer")}, scale=1.0)
        # end_effector_velocity = ObsTerm(func=mdp.end_effector_velocity, noise=Unoise(n_min=-0.1, n_max=0.1), params={"transform_cfg": SceneEntityCfg("end_effector_frame_transformer")}, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25
        )
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        base_quat = ObsTerm(
            func=mdp.base_quaternion, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=0.0)
        # velocity_commands = ObsTerm(func=mdp.base_lin_vel_cmd,scale=0.0)
        position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
            scale=1.0,
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01), scale=1.0
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        base_pos = ObsTerm(func=mdp.base_position, scale=0.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    skill_discovery: SkillDiscoveryObsCfg = SkillDiscoveryObsCfg()
    skill_conditioning: SkillConditioningCfg = SkillConditioningCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    no_contact_base = RewTerm(
        func=mdp.body_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "LF_THIGH",
                    "RF_THIGH",
                    "LH_THIGH",
                    "RH_THIGH",
                    "LF_HIP",
                    "RF_HIP",
                    "LH_HIP",
                    "RH_HIP",
                    "LF_SHANK",
                    "RF_SHANK",
                    "LH_SHANK",
                    "RH_SHANK",
                    "base",
                ],
            ),
        },
        weight=20.0,
    )

    # # undesired_contacts_thigh = RewTerm(
    # #     func=mdp.undesired_contacts,
    # #     weight=-1.0,
    # #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # # )

    # # undesired_contacts_shank = RewTerm(
    # #     func=mdp.undesired_contacts,
    # #     weight=-1.0,
    # #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*SHANK"), "threshold": 1.0},
    # # )

    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, params={"soft_ratio": 1.0}, weight=-1.0
    )
    # # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.)

    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.1)
    energy = RewTerm(func=mdp.energy_reward, weight=-1e-3)

    # dof_pos_close_to_nominal = RewTerm(func=mdp.joint_deviation_l1, weight=-2.0)

    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)

    feet_air_time = RewTerm(
        func=mdp.sd_feet_air_time,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "skills",
            "threshold": 0.5,
        },
    )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)

    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.6}
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-10.0, 10.0)},
    )


@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2,0.2), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
    #         "threshold": 1.0,
    #     },
    # )

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})

    evaluation = DoneTerm(func=mdp.evaluate_envs)


@configclass
class CurriculumCfg:
    """Curriculum settings for the MDP."""

    terrain_levels = None


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Stucco.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    foot_frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/.*FOOT"),
        ],
        debug_vis=False,
    )

    # objects
    # box = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/box",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 10.0, 0.6),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             retain_accelerations=False,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=1000.0,
    #             max_depenetration_velocity=1.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=10000.0, density=400.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(5.0, 0.0, 0.56),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )

    # end_effector_frame_transformer = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Robot/gripperStator"),
    #     ],
    #     debug_vis=False,
    # )


@configclass
class EventPlayCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (1.0,1.0),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.add_body_mass,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (0.0, 0.0)},
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.8, 0.8),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class TerminationsPlayCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})


@configclass
class AnymalDHierarchicalEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4296, env_spacing=2.5)
    # Basic settings
    commands: SkillsCfg = SkillsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()  # inherit the one from parent
    curriculum: CurriculumCfg = CurriculumCfg()

    actions: ActionsCfg = ActionsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    is_finite_horizon: bool = True

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # switch robot to anymal-c
        """Post initialization."""

        # self.scene.robot = ANYMAL_C_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.episode_length_s = 6.0


@configclass
class AnymalDHierarchicalEnvCfg_PLAY(AnymalDHierarchicalEnvCfg):
    events: EventCfg = EventPlayCfg()
    terminations: TerminationsCfg = TerminationsPlayCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 50
        self.scene.env_spacing = 0  # 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (20.0, 20.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.low_level_policy.enable_corruption = False
        #     # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.episode_length_s = 12.0
