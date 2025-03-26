# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
##
# Pre-defined configs
##
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import constrained_skill_discovery.tasks.skill_discovery.mdp as mdp_sd
from constrained_skill_discovery.lab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab_assets.ant import ANT_CFG  # isort: skip


@configclass
class SkillsCfg:
    """Command specifications for the MDP."""

    skills = mdp_sd.SkillCommandCfg(
        asset_name="robot",
        discrete_skills=False,
        skill_dim=2,
        max_magnitude=1.5,
        normalise=False,
        resampling_time_range=(0, 0),
        debug_vis=False,
    )

    position_commands = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-10.0, 10.0), pos_y=(-10.0, 10.0), heading=(-3.14, 3.14)
        ),
        simple_heading=True,
        debug_vis=False,
        resampling_time_range=(6, 6),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """This is the low level policy that takes some observations and returns an action."""

        base_pos = ObsTerm(func=mdp.root_pos_w)
        # base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        base_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        base_yaw_roll = ObsTerm(func=mdp_sd.base_roll_pitch_yaw)
        # base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        # base_up_proj = ObsTerm(func=mdp.base_up_proj)
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel, scale=0.05)
        # feet_body_forces = ObsTerm(
        #     func=mdp.body_incoming_wrench,
        #     scale=0.1,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
        #         )
        #     },
        # )
        actions = ObsTerm(func=mdp.last_action)

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

        base_pos = ObsTerm(func=mdp.root_pos_w)
        # base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        base_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        base_yaw_roll = ObsTerm(func=mdp_sd.base_roll_pitch_yaw)
        # base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        # base_up_proj = ObsTerm(func=mdp.base_up_proj)
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel, scale=0.05)
        # feet_body_forces = ObsTerm(
        #     func=mdp.body_incoming_wrench,
        #     scale=0.1,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
        #         )
        #     },
        # )
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    skill_discovery: SkillDiscoveryObsCfg = SkillDiscoveryObsCfg()
    skill_conditioning: SkillConditioningCfg = SkillConditioningCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()
    policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (2) Stay alive bonus
    # alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # (3) Reward for non-upright posture
    upright = RewTerm(
        func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93}
    )

    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}}
    )
    # (7) Penalty for reaching close to joint limits
    joint_limits = RewTerm(
        func=mdp.joint_limits_penalty_ratio,
        weight=-0.1,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=7.5
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp_sd.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})

    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle":1.5})

    evaluation = DoneTerm(func=mdp_sd.evaluate_envs)


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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class AntSDEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4296, env_spacing=5.0)
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
        self.decimation = 2
        self.episode_length_s = 6.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

        """Post initialization."""

        # self.scene.robot = ANYMAL_C_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum


@configclass
class AntSDEnvCfg_PLAY(AntSDEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 50
        self.scene.env_spacing = 0  # 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory

        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (20.0, 20.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.low_level_policy.enable_corruption = False
        self.terminations.torso_height = None
        #     # remove random pushing

        # self.episode_length_s = 1000.0
