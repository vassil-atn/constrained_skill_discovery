# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import constrained_skill_discovery.tasks.skill_discovery.mdp as mdp

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##
@configclass
class SkillsCfg:
    """Command specifications for the MDP."""

    skills = mdp.SkillCommandCfg(
        asset_name="robot",
        discrete_skills=False,
        skill_dim=2,
        max_magnitude=1.0,
        normalise=True,
        resampling_time_range=(0, 0),
        debug_vis=False,
    )

    position_commands = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.6, 0.6),
            pos_y=(-0.6, 0.6),
            pos_z=(0.0, 0.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        body_name="panda_hand",
        debug_vis=True,
        resampling_time_range=(6, 6),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee_pos = ObsTerm(
            func=mdp.end_effector_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
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
        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee_pos = ObsTerm(
            func=mdp.end_effector_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        ee_pos = ObsTerm(
            func=mdp.end_effector_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    skill_discovery: SkillDiscoveryObsCfg = SkillDiscoveryObsCfg()
    skill_conditioning: SkillConditioningCfg = SkillConditioningCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # pass
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: SkillsCfg = SkillsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
