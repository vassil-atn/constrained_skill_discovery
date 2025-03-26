# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip

import constrained_skill_discovery.tasks.navigation.mdp as mdp
from constrained_skill_discovery.lab.terrains.config.rough import MAZE_TERRAINS_CFG

# from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from constrained_skill_discovery.tasks.skill_discovery.config.anymal_d.hierarchical_skill_disc_env_cfg import (
    AnymalDHierarchicalEnvCfg,
)

LOW_LEVEL_ENV_CFG = AnymalDHierarchicalEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
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


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.SDPreTrainedPolicyActionCfg = mdp.SDPreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        policy_path="exts/constrained_skill_discovery/constrained_skill_discovery/lab_assets/policies/policy_2024-09-27_14-33-18.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.low_level_policy,
        low_level_skills=LOW_LEVEL_ENV_CFG.observations.skill_conditioning,
        max_magnitude=50.0,
        scale_magnitude=50.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 4.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    vel_to_goal = RewTerm(
        func=mdp.velocity_to_goal,
        weight=5.0,
        params={"command_name": "pose_command"},
    )
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-0.2,
    #     params={"command_name": "pose_command"},
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(16.0, 16.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(4.0, 4.0), pos_y=(7.0, 7.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MAZE_TERRAINS_CFG,
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


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: MySceneCfg = MySceneCfg(num_envs=4296, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 25
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.terrain.terrain_generator = MAZE_TERRAINS_CFG
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation
                * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (20.0, 20.0)
