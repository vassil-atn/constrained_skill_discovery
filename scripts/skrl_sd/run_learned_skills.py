# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher
from rl_games.algos_torch.flatten import TracingAdapter

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=100,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=600,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Anymal-Skill-Discovery", help="Name of the task."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--sd_version",
    type=str,
    default=None,
    help="Which skill discovery method to use (base vs MSE).",
)
parser.add_argument(
    "--skill_discovery_distance_metric",
    type=str,
    default=None,
    help="Which constraint to use ('l2' or 'one').",
)
parser.add_argument(
    "--max_magnitude",
    type=float,
    default=None,
    help="Maximum magnitude for sampling skills.",
)
parser.add_argument(
    "--normalise_encoder_states",
    type=int,
    default=None,
    help="Whether to normalise the encoder inputs with a running mean/variance.",
)
parser.add_argument(
    "--skill_dim",
    type=int,
    default=None,
    help="Skill space dimension.",
)
parser.add_argument(
    "--discrete_skills",
    type=bool,
    default=None,
    help="Whether to use discrete skills.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""


import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

# from isaaclab_tasks.utils.wrappers.skrl_LSD import SkrlVecEnvWrapper, process_skrl_cfg
from constrained_skill_discovery.tasks.utils.wrappers.skrl_LSD import (
    SkrlVecEnvWrapper,
    process_skrl_cfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

# from omni.isaac.core.utils.extensions import enable_extension
from isaacsim.core.utils.extensions import enable_extension
from matplotlib import pyplot as plt

from skrl.agents.torch.exploration.ppo_hierarchical import (
    PPO_DEFAULT_CONFIG,
    HierarchicalLSDPPO,
)
from skrl.agents.torch.exploration.skill_discovery import TrajectoryEncoder
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import (
    deterministic_model,
    gaussian_model,
    shared_model,
)
import isaaclab.utils.math as math_utils
enable_extension("omni.isaac.debug_draw")

from isaacsim.util.debug_draw import _debug_draw


def main():
    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    if args_cli.max_magnitude is not None:
        env_cfg.commands.skills.max_magnitude = args_cli.max_magnitude

    if args_cli.skill_dim is not None:
        env_cfg.commands.skills.skill_dim = args_cli.skill_dim

    if args_cli.discrete_skills is not None:
        env_cfg.commands.skills.discrete_skills = args_cli.discrete_skills

    if args_cli.normalise_encoder_states is not None:
        experiment_cfg["agent"]["curiosity_preprocessor"] = (
            "RunningStandardScaler" if args_cli.normalise_encoder_states == 1 else None
        )
    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    experiment_cfg["agent"]["skill_space"] = env.command_manager.cfg.skills.skill_dim
    experiment_cfg["agent"]["discrete_skills"] = (
        env.command_manager.cfg.skills.discrete_skills
    )
    experiment_cfg["agent"]["evaluate_envs"] = False

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/modules/skrl.utils.model_instantiators.html
    models = {}
    high_level_obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env.observation_space["policy"].shape[0],),
        dtype=np.float32,
    )
    low_level_obs_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(
            env.observation_space["low_level_policy"].shape[0]
            + env.observation_space["skill_conditioning"].shape[0],
        ),
        dtype=np.float32,
    )

    high_level_action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(experiment_cfg["agent"]["skill_space"],),
        dtype=np.float32,
    )
    low_level_action_space = env.action_space
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["low_level_policy"] = gaussian_model(
            observation_space=low_level_obs_space,
            action_space=low_level_action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["low_level_policy"]),
        )

        models["policy"] = gaussian_model(
            observation_space=high_level_obs_space,
            action_space=high_level_action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=high_level_obs_space,
            action_space=env.action_space,
            device=env.device,
            **process_skrl_cfg(experiment_cfg["models"]["value"]),
        )
    # shared models
    else:
        # raise "Not implemented yet."
        models["low_level_policy"] = shared_model(
            observation_space=low_level_obs_space,
            action_space=low_level_action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["low_level_policy"]),
                process_skrl_cfg(experiment_cfg["models"]["low_level_value"]),
            ],
        )

        models["low_level_value"] = models["low_level_policy"]

        models["high_level_policy"] = shared_model(
            observation_space=high_level_obs_space,
            action_space=high_level_action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                process_skrl_cfg(experiment_cfg["models"]["high_level_policy"]),
                process_skrl_cfg(experiment_cfg["models"]["high_level_value"]),
            ],
        )
        models["high_level_value"] = models["high_level_policy"]

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/modules/skrl.memories.random.html
    memory_size = experiment_cfg["agent"][
        "rollouts"
    ]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(
        memory_size=memory_size, num_envs=env.num_envs, device=env.device
    )

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = (
        None  # avoid 'dictionary changed size during iteration'
    )
    agent_cfg.update(process_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["low_level_state_preprocessor_kwargs"].update(
        {"size": env.observation_space["low_level_policy"], "device": env.device}
    )
    agent_cfg["low_level_value_preprocessor_kwargs"].update(
        {"size": 1, "device": env.device}
    )

    agent_cfg["high_level_state_preprocessor_kwargs"].update(
        {"size": env.observation_space["policy"], "device": env.device}
    )
    agent_cfg["high_level_value_preprocessor_kwargs"].update(
        {"size": 1, "device": env.device}
    )

    # dual optimisation stuff:

    agent_cfg["dual_lambda"] = nn.Parameter(
        torch.log(torch.tensor(agent_cfg["dual_lambda"]))
    )

    agent_cfg["curiosity_obs_space"] = env.observation_space["skill_discovery"].shape[0]

    if (
        not agent_cfg["dual_optimisation"]
        and agent_cfg["skill_discovery_distance_metric"] == "l2"
    ):
        use_spectral_norm = True
    else:
        use_spectral_norm = False

    if agent_cfg["sd_specific_state_ind"] != "None":
        sd_encoder_input_size = len(agent_cfg["sd_specific_state_ind"])
    else:
        sd_encoder_input_size = env.observation_space["skill_discovery"].shape[0]

    agent_cfg["curiosity_obs_space"] = sd_encoder_input_size
    agent_cfg["full_curiosity_obs_space"] = env.observation_space[
        "skill_discovery"
    ].shape[0]

    if agent_cfg["skill_discovery"]:
        agent_cfg["curiosity_preprocessor"] = (
            RunningStandardScaler
            if (agent_cfg["curiosity_preprocessor"] == "RunningStandardScaler")
            else None
        )
        agent_cfg["curiosity_preprocessor_kwargs"] = {
            "size": sd_encoder_input_size,
            "device": env.device,
        }
    encoder_output = agent_cfg["skill_space"]

    agent_cfg["lsd_model"] = TrajectoryEncoder(
        sd_encoder_input_size,
        experiment_cfg["agent"]["encoder_dims"],
        encoder_output,
        min_log_std=-5,
        max_log_std=2,
        use_spectral_norm=use_spectral_norm,
        device=env.device,
    )

    agent_cfg["num_envs"] = env.num_envs

    agent_cfg["experiment"]["wandb"] = False  # whether to use Weights & Biases

    agent = HierarchicalLSDPPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space["policy"],  # TODO: check this
        action_space=high_level_action_space.shape[0],  # TODO: check this
        low_level_observation_space=env.observation_space["low_level_policy"].shape[0],
        low_level_action_space=low_level_action_space.shape[0],
        device=env.device,
    )

    agent.init()


    # else:
    agent.load(resume_path)

    agent.set_running_mode("eval")


    # reset environment
    states_dict, infos = env.reset()
    skills = states_dict["skill_conditioning"]
    curiosity_states = states_dict["skill_discovery"]
    i = -1

    base_pos = []
    curiosity_states_store = []
    latent_trajectory = []
    reset_idx = np.zeros(env.num_envs)
    skills_store = []
    reset_idx_store = []
    reset_store = []

    # simulate environment

    failed_counter = 0
    duration = 295

    while simulation_app.is_running() and i < duration:
        # run everything in inference mode
        with torch.inference_mode():
            i += 1

            if env_cfg.commands.skills.skill_dim == 2:
                angles = torch.linspace(
                    0, 2 * np.pi, env.num_envs, device=env.device
                ).repeat(1)
                skills = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
                skills = skills * torch.linspace(0.0, env_cfg.commands.skills.max_magnitude, 20, device=env.device).repeat(
                    10
                ).unsqueeze(1)
            else:
                skills = states_dict["skill_conditioning"]
           
            obs_low = states_dict["low_level_policy"]

            actions_low = agent.low_level_act(obs_low, skills, timestep=0, timesteps=0)[
                0
            ]
            actions_scaled = actions_low * agent.action_scale

            # env stepping
            next_states_dict, rewards, terminated, _, infos = env.step(actions_scaled)

            if torch.any(terminated):
                failed_counter += 1

            base_pos.append(
                (
                    env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
                    - env.unwrapped.scene.env_origins[:, :2]
                )
                .cpu()
                .numpy()
            )
            reset_idx[terminated.squeeze().cpu().numpy()] = 1

            reset_idx_store.append(terminated.squeeze().cpu().numpy().copy())

            reset_store.append((terminated).cpu().numpy().copy())

            curiosity_states = states_dict["skill_discovery"].clone()

            curiosity_states = agent._curiosity_state_selector(curiosity_states)
            curiosity_states = agent._curiosity_preprocessor(curiosity_states)
            latent_states_mean = agent.lsd_model.get_distribution(curiosity_states).mean

            curiosity_states_store.append(curiosity_states.cpu().numpy().copy())

            latent_trajectory.append(latent_states_mean.cpu().numpy().copy())
            skills_store.append(skills.cpu().numpy().copy())

            states_dict = next_states_dict.copy()


    curiosity_states_store = np.array(curiosity_states_store).reshape(
        (duration + 1) * env.num_envs, -1
    )
    # np.savez("curiosity_preprocessor_means_stds.npz", sd_means = np.mean(curiosity_states_store,axis=0), sd_stds = np.std(curiosity_states_store,axis=0))
    # fig,ax = plt.subplots(4,3)
    # ax = ax.ravel()
    # for k in range(12):
    print(f"Failed counter: {failed_counter}")

    # fig,axis = plt.subplots(1,3)
    fig = plt.figure(figsize=(15, 5))
    grid = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    # Create three subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])

    fig.suptitle("Skills, Latent Trajectories, and Real XY Trajectories")
    latent_trajectory = np.array(latent_trajectory)
    skills = skills.cpu().numpy()
    curiosity_states_store = np.array(curiosity_states_store)

    base_pos = np.array(base_pos)
    colormap = plt.cm.hsv
    if agent.discrete_skills:
        pass
    else:
        for i in range(env.num_envs):
            if reset_idx[i]:
                continue
            # Distance metric:
            skills_normalized = skills / (
                np.linalg.norm(skills, axis=1)[:, None] + 1e-8
            )
            metric = np.arctan2(skills_normalized[:, 1], skills_normalized[:, 0])
            metric = (metric + 2 * np.pi) % (2 * np.pi)
            # Normalize metric
            metric = (
                metric / metric.max()
            )  # (metric - metric.min()) / (metric.max() - metric.min())
            color = colormap(metric[i])

            ax1.plot([0, skills[i, 0]], [0, skills[i, 1]], color=color)
            ax1.set_title("Skill Vectors")
            ax1.grid()
            # ax1.set_aspect('equal', 'box')

            ax2.plot(
                latent_trajectory[:, i, 0] - latent_trajectory[0, i, 0],
                latent_trajectory[:, i, 1] - latent_trajectory[0, i, 1],
                color=color,
            )
            ax2.set_title("Latent Trajectories")
            ax2.grid()
            # ax2.set_aspect('equal', 'box')
            # Also plot the skill unit vector in the same colour:

            ax3.plot(base_pos[:, i, 0], base_pos[:, i, 1], color=color)
            ax3.set_title("State (XY) Trajectories")
            # ax3.set_aspect('equal', 'box')
            ax3.grid()

        # Set same aspect ratio for all subplots
        # for ax in [ax1, ax2, ax3]:
        #     ax.set_aspect('equal')
        # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
        cax = fig.add_subplot(grid[0, 3])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap), cax=cax)
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        colorbar.set_ticklabels(["0", "0.5π", "π", "1.5π", "2π"])
        colorbar.set_label("Skill angle")
        # plt.savefig('output.pdf', bbox_inches='tight', pad_inches=0)
        #  Define custom normalization function

    # fig,axis = plt.subplots(1,3)
    fig = plt.figure(figsize=(15, 5))
    grid = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    # Create three subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])

    fig.suptitle("Skills, Latent Trajectories, and Real XY Trajectories")

    colormap = plt.cm.hsv
    if agent.discrete_skills:
        pass
    else:
        for i in range(env.num_envs):
            if reset_idx[i]:
                continue
            # Distance metric:
            skills_normalized = skills / (
                np.linalg.norm(skills, axis=1)[:, None] + 1e-8
            )
            metric = np.arctan2(skills_normalized[:, 1], skills_normalized[:, 0])
            metric = (metric + 2 * np.pi) % (2 * np.pi)
            # Normalize metric
            metric = (
                metric / metric.max()
            )  # (metric - metric.min()) / (metric.max() - metric.min())
            color = colormap(metric[i])

            ax1.plot([0, skills[i, 0]], [0, skills[i, 1]], color=color)
            ax1.set_title("Skill Vectors")
            ax1.grid()
            # ax1.set_aspect('equal', 'box')

            ax2.scatter(
                latent_trajectory[-1, i, 0] - latent_trajectory[0, i, 0],
                latent_trajectory[-1, i, 1] - latent_trajectory[0, i, 1],
                color=color,
            )
            ax2.set_title("Latent Trajectories")
            ax2.grid()
            # ax2.set_aspect('equal', 'box')
            # Also plot the skill unit vector in the same colour:

            ax3.scatter(
                base_pos[-1, i, 0] - base_pos[0, i, 0],
                base_pos[-1, i, 1] - base_pos[0, i, 1],
                color=color,
            )
            ax3.set_title("State (XY) Trajectories")
            # ax3.set_aspect('equal', 'box')
            ax3.grid()

        # Set same aspect ratio for all subplots
        # for ax in [ax1, ax2, ax3]:
        #     ax.set_aspect('equal')
        # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
        cax = fig.add_subplot(grid[0, 3])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap), cax=cax)
        colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        colorbar.set_ticklabels(["0", "0.5π", "π", "1.5π", "2π"])
        colorbar.set_label("Skill angle")

    plt.show()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
