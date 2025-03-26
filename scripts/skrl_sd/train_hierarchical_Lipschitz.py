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
import sys

print(os.path.dirname(sys.executable))
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=9600,
    help="Interval between video recordings (in steps).",
)
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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
    "--task",
    type=str,
    default="Anymal-Skill-Discovery-D-Hierarchical",
    help="Name of the task.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--logging", action="store_false", default=True, help="Disable logging."
)


# For sweep:
# hyperparameter options: ac_learning_rate, entropy_loss_scale,lsd_lr, dual_lambda_lr, dual_lambda, encoder_dims
# parser.add_argument("--param_ac_learning_rate", type=float, default=3e-4, help="Learning rate for the actor critic.")
# parser.add_argument("--param_entropy_loss_scale", type=float, default=0.01, help="Entropy loss scale.")
parser.add_argument(
    "--param_lsd_lr",
    type=float,
    default=None,
    help="Learning rate for the Lipschitz Skill Discovery.",
)
parser.add_argument(
    "--param_intrinsic_reward_scale",
    type=float,
    default=None,
    help="Intrinsic reward scale.",
)
# parser.add_argument("--param_dual_lambda_lr", type=float, default=3e-4, help="Learning rate for the dual lambda.")
# parser.add_argument("--param_dual_lambda", type=float, default=0.1, help="Initial value for the dual lambda.")
# parser.add_argument("--param_encoder_dims", type=list, default=[64,32], help="Dimensions for the encoder.")
parser.add_argument(
    "--param_learning_epochs", type=int, default=None, help="Grad Steps."
)
parser.add_argument(
    "--param_lsd_loss_scale", type=float, default=None, help="LSD loss scale."
)
parser.add_argument(
    "--param_lsd_norm_clip", type=float, default=None, help="LSD norm clip."
)
parser.add_argument(
    "--param_dual_lambda_lr", type=float, default=None, help="Learning rate for the Lagrange multiplier."
)
parser.add_argument(
    "--param_dual_lambda_init", type=float, default=None, help="Initial Lagrange Multiplier value."
)
# parser.add_argument("--mini_batches", type=int, default=8, help="Mini-batches.")
# parser.add_argument("--rollouts", type=int, default=24, help="Rollouts.")
# parser.add_argument("--intrinsic_reward_scale", type=float, default=1.0, help="Intrinsic reward scale.")
# parser.add_argument("--extrinsic_reward_scale", type=float, default=1.0, help="Extrinsic reward scale.")
parser.add_argument("--sd_version", type=str, default=None, help="Which skill discovery method to use (base vs MSE).")
parser.add_argument(
    "--skill_discovery_distance_metric",
    type=str,
    default=None,
    help="Which constraint to use ('l2' or 'one').",
)
parser.add_argument("--max_magnitude", type=float, default=None, help="Maximum magnitude for sampling skills.")
parser.add_argument("--normalise_encoder_states", type=int, default=None, help="Whether to normalise the encoder inputs with a running mean/variance. 0: False, 1: True.")
parser.add_argument(
    "--skill_dim",
    type=int,
    default=None,
    help="Skill space dimension.",
)
parser.add_argument(
    "--discrete_skills",
    type=int,
    default=None,
    help="Whether to use discrete skills. 0: False, 1: True.",
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


from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from constrained_skill_discovery.tasks.utils.wrappers.skrl_LSD import (
    SkrlSkillDiscoverySequentialLogTrainer,
    SkrlVecEnvWrapper,
    process_skrl_cfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo_hierarchical import (
    PPO_DEFAULT_CONFIG,
    HierarchicalLSDPPO,
)
from skrl.agents.torch.exploration.skill_discovery import TrajectoryEncoder
from skrl.agents.torch.exploration.successor_feature import SuccessorFeature
from skrl.lbs_exploration.models import *
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import (
    deterministic_model,
    gaussian_model,
    shared_model,
)


def main():
    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{experiment_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # Update hyperparameters from command line

    if args_cli.param_intrinsic_reward_scale is not None:
        experiment_cfg["agent"]["intrinsic_reward_scale"] = (
            args_cli.param_intrinsic_reward_scale
        )
    if args_cli.param_lsd_lr is not None:
        experiment_cfg["agent"]["lsd_lr"] = args_cli.param_lsd_lr
    if args_cli.param_learning_epochs is not None:
        experiment_cfg["agent"]["learning_epochs"] = args_cli.param_learning_epochs
    if args_cli.param_lsd_loss_scale is not None:
        experiment_cfg["agent"]["lsd_loss_scale"] = args_cli.param_lsd_loss_scale
    if args_cli.sd_version is not None:
        experiment_cfg["agent"]["sd_version"] = args_cli.sd_version
    if args_cli.skill_discovery_distance_metric is not None:
        experiment_cfg["agent"]["skill_discovery_distance_metric"] = (
            args_cli.skill_discovery_distance_metric
        )
    if args_cli.param_lsd_norm_clip is not None:
        experiment_cfg["agent"]["lsd_norm_clip"] = args_cli.param_lsd_norm_clip
    if args_cli.param_dual_lambda_lr is not None:
        experiment_cfg["agent"]["dual_lambda_lr"] = args_cli.param_dual_lambda_lr
    if args_cli.param_dual_lambda_init is not None:
        experiment_cfg["agent"]["dual_lambda"] = args_cli.param_dual_lambda_init
    if args_cli.max_magnitude is not None:
       env_cfg.commands.skills.max_magnitude = args_cli.max_magnitude

    if args_cli.skill_dim is not None:
        env_cfg.commands.skills.skill_dim = args_cli.skill_dim

    if args_cli.discrete_skills is not None:
        env_cfg.commands.skills.discrete_skills = args_cli.discrete_skills == 1

    if args_cli.normalise_encoder_states is not None:
        experiment_cfg["agent"]["curiosity_preprocessor"] = "RunningStandardScaler" if args_cli.normalise_encoder_states == 1 else None

    #
    print("[INFO] Experiment configuration:")
    print("SD VERSION:", experiment_cfg["agent"]["sd_version"])
    print(
        "SKILL DISCOVERY DISTANCE METRIC:",
        experiment_cfg["agent"]["skill_discovery_distance_metric"],
    )
    print("SKILL DIMENSION:", env_cfg.commands.skills.skill_dim)
    print("DISCRETE SKILLS:", env_cfg.commands.skills.discrete_skills)
    print("MAX MAGNITUDE:", env_cfg.commands.skills.max_magnitude)
    print("NORMALISE ENCODER STATES:", args_cli.normalise_encoder_states, experiment_cfg["agent"]["curiosity_preprocessor"])

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

    # dump the configuration into log-directory
    if args_cli.logging:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
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
    if experiment_cfg["agent"]["evaluate_envs"]:
        num_eval_envs = experiment_cfg["agent"]["evaluate_envs_num"]
    else:
        num_eval_envs = 0
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=env.num_envs - num_eval_envs,
        device=env.device,
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

    agent_cfg["episode_length"] = env.unwrapped.max_episode_length

    # if args_cli.intrinsic_reward_scale is not None:
    #     agent_cfg["intrinsic_reward_scale"] = args_cli.intrinsic_reward_scale

    if (
        not agent_cfg["dual_optimisation"]
        and agent_cfg["skill_discovery_distance_metric"] == "l2"
    ):
        use_spectral_norm = True
    else:
        use_spectral_norm = False

    # use_spectral_norm = True

    if agent_cfg["sd_specific_state_ind"] != "None":
        sd_encoder_input_size = len(agent_cfg["sd_specific_state_ind"])
    else:
        sd_encoder_input_size = env.observation_space["skill_discovery"].shape[0]

    if agent_cfg["sd_version"] == "ASE":
        sd_encoder_input_size *= 2

    agent_cfg["curiosity_obs_space"] = sd_encoder_input_size
    agent_cfg["full_curiosity_obs_space"] = env.observation_space[
        "skill_discovery"
    ].shape[0]

    agent_cfg["rewards_preprocessor"] = None
    agent_cfg[
        "rewards_preprocessor_kwargs"
    ] = {}  # {"size": env.observation_space["skill_discovery"].shape[0], "device": env.device}

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

    if agent_cfg["skill_discovery"]:
        
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

    if args_cli.logging:
        agent_cfg["experiment"]["wandb"] = True  # whether to use Weights & Biases
        agent_cfg["experiment"]["wandb_kwargs"] = {
            "project": "Lipschitz Skill Learning",
            "mode": "online",
        }  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

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

    if experiment_cfg["agent"]["load_low_level_policy"]:
        path = experiment_cfg["agent"]["low_level_policy_path"]
        agent.load_hierarchical_policies(log_root_path + path, level="low")
    # path = "logs/skrl/anymal/2024-04-17_10-25-29/checkpoints/agent_100000.pt"
    # agent.load_hierarchical_policies(path, level="low")

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
    trainer_cfg = experiment_cfg["trainer"]
    trainer = SkrlSkillDiscoverySequentialLogTrainer(
        cfg=trainer_cfg, env=env, agents=agent
    )

    # train the agent
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
