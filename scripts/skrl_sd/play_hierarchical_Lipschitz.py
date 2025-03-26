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
    type=bool,
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

# from skrl.resources.preprocessors.torch import RunningStandardScaler
# from source.video_recorder_wrappers.record_video import RecordVideo
# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
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


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model, preprocessor):
        torch.nn.Module.__init__(self)
        self._model = model
        self._preprocessor = preprocessor
        # self.to(device)

    def forward(self, inputs):
        inputs = inputs.to(self._model.device)

        inputs = self._preprocessor(inputs)

        output = self._model.get_distribution(inputs).mean

        return output


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        # self.to(device)

    def forward(self, inputs):
        inputs = inputs.to(self._model.device)
        actions, log_prob, outputs, _ = self._model.act({"states": inputs}, "policy")
        return outputs["mean_actions"]


class ModelWrapperTRT(torch.nn.Module):
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model

    def forward(self, inputs):
        inputs = inputs.to(self._model.device)
        actions = self._model.act_no_dict(inputs, "policy")
        return actions


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
        experiment_cfg["agent"]["RunningStandardScaler"] = (
            "RunningStandardScaler" if args_cli.normalise_encoder_states else None
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

    # models["policy"] = Policy(obs_space, env.action_space, skill_space, env.device)
    # models["value"] = Value(obs_space, env.action_space, skill_space,  env.device)
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

    # if experiment_cfg["agent"]["play_load_low_level_only"]:

    #     agent.load_hierarchical_policies(resume_path, level="low")

    # else:
    agent.load(resume_path)

    # if experiment_cfg["agent"]["load_low_level_policy"]:
    #     path = experiment_cfg["agent"]["low_level_policy_path"]
    #     agent.load_hierarchical_policies(log_root_path+path, level="low")
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # trainer_cfg = experiment_cfg["trainer"]
    # trainer = SkrlSkillDiscoverySequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # train the agent
    # trainer.train()

    # reset environment
    states_dict, infos = env.reset()
    states = states_dict["policy"]
    skills = states_dict["skill_conditioning"]
    curiosity_states = states_dict["skill_discovery"]
    if agent.hierarchical:
        high_level_states = states_dict["policy"]
        low_level_states = states_dict["low_level_policy"]
    i = -1
    robot_id = 0

    deploy_policy = False
    deploy_device = "cpu"
    if deploy_policy:
        example = torch.randn(
            (1, low_level_obs_space.shape[0]), dtype=torch.float32, device=env.device
        ).to(deploy_device)
        adapter = TracingAdapter(
            ModelWrapper(agent.models["low_level_policy"]),
            example,
            allow_non_tensor=True,
        )
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        traced.eval()
        import time

        start = time.time()
        output = traced.forward(example)[0]
        # print("Jit script time: ", time.time() - start)
        # print(output)
        start = time.time()
        # output_net = (agent.models["low_level_policy"]).forward(
        #     {"states": example}, role="policy"
        # )[2]["mean_actions"]
        # print("Base time: ", time.time() - start)
        fname = "policy_" + resume_path.split("/")[-3] + ".pt"
        torch.jit.save(traced, fname)

        # Get encoder too:
        example_encoder = torch.randn(
            (1, env.observation_space["skill_discovery"].shape[0]),
            dtype=torch.float32,
            device=env.device,
        ).to(deploy_device)
        encoder = EncoderWrapper(agent.lsd_model, agent._curiosity_preprocessor)
        # example_encoder = agent._curiosity_preprocessor(example_encoder)
        # example_encoder = agent._curiosity_state_selector(example_encoder)
        traced = torch.jit.trace(encoder, example_encoder, check_trace=False)
        traced.eval()
        fname = "encoder_" + resume_path.split("/")[-3] + ".pt"
        torch.jit.save(traced, fname)

        model_wrapped = ModelWrapperTRT(agent.models["low_level_policy"])
        # from torch2trt import torch2trt
        # model_trt = torch2trt(model_wrapped, [example])
        # start = time.time()
        # output_trt = model_trt(example)

        # for i in range(10):
        #     example = torch.randn((1, low_level_obs_space.shape[0]), dtype=torch.float32, device=env.device).to('cuda')

        #     start = time.time()
        #     output = traced.forward(example)[0]
        #     print("Jit script time (ms): ", (time.time() - start) * 1e3 )
        #     start = time.time()
        #     output_net = (agent.models["low_level_policy"].to('cuda')).forward({"states": example},role="policy")[2]["mean_actions"]
        #     print("Base time (ms): ", (time.time() - start) * 1e3 )

        #     start = time.time()
        #     output_trt = model_trt(example)

        #     print("TRT time (ms): ", (time.time() - start) * 1e3 )

        # print("TRT time: ", time.time() - start)
        # print("Jit script is close to real:",  torch.isclose(output,output_net))
        # print("TRT is close to real:",  torch.isclose(output,output_net))
        # print("TRT is close to Jit:",  torch.isclose(output,output_trt))
        # torch.onnx.export(model_wrapped, example, "model.onnx")
        # torch.onnx.export(encoder, example_encoder, "encoder.onnx")
        print("-")

    # plt.ion()
    # fig,ax = plt.subplots(4,3)
    # ax = ax.ravel()
    # lines = [ax[k].plot(0,0, color='r')[0] for k in range(12)]
    base_pos = []
    joint_vel = []
    base_vel = []
    base_ang_vel = []
    energy = []
    curiosity_states_store = []
    latent_trajectory = []
    reset_idx = np.zeros(env.num_envs)
    skills_store = []
    contacts = []
    reset_idx_store = []
    des_base_pos = []
    cst_penalty_store = []
    commands_store = []
    reset_store = []

    # simulate environment

    failed_counter = 0
    duration = np.inf # np.inf  # 295
    goal_tracking = True

    while simulation_app.is_running() and i < duration:
        asset = env.scene.articulations["robot"]
        # skills = torch.zeros(env.num_envs,2,device=env.device)
        # run everything in inference mode
        with torch.inference_mode():
            i += 1

            skills = states_dict["skill_conditioning"]
            # angles = torch.linspace(
            #     0, 2 * np.pi, env.num_envs, device=env.device
            # ).repeat(1)
            # skills = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
            # skills = skills * torch.linspace(0.0, 1., 20, device=env.device).repeat(
            #     10
            # ).unsqueeze(1)

            # # Get actions based on desired position:
            if goal_tracking:
                encoder_obs = states_dict["skill_discovery"].clone()
                # # encoder_obs[:,236:238] = 0.
                encoder_next_obs = states_dict["skill_discovery"][:].clone()
                # encoder_next_obs[:,236] = 0.0
                # encoder_next_obs[:,237] = 0.0
                encoder_next_obs[:, -3:-1] = (
                    env.unwrapped.command_manager._terms["pose_command"].pos_command_w[
                        :, :2
                    ]
                    - env.unwrapped.scene.env_origins[:, :2]
                )
                # encoder_next_obs[:,-1] = 0.6
                # heading = torch.atan2(
                #     env.unwrapped.command_manager._terms[
                #         "position_commands"
                #     ].pos_command_w[:, 1],
                #     env.unwrapped.command_manager._terms[
                #         "position_commands"
                #     ].pos_command_w[:, 0],
                # )  # env.unwrapped.command_manager._terms["position_commands"].heading_command_w[:]
                current_heading = math_utils.euler_xyz_from_quat(encoder_obs[:, 6:10])[-1]
                # current_heading = encoder_obs[:, 9]

                # Wrap the angle to the range -pi to pi
                # current_heading = current_heading + torch.floor((torch.pi-current_heading)/(2*torch.pi))*2*torch.pi
                # print(f"Current heading: {current_heading}")
                # print(f"Goal heading: {heading}")
                # heading = -torch.pi/2 + 0*heading
                heading = env.unwrapped.command_manager._terms["pose_command"].heading_command_w.clone()
                encoder_next_obs[:,6:10] = math_utils.quat_from_euler_xyz(0*heading,0*heading,heading)
                # encoder_next_obs[:,9] = heading
                # encoder_next_obs[:,5] = 10.0
                # encoder_next_obs[:,13:25] = env.scene.articulations["robot"].data.default_joint_pos

                # dist_to_goal = (
                #     encoder_next_obs[:, 236:238] - encoder_obs[:, 236:238]
                # ).norm(dim=-1)

                # encoder_obs = agent._curiosity_preprocessor(encoder_obs)
                # encoder_next_obs = agent._curiosity_preprocessor(encoder_next_obs)

                encoder_obs = agent._curiosity_preprocessor(
                    agent._curiosity_state_selector(encoder_obs)
                )
                encoder_next_obs = agent._curiosity_preprocessor(
                    agent._curiosity_state_selector(encoder_next_obs)
                )

                latent_states_mean = agent.lsd_model.get_distribution(encoder_obs).mean
                latent_next_states_mean = agent.lsd_model.get_distribution(
                    encoder_next_obs
                ).mean

                max_magnitude = 1.

                skills_base = (latent_next_states_mean - latent_states_mean) / 300.
                # skills = skills_base / skills_base.norm(dim=-1,keepdim=True).clamp(min=1e-6)
                # skills = torch.linspace(5.0,30.,env.num_envs,device=env.device).unsqueeze(1)*skills_base / skills_base.norm(dim=-1,keepdim=True).clamp(min=1e-6)
                skills = (5*skills_base.norm(dim=-1,keepdim=True)).clamp(max=max_magnitude,min=0) * skills_base / skills_base.norm(dim=-1,keepdim=True).clamp(min=1e-6)
 

                print(
                    f"Dist to goal: {env.unwrapped.command_manager._terms['pose_command'].pos_command_b[:, :2].norm(dim=1).item()}"
                )
                print(f" Heading error: {env.unwrapped.command_manager._terms['pose_command'].heading_command_b[:].item()}")

            obs_high = states_dict["policy"]
            obs_low = states_dict["low_level_policy"]

            actions_low = agent.low_level_act(obs_low, skills, timestep=0, timesteps=0)[
                0
            ]
            actions_scaled = actions_low * agent.action_scale

            # env stepping
            next_states_dict, rewards, terminated, _, infos = env.step(actions_scaled)

            if torch.any(terminated):
                failed_counter += 1
            if i == 1:
                agent._init_states = next_states_dict["skill_discovery"].clone()

            base_pos.append(
                (
                    env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
                    - env.unwrapped.scene.env_origins[:, :2]
                )
                .cpu()
                .numpy()
            )
            # base_pos.append(states_dict["skill_discovery"][:,260:263].cpu().numpy().copy())
            base_vel.append(
                env.unwrapped.scene["robot"]
                .data.root_lin_vel_w[:, :2]
                .cpu()
                .numpy()
                .copy()
            )
            base_ang_vel.append(
                env.unwrapped.scene["robot"]
                .data.root_ang_vel_w[:, 2]
                .cpu()
                .numpy()
                .copy()
            )
            energy.append(
                states_dict["skill_discovery"][:, -1].cpu().numpy().copy() / 0.1
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

            next_curiosity_states = next_states_dict["skill_discovery"][:].clone()

            next_curiosity_states = agent._curiosity_preprocessor(next_curiosity_states)
            next_curiosity_states = agent._curiosity_state_selector(
                next_curiosity_states
            )
            latent_next_states_mean = agent.lsd_model.get_distribution(
                next_curiosity_states
            ).mean

            skills_store.append(skills.cpu().numpy().copy())

            # # Check constraints:
            # skill_angle = torch.arctan2(skills[:,1],skills[:,0])
            cst_dist = torch.square(next_curiosity_states - curiosity_states).mean(
                dim=1
            )
            # cst_dist = torch.square(next_curiosity_states[:,-3:] - curiosity_states[:,-3:]).mean(
            #     dim=1
            # )
            # cst_dist = torch.ones(env.num_envs,device=env.device)

            # cst_penalty_lower_bound = 2*0.5*(torch.cos(skill_angle) + 1.)*torch.square(latent_next_states_mean - latent_states_mean).mean(dim=1) - cst_dist
            cst_penalty = cst_dist - torch.square(
                latent_next_states_mean - latent_states_mean
            ).mean(dim=1)

            cst_penalty = cst_penalty.clamp(max=1.0e-6)

            cst_penalty_store.append(cst_penalty.cpu().numpy().copy())

            # intrinsic_reward = torch.sum((latent_next_states_mean - latent_states_mean) * skills, dim=1) / (torch.norm((latent_next_states_mean - latent_states_mean),dim=-1) * torch.norm(skills,dim=-1) + 1e-8)
            draw = _debug_draw.acquire_debug_draw_interface()
            cmap = plt.cm.get_cmap("hsv")

            skills_temp = skills.cpu().numpy()
            skills_normalized = skills_temp / (
                np.linalg.norm(skills_temp, axis=1)[:, None] + 1e-8
            )
            metric = np.arctan2(skills_normalized[:, 1], skills_normalized[:, 0])
            metric = (metric + 2 * np.pi) % (2 * np.pi)
            # Normalize metric
            metric = metric / (
                2 * np.pi
            )  # (metric - metric.min()) / (metric.max() - metric.min())
            # Convert the list of arrays to a list of lists
            # draw = _debug_draw.acquire_debug_draw_interface()
            cmap = plt.cm.hsv  #
            # cmap = plt.cm.get_cmap("tab20")

            # Convert the list of arrays to a list of lists
            env_origins = env.unwrapped.scene.env_origins[:, :2]
            if i == 0:
                point_list_1 = [(0.0 + env_origins[i, 0], 0.0 + env_origins[i, 1], 0.5)]
            else:
                point_list_1 = [
                    (
                        base_pos[-2][i, 0] + env_origins[i, 0],
                        base_pos[-2][i, 1] + env_origins[i, 1],
                        0.5,
                    )
                    for i in range(env.num_envs)
                ]
            point_list_2 = [
                (
                    base_pos[-1][i, 0] + env_origins[i, 0],
                    base_pos[-1][i, 1] + env_origins[i, 1],
                    0.5,
                )
                for i in range(env.num_envs)
            ]

            colors = [cmap(i) for i in range(env.num_envs)]
            sizes = [5 for i in range(env.num_envs)]

            # if i%299>=0 and i%299<2:
            #     draw.clear_lines()
            # else:
            # draw.draw_lines(point_list_1, point_list_2, colors, sizes)

            # point_list = [(goals[i,0] + env_origins[i,0], goals[i,1] + env_origins[i,1],1.0) for i in range(env.num_envs)]

            draw.draw_lines(point_list_1, point_list_2, colors, sizes)

            states_dict = next_states_dict.copy()

    # np.savez("data_final/rudin_reward_MSE_tracking.npz",des_base_pos=des_base_pos,base_pos=base_pos,base_vel=base_vel,base_ang_vel=base_ang_vel,\
    #  energy=energy,curiosity_states=curiosity_states_store,latent_trajectory=latent_trajectory,skills=skills_store, \
    # contacts=contacts, reset_idx=reset_idx_store)
    np.savez("data_final/test_3d_stuff_2.npz",base_pos=base_pos,base_vel=base_vel,base_ang_vel=base_ang_vel,energy=energy,resets=reset_store,curiosity_states=curiosity_states_store,latent_trajectory=latent_trajectory,skills=skills_store, contacts=contacts,commands=commands_store)

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
    curiosity_states_norms = np.linalg.norm(curiosity_states_store, axis=-1)

    # Norms of differences:
    curiosity_transition_norms = np.linalg.norm(
        curiosity_states_store[1:] - curiosity_states_store[:-1], axis=-1
    )
    latent_transition_norms = np.linalg.norm(
        latent_trajectory[1:] - latent_trajectory[:-1], axis=-1
    )

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
        # plt.savefig('output.pdf', bbox_inches='tight', pad_inches=0)
        #  Define custom normalization function

    cst_penalty_store = np.array(cst_penalty_store)
    # print(f"Mean constraint violation: {np.mean(cst_penalty_store,axis=0)}")

    plt.figure()
    plt.suptitle("Position Trajectories")
    # skills = skills.cpu().numpy()
    # base_pos = np.array(base_pos)
    if agent.discrete_skills:
        cmap = plt.get_cmap("tab20", agent_cfg["skill_space"])
        # cmap = generate_colormap(agent_cfg["skill_space"]/2*agent_cfg["skill_space"]/2)
        for i in range(env.num_envs):
            if reset_idx[i]:
                continue
            color = cmap(np.argmax(skills[i]))
            plt.plot(
                base_pos[:, i, 0],
                base_pos[:, i, 1],
                color=color,
                label=f"Skill {np.argmax(skills[i])}",
            )
            # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor=cmap(i),
                markersize=10,
                label=f"Skill {i}",
            )
            for i in range(agent_cfg["skill_space"])
        ]
        plt.legend(handles=legend_handles, loc="upper left")
    else:
        # Distance metric:
        metric = np.arctan2(skills[:, 1], skills[:, 0])
        # Normalize metric
        metric = (metric - metric.min()) / (metric.max() - metric.min())
        # Create colormap based on cosine similarity
        colormap = plt.cm.hsv  # You can use any other colormap

        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

        for i in range(env.num_envs):
            color = colormap(metric[i])
            plt.plot(
                base_pos[:, i, 0],
                base_pos[:, i, 1],
                color=color,
                label=f"Skill {skills[i]}",
            )
        plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label="Skill angle")

    plt.figure()
    plt.suptitle("Velocity Trajectories")
    base_vel = np.array(base_vel)
    if agent.discrete_skills:
        cmap = plt.get_cmap("tab20", agent_cfg["skill_space"])
        # cmap = generate_colormap(agent_cfg["skill_space"]/2*agent_cfg["skill_space"]/2)

        for i in range(env.num_envs):
            if reset_idx[i]:
                continue
            color = cmap(np.argmax(skills[i]))
            plt.plot(
                base_vel[:, i, 0],
                base_vel[:, i, 1],
                color=color,
                label=f"Skill {np.argmax(skills[i])}",
            )
            # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor=cmap(i),
                markersize=10,
                label=f"Skill {i}",
            )
            for i in range(agent_cfg["skill_space"])
        ]
        plt.legend(handles=legend_handles, loc="upper left")
    else:
        # Distance metric:
        metric = np.arctan2(skills[:, 1], skills[:, 0])
        # Normalize metric
        metric = (metric - metric.min()) / (metric.max() - metric.min())
        # Create colormap based on cosine similarity
        colormap = plt.cm.hsv  # You can use any other colormap

        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

        for i in range(env.num_envs):
            color = colormap(metric[i])
            plt.plot(
                base_vel[:, i, 0],
                base_vel[:, i, 1],
                color=color,
                label=f"Skill {skills[i]}",
            )
        plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label="Skill angle")

    # fig, ax = plt.subplots(2,1)
    # mean_base_vel = np.mean(base_vel,axis=0)
    # gamma_matrix = agent_cfg["succ_feat_gamma"] ** np.arange(0,base_vel.shape[0])
    # succ_base_vel = np.sum(base_vel * gamma_matrix[:,None,None],axis=0)
    # for i in range(env.num_envs):
    #     if reset_idx[i]:
    #         continue
    #     color = cmap(np.argmax(skills[i]))
    #     ax[0].scatter(mean_base_vel[i,0],mean_base_vel[i,1],color=color,label="Mean Vel")
    #     ax[1].scatter(succ_base_vel[i,0],succ_base_vel[i,1],color=color,label="Succ Vel")

    # fig, ax = plt.subplots(2,2)
    # ax = ax.ravel()
    # mean_base_vel = np.mean(base_vel,axis=0)
    # energy = np.mean(np.array(energy),axis=0)
    # if agent_cfg["skill_space"] > 24:
    #     markers = ['o' if i % 3 == 0 else "x" if i % 3 == 1 else "+" for i in range(agent_cfg["skill_space"])]
    # else:
    #     markers = ['o' for i in range(agent_cfg["skill_space"])]
    # if agent.discrete_skills:

    #     cmap = plt.get_cmap('tab20', agent_cfg["skill_space"])
    #     # cmap = generate_colormap(agent_cfg["skill_space"]/2*agent_cfg["skill_space"]/2)

    #     for i in range(env.num_envs):
    #         if reset_idx[i]:
    #             continue
    #         color = cmap(np.argmax(skills[i]))
    #         ax[0].scatter(np.argmax(skills[i]),mean_base_vel[i,0], color=color, label=f'Skill {np.argmax(skills[i])}', marker=markers[np.argmax(skills[i])], s=30)
    #         ax[0].set_title("Mean X Vel")
    #         ax[1].scatter(np.argmax(skills[i]),mean_base_vel[i,1], color=color, label=f'Skill {np.argmax(skills[i])}', marker=markers[np.argmax(skills[i])], s=30)
    #         ax[1].set_title("Mean Y Vel")
    #         ax[2].scatter(np.argmax(skills[i]),mean_base_vel[i,2], color=color, label=f'Skill {np.argmax(skills[i])}', marker=markers[np.argmax(skills[i])], s=30)
    #         # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
    #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker=markers[i], color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left', ncols=2)
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv  # You can use any other colormap

    #     # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.plot(mean_base_vel[i,0],mean_base_vel[i,1], color=color, label=f'Skill {skills[i]}')
    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    # plt.figure()
    # plt.suptitle("XZ Velocity Trajectories")
    # if agent.discrete_skills:

    #     cmap = plt.get_cmap('tab20', agent_cfg["skill_space"])
    #     # cmap = generate_colormap(agent_cfg["skill_space"]/2*agent_cfg["skill_space"]/2)

    #     for i in range(env.num_envs):
    #         if reset_idx[i]:
    #             continue
    #         color = cmap(np.argmax(skills[i]))
    #         plt.plot(base_vel[:,i,0],base_vel[:,i,2], color=color, label=f'Skill {np.argmax(skills[i])}')
    #         # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
    #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left')
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv  # You can use any other colormap

    #     # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.plot(base_vel[:,i,0],base_vel[:,i,2], color=color, label=f'Skill {skills[i]}')
    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    # fig,ax = plt.subplots(3,1)
    # fig.suptitle("Angular Velocity Trajectories")
    # base_ang_vel = np.array(base_ang_vel)
    # if agent.discrete_skills:

    #     cmap = plt.get_cmap('tab20', agent_cfg["skill_space"])
    #     # cmap = generate_colormap(agent_cfg["skill_space"]/2*agent_cfg["skill_space"]/2)

    #     for i in range(env.num_envs):
    #         if reset_idx[i]:
    #             continue
    #         color = cmap(np.argmax(skills[i]))
    #         ax[0].plot(base_ang_vel[:,i,0], color=color, label=f'Skill {np.argmax(skills[i])}')
    #         ax[0].set_title("Roll Angular Velocity")

    #         ax[1].plot(base_ang_vel[:,i,1], color=color, label=f'Skill {np.argmax(skills[i])}')
    #         ax[1].set_title("Pitch Angular Velocity")

    #         ax[2].plot(base_ang_vel[:,i,2], color=color, label=f'Skill {np.argmax(skills[i])}')
    #         ax[2].set_title("Yaw Angular Velocity")

    #         # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
    #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left')
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv  # You can use any other colormap

    #     # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.plot(base_ang_vel[:,i,0],base_ang_vel[:,i,1], color=color, label=f'Skill {skills[i]}')
    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    # fig,ax = plt.subplots(2,2)
    # ax = ax.ravel()
    # plt.suptitle("Global Successor features for each skill")

    # if agent_cfg["skill_discovery"] and agent_cfg["domino"]:
    #     cmap = plt.get_cmap('tab20', agent_cfg["skill_space"])

    #     agent._update_global_succ_feat(agent._curiosity_preprocessor((agent._init_states), train=False))
    #     for i in range(min(env.num_envs,agent.skill_space)):
    #         color = cmap(i)
    #         feats = agent.global_succ_feat[i].detach().cpu().numpy()

    #         # ax[0].scatter(feats[0],feats[1], color=color, label=f'Skill {i}')
    #         # ax[0].set_title("XY Vel Global Successor features for each skill")
    #         # ax[1].scatter(feats[2],feats[3], color=color, label=f'Skill {i}')
    #         # ax[1].set_title("Z Vel vs Energy Global Successor features for each skill")
    #         # # ax[2].scatter(feats[2],feats[5], color=color, label=f'Skill {i}')
    #         # ax[2].set_title("Z-Y Vel/Ang vel Global Successor features for each skill")

    #         ax[0].scatter(np.argmax(skills[i]),feats[0], color=color, label=f'Skill {i}')
    #         ax[0].set_title("Skill vs X Vel Global Successor features for each skill")
    #         ax[1].scatter(np.argmax(skills[i]),feats[1], color=color, label=f'Skill {i}')
    #         ax[1].set_title("Skill vs Y Vel Global Successor features for each skill")
    #         # ax[2].scatter(np.argmax(skills[i]),feats[2], color=color, label=f'Skill {i}')
    #         # ax[2].set_title("Skill vs Z Vel Global Successor features for each skill")
    #         # ax[3].scatter(np.argmax(skills[i]),feats[3], color=color, label=f'Skill {i}')
    #         # ax[3].set_title("Skill vs Energy Global Successor features for each skill")

    #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left')

    # # Plot histogram of mean velocities per agent:

    # base_vel = np.array(base_vel)
    # mean_base_vel = np.mean(base_vel,axis=0)
    # norm_mean_base_vel = np.linalg.norm(mean_base_vel,axis=-1)
    # plt.figure()
    # plt.suptitle("Velocity Histograms")

    # plt.hist(norm_mean_base_vel, bins=50, edgecolor='black')
    # plt.xlabel('Mean Velocity Magnitude')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Velocities')

    # # Final position histogram:
    # plt.figure()
    # plt.suptitle("Final Position Histograms")

    # plt.hist(np.linalg.norm(base_pos[-1,:,:2],axis=-1), bins=50, edgecolor='black')

    # plt.figure()
    # base_vel = np.array(base_vel)
    # plt.suptitle("Velocity Clusters")
    # if agent.discrete_skills:
    #     cmap = plt.get_cmap('tab20', agent_cfg["skill_space"])
    #     for i in range(env.num_envs):
    #         plt.scatter(base_vel[:,i,0],base_vel[:,i,1], color=cmap(skills[i]), label=f'Skill {skills[i]}')
    #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range( agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left')
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv  # You can use any other colormap

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.scatter(base_vel[:,i,0],base_vel[:,i,1], label=f'Skill {skills[i]}', color=color)
    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
    # plt.figure()
    # plt.suptitle("Global Successor features for each skill")

    # if agent_cfg["skill_discovery"] and agent_cfg["domino"]:
    #     obs =
    #     agent._update_global_succ_feat(obs)

    #     plt.plot(global_succ_feat[:][])

    # fig,ax = plt.subplots(4,3)
    # fig.suptitle("Velocity Trajectories")
    # ax = ax.ravel()
    # base_vel = np.array(base_vel)

    # if agent.discrete_skills:

    #     # cmap = plt.get_cmap('tab', agent_cfg["skill_space"])
    #     for i in range(env.num_envs):
    #         j = i // 4
    #         color = cmap(np.argmax(skills[i]))
    #         ax[j].plot(base_vel[:,i,0],base_vel[:,i,1], label=f'Skill {np.argmax(skills[i])}')
    #         # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
    #     # legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(agent_cfg["skill_space"])]
    #     plt.legend(handles=legend_handles, loc='upper left')
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv  # You can use any other colormap

    #     # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.plot(base_vel[:,i,0],base_vel[:,i,1], color=color, label=f'Skill {skills[i]}')
    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    # plt.figure()
    # plt.suptitle("Energy")
    # energy = np.array(energy)

    # metric = np.arctan2(skills[:,1], skills[:,0])
    # # Normalize metric
    # metric = (metric - metric.min()) / (metric.max() - metric.min())
    # # Create colormap based on cosine similarity
    # colormap = plt.cm.hsv  # You can use any other colormap

    # for i in range(env.num_envs):
    #     color = colormap(metric[i])
    #     plt.plot(energy[:,i], color=color, label=f'Skill {skills[i]}')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    # Plot vectors of final - initial velocity:
    # plt.figure()
    # plt.suptitle("Velocity Vectors")
    # # base_vel = np.array(base_vel)

    # if agent.discrete_skills:
    #     pass
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         # plt.quiver(base_pos[0,i,0],base_pos[0,i,1],base_vel[-1,i,0]-base_vel[0,i,0],base_vel[-1,i,1]-base_vel[0,i,1], color=color, label=f'Skill {skills[i]}')
    #         plt.plot([base_pos[0,i,0],base_pos[-1,i,0]],[base_pos[0,i,1],base_pos[-1,i,1]], color=color, label=f'Skill {skills[i]}')

    # # Plot vectors of final - initial position:
    # plt.figure()
    # plt.suptitle("Position Vectors")

    # if agent.discrete_skills:
    #     pass
    # else:
    #     # Distance metric:
    #     metric = np.arctan2(skills[:,1], skills[:,0])
    #     # Normalize metric
    #     metric = (metric - metric.min()) / (metric.max() - metric.min())
    #     # Create colormap based on cosine similarity
    #     colormap = plt.cm.hsv

    #     for i in range(env.num_envs):
    #         color = colormap(metric[i])
    #         plt.plot([base_pos[0,i,0],base_pos[-1,i,0]],[base_pos[0,i,1],base_pos[-1,i,1]], color=color, label=f'Skill {skills[i]}')
    # Plot Trajectories in latent space:

    # fig,axis = plt.subplots(1,3)
    # fig.suptitle("Skills, Latent Trajectories, and Real XY Trajectories")
    # latent_trajectory = np.array(latent_trajectory)

    # colormap = plt.cm.hsv
    # if agent.discrete_skills:
    #     pass
    # else:
    #     for i in range(env.num_envs):

    #         color = colormap(metric[i])

    #         axis[0].plot([0,skills[i,0]], [0,skills[i,1]], color=color)
    #         axis[0].set_title("Skill Vectors")

    #         axis[1].plot(latent_trajectory[:,i,0],latent_trajectory[:,i,1], color=color)
    #         axis[1].set_title("Latent Trajectories")
    #         # Also plot the skill unit vector in the same colour:

    #         axis[2].plot(base_pos[:,i,0], base_pos[:,i,1], color=color)
    #         axis[2].set_title("Real XY Trajectories")

    #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

    plt.show()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
