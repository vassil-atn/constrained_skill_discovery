# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env

    env = wrap_env(env, wrapper="isaac-orbit")

"""

from __future__ import annotations

import copy
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: F401
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils.model_instantiators.torch import Shape  # noqa: F401
from tqdm.auto import tqdm

import wandb

"""
Configuration Parser.
"""


def process_skrl_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to skrl classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "state_preprocessor",
        "value_preprocessor",
        "low_level_state_preprocessor",
        "low_level_value_preprocessor",
        "high_level_state_preprocessor",
        "high_level_value_preprocessor",
        "input_shape",
        "output_shape",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, timestep, timesteps):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)

        return d

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(env: ManagerBasedRLEnv):
    """Wraps around Orbit environment for skrl.

    This function wraps around the Orbit environment. Since the :class:`ManagerBasedRLEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.

    Raises:
        ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.envs.wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, ManagerBasedRLEnv):
        raise ValueError(
            f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}"
        )
    # wrap and return the environment
    return wrap_env(env, wrapper="isaac-orbit")


def plot_skill_transitions(latent_trajectory, skills, base_pos, reset_idx):
    num_envs = latent_trajectory.shape[1]

    fig = plt.figure(figsize=(15, 5))
    grid = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    # Create three subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])

    fig.suptitle("Skills, Latent Trajectories, and Real XY Trajectories")

    latent_trajectory = latent_trajectory.cpu().numpy()
    skills = skills.cpu().numpy()
    base_pos = base_pos.cpu().numpy()
    reset_idx = reset_idx.cpu().numpy()[:, 0]

    colormap = plt.cm.hsv

    for i in range(num_envs):
        if reset_idx[i]:
            continue
        # Distance metric:
        skills_normalized = skills / (np.linalg.norm(skills, axis=1)[:, None] + 1e-8)
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
            latent_trajectory[2:, i, 0] - latent_trajectory[2, i, 0],
            latent_trajectory[2:, i, 1] - latent_trajectory[2, i, 1],
            color=color,
        )
        ax2.set_title("Latent Trajectories")
        ax2.grid()
        # ax2.set_aspect('equal', 'box')
        # Also plot the skill unit vector in the same colour:

        ax3.plot(
            base_pos[2:, i, 0] - base_pos[2, i, 0],
            base_pos[2:, i, 1] - base_pos[2, i, 1],
            color=color,
        )
        ax3.set_title("State (XY) Trajectories")
        # ax3.set_aspect('equal', 'box')
        ax3.grid()

    # Plot a circle of radius max latent distance
    # max_latent_distance = np.max(np.linalg.norm(latent_trajectory[2:,reset_idx==0,0:2]-latent_trajectory[2,reset_idx==0,0:2],axis=-1))
    # ax2.plot(max_latent_distance*np.cos(np.linspace(0,2*np.pi,100)), max_latent_distance*np.sin(np.linspace(0,2*np.pi,100)), color='black', linestyle='--')

    # Plot a circle of radius max XY distance
    # max_XY_distance = np.max(np.linalg.norm(base_pos[2:,reset_idx==0,0:2]-base_pos[2,reset_idx==0,0:2],axis=-1))
    # ax3.plot(max_XY_distance*np.cos(np.linspace(0,2*np.pi,100)), max_XY_distance*np.sin(np.linspace(0,2*np.pi,100)), color='black', linestyle='--')

    # Set same aspect ratio for all subplots
    # for ax in [ax1, ax2, ax3]:
    #     ax.set_aspect('equal')
    # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
    cax = fig.add_subplot(grid[0, 3])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap), cax=cax)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_ticklabels(["0", "0.5π", "π", "1.5π", "2π"])
    colorbar.set_label("Skill angle")

    # Log the figure to wandb
    wandb.log({"Plots/Performance": wandb.Image(fig)})

    # Close the figure
    plt.close(fig)


"""
Custom trainer for skrl.
"""


class SkrlSkillDiscoverySequentialLogTrainer(Trainer):
    """Sequential trainer with logging of episode information.

    This trainer inherits from the :class:`skrl.trainers.base_class.Trainer` class. It is used to
    train agents in a sequential manner (i.e., one after the other in each interaction with the
    environment). It is most suitable for on-policy RL agents such as PPO, A2C, etc.

    It modifies the :class:`skrl.trainers.torch.sequential.SequentialTrainer` class with the following
    differences:

    * It also log episode information to the agent's logger.
    * It does not close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
    """

    def __init__(
        self,
        env: Wrapper,
        agents: Agent | list[Agent],
        agents_scope: list[int] | None = None,
        cfg: dict | None = None,
    ):
        """Initializes the trainer.

        Args:
            env: Environment to train on.
            agents: Agents to train.
            agents_scope: Number of environments for each agent to
                train on. Defaults to None.
            cfg: Configuration dictionary. Defaults to None.
        """
        # update the config
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        # store agents scope
        agents_scope = agents_scope if agents_scope is not None else []
        # initialize the base class
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        self.init_at_random_ep_len = _cfg.get("init_at_random_ep_len", False)
        # init agents
        if self.env.num_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self):
        """Train the agents sequentially.

        This method executes the training loop for the agents. It performs the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states_dict, infos = self.env.reset()
        states = states_dict["policy"]
        if self.agents.hierarchical:
            high_level_states = states_dict["policy"]
            low_level_states = states_dict["low_level_policy"]
        curiosity_states = states_dict["skill_discovery"]
        skills = states_dict["skill_conditioning"]
        prev_low_level_actions, _, _, _ = self.agents.low_level_act(
            low_level_states, skills, timestep=0, timesteps=self.timesteps
        )
        prev_actions_scaled = prev_low_level_actions * self.agents.action_scale

        travelled_distance = torch.zeros(self.env.unwrapped.scene.num_envs).to(
            self.agents.device
        )
        initial_base_pos = self.env.initial_root_states[:, :3].clone()
        base_pos = self.env.initial_root_states[:, :3].clone()

        # Init preprocessor:
        # from source.normalizer_statistics import STATE_PREPROCESSOR_MEAN,STATE_PREPROCESSOR_VAR,CURIOSITY_PREPROCESSOR_MEAN,CURIOSITY_PREPROCESSOR_VAR

        # self.agents._state_preprocessor.running_mean = STATE_PREPROCESSOR_MEAN
        # self.agents._state_preprocessor.running_var = STATE_PREPROCESSOR_VAR
        # self.agents._curiosity_preprocessor.running_mean = CURIOSITY_PREPROCESSOR_MEAN
        # self.agents._curiosity_preprocessor.running_var = CURIOSITY_PREPROCESSOR_VAR
        if self.init_at_random_ep_len:
            self.env.unwrapped.episode_length_buf = torch.randint_like(
                self.env.unwrapped.episode_length_buf,
                high=int(self.env.unwrapped.max_episode_length),
            )

        progress_bar = tqdm(
            range(self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        )
        start_time = time.time()
        data_collect_time_bef = time.time()
        data_collect_time_aft = time.time()
        update_time_bef = time.time()
        update_time_aft = time.time()
        data_collect_time_bef_prev = data_collect_time_bef
        episode_trajectory = torch.zeros(
            self.env.unwrapped.scene.num_envs, self.env.unwrapped.max_episode_length, 6
        ).to(self.agents.device)
        visited_pos_cells_count = 0
        visited_vel_cells_count = 0
        eval_envs_num = self.agents.evaluate_envs_num
        eval_latent_states = torch.zeros(
            self.agents.episode_length, eval_envs_num, 2
        ).to(self.agents.device)
        eval_XY_states = torch.zeros(self.agents.episode_length, eval_envs_num, 2).to(
            self.agents.device
        )
        eval_dones = torch.zeros(eval_envs_num, 1).to(self.agents.device)
        eval_curiosity_states = torch.zeros(
            self.agents.episode_length, eval_envs_num, 3
        ).to(self.agents.device)
        eval_envs_ids = self.agents.evaluate_envs_ids
        if self.agents.evaluate_envs:
            evaluation_envs = torch.arange(self.env.unwrapped.scene.num_envs).to(
                self.agents.device
            )[eval_envs_ids]

        # training loop
        for timestep in range(self.timesteps):
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            with torch.no_grad():
                #
                if self.agents.evaluate_envs:
                    if timestep % self.agents.evaluate_envs_interval == 0:
                        self.env.reset_to_evaluate[eval_envs_ids] = 1
                    elif timestep % self.agents.evaluate_envs_interval == 1:
                        self.env.command_manager._terms["skills"]._evaluation_commands(
                            evaluation_envs
                        )
                        # Save commands, latent transition and real XY positions
                        eval_commands = (
                            self.env.command_manager._terms["skills"]
                            .skills[evaluation_envs]
                            .clone()
                        )
                        eval_latent_states[:] = 0.0
                        eval_XY_states[:] = 0.0
                        eval_dones[:] = 0.0
                        eval_curiosity_states[:] = 0.0
                    elif ((timestep % self.agents.evaluate_envs_interval) > 1) and (
                        (timestep % self.agents.evaluate_envs_interval)
                        < self.agents.episode_length
                    ):
                        # Save commands, latent transition and real XY positions
                        curiosity_states_selected = self.agents._curiosity_preprocessor(
                            self.agents._curiosity_state_selector(curiosity_states)
                        )

                        eval_curiosity_states[
                            timestep % self.agents.evaluate_envs_interval
                        ] = curiosity_states_selected[evaluation_envs, -3:].clone()

                        eval_latent_states[
                            timestep % self.agents.evaluate_envs_interval
                        ] = (
                            self.agents.lsd_model.get_distribution(
                                curiosity_states_selected
                            )
                            .mean[evaluation_envs]
                            .clone()
                        )[:, :2]

                        # asset = self.env.scene[asset_cfg.name]
                        # eval_XY_states[
                        #     timestep % self.agents.evaluate_envs_interval
                        # ] = asset.data.root_pos_w[evaluation_envs, :2].clone()
                        asset_cfg = SceneEntityCfg("robot")
                        asset = self.env.scene[asset_cfg.name]
                        ### NOTE: The line below should not do anything but it significantly changes the training curve! Do not remove it!
                        target = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()

                        # eval_XY_states[
                        #     timestep % self.agents.evaluate_envs_interval
                        # ] = target.reshape(-1,3)[evaluation_envs,:2]

                        eval_XY_states[
                            timestep % self.agents.evaluate_envs_interval
                        ] = asset.data.root_pos_w[evaluation_envs, :2].clone()

                        eval_dones[terminated[evaluation_envs]] = 1.0

                    if (
                        timestep % self.agents.evaluate_envs_interval
                        == self.agents.episode_length - 1
                    ):
                        # plot on wandb
                        plot_skill_transitions(
                            eval_latent_states,
                            eval_commands,
                            eval_XY_states,
                            eval_dones,
                        )
                    # elif timestep%self.agents.evaluate_envs_interval == self.agents.episode_length:
                    # eval_envs_ids[:] = False

                # If using the hierarchical controller:
                if self.agents.hierarchical:
                    # For high-level control:
                    if self.agents.train_high_level and (
                        timestep % self.agents.high_level_action_frequency == 0
                    ):
                        # get the skills from the high level policy:
                        high_level_actions, _, _, high_level_actions_pretanh = (
                            self.agents.act(
                                high_level_states,
                                timestep=timestep,
                                timesteps=self.timesteps,
                            )
                        )
                        skills = high_level_actions / torch.norm(
                            high_level_actions, dim=-1, keepdim=True
                        )
                    elif self.agents.train_high_level and not (
                        timestep % self.agents.high_level_action_frequency == 0
                    ):
                        # Use the previously computed skills
                        pass
                    elif self.agents.train_low_level:
                        high_level_actions = torch.zeros(
                            self.env.unwrapped.scene.num_envs, self.agents.action_space
                        ).to(self.agents.device)
                        high_level_actions_pretanh = torch.zeros(
                            self.env.unwrapped.scene.num_envs, self.agents.action_space
                        ).to(self.agents.device)
                        skills = states_dict["skill_conditioning"]
                    else:
                        raise NotImplementedError(
                            "Both high and low level control set to False."
                        )

                    # get the low level actions:
                    low_level_actions, _, _, low_level_actions_pretanh = (
                        self.agents.low_level_act(
                            low_level_states,
                            skills,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                    )
                    actions_scaled = low_level_actions * self.agents.action_scale

                else:
                    actions, _, _, actions_pretanh = self.agents.act(
                        states, skills, timestep=timestep, timesteps=self.timesteps
                    )
                    actions_scaled = actions * self.agents.action_scale

                prev_infos = infos.copy()
                # step the environments

                if self.agents.hierarchical:
                    if self.agents.delay_actions:
                        next_states_dict, rewards, terminated, truncated, infos = (
                            self.env.step(prev_actions_scaled)
                        )
                    else:
                        next_states_dict, rewards, terminated, truncated, infos = (
                            self.env.step(actions_scaled)
                        )

                    next_high_level_states = next_states_dict["policy"]
                    next_low_level_states = next_states_dict["low_level_policy"]
                    next_high_level_actions = self.agents.act(
                        next_high_level_states,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )[0]
                    if self.agents.train_high_level and (
                        (timestep + 1) % self.agents.high_level_action_frequency == 0
                    ):
                        next_skills = next_high_level_actions / torch.norm(
                            next_high_level_actions, dim=-1, keepdim=True
                        )
                    elif self.agents.train_high_level and not (
                        (timestep + 1) % self.agents.high_level_action_frequency == 0
                    ):
                        next_skills = skills
                    else:
                        next_skills = next_states_dict["skill_conditioning"]

                else:
                    next_states_dict, rewards, terminated, truncated, infos = (
                        self.env.step(actions_scaled)
                    )
                    next_states = next_states_dict["policy"]
                    next_skills = next_states_dict["skill_conditioning"]

                next_curiosity_states = next_states_dict["skill_discovery"]

                # After the very first step, compute the initial states (so that randomisation is applied):
                if timestep == 0:
                    self.agents._init_states = next_curiosity_states.clone()

                # Compute the travelled distance in the episode:
                env_ids = (terminated + truncated).squeeze()
                if torch.any(env_ids):
                    travelled_distance[env_ids * ~eval_envs_ids] = torch.norm(
                        base_pos[env_ids * ~eval_envs_ids, :2]
                        - initial_base_pos[env_ids * ~eval_envs_ids, :2],
                        dim=-1,
                    )
                    initial_base_pos[env_ids * ~eval_envs_ids] = (
                        self.env.initial_root_states[
                            env_ids * ~eval_envs_ids, :3
                        ].clone()
                    )
                base_pos[~env_ids * ~eval_envs_ids] = (
                    self.env.scene.articulations["robot"]
                    .data.root_pos_w[~env_ids * ~eval_envs_ids, :3]
                    .clone()
                )

                # episode_trajectory[torch.arange(self.env.unwrapped.scene.num_envs),self.env.unwrapped.episode_length_buf-1,:] = torch.cat([curiosity_states[:,-3:],curiosity_states[:,0:3]],dim=-1)
                episode_trajectory[
                    torch.arange(self.env.unwrapped.scene.num_envs),
                    self.env.unwrapped.episode_length_buf - 1,
                    :,
                ] = torch.cat(
                    [curiosity_states[:, -3:], curiosity_states[:, 0:3]], dim=-1
                )

                # # Check the number of visited cells:
                if torch.any(env_ids):
                    grid_resolution = (
                        torch.tensor([0.2, 0.2, 0.05])
                        .to(self.agents.device)
                        .reshape(1, 1, 3)
                    )
                    position_cell_indices = (
                        (episode_trajectory[env_ids, :, 0:3] // grid_resolution)
                        .int()
                        .reshape(-1, 3)
                    )
                    visited_pos_cells_count = (
                        len(torch.unique(position_cell_indices[:, 0]))
                        + len(torch.unique(position_cell_indices[:, 1]))
                        + len(torch.unique(position_cell_indices[:, 2]))
                    )

                    grid_resolution = (
                        torch.tensor([0.05, 0.05, 0.05])
                        .to(self.agents.device)
                        .reshape(1, 1, 3)
                    )
                    velocity_cell_indices = (
                        (episode_trajectory[env_ids, :, 3:6] // grid_resolution)
                        .int()
                        .reshape(-1, 3)
                    )
                    visited_vel_cells_count = (
                        len(torch.unique(velocity_cell_indices[:, 0]))
                        + len(torch.unique(velocity_cell_indices[:, 1]))
                        + len(torch.unique(velocity_cell_indices[:, 2]))
                    )

                    episode_trajectory[env_ids, :, :] = 0.0

                if self.agents.hierarchical:
                    high_level_extrinsic_rewards = 0 * rewards.view(-1, 1)
                    low_level_extrinsic_rewards = rewards.view(-1, 1)

                    high_level_rewards = 0 * rewards.view(-1, 1)
                    low_level_rewards = rewards.view(-1, 1)

                    intrinsic_rewards = torch.zeros_like(low_level_extrinsic_rewards)

                else:
                    extrinsic_rewards = rewards.clone()
                    intrinsic_rewards = torch.zeros_like(rewards)

                if self.agents.skill_discovery:
                    if self.agents.hierarchical and not self.agents.train_low_level:
                        # If we don't want to train the low level controller we don't need this.
                        pass
                    else:
                        # Compute the reward for skill discovery (LSD):

                        # Need to handle resets (the difference between current and next state is not meaningful):
                        reset_idx = (truncated + terminated).squeeze()

                        if (
                            self.agents.skill_discovery_dist == "l2"
                            and self.agents.sd_version == "base"
                        ):
                            intrinsic_rewards = self.agents._compute_intrinsic_reward(
                                curiosity_states, next_curiosity_states, skills
                            ).detach()
                        elif (
                            self.agents.skill_discovery_dist == "l2"
                            and self.agents.sd_version == "MSE"
                        ):
                            if (self.agents._curiosity_preprocessor != self.agents._empty_preprocessor):

                                intrinsic_rewards = torch.exp(
                                    10
                                    * self.agents._compute_intrinsic_reward(
                                        curiosity_states, next_curiosity_states, skills
                                    ).detach()
                                )
                            else:
                                intrinsic_rewards = 1 / (
                                    1
                                    - 0.05
                                    * self.agents._compute_intrinsic_reward(
                                        curiosity_states, next_curiosity_states, skills
                                    ).detach()
                                )
                        elif (
                            self.agents.skill_discovery_dist == "one"
                            and self.agents.sd_version == "base"
                        ):
                            intrinsic_rewards = self.agents._compute_intrinsic_reward(
                                curiosity_states, next_curiosity_states, skills
                            ).detach()


                        elif (
                            self.agents.skill_discovery_dist == "one"
                            and self.agents.sd_version == "MSE"
                        ):
                            intrinsic_rewards = torch.exp(
                                10
                                * self.agents._compute_intrinsic_reward(
                                    curiosity_states, next_curiosity_states, skills
                                ).detach()
                            )
                        

                        else:
                            raise NotImplementedError(
                                "The distance metric for skill discovery is not implemented."
                            )
                        # intrinsic_rewards = (self.agents._compute_intrinsic_reward(curiosity_states, next_curiosity_states, skills).detach())
                        # intrinsic_rewards += 0.1*(self.agents._compute_regularisation_reward(curiosity_states, next_curiosity_states, skills).detach())
                        intrinsic_rewards[reset_idx] = 0.0

                        if self.agents.hierarchical:
                            # For hierarchical control we use the extrinsic rewards as the high level rewards
                            pass

                        if self.agents.on_policy_rewards:
                            low_level_rewards = (
                                self.agents.extrinsic_reward_scale
                                * low_level_extrinsic_rewards
                                + intrinsic_rewards.reshape(-1, 1)
                                * self.agents.intrinsic_reward_scale
                            )
                            high_level_rewards = high_level_extrinsic_rewards

                else:
                    raise NotImplementedError(
                        "The agent is not using any form of intrinsic reward."
                    )

                # record the environments' transitions
                # TODO: fix size of infos when evaluating
                if self.agents.hierarchical:
                    self.agents.record_transition(
                        low_level_states=low_level_states[~eval_envs_ids],
                        low_level_actions=low_level_actions[~eval_envs_ids],
                        low_level_actions_pretanh=low_level_actions_pretanh[
                            ~eval_envs_ids
                        ],
                        low_level_rewards=low_level_rewards[~eval_envs_ids],
                        low_level_rewards_extrinsic=low_level_extrinsic_rewards[
                            ~eval_envs_ids
                        ],
                        low_level_rewards_intrinsic=intrinsic_rewards.reshape(-1, 1)[
                            ~eval_envs_ids
                        ],
                        next_low_level_states=next_low_level_states[~eval_envs_ids],
                        high_level_states=high_level_states[~eval_envs_ids],
                        high_level_actions=high_level_actions[~eval_envs_ids],
                        high_level_actions_pretanh=high_level_actions_pretanh[
                            ~eval_envs_ids
                        ],
                        next_high_level_states=next_high_level_states[~eval_envs_ids],
                        high_level_rewards=high_level_rewards[~eval_envs_ids],
                        curiosity_states=curiosity_states[~eval_envs_ids],
                        skills=skills[~eval_envs_ids],
                        next_skills=next_skills[~eval_envs_ids],
                        next_curiosity_states=next_curiosity_states[~eval_envs_ids],
                        terminated=terminated[~eval_envs_ids],
                        truncated=truncated[~eval_envs_ids],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                # Record some statistic for the normalizers:
                if (
                    self.agents.pretrain_normalizers
                    and timestep < self.agents._learning_starts
                ):
                    if (
                        self.agents._low_level_state_preprocessor
                        != self.agents._empty_preprocessor
                    ):
                        self.agents.normalizer_states[timestep] = states[:]
                    if (
                        self.agents._curiosity_preprocessor
                        != self.agents._empty_preprocessor
                    ):
                        _curiosity_states = self.agents._curiosity_state_selector(
                            curiosity_states
                        )
                        _curiosity_states = self.agents._curiosity_preprocessor(
                            _curiosity_states, train=False
                        )

                        self.agents.normalizer_curiosity_states[timestep] = (
                            _curiosity_states[:]
                        )
            # log custom environment data
            # if "episode" in infos:
            #     for k, v in infos["episode"].items():
            #         if isinstance(v, torch.Tensor) and v.numel() == 1:
            #             self.agents.track_data(f"EpisodeInfo / {k}", v.item())

            # Log the individual reward terms:
            if "log" in infos:
                for k, v in infos["log"].items():
                    if "Episode Reward" in k or "Episode_Reward" in k:
                        if "low_level" in k:
                            name = f"Episode Reward Low Level / {k.split('/')[-1]}"
                        elif "high_level" in k:
                            name = f"Episode Reward High Level / {k.split('/')[-1]}"
                        else:
                            name = f"Episode Reward  Low Level / {k.split('/')[-1]}"
                        self.agents.track_data(name, v.cpu().numpy())
                    if "Metrics" in k:
                        self.agents.track_data(f"Metrics / {k.split('/')[-1]}", v)

            # Log the distance travelled:
            if self.agents.skill_discovery:
                self.agents.tracking_data["Task/Distance travelled (max)"].append(
                    np.max(travelled_distance[~eval_envs_ids].cpu().numpy())
                )
                self.agents.tracking_data["Task/Distance travelled (mean)"].append(
                    np.mean(travelled_distance[~eval_envs_ids].cpu().numpy())
                )

                self.agents.tracking_data["Task/XYZ Bins Visited"].append(
                    visited_pos_cells_count
                )
                self.agents.tracking_data["Task/Vel Bins Visited"].append(
                    visited_vel_cells_count
                )

            if self.agents.skill_discovery:
                self.agents.tracking_data["Task/Intrinsic Reward (mean)"].append(
                    np.mean(intrinsic_rewards[~eval_envs_ids].cpu().numpy())
                )

            self.agents.tracking_data["Task/Extrinsic Reward (mean)"].append(
                np.mean(low_level_extrinsic_rewards[~eval_envs_ids].cpu().numpy())
            )

            self.agents.tracking_data["Task/Total Reward (mean)"].append(
                np.mean(low_level_rewards[~eval_envs_ids].cpu().numpy())
            )
            # post-interaction

            # If network will be updated:
            if not (self.agents._rollout + 1) % self.agents._rollouts:
                data_collect_time_bef_prev = data_collect_time_bef
                data_collect_time_aft = time.time()
                update_time_bef = time.time()

            # if self.agents.hierarchical and self.agents.train_high_level and not (timestep%self.agents.high_level_action_frequency == 0):
            #     pass
            # else:
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset the environments
            # note: here we do not call reset scene since it is done in the env.step() method
            # update states

            if self.agents.hierarchical:
                high_level_states = next_high_level_states
                low_level_states = next_low_level_states

            else:
                states.copy_(next_states)

            skills.copy_(next_skills)
            curiosity_states.copy_(next_curiosity_states)
            if self.agents.delay_actions:
                prev_actions_scaled.copy_(actions_scaled)
            # If network has been updated:
            if not self.agents._rollout % self.agents._rollouts:
                update_time_aft = time.time()
                data_collect_time_bef = time.time()

            progress_bar.set_description(
                f"Rollout : {(data_collect_time_aft - data_collect_time_bef_prev):.4f} s, Update {(update_time_aft - update_time_bef):.4f} s",
                refresh=True,
            )
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    def eval(self) -> None:
        """Evaluate the agents sequentially.

        This method executes the following steps in loop:

        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        """
        # set running mode
        if self.num_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")
        # single agent
        if self.num_agents == 1:
            self.single_agent_eval()
            return

        # reset env
        states, infos = self.env.reset()
        # evaluation loop
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
        ):
            # compute actions
            with torch.no_grad():
                actions = torch.vstack(
                    [
                        agent.act(
                            states[scope[0] : scope[1]],
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )[0]
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            with torch.no_grad():
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    # track data
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    # log custom environment data
                    if "log" in infos:
                        for k, v in infos["log"].items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                agent.track_data(k, v.item())
                    # perform post-interaction
                    super(type(agent), agent).post_interaction(
                        timestep=timestep, timesteps=self.timesteps
                    )

                # reset environments
                # note: here we do not call reset scene since it is done in the env.step() method
                states.copy_(next_states)
