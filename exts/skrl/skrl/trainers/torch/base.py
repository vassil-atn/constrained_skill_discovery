from typing import List, Optional, Union

import atexit
import sys
import tqdm

import torch

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
# from torchviz import make_dot
import numpy as np
# from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from matplotlib import pyplot as plt

def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})")
    return scopes


class Trainer:
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)

        self.initial_timestep = 0
        self.init_at_random_ep_len = self.cfg.get("init_at_random_ep_len", False)

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})")
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})")
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})")
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()
        curiosity_states = torch.zeros((self.env.num_envs, self.agents.curiosity_obs_size), dtype=torch.float, device=self.env.device)
        
        if self.init_at_random_ep_len:
            self.env.progress_buf = torch.randint_like(self.env.progress_buf, high=int(self.env.max_episode_length))
        

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                


                # If using the hierarchical controller:
                if self.agents.hierarchical:
                    # get the skills from the high level policy:
                    actions, _, _, actions_pretanh = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                    actions = self.env.skills[:]
                    # get the low level actions:
                    actions_lowlevel, _, _, _ = self.agents.low_level_act(states,actions, timestep=timestep, timesteps=self.timesteps)
                    actions_scaled = actions_lowlevel * self.agents.action_scale
                else:
                    skills = self.env.skills.clone().to(self.agents.device)
                    actions, _, _, actions_pretanh = self.agents.act(states,skills, timestep=timestep, timesteps=self.timesteps)
                    actions_scaled = actions * self.agents.action_scale
                

                prev_infos = infos.copy()
                # step the environments

                next_states, rewards, terminated, truncated, infos = self.env.step(actions_scaled)

                next_curiosity_states = infos["curiosity_states"][:]

                next_skills = self.env.skills.clone().to(self.agents.device)

                
                # Compute the curiosity latent:
                if not prev_infos:
                    prev_infos = infos

                extrinsic_rewards = rewards.clone()
                intrinsic_rewards = torch.zeros_like(rewards)

                # if self.agents.use_curiosity:

                #     if self.agents.curiosity_observation_type == "full_state":
                #         curiosity_states = states[:]
                #         next_curiosity_states = next_states[:]

                #     elif self.agents.curiosity_observation_type == "velocities":
                #         curiosity_states = states[:, 0:3]
                #         next_curiosity_states = next_states[:, 0:3]

                #     elif self.agents.curiosity_observation_type == "positions":
                #         curiosity_states = prev_infos["base_pos"][:]#self.agents._curiosity_preprocessor(states)
                #         next_curiosity_states = infos["base_pos"][:]#self.agents._curiosity_preprocessor(next_states)

                #     elif self.agents.curiosity_observation_type == "pos_vel":
                #         curiosity_states = torch.cat((prev_infos["base_pos"][:],states[:,0:6]),dim=1)
                #         next_curiosity_states = torch.cat((infos["base_pos"][:],next_states[:,0:6]),dim=1)

                #     elif self.agents.curiosity_observation_type == "xvel_yaw":
                #         curiosity_states = torch.cat((states[:,0:1],states[:,5:6]),dim=1)
                #         next_curiosity_states = torch.cat((next_states[:,0:1],next_states[:,5:6]),dim=1)

                #     curiosities = self.agents.curiosity_model.curiosity(self.agents._curiosity_preprocessor(curiosity_states),actions, \
                #                                                         self.agents._curiosity_preprocessor(next_curiosity_states)).detach()
                #     intrinsic_rew = (curiosities).clip(-3*torch.sqrt(self.agents.intr_ret_rms.running_variance), 3*torch.sqrt(self.agents.intr_ret_rms.running_variance)) # ADD CLIP

                #     # Compute the intrinsic return:
                #     self.agents.intr_ret = self.agents.intr_ret * self.agents._discount_factor + intrinsic_rew.unsqueeze(1)
                #     # Train the running mean and variance:
                #     _ = self.agents.intr_ret_rms(self.agents.intr_ret,train=True) 

                #     # Normalise the intrinsic reward by the std of the running return (comes from RND paper):
                #     intrinsic_rew = intrinsic_rew / torch.sqrt(self.agents.intr_ret_rms.running_variance + 1e-8)
                    
                    
                #     intrinsic_rewards = intrinsic_rew.unsqueeze(1) * self.agents._curiosity_scale
                #     rewards = rewards + intrinsic_rewards
                
                if self.agents.hierarchical:
                    # Reward is just the extrinsic reward from the environment
                    pass

                elif self.agents.skill_discovery:
                    # Compute the reward for skill discovery (LSD):

                    # Need to handle resets (the difference between current and next state is not meaningful):
                    reset_idx = (truncated + terminated).squeeze()

                    curiosity_states_processed = self.agents._curiosity_preprocessor(curiosity_states)
                    next_curiosity_states_processed = self.agents._curiosity_preprocessor(next_curiosity_states)


                    if self.agents.sd_specific_state_ind is not None:
                        mask = torch.ones_like(curiosity_states_processed, dtype=torch.bool, device=self.agents.device)
                        mask[:,self.agents.sd_specific_state_ind] = False
                        curiosity_states_processed[mask] = 0.
                        next_curiosity_states_processed[mask] = 0.



                    intrinsic_rewards = self.agents._compute_intrinsic_reward(curiosity_states_processed, next_curiosity_states_processed, skills)
                    intrinsic_rewards[reset_idx] = 0.

                    if self.agents.on_policy_rewards:
                        rewards = self.agents.extrinsic_reward_scale * rewards + intrinsic_rewards.reshape(-1,1) * self.agents.intrinsic_reward_scale
                else:
                    raise NotImplementedError("The agent is not using any form of intrinsic reward.")


                # render scene
                if not self.headless:
                    self.env.render()

                # Process the rewards for the nstep return:
                # if self.agents.use_nstep_return: 
                #     states, actions, rewards, next_states, terminated = self.agents.n_step_buffer.add_to_buffer(states, actions, rewards, next_states, terminated)
                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              actions_pretanh=actions_pretanh,
                                              rewards=rewards,
                                              rewards_extrinsic=extrinsic_rewards,
                                              rewards_intrinsic=intrinsic_rewards,
                                              next_states=next_states,
                                              curiosity_states=curiosity_states,
                                              next_curiosity_states=next_curiosity_states,
                                              skills=skills,
                                              next_skills = next_skills,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

                # Record some statistic for the normalizers:
                if self.agents.pretrain_normalizers and timestep < self.agents._learning_starts:
                    self.agents.normalizer_states[timestep] = states[:]
                    self.agents.normalizer_curiosity_states[timestep] = curiosity_states[:]

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # Log some stuff:
            # Log rewards:
            for reward in self.env.episode_sums:
                
                if self.env.rew_scales[reward] != 0:
                    self.agents.tracking_data[f"Rollout Rewards / {reward}"].append(np.mean(self.env.episode_sums[reward].cpu().numpy()))

            # Log the distance travelled:
            if self.agents.skill_discovery:
                self.agents.tracking_data["Task/Distance travelled (max)"].append(self.env.travelled_distance_episode.max().cpu().numpy())
                self.agents.tracking_data["Task/Distance travelled (mean)"].append(self.env.travelled_distance_episode.mean().cpu().numpy())

                self.agents.tracking_data["Task/XYZ Bins Visited"].append(self.env.visited_pos_cells_count)
                self.agents.tracking_data["Task/Vel Bins Visited"].append(self.env.visited_vel_cells_count)


            # reset environments
            if self.env.num_envs > 1:
                states = next_states
                curiosity_states = next_curiosity_states[:]
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                        curiosity_states = torch.zeros((self.env.num_envs, self.agents.curiosity_obs_size), dtype=torch.float, device=self.env.device)
                else:
                    states = next_states
                    curiosity_states = next_curiosity_states[:]

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()
        curiosity_states = torch.zeros((self.env.num_envs, self.agents.curiosity_obs_size), dtype=torch.float, device=self.env.device)
        next_states = states.clone()
        # Store env states for plotting:    
        base_lin_vels = []
        base_ang_vels = []
        base_pos = []
        commands = []
        base_lin_vels_global = []
        base_ang_vels_global = []
        curiosities_stored = []
        joint_pos = []
        commands_obs = []
        sampled_skills = []
        sampled_traj = []
        reset_envs = np.zeros(self.env.num_envs)
        sampled_curio_states = []
        sampled_latent_traj = []
        sampled_vel_traj = []
        joint_vel_trajectory = []

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # compute actions
            with torch.no_grad():
                
                manual_skill = False
                uniform_skill = False

                if manual_skill:
                    # # Zero-shot point following:
                    # des_base_pos = torch.tensor([10.0, 10.0, 0.62], device=self.env.device).repeat(self.env.num_envs,1)
                    des_base_pos = self.env.commands_given[:]
                    des_curiosity_state = 0*curiosity_states[:,:]
                    des_curiosity_state[:,-3:] = des_base_pos[:]
                    des_curiosity_state[:,9] = 1. # quaternion
                    next_states_unmasked_processed = self.agents._curiosity_preprocessor(des_curiosity_state)
                    states_unmasked_processed = self.agents._curiosity_preprocessor(curiosity_states)

                    if self.agents.sd_specific_state_ind is not None:
                        mask = torch.ones_like(states_unmasked_processed, dtype=torch.bool, device=self.agents.device)
                        mask[:,self.agents.sd_specific_state_ind] = False
                        states_unmasked_processed[mask] = 0.
                        next_states_unmasked_processed[mask] = 0.

                    des_base_pos_latent = self.agents.lsd_model.get_distribution(next_states_unmasked_processed).mean
                    base_pos_latent = self.agents.lsd_model.get_distribution(states_unmasked_processed).mean

                    skills = 1. * (des_base_pos_latent - base_pos_latent) / torch.clamp(torch.norm(des_base_pos_latent - base_pos_latent, dim=1).unsqueeze(1),min=1e-4)
                
                    
                elif uniform_skill:
                    if self.agents.discrete_skills:
                        skills = torch.randint(0, self.env.skill_dim, (self.env.num_envs,1), device=self.env.device)
                        skills = torch.nn.functional.one_hot(skills.to(torch.int64), self.env.skill_dim).to(torch.float32)

                    else:
                        angles = torch.linspace(0,2*np.pi,self.env.num_envs,device=self.env.device)
                        skills = torch.stack((torch.cos(angles),torch.sin(angles)),dim=1)


                else:
                    skills = self.env.skills

                # if timestep % 100 == 0:
                #     print("Skill switched.")
                #     skills = torch.randn((self.env.num_envs,self.env.skill_dim), dtype=torch.float, device=self.env.device)
                # skills = self.env.skills
                    
                # if timestep == 0:
                #     metric = np.arctan2(skills[:,1].cpu().numpy(), skills[:,0].cpu().numpy())
                #     # Normalize metric
                #     metric = (metric - metric.min()) / (metric.max() - metric.min())
                #     colormap = plt.cm.hsv
                #     plt.ion()
                #     fig,ax = plt.subplots(1,2)
                #     latent_trajectory_line = [ax[0].plot(0,0, color=colormap(metric[i]))[0] for i in range(self.env.num_envs)]
                #     real_trajectory_line = [ax[1].plot(0,0, color=colormap(metric[i]))[0] for i in range(self.env.num_envs)]



                # If using the hierarchical controller:
                if self.agents.hierarchical:
                    # get the skills from the high level policy:
                    # actions, _, _, actions_pretanh = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)
                    # get the low level actions:
                    skills = self.env.skills.clone().to(self.agents.device)
                    actions_lowlevel, _, _, _ = self.agents.low_level_act(states,skills, timestep=timestep, timesteps=self.timesteps)
                    actions_scaled = actions_lowlevel * self.agents.action_scale

                else:
                    actions, _, _, actions_pretanh = self.agents.act(states,skills, timestep=timestep, timesteps=self.timesteps)
                    actions_scaled = actions * self.agents.action_scale

                prev_infos = infos.copy()
                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions_scaled)

                if manual_skill:
                    self.env._draw_zero_shot_goal(des_base_pos)
                next_curiosity_states = infos["curiosity_states"][:]

                # Update rewards based on intrinsic reward:
                
                # Compute the curiosity latent:
                if not prev_infos:
                    prev_infos = infos


                # if self.agents.use_curiosity:

                #     if self.agents.curiosity_observation_type == "full_state":
                #         curiosity_states = states[:]
                #         next_curiosity_states = next_states[:]

                #     elif self.agents.curiosity_observation_type == "velocities":
                #         curiosity_states = states[:, 0:3]
                #         next_curiosity_states = next_states[:, 0:3]

                #     elif self.agents.curiosity_observation_type == "positions":
                #         curiosity_states = prev_infos["base_pos"][:]#self.agents._curiosity_preprocessor(states)
                #         next_curiosity_states = infos["base_pos"][:]#self.agents._curiosity_preprocessor(next_states)

                #     elif self.agents.curiosity_observation_type == "pos_vel":
                #         curiosity_states = torch.cat((prev_infos["base_pos"][:],states[:,0:6]),dim=1)
                #         next_curiosity_states = torch.cat((infos["base_pos"][:],next_states[:,0:6]),dim=1)

                #     elif self.agents.curiosity_observation_type == "xvel_yaw":
                #         curiosity_states = torch.cat((states[:,0:1],states[:,5:6]),dim=1)
                #         next_curiosity_states = torch.cat((next_states[:,0:1],next_states[:,5:6]),dim=1)

                #     elif self.agents.curiosity_observation_type == "debugLSD":
                #         curiosity_states = torch.cat((states[:,:-2]),dim=1)
                #         next_curiosity_states = torch.cat((next_states[:,:-2]),dim=1)

                #     curiosities = self.agents.curiosity_model.curiosity(self.agents._curiosity_preprocessor(curiosity_states),actions, \
                #                                                         self.agents._curiosity_preprocessor(next_curiosity_states)).detach()
                #     intrinsic_rew = (curiosities).clip(-3*torch.sqrt(self.agents.intr_ret_rms.running_variance), 3*torch.sqrt(self.agents.intr_ret_rms.running_variance)) # ADD CLIP

                #     # Compute the intrinsic return:
                #     self.agents.intr_ret = self.agents.intr_ret * self.agents._discount_factor + intrinsic_rew.unsqueeze(1)
                #     # Train the running mean and variance:
                #     _ = self.agents.intr_ret_rms(self.agents.intr_ret,train=True) 

                #     # Normalise the intrinsic reward by the std of the running return (comes from RND paper):
                #     intrinsic_rew = intrinsic_rew / torch.sqrt(self.agents.intr_ret_rms.running_variance + 1e-8)
                    
                    
                #     intrinsic_rewards = intrinsic_rew.unsqueeze(1) * self.agents._curiosity_scale 
                #     rewards = rewards + intrinsic_rewards
                

                if self.agents.skill_discovery:
                    # Store the sampled skills and corresponding trajectory:
                    # skills = prev_infos["skills"]
                    curiosity_states_processed = self.agents._curiosity_preprocessor(curiosity_states)
                    next_curiosity_states_processed = self.agents._curiosity_preprocessor(next_curiosity_states)

                    if self.agents.sd_specific_state_ind is not None:
                        mask = torch.ones_like(curiosity_states_processed, dtype=torch.bool, device=self.agents.device)
                        mask[:,self.agents.sd_specific_state_ind] = False
                        curiosity_states_processed[mask] = 0.
                        next_curiosity_states_processed[mask] = 0.

                    if self.agents.discrete_skills:
                        sampled_skills.append( torch.argmax(skills, dim=1).cpu().numpy())
                    else:
                        sampled_skills.append( skills.cpu().numpy())
                    rel_base_pos = self.env.root_states[:, 0:3]
                    rel_base_pos[:,:2] = self.env.root_states[:, 0:2] - self.env.initial_root_states[:,0:2]
                    sampled_traj.append((rel_base_pos).cpu().numpy())
                    sampled_vel_traj.append(self.env.root_states[:, 7:13].cpu().numpy())
                    sampled_curio_states.append(curiosity_states.cpu().numpy())
                    latent_trajectory = self.agents.lsd_model.get_distribution(curiosity_states_processed).mean
                    sampled_latent_traj.append(latent_trajectory.cpu().numpy())
                    distance_travelled = self.env.travelled_distance_episode

                    

                    # Check the penalty violation when using metra:
                    cst_dist = torch.ones_like(curiosity_states_processed[:,0])
                    latent_states_mean = self.agents.lsd_model.get_distribution(curiosity_states_processed).mean
                    latent_next_states_mean = self.agents.lsd_model.get_distribution(next_curiosity_states_processed).mean
                    cst_penalty = (cst_dist - torch.square(latent_next_states_mean - latent_states_mean).mean(dim=1)).clamp(max=1e-3)


                    # sampled_traj.append((self.env.root_states[:, 0:3] - self.env.initial_root_states[:,0:3]).cpu().numpy())
                    reset_envs[self.env.reset_buf.cpu().numpy().astype(bool)] = 1

                # # render scene
                if not self.headless:
                    self.env.render()

                # Do the real-time plot:
                # if timestep % 5 == 0:
                #     # Real trajectory:
                #     x = rel_base_pos.cpu().numpy()[:,0]
                #     y = rel_base_pos.cpu().numpy()[:,1]
 

                #     # Latent:
                #     latent_x = latent_trajectory.cpu().numpy()[:,0]
                #     latent_y = latent_trajectory.cpu().numpy()[:,1]

                #     for i in range(self.env.num_envs):
                #         if timestep == 0:
                #             latent_x_data = latent_x[i]
                #             latent_y_data = latent_y[i]
                #             x_data = x[i]
                #             y_data = y[i]
                #         else:
                #             latent_x_data = np.append(latent_trajectory_line[i].get_xdata(),latent_x[i])
                #             latent_y_data = np.append(latent_trajectory_line[i].get_ydata(),latent_y[i])
                #             x_data = np.append(real_trajectory_line[i].get_xdata(),x[i])
                #             y_data = np.append(real_trajectory_line[i].get_ydata(),y[i])

                #         real_trajectory_line[i].set_data(x_data,y_data)
                #         latent_trajectory_line[i].set_data(latent_x_data,latent_y_data)
                #     ax[0].relim()  # Update the axes limits
                #     ax[0].autoscale_view()  # Auto-scale the axes
                #     ax[1].relim()  # Update the axes limits
                #     ax[1].autoscale_view()  # Auto-scale the axes
                #     plt.draw()
                #     plt.pause(0.001)

                # if plot
                plot = False
                

         
                base_quat = self.env.root_states[:, 3:7]
                base_lin_vels_global.append(self.env.root_states[:, 7:10].cpu().numpy().copy())
                base_ang_vels_global.append(self.env.root_states[:, 10:13].cpu().numpy().copy())
                # base_lin_vels.append(quat_rotate_inverse(base_quat, self.env.root_states[:, 7:10]).cpu().numpy().copy())
                # base_ang_vels.append(quat_rotate_inverse(base_quat, self.env.root_states[:, 10:13]).cpu().numpy().copy())
                base_pos.append(self.env.root_states[:, 0:3].cpu().numpy().copy())
                commands.append(self.env.commands_given.cpu().numpy().copy())
                commands_obs.append(self.env.commands.cpu().numpy().copy())
                joint_pos.append(self.env.dof_pos.cpu().numpy().copy())
                joint_vel_trajectory.append(self.env.dof_vel.cpu().numpy().copy())

                # if self.agents.use_curiosity:
                #     curiosities_stored.append(curiosities.cpu().numpy().copy())
                # write data to TensorBoard
                # self.agents.record_transition(states=states,
                #                               actions=actions,
                #                               rewards=rewards,
                #                               next_states=next_states,
                #                               terminated=terminated,
                #                               truncated=truncated,
                #                               infos=infos,
                #                               timestep=timestep,
                #                               timesteps=self.timesteps)
                # super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
                curiosity_states = next_curiosity_states[:]
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                        curiosity_states = torch.zeros((self.env.num_envs, self.agents.curiosity_obs_size), dtype=torch.float, device=self.env.device)
                else:
                    states = next_states
                    curiosity_states = next_curiosity_states[:]


        base_lin_vels = np.array(base_lin_vels)
        base_ang_vels = np.array(base_ang_vels)
        commands = np.array(commands)
        base_lin_vels_global = np.array(base_lin_vels_global)
        base_ang_vels_global = np.array(base_ang_vels_global)
        curiosities_stored = np.array(curiosities_stored)
        joint_pos = np.array(joint_pos)
        base_pos = np.array(base_pos)
        commands_obs = np.array(commands_obs)
        sampled_skills = np.array(sampled_skills)
        sampled_traj = np.array(sampled_traj)
        sampled_curio_states = np.array(sampled_curio_states)
        sampled_latent_traj = np.array(sampled_latent_traj)
        sampled_vel_traj = np.array(sampled_vel_traj)
        joint_vel_trajectory = np.array(joint_vel_trajectory)

        # np.savez("/home/vassil/skrl/data_default_ppo", base_lin_vels=base_lin_vels, base_ang_vels=base_ang_vels, commanded_vels=commanded_vels)
        plt.ioff()
        if plot:
            base_lin_vels = base_lin_vels[:,0,:]
            base_ang_vels = base_ang_vels[:,0,:]
            commands = commands[:,0:5,:]
            joint_pos = joint_pos[:,0,:]
            base_pos = base_pos[:,0:5,:]
            commands_obs = commands_obs[:,0,:]

            base_lin_vels_global = base_lin_vels_global[:,0,:]
            base_ang_vels_global = base_ang_vels_global[:,0,:]
            # if self.agents.use_curiosity:
            #     curiosities_stored = curiosities_stored[:,0]

        # Plot correlation matrix:
        state_dim = sampled_curio_states.shape[-1]
        skill_dim = self.env.skill_dim
        
        correlation_matrix = np.corrcoef(np.concatenate((sampled_curio_states.reshape(-1,state_dim), sampled_latent_traj.reshape(-1,skill_dim)), axis=1), rowvar=False)
        correlations_X_to_y = correlation_matrix[:state_dim, state_dim:]  # Extract the correlation values between X and y
        plt.figure()
        plt.imshow(correlations_X_to_y.T, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Correlation')
        plt.title('Correlation Matrix between X and y')
        plt.ylabel('Output Variables (y)')
        plt.xlabel('Input Features (X)')
        plt.yticks(np.arange(skill_dim))#, ['y1', 'y2','y3'])  # Label output variables
        plt.xticks(np.arange(state_dim))  # Label input features
        plt.tight_layout()

        if self.agents.skill_discovery:
            plt.figure()
            plt.suptitle("Position Trajectories")
            if self.agents.discrete_skills:

                cmap = plt.get_cmap('tab20', self.env.skill_dim)
                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    plt.plot(sampled_traj[:,i,0],sampled_traj[:,i,1], color=cmap(sampled_skills[0,i]), label=f'Skill {sampled_skills[0,i]}')
                    # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
                legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(self.env.skill_dim)]
                plt.legend(handles=legend_handles, loc='upper left')
            else:
                # Distance metric:
                metric = np.arctan2(sampled_skills[0,:,1], sampled_skills[0,:,0])
                # Normalize metric
                metric = (metric - metric.min()) / (metric.max() - metric.min())
                # Create colormap based on cosine similarity
                colormap = plt.cm.hsv  # You can use any other colormap

                # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.env.num_envs)))

                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    if manual_skill:
                        color = 'g'
                    else:
                        color = colormap(metric[i])
                    plt.plot(sampled_traj[:,i,0],sampled_traj[:,i,1], label=f'Skill {sampled_skills[0,i]}', color=color)
                plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
                plt.figure()
                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    # Plot z trajectory:
                    if manual_skill:
                        color = 'g'
                    else:
                        color = colormap(metric[i])
                    plt.plot(sampled_traj[:,i,2], label=f'Skill {i}',color=color)
                    # print(sampled_skills[0,i])
                plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')
                    # plt.scatter(sampled_traj[:,i,0],sampled_traj[:,i,1], c=sampled_skills[:,i], cmap='jet', label=f'Trajectory {i+1}')
                # legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(self.env.skill_dim)]
                # plt.legend(handles=legend_handles, loc='upper left')
            # plt.figure()
            # plt.suptitle("Velocity Trajectories")
            # if self.agents.discrete_skills:

            #     cmap = plt.get_cmap('tab20', self.env.skill_dim)
            #     for i in range(self.env.num_envs):
            #         if reset_envs[i] == 1:
            #             continue
            #         plt.plot(sampled_vel_traj[:,i,0],sampled_vel_traj[:,i,1], color=cmap(sampled_skills[0,i]), label=f'Skill {sampled_skills[0,i]}')
            #     legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(self.env.skill_dim)]
            #     plt.legend(handles=legend_handles, loc='upper left')
            # else:
            #     # Distance metric:
            #     metric = np.arctan2(sampled_skills[0,:,1], sampled_skills[0,:,0])
            #     # Normalize metric
            #     metric = (metric - metric.min()) / (metric.max() - metric.min())
            #     # Create colormap based on cosine similarity
            #     colormap = plt.cm.hsv  # You can use any other colormap

            #     for i in range(self.env.num_envs):
            #         if reset_envs[i] == 1:
            #             continue
            #         if manual_skill:
            #             color = 'g'
            #         else:
            #             color = colormap(metric[i])
            #         plt.plot(sampled_vel_traj[:,i,0],sampled_vel_traj[:,i,1], label=f'Skill {sampled_skills[0,i]}', color=color)
            #     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')

            plt.figure()
            plt.suptitle("Velocity Clusters")
            if self.agents.discrete_skills:
                cmap = plt.get_cmap('tab20', self.env.skill_dim)
                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    plt.scatter(sampled_vel_traj[:,i,0],sampled_vel_traj[:,i,1], color=cmap(sampled_skills[0,i]), label=f'Skill {sampled_skills[0,i]}')
                legend_handles = [plt.Line2D([0], [0], linestyle='', marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=f"Skill {i}") for i in range(self.env.skill_dim)]
                plt.legend(handles=legend_handles, loc='upper left')
            else:
                # Distance metric:
                metric = np.arctan2(sampled_skills[0,:,1], sampled_skills[0,:,0])
                # Normalize metric
                metric = (metric - metric.min()) / (metric.max() - metric.min())
                # Create colormap based on cosine similarity
                colormap = plt.cm.hsv  # You can use any other colormap

                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    if manual_skill:
                        color = 'g'
                    else:
                        color = colormap(metric[i])
                    plt.scatter(sampled_vel_traj[:,i,0],sampled_vel_traj[:,i,1], label=f'Skill {sampled_skills[0,i]}', color=color)
                plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')                


            fig,axis = plt.subplots(1,2)
            fig.suptitle("Latent Trajectories and Skills")


            colormap = plt.cm.hsv 
            if self.agents.discrete_skills:
                pass
            else:
                for i in range(self.env.num_envs):
                    if reset_envs[i] == 1:
                        continue
                    color = colormap(metric[i])
                    axis[0].plot(sampled_latent_traj[:,i,0],sampled_latent_traj[:,i,1], color=color)
                    # Also plot the skill unit vector in the same colour:
                    
                    axis[1].plot([0,sampled_skills[0,i,0]], [0,sampled_skills[0,i,1]], color=color)

                plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label='Skill angle')  
            # plt.figure()
            # plt.suptitle("Latent Trajectories")
            # # First plot the latent space:
            # n = 5000
            # states_space = np.zeros((n,curiosity_states.shape[1]))

            # pos = np.zeros((n,3))
            # pos[:,:2] = np.random.uniform(-50.,50.,(n,2))
            # pos[:,2] = np.random.uniform(0.,0.8,(n))
            # vel = np.zeros((n,3))
            # vel[:,:2] = np.random.uniform(-6.,6.,(n,2))
            # vel[:,2] = np.random.uniform(-0.8,0.8,(n))
            # ang_vel = np.zeros((n,3))
            # ang_vel[:,:2] = np.random.uniform(-1.,1.,(n,2))
            # ang_vel[:,2] = np.random.uniform(-3.14,3.14,(n))

            # states_space[:,0:6] = np.concatenate((vel,ang_vel),axis=1)
            # states_space[:,-3:] = pos

            # states_space_processed = self.agents._curiosity_preprocessor(torch.tensor(states_space, dtype=torch.float, device=self.env.device))
            # latent_space = self.agents.lsd_model.get_distribution(states_space_processed).mean.detach().cpu().numpy()

            # plt.scatter(latent_space[:,0],latent_space[:,1], c='gray', alpha=0.5, label=f'Latent space')

            # # Plot the latent space trajectory:
            # robot_id = 0
            # plt.plot(sampled_latent_traj[:,robot_id,0],sampled_latent_traj[:,robot_id,1], label=f'Latent trajectory', color='r')


            fig,ax = plt.subplots(4,3)
            fig.suptitle("Joint velocity")
            for i in range(4):
                for j in range(3):
                    ax[i,j].plot(joint_vel_trajectory[:,0,i*3+j], label=f"Joint {i*3+j}")
                    ax[i,j].legend()

            # Plot visited sets of the XY state space:
            # grid_resolution = 0.5
            # cell_indices = np.int64(sampled_traj.reshape(-1,2) // grid_resolution)
            # visited_cells = set(map(tuple, cell_indices))
            # print(f"Visited cells number: {len(visited_cells)}")
            # plt.figure()
            # plt.suptitle("Visited cells")
            # plt.scatter(cell_indices[:,0], cell_indices[:,1], s=1)

            


        if plot:
            fig,ax = plt.subplots(5,1)
            for i in range(5):
                ax[i].plot(base_pos[:,i,0],base_pos[:,i,1], label=f"Base pos {i}")
                ax[i].plot(commands[-1,i,0],commands[-1,i,1], label=f"Commanded pos {i}", marker="o", markersize=5)
                ax[i].legend()
            # ax[0].plot(base_pos[:,0], label="Base x pos")
            # ax[0].plot(commands[:,0], label="Commanded x pos")
            # ax[0].legend()
            # ax[1].plot(base_pos[:,1], label="Base y pos")
            # ax[1].plot(commands[:,1], label="Commanded y pos")
            # ax[1].legend()

            fig,ax = plt.subplots(2,1)
            ax[0].plot(commands_obs[:,0], label="Command x obs")
            ax[0].plot(commands[:,0], label="Commanded x pos")
            ax[0].legend()
            ax[1].plot(commands_obs[:,1], label="Command y obs")
            ax[1].plot(commands[:,1], label="Commanded y pos")
            ax[1].legend()      
            
            fig,ax = plt.subplots(3,1)
            ax[0].plot(base_lin_vels[:,0], label="Base linear velocity(x)")
            ax[0].plot(base_lin_vels_global[:,0], label="Base linear velocity global (x)")
            # ax[0].plot(commands[:,0], label="Commanded linear velocity(x)")
            ax[0].legend()
            ax[1].plot(base_lin_vels[:,1], label="Base linear velocity(y)")
            ax[1].plot(base_lin_vels_global[:,1], label="Base linear velocity global (y)")
            # ax[1].plot(commands[:,1], label="Commanded linear velocity(y)")
            ax[1].legend()
            ax[2].plot(base_ang_vels[:,2], label="Base angular velocity(z)")
            ax[2].plot(base_ang_vels_global[:,2], label="Base angular velocity global (z)")
            # ax[2].plot(commands[:,2], label="Commanded angular velocity(z)")
            ax[2].legend()

            if self.agents.use_curiosity:
                fig,ax = plt.subplots(1,1)
                ax.plot(curiosities_stored[:], label="Curiosity")
                ax.legend()

            fig,ax = plt.subplots(4,3)
            for i in range(4):
                for j in range(3):
                    ax[i,j].plot(joint_pos[:,i*3+j], label=f"Joint {i*3+j}")
                    ax[i,j].legend()

        plt.show()
    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = infos.get("shared_states", None)

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = infos.get("shared_states", None)
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with torch.no_grad():
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = infos.get("shared_states", None)
                else:
                    states = next_states
                    shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = infos.get("shared_states", None)

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = infos.get("shared_states", None)
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # reset environments
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = infos.get("shared_states", None)
                else:
                    states = next_states
                    shared_states = shared_next_states
