from typing import Any, Dict, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
# from torchviz import make_dot
from skrl.resources.preprocessors.torch import RunningStandardScaler

import numpy as np
import collections

# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "curiosity_observation_type" : "full_state",

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class LBSPPO(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._entropy_reward_scale = self.cfg["entropy_reward_scale"]
        self.use_nstep_return = self.cfg["use_nstep_return"]

        self.num_envs = self.cfg["num_envs"]

        self._track_extrinsic_rewards = collections.deque(maxlen=100)
        self._track_intrinsic_rewards = collections.deque(maxlen=100)
        self._new_track_timesteps = collections.deque(maxlen=100)
        self._cumulative_extrinsic_rewards = None
        self._cumulative_intrinsic_rewards = None
        self._new_cumulative_timesteps = None

        self.intr_ret_rms = RunningStandardScaler(1,device=self.device)
        self.intr_ret = torch.zeros((self.num_envs, 1),device=self.device)
        self.curiosity_model = self.cfg["curiosity_model"]
        self._curiosity_lr = self.cfg["curiosity_lr"]
        self._curiosity_scale = self.cfg["curiosity_scale"]
        self.anneal_curiosity_warmup = self.cfg["anneal_curiosity_warmup"]
        self.anneal_curiosity_end = self.cfg["anneal_curiosity_end"]
        self.curiosity_optimizer = torch.optim.Adam(self.curiosity_model.parameters(), lr=self._curiosity_lr)

        self.use_curiosity = self.cfg["use_curiosity"]
        self.skill_discovery = self.cfg["skill_discovery"]
        self.mask_xy_pos = self.cfg["mask_xy_pos"]

        self.curiosity_observation_type = self.cfg["curiosity_observation_type"]
        self.curiosity_obs_size = self.curiosity_model.obs_size
        self._curiosity_preprocessor = self.cfg["curiosity_preprocessor"]
        self._curio_mini_batches = self.cfg["curiosity_mini_batches"]
        self._curiosity_epochs = self.cfg["curiosity_epochs"]

        self.checkpoint_modules["curiosity"] = self.curiosity_model
        self.checkpoint_modules["curiosity_optimizer"] = self.curiosity_optimizer

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._curiosity_preprocessor:
            self._curiosity_preprocessor = self._curiosity_preprocessor(**self.cfg["curiosity_preprocessor_kwargs"])
            self.checkpoint_modules["curiosity_preprocessor"] = self._curiosity_preprocessor
        else:
            self._curiosity_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="curiosity_states", size=self.curiosity_obs_size, dtype=torch.float32)
            self.memory.create_tensor(name="next_curiosity_states", size=self.curiosity_obs_size, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "next_states", "curiosity_states","next_curiosity_states","returns", "advantages","rewards"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        self._current_log_prob = log_prob

        return actions, log_prob, outputs


    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(k, np.min(v), timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(k, np.max(v), timestep)
            else:
                self.writer.add_scalar(k, np.mean(v), timestep)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_extrinsic_rewards.clear()
        self._track_intrinsic_rewards.clear()
        self._track_timesteps.clear()
        self._new_track_timesteps.clear()
        self.tracking_data.clear()

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          rewards_extrinsic: torch.Tensor,
                          rewards_intrinsic: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int,
                          curiosity_states: torch.Tensor = None,
                          next_curiosity_states: torch.Tensor = None,
                          skills: torch.Tensor = None) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions,rewards,next_states, terminated, truncated, infos, timestep, timesteps)
        if self.write_interval > 0:
            # compute the cumulative sum of the rewards and timesteps



            if self._cumulative_extrinsic_rewards is None:
                self._cumulative_extrinsic_rewards = torch.zeros_like(rewards_extrinsic, dtype=torch.float32)
            if self._cumulative_intrinsic_rewards is None:
                self._cumulative_intrinsic_rewards = torch.zeros_like(rewards_intrinsic, dtype=torch.float32)
                self._new_cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_extrinsic_rewards.add_(rewards_extrinsic)
            self._cumulative_intrinsic_rewards.add_(rewards_intrinsic)
            self._new_cumulative_timesteps.add_(1)


            finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():

                # storage cumulative rewards and timesteps
                self._track_extrinsic_rewards.extend(self._cumulative_extrinsic_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_intrinsic_rewards.extend(self._cumulative_intrinsic_rewards[finished_episodes][:, 0].reshape(-1).tolist())

                self._new_track_timesteps.extend(self._new_cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_extrinsic_rewards[finished_episodes] = 0
                self._cumulative_intrinsic_rewards[finished_episodes] = 0
                self._new_cumulative_timesteps[finished_episodes] = 0


            self.tracking_data["Reward / Extrinsic instantenous reward (max)"].append(torch.max(rewards_extrinsic).item())
            self.tracking_data["Reward / Extrinsic instantenous reward (min)"].append(torch.min(rewards_extrinsic).item())
            self.tracking_data["Reward / Extrinsic instantenous reward (mean)"].append(torch.mean(rewards_extrinsic).item())

            self.tracking_data["Reward / Intrinsic instantenous reward (max)"].append(torch.max(rewards_intrinsic).item())
            self.tracking_data["Reward / Intrinsic instantenous reward (min)"].append(torch.min(rewards_intrinsic).item())
            self.tracking_data["Reward / Intrinsic instantenous reward (mean)"].append(torch.mean(rewards_intrinsic).item())

            if len(self._track_rewards):
                track_extrinsic_rewards = np.array(self._track_extrinsic_rewards)
                track_intrinsic_rewards = np.array(self._track_intrinsic_rewards)
                track_timesteps = np.array(self._new_track_timesteps)

                self.tracking_data["Reward / Total extrinsic reward (max)"].append(np.max(track_extrinsic_rewards))
                self.tracking_data["Reward / Total extrinsic reward (min)"].append(np.min(track_extrinsic_rewards))
                self.tracking_data["Reward / Total extrinsic reward (mean)"].append(np.mean(track_extrinsic_rewards))

                self.tracking_data["Reward / Total intrinsic reward (max)"].append(np.max(track_intrinsic_rewards))
                self.tracking_data["Reward / Total intrinsic reward (min)"].append(np.min(track_intrinsic_rewards))
                self.tracking_data["Reward / Total intrinsic reward (mean)"].append(np.mean(track_intrinsic_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated


 
            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, curiosity_states=curiosity_states, next_curiosity_states=next_curiosity_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float())}, role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")

        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=values,
                                          next_values=last_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        sampled_curiosity_batches = self.memory.sample_all(names=["curiosity_states","actions","next_curiosity_states"], mini_batches=self._curio_mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        entropy_reward_cumulative = 0
        sampled_rewards_cumulative = 0
        cumulative_exploration_loss = 0
        cumulative_prior_loss = 0
        cumulative_target_loss = 0
        # learning epochs

        if self.cfg["anneal_curiosity"] and timestep > self.anneal_curiosity_warmup:
            self._curiosity_scale = (self.cfg["curiosity_scale"] - self.cfg["curiosity_scale"]*((timestep-self.anneal_curiosity_warmup)/(self.anneal_curiosity_end-self.anneal_curiosity_warmup))).clip(min=0.)

        for curio_epoch in range(self._curiosity_epochs):
            for sampled_curiosity_states, sampled_actions, sampled_next_curiosity_states in sampled_curiosity_batches:
                # curiosity loss
                self.curiosity_optimizer.zero_grad()


                sampled_curiosity_states = self._curiosity_preprocessor(sampled_curiosity_states, train=not curio_epoch)
                sampled_curiosity_next_states = self._curiosity_preprocessor(sampled_next_curiosity_states, train=not curio_epoch)

                loss, loss_prior, loss_target = self.curiosity_model.loss(sampled_curiosity_states, sampled_actions, sampled_curiosity_next_states)
                loss = loss.mean()
                loss_prior = loss_prior.mean()
                loss_target = loss_target.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.curiosity_model.parameters(), self._grad_norm_clip)
                cumulative_exploration_loss += loss.item()
                cumulative_prior_loss += loss_prior.item()
                cumulative_target_loss += loss_target.item()



        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for sampled_states, sampled_actions, sampled_log_prob, sampled_values, sampled_next_states, sampled_curiosity_states, sampled_next_curiosity_states, \
            sampled_returns, sampled_advantages, sampled_rewards in sampled_batches:

                
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=not epoch)
                mean_test,std_test,outputs_test = self.policy.compute({"states": sampled_states}, role="policy")
                _, next_log_prob, _ = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions}, role="policy")
                # next_log_prob_alt = self.policy.distribution(role="policy").log_prob(sampled_actions).sum(dim=-1)
                # test_entropy = self.policy.get_entropy(role="policy").sum(dim=-1)
                # test_log_prob = 0.5*(1 + torch.log(2*torch.pi*self.policy.distribution(role="policy").stddev**2)).sum(dim=-1)


                self.curiosity_optimizer.step()

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                # Add entropy reward:
                # _,log_probs,_ = self.policy.act({"states": self._state_preprocessor(sampled_states)}, role="policy")
                # entropy_reward = self._entropy_reward_scale * -next_log_prob
                # entropy_reward_cumulative = entropy_reward.detach() + entropy_reward_cumulative
                # sampled_rewards_cumulative = sampled_rewards + sampled_rewards_cumulative

                surrogate = sampled_advantages * ratio
                # make_dot(entropy_reward,params=dict(self.policy.named_parameters())).render("entropy_reward", format="png")
                # make_dot(sampled_advantages,params=dict(self.policy.named_parameters())).render("advantages", format="png")
                # make_dot(ratio,params=dict(self.policy.named_parameters())).render("ratio", format="png")
                # make_dot(surrogate,params=dict(self.policy.named_parameters())).render("surrogate", format="png")
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip)
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Exploration loss", cumulative_exploration_loss / (self._curiosity_epochs * self._curio_mini_batches))
        self.track_data("Loss / Prior loss", cumulative_prior_loss / (self._curiosity_epochs * self._curio_mini_batches))
        self.track_data("Loss / Target loss", cumulative_target_loss / (self._curiosity_epochs * self._curio_mini_batches))
        self.track_data("Policy / Curiosity scale", self._curiosity_scale)
        self.track_data("Loss / MSE posterior prior ", self.curiosity_model.last_prior_mse)
        self.track_data("Loss / MSE reconstruction ", self.curiosity_model.last_post_mse)

        # self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        # sampled_rewards_mean = sampled_rewards_cumulative / (self._learning_epochs * self._mini_batches)
        # entropy_reward_mean = entropy_reward_cumulative / (self._learning_epochs * self._mini_batches)
        # self.track_data("Reward / Total reward (mean) base", sampled_rewards_mean.cpu().numpy())
        # self.track_data("Reward / Total reward (mean) with entropy", sampled_rewards_mean.cpu().numpy() + entropy_reward_mean.cpu().numpy())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
