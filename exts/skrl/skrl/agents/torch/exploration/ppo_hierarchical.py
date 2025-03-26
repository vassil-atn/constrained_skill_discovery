import collections
import copy
import itertools
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

# import torch_optimizer as optim

# [start-config-dict-torch]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,  # number of rollouts before updating
    "learning_epochs": 8,  # number of learning epochs during each update
    "mini_batches": 2,  # number of mini batches during each learning epoch
    "discount_factor": 0.99,  # discount factor (gamma)
    "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0.5,  # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,  # clip predicted values during value loss computation
    "entropy_loss_scale": 0.0,  # entropy loss scaling factor
    "value_loss_scale": 1.0,  # value loss scaling factor
    "kl_threshold": 0,  # KL divergence threshold for early stopping
    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "curiosity_observation_type": "full_state",
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": 250,  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}
# [end-config-dict-torch]
from math import inf


def compute_total_norm(parameters, norm_type=2):
    # Code adopted from clip_grad_norm_().
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class HierarchicalLSDPPO(Agent):
    """Hierarchical LSDPPO agent"""

    def __init__(
        self,
        models: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        action_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        low_level_observation_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        low_level_action_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.high_level_policy = self.models.get("high_level_policy", None)
        self.high_level_value = self.models.get("high_level_value", None)
        self.low_level_policy = self.models.get("low_level_policy", None)
        self.low_level_value = self.models.get("low_level_value", None)

        self.low_level_observation_space = low_level_observation_space
        self.low_level_action_space = low_level_action_space

        # checkpoint models
        self.checkpoint_modules = {}
        self.checkpoint_modules["high_level_policy"] = self.high_level_policy
        self.checkpoint_modules["high_level_value"] = self.high_level_value
        self.checkpoint_modules["low_level_policy"] = self.low_level_policy
        self.checkpoint_modules["low_level_value"] = self.low_level_value

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._sd_mini_batches = self.cfg.get("sd_mini_batches", self._mini_batches)
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

        self._low_level_state_preprocessor = self.cfg["low_level_state_preprocessor"]
        self._low_level_value_preprocessor = self.cfg["low_level_value_preprocessor"]

        self._high_level_state_preprocessor = self.cfg["high_level_state_preprocessor"]
        self._high_level_value_preprocessor = self.cfg["high_level_value_preprocessor"]

        self._rewards_preprocessor = self.cfg.get("rewards_preprocessor", None)

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._entropy_reward_scale = self.cfg.get("entropy_reward_scale", 0.0)
        self.use_nstep_return = self.cfg.get("use_nstep_return", False)

        self.num_envs = self.cfg["num_envs"]

        self.episode_length = self.cfg.get("episode_length", 300)

        self._track_extrinsic_rewards = collections.deque(maxlen=100)
        self._track_intrinsic_rewards = collections.deque(maxlen=100)
        self._new_track_timesteps = collections.deque(maxlen=100)
        self._cumulative_extrinsic_rewards = None
        self._cumulative_intrinsic_rewards = None
        self._new_cumulative_timesteps = None

        self.intr_ret_rms = RunningStandardScaler(1, device=self.device)
        self.intr_ret = torch.zeros((self.num_envs, 1), device=self.device)
        # self.curiosity_model = self.cfg.get("curiosity_model",None)
        # self._curiosity_lr = self.cfg.get("curiosity_lr",0.0)
        # self._curiosity_scale = self.cfg.get("curiosity_scale",0.0)
        # self.anneal_curiosity_warmup = self.cfg.get("anneal_curiosity_warmup",0.)
        # if self.curiosity_model is not None:
        # self.curiosity_optimizer = torch.optim.Adam(self.curiosity_model.parameters(), lr=self._curiosity_lr)
        # else:
        # self.curiosity_optimizer = None

        self.train_high_level = self.cfg.get("train_high_level", True)
        self.train_low_level = self.cfg.get("train_low_level", True)

        self.delay_actions = self.cfg.get("delay_actions", False)

        # self.use_curiosity = self.cfg.get("use_curiosity",False)
        self.skill_discovery = self.cfg.get("skill_discovery", False)
        # self.mask_xy_pos = self.cfg.get("mask_xy_pos",False)

        # self.curiosity_observation_type = self.cfg["curiosity_observation_type"]
        self.full_curiosity_obs_size = self.cfg["full_curiosity_obs_space"]
        self.curiosity_obs_size = self.cfg["curiosity_obs_space"]
        self._curiosity_preprocessor = self.cfg["curiosity_preprocessor"]
        # self._curio_mini_batches = self.cfg["curiosity_mini_batches"]
        # self._curiosity_epochs = self.cfg["curiosity_epochs"]

        self.discrete_skills = self.cfg["discrete_skills"]
        self.skill_space = self.cfg["skill_space"]
        self._skill_epochs = self.cfg["skill_discovery_epochs"]

        self.lsd_model = self.cfg.get("lsd_model", None)
        self.lsd_lr = self.cfg["lsd_lr"]

        if self.lsd_model is not None:
            self.lsd_optimizer = torch.optim.Adam(
                self.lsd_model.parameters(), lr=self.lsd_lr
            )
            # self.lsd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.lsd_optimizer, mode='min', factor=0.9, patience=3000, verbose=True)
            # self.lsd_optimizer = optim.Lamb(self.lsd_model.parameters(), lr=self.lsd_lr)
        else:
            self.lsd_optimizer = None

        self.skill_space = self.cfg["skill_space"]
        self.pretrain_normalizers = self.cfg["pretrain_normalizers"]
        self.intrinsic_reward_scale = self.cfg.get("intrinsic_reward_scale", 1.0)
        self.extrinsic_reward_scale = self.cfg.get("extrinsic_reward_scale", 0.0)

        self.skill_dual_reg = self.cfg["dual_optimisation"]
        self.log_dual_lambda = self.cfg["dual_lambda"]
        self.dual_slack = self.cfg["dual_slack"]
        self.lambda_lr = self.cfg["dual_lambda_lr"]
        self.dual_lambda_optimizer = torch.optim.Adam(
            [self.log_dual_lambda], lr=self.lambda_lr
        )
        self.learn_dual_lambda = self.cfg["learn_dual_lambda"]
        self.action_scale = 1.0
        self.running_max_intr_reward = self.cfg["loss_intr_rew_scale"]
        self.lsd_loss_scale = self.cfg["lsd_loss_scale"]
        self.lsd_norm_clip = self.cfg["lsd_norm_clip"]

        self.evaluate_envs = self.cfg.get("evaluate_envs", False)
        self.evaluate_envs_num = self.cfg.get("num_eval_envs", 200)
        self.evaluate_envs_ids = torch.zeros(self.num_envs, dtype=torch.bool).to(
            self.device
        )
        if self.evaluate_envs:
            self.evaluate_envs_ids[
                torch.arange(0, self.evaluate_envs_num, dtype=torch.int)
            ] = 1
        self.evaluate_envs_interval = self.cfg.get("evaluate_envs_interval", 3000)

        self.hierarchical = True

        self.skill_discovery_dist = self.cfg["skill_discovery_distance_metric"]
        self.dist_predictor = self.cfg.get("dist_predictor", None)
        self.dist_model_lr = self.cfg.get("dist_model_lr", 5e-3)
        self.sd_version = self.cfg.get("sd_version", "MSE")
        if self.dist_predictor is not None:
            self.dist_metric_optimizer = torch.optim.Adam(
                self.dist_predictor.parameters(), lr=self.dist_model_lr
            )
        else:
            self.dist_metric_optimizer = None
        self.sd_specific_state_ind = self.cfg["sd_specific_state_ind"]
        if self.sd_specific_state_ind == "None":
            self.sd_specific_state_ind = None

        self.on_policy_rewards = self.cfg["on_policy_rewards"]

        self._init_states = torch.zeros(
            (self.num_envs, self.full_curiosity_obs_size), device=self.device
        )

        self.high_level_action_frequency = self.cfg.get(
            "high_level_action_frequency", 0
        )
        # self.checkpoint_modules["curiosity"] = self.curiosity_model
        # self.checkpoint_modules["curiosity_optimizer"] = self.curiosity_optimizer
        self.checkpoint_modules["lsd_model"] = self.lsd_model
        self.checkpoint_modules["lsd_optimizer"] = self.lsd_optimizer

        if self.pretrain_normalizers:
            if self._low_level_state_preprocessor != self._empty_preprocessor:
                self.normalizer_states = torch.zeros(
                    (
                        self._learning_starts,
                        self.num_envs,
                        self.low_level_observation_space,
                    ),
                    device=self.device,
                )

            if self._curiosity_preprocessor != self._empty_preprocessor:
                self.normalizer_curiosity_states = torch.zeros(
                    (self._learning_starts, self.num_envs, self.curiosity_obs_size),
                    device=self.device,
                )

        # set up optimizer and learning rate scheduler
        if self.high_level_policy is not None and self.high_level_value is not None:
            if self.high_level_policy is self.high_level_value:
                self.high_level_optimizer = torch.optim.Adam(
                    self.high_level_policy.parameters(), lr=self._learning_rate
                )
            else:
                self.high_level_optimizer = torch.optim.Adam(
                    itertools.chain(
                        self.high_level_policy.parameters(),
                        self.high_level_value.parameters(),
                    ),
                    lr=self._learning_rate,
                )
            if self._learning_rate_scheduler is not None:
                self.high_level_scheduler = self._learning_rate_scheduler(
                    self.high_level_optimizer,
                    **self.cfg["learning_rate_scheduler_kwargs"],
                )

            self.checkpoint_modules["high_level_optimizer"] = self.high_level_optimizer

        if self.low_level_policy is not None and self.low_level_value is not None:
            if self.low_level_policy is self.low_level_value:
                self.low_level_optimizer = torch.optim.Adam(
                    self.low_level_policy.parameters(), lr=self._learning_rate
                )
            else:
                self.low_level_optimizer = torch.optim.Adam(
                    itertools.chain(
                        self.low_level_policy.parameters(),
                        self.low_level_value.parameters(),
                    ),
                    lr=self._learning_rate,
                )
            if self._learning_rate_scheduler is not None:
                self.low_level_scheduler = self._learning_rate_scheduler(
                    self.low_level_optimizer,
                    **self.cfg["learning_rate_scheduler_kwargs"],
                )

            self.checkpoint_modules["low_level_optimizer"] = self.low_level_optimizer

        # set up preprocessors
        if self._low_level_state_preprocessor:
            self._low_level_state_preprocessor = self._low_level_state_preprocessor(
                **self.cfg["low_level_state_preprocessor_kwargs"]
            )
            self.checkpoint_modules["low_level_state_preprocessor"] = (
                self._low_level_state_preprocessor
            )
        else:
            self._low_level_state_preprocessor = self._empty_preprocessor

        if self._high_level_state_preprocessor:
            self._high_level_state_preprocessor = self._high_level_state_preprocessor(
                **self.cfg["high_level_state_preprocessor_kwargs"]
            )
            self.checkpoint_modules["high_level_state_preprocessor"] = (
                self._high_level_state_preprocessor
            )
        else:
            self._high_level_state_preprocessor = self._empty_preprocessor

        if self._low_level_value_preprocessor:
            self._low_level_value_preprocessor = self._low_level_value_preprocessor(
                **self.cfg["low_level_value_preprocessor_kwargs"]
            )
            self.checkpoint_modules["low_level_value_preprocessor"] = (
                self._low_level_value_preprocessor
            )
        else:
            self._low_level_value_preprocessor = self._empty_preprocessor

        if self._high_level_value_preprocessor:
            self._high_level_value_preprocessor = self._high_level_value_preprocessor(
                **self.cfg["high_level_value_preprocessor_kwargs"]
            )
            self.checkpoint_modules["high_level_value_preprocessor"] = (
                self._high_level_value_preprocessor
            )
        else:
            self._high_level_value_preprocessor = self._empty_preprocessor

        if self._curiosity_preprocessor:
            self._curiosity_preprocessor = self._curiosity_preprocessor(
                **self.cfg["curiosity_preprocessor_kwargs"]
            )
            self.checkpoint_modules["curiosity_preprocessor"] = (
                self._curiosity_preprocessor
            )
        else:
            self._curiosity_preprocessor = self._empty_preprocessor

        if (
            self.cfg["load_curiosity_preprocesor_params"]
            and self._curiosity_preprocessor != self._empty_preprocessor
        ):
            data = np.load("curiosity_preprocessor_means_stds.npz")
            self._curiosity_preprocessor.mean = torch.tensor(data["sd_means"]).to(
                self.device
            )
            self._curiosity_preprocessor.variance = torch.tensor(data["sd_stds"]).to(
                self.device
            )

        if self._rewards_preprocessor:
            self._rewards_preprocessor = self._rewards_preprocessor(
                **self.cfg["rewards_preprocessor_kwargs"]
            )
            self.checkpoint_modules["rewards_preprocessor"] = self._rewards_preprocessor
        else:
            self._rewards_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="low_level_states",
                size=self.low_level_observation_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="low_level_actions",
                size=self.low_level_action_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="low_level_actions_pretanh",
                size=self.low_level_action_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="low_level_rewards", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_rewards_extrinsic", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_rewards_intrinsic", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="next_low_level_states",
                size=self.low_level_observation_space,
                dtype=torch.float32,
            )

            self.memory.create_tensor(
                name="high_level_states",
                size=self.observation_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="high_level_actions", size=self.action_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="high_level_actions_pretanh",
                size=self.action_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="high_level_rewards", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="next_high_level_states",
                size=self.observation_space,
                dtype=torch.float32,
            )

            self.memory.create_tensor(
                name="curiosity_states",
                size=self.full_curiosity_obs_size,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="next_curiosity_states",
                size=self.full_curiosity_obs_size,
                dtype=torch.float32,
            )
            self.memory.create_tensor(
                name="skills", size=self.skill_space, dtype=torch.float32
            )
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self.memory.create_tensor(
                name="high_level_log_prob", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="high_level_values", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="high_level_returns", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="high_level_advantages", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_log_prob", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_values", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_returns", size=1, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="low_level_advantages", size=1, dtype=torch.float32
            )

            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            # tensors sampled during training
            self._tensors_names = [
                "low_level_states",
                "low_level_actions",
                "low_level_actions_pretanh",
                "low_level_rewards",
                "low_level_rewards_extrinsic",
                "low_level_rewards_intrinsic",
                "next_low_level_states",
                "high_level_states",
                "high_level_actions",
                "high_level_actions_pretanh",
                "high_level_rewards",
                "next_high_level_states",
                "curiosity_states",
                "next_curiosity_states",
                "terminated",
                "high_level_log_prob",
                "high_level_values",
                "high_level_returns",
                "high_level_advantages",
                "low_level_log_prob",
                "low_level_values",
                "low_level_returns",
                "low_level_advantages",
                "truncated",
            ]

        # create temporary variables needed for storage and computation
        self._current_low_level_log_prob = None
        self._current_next_low_level_states = None
        self._current_high_level_log_prob = None
        self._current_next_high_level_states = None
        self._current_next_skills = None
        self._current_next_curiosity_states = None

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

    def record_transition(
        self,
        low_level_states: torch.Tensor,
        low_level_actions: torch.Tensor,
        low_level_actions_pretanh: torch.Tensor,
        low_level_rewards: torch.Tensor,
        low_level_rewards_extrinsic: torch.Tensor,
        low_level_rewards_intrinsic: torch.Tensor,
        next_low_level_states: torch.Tensor,
        high_level_states: torch.Tensor,
        high_level_actions: torch.Tensor,
        high_level_actions_pretanh: torch.Tensor,
        next_high_level_states: torch.Tensor,
        high_level_rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
        curiosity_states: torch.Tensor = None,
        next_curiosity_states: torch.Tensor = None,
        skills: torch.Tensor = None,
        next_skills: torch.Tensor = None,
    ) -> None:
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
        if self.write_interval > 0:
            # compute the cumulative sum of the rewards and timesteps

            if self._cumulative_extrinsic_rewards is None:
                self._cumulative_extrinsic_rewards = torch.zeros_like(
                    low_level_rewards_extrinsic, dtype=torch.float32
                )
            if self._cumulative_intrinsic_rewards is None:
                self._cumulative_intrinsic_rewards = torch.zeros_like(
                    low_level_rewards_extrinsic, dtype=torch.float32
                )
                self._new_cumulative_timesteps = torch.zeros_like(
                    low_level_rewards_extrinsic, dtype=torch.int32
                )

            self._cumulative_extrinsic_rewards.add_(low_level_rewards_extrinsic)
            self._cumulative_intrinsic_rewards.add_(low_level_rewards_intrinsic)
            self._new_cumulative_timesteps.add_(1)

            finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                # storage cumulative rewards and timesteps
                self._track_extrinsic_rewards.extend(
                    self._cumulative_extrinsic_rewards[finished_episodes][:, 0]
                    .reshape(-1)
                    .tolist()
                )
                self._track_intrinsic_rewards.extend(
                    self._cumulative_intrinsic_rewards[finished_episodes][:, 0]
                    .reshape(-1)
                    .tolist()
                )

                self._new_track_timesteps.extend(
                    self._new_cumulative_timesteps[finished_episodes][:, 0]
                    .reshape(-1)
                    .tolist()
                )

                # reset the cumulative rewards and timesteps
                self._cumulative_extrinsic_rewards[finished_episodes] = 0
                self._cumulative_intrinsic_rewards[finished_episodes] = 0
                self._new_cumulative_timesteps[finished_episodes] = 0

            # self.tracking_data["Reward / Extrinsic instantenous reward (max)"].append(torch.max(rewards_extrinsic).item())
            # self.tracking_data["Reward / Extrinsic instantenous reward (min)"].append(torch.min(rewards_extrinsic).item())
            # self.tracking_data["Reward / Extrinsic instantenous reward (mean)"].append(torch.mean(rewards_extrinsic).item())

            # self.tracking_data["Reward / Intrinsic instantenous reward (max)"].append(torch.max(rewards_intrinsic).item())
            # self.tracking_data["Reward / Intrinsic instantenous reward (min)"].append(torch.min(rewards_intrinsic).item())
            # self.tracking_data["Reward / Intrinsic instantenous reward (mean)"].append(torch.mean(rewards_intrinsic).item())

            # self.tracking_data["Normaliser / Running mean (mean over states)"].append(torch.mean(self._state_preprocessor.running_mean).item())
            # self.tracking_data["Normaliser / Running variance (mean over states)"].append(torch.mean(self._state_preprocessor.running_variance).item())

            if len(self._track_rewards):
                track_extrinsic_rewards = np.array(self._track_extrinsic_rewards)
                track_intrinsic_rewards = np.array(self._track_intrinsic_rewards)
                track_timesteps = np.array(self._new_track_timesteps)

                # self.tracking_data["Reward / Total extrinsic reward (max)"].append(np.max(track_extrinsic_rewards))
                # self.tracking_data["Reward / Total extrinsic reward (min)"].append(np.min(track_extrinsic_rewards))
                self.tracking_data["Reward / Total extrinsic reward (mean)"].append(
                    np.mean(track_extrinsic_rewards)
                )

                # self.tracking_data["Reward / Total intrinsic reward (max)"].append(np.max(track_intrinsic_rewards))
                # self.tracking_data["Reward / Total intrinsic reward (min)"].append(np.min(track_intrinsic_rewards))
                self.tracking_data["Reward / Total intrinsic reward (mean)"].append(
                    np.mean(track_intrinsic_rewards)
                )

                self.tracking_data["Episode / Total timesteps (max)"].append(
                    np.max(track_timesteps)
                )
                self.tracking_data["Episode / Total timesteps (min)"].append(
                    np.min(track_timesteps)
                )
                self.tracking_data["Episode / Total timesteps (mean)"].append(
                    np.mean(track_timesteps)
                )

        if self.memory is not None:
            # TODO: what to do here?
            self._current_next_low_level_states = next_low_level_states
            self._current_next_high_level_states = next_high_level_states
            self._current_next_curiosity_states = next_curiosity_states
            self._current_next_skills = next_skills

            # reward shaping
            if self._rewards_shaper is not None:
                high_level_rewards = self._rewards_shaper(
                    high_level_rewards, timestep, timesteps
                )
                low_level_rewards = self._rewards_shaper(
                    low_level_rewards, timestep, timesteps
                )

            high_level_rewards = self._rewards_preprocessor(
                high_level_rewards, train=False
            )
            low_level_rewards = self._rewards_preprocessor(
                low_level_rewards, train=False
            )
            #
            # TODO: fix processing:
            # states_processed = self._state_preprocessor(states, train=False)
            # compute values
            high_level_values, _, _ = self.high_level_value.act(
                {
                    "states": self._high_level_state_preprocessor(
                        high_level_states, train=False
                    )
                },
                role="value",
            )
            high_level_values = self._high_level_value_preprocessor(
                high_level_values, inverse=True
            )

            low_level_values, _, _ = self.low_level_value.act(
                {
                    "states": self._low_level_state_preprocessor(
                        low_level_states, train=False
                    ),
                    "skills": skills,
                },
                role="value",
            )
            low_level_values = self._low_level_value_preprocessor(
                low_level_values, inverse=True
            )

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                low_level_rewards += (
                    self._discount_factor * low_level_values * truncated
                )
                high_level_rewards += (
                    self._discount_factor * high_level_values * truncated
                )

            # storage transition in memory

            self.memory.add_samples(
                low_level_states=low_level_states,
                low_level_actions=low_level_actions,
                low_level_actions_pretanh=low_level_actions_pretanh,
                low_level_rewards=low_level_rewards,
                low_level_rewards_extrinsic=low_level_rewards_extrinsic,
                low_level_rewards_intrinsic=low_level_rewards_intrinsic,
                next_low_level_states=next_low_level_states,
                high_level_states=high_level_states,
                high_level_actions=high_level_actions,
                high_level_actions_pretanh=high_level_actions_pretanh,
                high_level_rewards=high_level_rewards,
                next_high_level_states=next_high_level_states,
                curiosity_states=curiosity_states,
                next_curiosity_states=next_curiosity_states,
                skills=skills,
                terminated=terminated,
                truncated=truncated,
                high_level_log_prob=self._current_high_level_log_prob[
                    ~self.evaluate_envs_ids
                ],
                high_level_values=high_level_values,
                low_level_log_prob=self._current_low_level_log_prob[
                    ~self.evaluate_envs_ids
                ],
                low_level_values=low_level_values,
            )

            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def low_level_act(
        self, states: torch.Tensor, skills: torch.Tensor, timestep: int, timesteps: int
    ) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the low-level policy

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
            (
                actions,
                _,
                _,
            ) = self.low_level_policy.random_act(
                {
                    "states": self._low_level_state_preprocessor(states, train=False),
                    "skills": skills,
                },
                role="policy",
            )
            log_prob = torch.zeros((self.num_envs, 1), device=self.device)
            outputs = 0 * actions
            actions_pretanh = actions.clone()

        else:
            # sample stochastic actions

            actions, log_prob, outputs, actions_pretanh = self.low_level_policy.act(
                {
                    "states": self._low_level_state_preprocessor(states, train=False),
                    "skills": skills,
                },
                role="policy",
            )

        self._current_low_level_log_prob = log_prob

        return actions, log_prob, outputs, actions_pretanh

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
            (
                actions,
                _,
                _,
            ) = self.high_level_policy.random_act(
                {"states": self._high_level_state_preprocessor(states, train=False)},
                role="policy",
            )
            log_prob = torch.zeros((self.num_envs, 1), device=self.device)
            outputs = 0 * actions
            actions_pretanh = actions.clone()

        else:
            # sample stochastic actions

            actions, log_prob, outputs, actions_pretanh = self.high_level_policy.act(
                {"states": self._high_level_state_preprocessor(states, train=False)},
                role="policy",
            )

        self._current_high_level_log_prob = log_prob

        return actions, log_prob, outputs, actions_pretanh

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        if (not self.train_high_level) or (
            self.train_high_level and (timestep % self.high_level_action_frequency == 0)
        ):
            self._rollout += 1

            if self.pretrain_normalizers and timestep >= self._learning_starts:
                if self._low_level_state_preprocessor != self._empty_preprocessor:
                    self._low_level_state_preprocessor.mean = torch.mean(
                        self.normalizer_states, dim=0
                    )
                    self._low_level_state_preprocessor.variance = torch.var(
                        self.normalizer_states, dim=0
                    )

                if self._curiosity_preprocessor != self._empty_preprocessor:
                    self._curiosity_preprocessor.mean = torch.mean(
                        self.normalizer_curiosity_states, dim=0
                    )
                    self._curiosity_preprocessor.variance = torch.var(
                        self.normalizer_curiosity_states, dim=0
                    )

            if not self._rollout % self._rollouts and timestep >= self._learning_starts:
                self.set_mode("train")
                self._update(timestep, timesteps)
                self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _compute_intrinsic_reward(
        self,
        curiosity_states: torch.Tensor = None,
        next_curiosity_states: torch.Tensor = None,
        skills: torch.Tensor = None,
        as_reward=False,
        train_preprocessors=False,
    ) -> torch.Tensor:
        if self.discrete_skills:
            diverse_states = self._curiosity_preprocessor(
                self._curiosity_state_selector(curiosity_states),
                train=train_preprocessors,
            )
            diverse_next_states = self._curiosity_preprocessor(
                self._curiosity_state_selector(next_curiosity_states),
                train=train_preprocessors,
            )

            # Get the skills as 0-centered onehot vectors
            masks = (
                (skills - skills.mean(dim=1, keepdim=True))
                * (self.skill_space)
                / (self.skill_space - 1 if self.skill_space != 1 else 1)
            )
            diversity_reward = torch.sum(
                (
                    self.lsd_model.get_distribution(diverse_next_states).mean
                    - self.lsd_model.get_distribution(diverse_states).mean
                )
                * masks,
                dim=1,
            )
        else:
            # For the states we want to diversify:
            diverse_states = self._curiosity_preprocessor(
                self._curiosity_state_selector(curiosity_states),
                train=train_preprocessors,
            )
            diverse_next_states = self._curiosity_preprocessor(
                self._curiosity_state_selector(next_curiosity_states),
                train=train_preprocessors,
            )

            next_latent = self.lsd_model.get_distribution(diverse_next_states).mean
            latent = self.lsd_model.get_distribution(diverse_states).mean

            if self.skill_discovery_dist == "l2" and self.sd_version == "MSE":

                if self._curiosity_preprocessor != self._empty_preprocessor:
                    diversity_reward = -torch.square(
                        (next_latent - latent) - skills
                    ).mean(dim=-1)
                else:
                    diversity_reward = -torch.nn.functional.smooth_l1_loss(
                        self.episode_length * (next_latent - latent),
                        skills,
                        reduction="none",
                        beta=5.0,
                    ).mean(dim=-1)

            elif self.skill_discovery_dist == "l2" and self.sd_version == "base":
                unit_skills = skills / torch.norm(skills, dim=1, keepdim=True)
                diversity_reward = torch.sum(
                    (next_latent - latent)
                    * unit_skills
                    / torch.norm(skills, dim=-1, keepdim=True),
                    dim=1,
                )

            elif self.skill_discovery_dist == "one" and self.sd_version == "MSE":
                diversity_reward = -torch.square((next_latent - latent) - skills).mean(
                    dim=-1
                )

            elif self.skill_discovery_dist == "one" and self.sd_version == "base":
                unit_skills = skills / torch.norm(skills, dim=1, keepdim=True)
                diversity_reward = torch.sum(
                    (next_latent - latent) * unit_skills, dim=1
                )

        return diversity_reward

    def _get_cst_distance(
        self, sampled_states_sd, sampled_next_states_sd, sampled_skills
    ):
        if self.skill_discovery_dist == "l2":
            cst_dist = torch.square(sampled_next_states_sd - sampled_states_sd).mean(
                dim=-1
            )

        elif self.skill_discovery_dist == "one":
            cst_dist = torch.ones_like(sampled_next_states_sd[:, 0])

        return cst_dist

    def _update_skill_discovery(self, batch, epoch):
        train_preprocessors = not epoch
        # sample the batch
        sampled_curiosity_states = batch["curiosity_states"]
        sampled_next_curiosity_states = batch["next_curiosity_states"]
        sampled_skills = batch["skills"]
        sampled_truncated = batch["truncated"]
        sampled_terminated = batch["terminated"]

        cst_penalty = torch.zeros_like(sampled_skills[:, 0], device=self.device)
        loss_dual_lambda = torch.zeros_like(sampled_skills[:, 0], device=self.device)
        if self.skill_discovery:
            # Lipschitz-constrained skill discovery:
            # Compute the lsd loss:

            lsd_loss = self.lsd_loss_scale * -self._compute_intrinsic_reward(
                sampled_curiosity_states,
                sampled_next_curiosity_states,
                sampled_skills,
                train_preprocessors=train_preprocessors,
            )

            # Discard the lsd_loss on reset (meaningless as there is no actual transition):
            reset_idx = (sampled_truncated + sampled_terminated).squeeze()
            lsd_loss[reset_idx] = 0.0

            self.lsd_optimizer.zero_grad()

            if self.skill_dual_reg:
                sampled_states_sd = self._curiosity_preprocessor(
                    self._curiosity_state_selector(sampled_curiosity_states)
                )
                sampled_next_states_sd = self._curiosity_preprocessor(
                    self._curiosity_state_selector(sampled_next_curiosity_states)
                )

                latent_states_mean = self.lsd_model.get_distribution(
                    sampled_states_sd
                ).mean
                latent_next_states_mean = self.lsd_model.get_distribution(
                    sampled_next_states_sd
                ).mean

                cst_dist = self._get_cst_distance(
                    sampled_states_sd, sampled_next_states_sd, sampled_skills
                )

                cst_penalty = cst_dist - torch.square(
                    latent_next_states_mean - latent_states_mean
                ).mean(dim=1)

                cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

                # Discard cst_penalty on reset transitions:
                cst_penalty[reset_idx] = 0.0

                lsd_loss_with_penalty = lsd_loss - (
                    self.log_dual_lambda.detach().exp() * cst_penalty
                )

            if self.skill_dual_reg:
                lsd_loss_with_penalty = lsd_loss_with_penalty[~reset_idx].mean()
                lsd_loss_with_penalty.backward()
                lsd_loss = lsd_loss[~reset_idx].mean()
            else:
                lsd_loss_with_penalty = 0 * lsd_loss.mean()
                lsd_loss = lsd_loss[~reset_idx].mean()
                lsd_loss.backward()

            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.lsd_model.parameters(), self.lsd_norm_clip
                )

            self.lsd_optimizer.step()

            if self.skill_dual_reg and self.learn_dual_lambda:
                # Recompute the latent encodings under the updated encoder:
                latent_states_mean = self.lsd_model.get_distribution(
                    sampled_states_sd
                ).mean
                latent_next_states_mean = self.lsd_model.get_distribution(
                    sampled_next_states_sd
                ).mean

                cst_dist = self._get_cst_distance(
                    sampled_states_sd, sampled_next_states_sd, sampled_skills
                )

                cst_penalty = cst_dist - torch.square(
                    latent_next_states_mean - latent_states_mean
                ).mean(dim=1)
                cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

                # Discard cst_penalty on reset transitions:
                reset_idx = (sampled_truncated + sampled_terminated).squeeze()
                cst_penalty[reset_idx] = 0.0

                # update the lambda
                self.dual_lambda_optimizer.zero_grad()

                loss_dual_lambda = (
                    self.log_dual_lambda * (cst_penalty).detach()
                ).mean()

                loss_dual_lambda.backward()

                self.dual_lambda_optimizer.step()

            return (
                lsd_loss,
                lsd_loss_with_penalty,
                cst_penalty,
                loss_dual_lambda,
            )

    def _update_hierarchical(self, level, kl_divergences, batch, epoch):
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0

        if level == "high":
            sampled_states = batch["high_level_states"]
            sampled_next_states = batch["next_high_level_states"]
            sampled_actions = batch["high_level_actions"]
            sampled_actions_pretanh = batch["high_level_actions_pretanh"]
            sampled_log_prob = batch["high_level_log_prob"]
            sampled_values = batch["high_level_values"]
            sampled_advantages = batch["high_level_advantages"]
            sampled_returns = batch["high_level_returns"]

            policy = self.high_level_policy
            value = self.high_level_value
            optimizer = self.high_level_optimizer
            scheduler = self.high_level_scheduler

            # TODO: make sure the right state processor for the right level is used
            train_preprocessors = not epoch
            # Skills shouldn't be normalized:
            sampled_states = self._high_level_state_preprocessor(
                sampled_states, train=train_preprocessors
            )
            sampled_next_states = self._high_level_state_preprocessor(
                sampled_next_states, train=train_preprocessors
            )

            _, next_log_prob, _, _ = self.high_level_policy.act(
                {
                    "states": sampled_states,
                    "taken_actions": sampled_actions,
                    "taken_actions_pretanh": sampled_actions_pretanh,
                },
                role="policy",
            )

        elif level == "low":
            sampled_states = batch["low_level_states"]
            sampled_next_states = batch["next_low_level_states"]
            sampled_skills = batch["skills"]
            sampled_actions = batch["low_level_actions"]
            sampled_actions_pretanh = batch["low_level_actions_pretanh"]
            sampled_log_prob = batch["low_level_log_prob"]
            sampled_values = batch["low_level_values"]
            sampled_advantages = batch["low_level_advantages"]
            sampled_returns = batch["low_level_returns"]

            policy = self.low_level_policy
            value = self.low_level_value
            optimizer = self.low_level_optimizer
            scheduler = self.low_level_scheduler

            train_preprocessors = not epoch
            sampled_states = self._low_level_state_preprocessor(
                sampled_states, train=train_preprocessors
            )
            sampled_next_states = self._low_level_state_preprocessor(
                sampled_next_states, train=train_preprocessors
            )

            _, next_log_prob, _, _ = self.low_level_policy.act(
                {
                    "states": sampled_states,
                    "skills": sampled_skills,
                    "taken_actions": sampled_actions,
                    "taken_actions_pretanh": sampled_actions_pretanh,
                },
                role="policy",
            )

        # compute approximate KL divergence
        with torch.no_grad():
            ratio = next_log_prob - sampled_log_prob
            kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
            kl_divergences.append(kl_divergence)

        # early stopping with KL divergence
        if self._kl_threshold and kl_divergence > self._kl_threshold:
            return kl_divergences, policy_loss, value_loss, entropy_loss

        # compute entropy loss
        if self._entropy_loss_scale:
            entropy_loss = (
                -self._entropy_loss_scale * policy.get_entropy(role="policy").mean()
            )
        else:
            entropy_loss = 0
        # compute policy loss
        ratio = torch.exp(next_log_prob - sampled_log_prob)

        surrogate = sampled_advantages * ratio

        surrogate_clipped = sampled_advantages * torch.clip(
            ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
        )

        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # compute value loss
        if level == "low":
            predicted_values, _, _ = value.act(
                {"states": sampled_states, "skills": sampled_skills}, role="value"
            )
        else:
            predicted_values, _, _ = value.act({"states": sampled_states}, role="value")

        if self._clip_predicted_values:
            predicted_values = sampled_values + torch.clip(
                predicted_values - sampled_values,
                min=-self._value_clip,
                max=self._value_clip,
            )
        value_loss = self._value_loss_scale * F.mse_loss(
            sampled_returns, predicted_values
        )

        # optimization step
        optimizer.zero_grad()
        (policy_loss + entropy_loss + value_loss).backward()

        if self._grad_norm_clip > 0:
            if policy is value:
                nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip)
            else:
                nn.utils.clip_grad_norm_(
                    itertools.chain(policy.parameters(), value.parameters()),
                    self._grad_norm_clip,
                )

        optimizer.step()

        # update learning rate
        if self._learning_rate_scheduler:
            if isinstance(scheduler, KLAdaptiveLR):
                scheduler.step(torch.tensor(kl_divergences).mean())
            else:
                scheduler.step()

        return kl_divergences, policy_loss, value_loss, entropy_loss

    def _curiosity_state_selector(self, curiosity_states):
        if self.sd_specific_state_ind is not None:
            # mask = torch.ones_like(curiosity_states, dtype=torch.bool, device=self.device)
            # mask[:,self.sd_specific_state_ind] = False

            # curiosity_states[mask] = 0.
            # curiosity_states[mask] = 0.
            curiosity_states_selected = curiosity_states[:, self.sd_specific_state_ind]

            return curiosity_states_selected

        return curiosity_states

    def encode_skills(self, skills: object) -> object:
        return (
            torch.nn.functional.one_hot(skills, num_classes=self.skill_space)
        ).squeeze(1)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            last_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
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
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor
                    * not_dones[i]
                    * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.high_level_value.train(False)

            current_next_states_processed = self._high_level_state_preprocessor(
                self._current_next_high_level_states
            )

            last_high_level_values, _, _ = self.high_level_value.act(
                {"states": current_next_states_processed.float()}, role="value"
            )
            self.high_level_value.train(True)

            self.low_level_value.train(False)

            current_next_states_processed = self._low_level_state_preprocessor(
                self._current_next_low_level_states
            )

            last_low_level_values, _, _ = self.low_level_value.act(
                {
                    "states": current_next_states_processed.float(),
                    "skills": self._current_next_skills,
                },
                role="value",
            )
            self.low_level_value.train(True)

        last_high_level_values = self._high_level_value_preprocessor(
            last_high_level_values, inverse=True
        )

        last_low_level_values = self._low_level_value_preprocessor(
            last_low_level_values, inverse=True
        )

        high_level_values = self.memory.get_tensor_by_name("high_level_values")

        high_level_returns, high_level_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("high_level_rewards"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=high_level_values,
            next_values=last_high_level_values,
            last_values=last_high_level_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        low_level_values = self.memory.get_tensor_by_name("low_level_values")

        low_level_returns, low_level_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("low_level_rewards"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=low_level_values,
            next_values=last_low_level_values,
            last_values=last_low_level_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name(
            "high_level_values",
            self._high_level_value_preprocessor(high_level_values, train=True),
        )
        self.memory.set_tensor_by_name(
            "high_level_returns",
            self._high_level_value_preprocessor(high_level_returns, train=True),
        )
        self.memory.set_tensor_by_name("high_level_advantages", high_level_advantages)

        self.memory.set_tensor_by_name(
            "low_level_values",
            self._low_level_value_preprocessor(low_level_values, train=True),
        )
        self.memory.set_tensor_by_name(
            "low_level_returns",
            self._low_level_value_preprocessor(low_level_returns, train=True),
        )
        self.memory.set_tensor_by_name("low_level_advantages", low_level_advantages)

        cumulative_hl_policy_loss = 0
        cumulative_hl_entropy_loss = 0
        cumulative_hl_value_loss = 0
        cumulative_ll_policy_loss = 0
        cumulative_ll_entropy_loss = 0
        cumulative_ll_value_loss = 0
        loss_dual_lambda_cumul = 0
        cst_penalty_cumul = 0
        lsd_loss_cumul = 0
        lsd_loss_penalised_cumul = 0
        lsd_norm = 0
        # learning epochs
        for epoch in range(self._learning_epochs):
            hl_kl_divergences = []
            ll_kl_divergences = []

            # mini-batches loop
            for batch in self.memory.sample_all(
                names=self._tensors_names, mini_batches=self._mini_batches
            ):
                if self.skill_discovery and self.train_low_level:
                    (
                        lsd_loss,
                        lsd_loss_with_penalty,
                        cst_penalty,
                        loss_dual_lambda,
                    ) = self._update_skill_discovery(batch, epoch)

                    cst_penalty_cumul += cst_penalty.mean().item()
                    lsd_loss_cumul += lsd_loss.item() / self.lsd_loss_scale
                    lsd_loss_penalised_cumul += (
                        lsd_loss_with_penalty.item() / self.lsd_loss_scale
                    )
                    loss_dual_lambda_cumul += loss_dual_lambda.mean().item()
                    with torch.no_grad():
                        lsd_norm += compute_total_norm(
                            self.lsd_model.parameters()
                        ).item()

                if self.train_high_level:
                    (
                        hl_kl_divergences,
                        hl_policy_loss,
                        hl_value_loss,
                        hl_entropy_loss,
                    ) = self._update_hierarchical(
                        level="high",
                        kl_divergences=hl_kl_divergences,
                        batch=batch,
                        epoch=epoch,
                    )
                    # update cumulative losses
                    cumulative_hl_policy_loss += hl_policy_loss.item()
                    cumulative_hl_value_loss += hl_value_loss.item()
                    if self._entropy_loss_scale:
                        cumulative_hl_entropy_loss += hl_entropy_loss.item()

                if self.train_low_level:
                    (
                        ll_kl_divergences,
                        ll_policy_loss,
                        ll_value_loss,
                        ll_entropy_loss,
                    ) = self._update_hierarchical(
                        level="low",
                        kl_divergences=ll_kl_divergences,
                        batch=batch,
                        epoch=epoch,
                    )

                    # update cumulative losses
                    cumulative_ll_policy_loss += ll_policy_loss.item()
                    cumulative_ll_value_loss += ll_value_loss.item()
                    if self._entropy_loss_scale:
                        cumulative_ll_entropy_loss += ll_entropy_loss.item()

        # record data
        self.track_data(
            "Loss / Low Level Policy loss",
            cumulative_ll_policy_loss / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / Low Level Value loss",
            cumulative_ll_value_loss / (self._learning_epochs * self._mini_batches),
        )
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Low Level Entropy loss",
                cumulative_ll_entropy_loss
                / (self._learning_epochs * self._mini_batches),
            )

        self.track_data(
            "Loss / High Level Policy loss",
            cumulative_hl_policy_loss / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / High Level Value loss",
            cumulative_hl_value_loss / (self._learning_epochs * self._mini_batches),
        )
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / High Level Entropy loss",
                cumulative_hl_entropy_loss
                / (self._learning_epochs * self._mini_batches),
            )

        self.track_data(
            "Loss / LSD loss",
            lsd_loss_cumul / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / LSD loss penalised",
            lsd_loss_penalised_cumul / (self._learning_epochs * self._mini_batches),
        )
        # self.track_data("Loss / Distance metric loss", loss_dist_metric_cumul/(self._learning_epochs * self._mini_batches))
        self.track_data(
            "Loss / Dual lambda loss",
            loss_dual_lambda_cumul / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / LSD Norm", lsd_norm / (self._learning_epochs * self._mini_batches)
        )

        self.track_data("Coefficient / Log Dual lambda", self.log_dual_lambda.item())
        self.track_data(
            "Coefficient / Penalty violation",
            cst_penalty_cumul / (self._learning_epochs * self._mini_batches),
        )

        # self.track_data("Policy / Standard deviation", self.high_level_policy.distribution(role="policy").stddev.mean().item())
        # sampled_rewards_mean = sampled_rewards_cumulative / (self._learning_epochs * self._mini_batches)
        # entropy_reward_mean = entropy_reward_cumulative / (self._learning_epochs * self._mini_batches)
        # self.track_data("Reward / Total reward (mean) base", sampled_rewards_mean.cpu().numpy())
        # self.track_data("Reward / Total reward (mean) with entropy", sampled_rewards_mean.cpu().numpy() + entropy_reward_mean.cpu().numpy())
        if self._learning_rate_scheduler:
            self.track_data(
                "Learning / Learning rate", self.low_level_scheduler.get_last_lr()[0]
            )
