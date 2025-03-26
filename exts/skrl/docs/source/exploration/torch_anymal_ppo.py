import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.exploration.ppo import LBSPPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.lbs_exploration.models import *

def main(mode="train"):
    # seed for reproducibility
    set_seed(42)  # e.g. `set_seed(42)` for fixed seed


    # define shared model (stochastic and deterministic models) using mixins
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction,squashedNormal=False)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ELU(),
                                    nn.Linear(512, 256),
                                    nn.ELU(),
                                    nn.Linear(256, 128),
                                    nn.ELU(),
                                    nn.Linear(128, self.num_actions))
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), self.log_std_parameter, {}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ELU(),
                                    nn.Linear(512, 256),
                                    nn.ELU(),
                                    nn.Linear(256, 128),
                                    nn.ELU(),
                                    nn.Linear(128, 1))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    if mode == "train":
        headless = False
        num_envs = 4096
    else:
        headless = False
        num_envs = 100

    # load and wrap the Isaac Gym environment
    env = load_isaacgym_env_preview4(task_name="Anymal", num_envs=num_envs,headless=headless)
    env = wrap_env(env)

    device = env.device


    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 24  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 3  # 24 * 4096 / 32768
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 2
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "runs/torch/Anymal"

    cfg["entropy_reward_scale"] = 0.0

    cfg["use_nstep_return"] = False


    cfg["num_envs"] = env.num_envs
    cfg["curiosity_lr"] = 5e-4
    cfg["curiosity_scale"] = 1e-1
    cfg["curiosity_observation_type"] = "velocities"


    if cfg["curiosity_observation_type"] == "full_state":
        curiosity_obs_space = env.observation_space.shape[0]
    elif cfg["curiosity_observation_type"] == "velocities":
        curiosity_obs_space = 6


    state_dim = curiosity_obs_space
    hidden_dim = 32
    cfg["curiosity_model"]= LBS(curiosity_obs_space,env.action_space.shape[0],state_dim,hidden_dim,device=device)


    if mode == "train":
        cfg["experiment"]["wandb"] = True           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                                "mode":"online"}   
        agent = LBSPPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 24000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # start training
        trainer.train()

    else:
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                                "mode":"offline"}   
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0

        agent = LBSPPO(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 1000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        # path = "runs/torch/Anymal/24-01-24_18-07-56-951000_LBSPPO/checkpoints/agent_24000.pt" # default ppo
        # path = "runs/torch/Anymal/24-01-24_18-22-14-559048_LBSPPO/checkpoints/agent_24000.pt" # intrinsic 0.01
        # path = "runs/torch/Anymal/24-01-24_18-40-40-933913_LBSPPO/checkpoints/agent_24000.pt" # intrinsic 0.1
        # path = "runs/torch/Anymal/24-01-24_18-50-55-539746_LBSPPO/checkpoints/agent_24000.pt" # intrinsic 1.0
        path = "runs/torch/Anymal/24-01-25_10-33-04-728161_LBSPPO/checkpoints/agent_24000.pt" # intrinsic 0.1, partial state
        # path = "runs/torch/Anymal/24-01-25_10-50-03-099341_LBSPPO/checkpoints/agent_24000.pt" # intrinsic 0.01, partial state
        agent.load(path)
        # # start evaluation
        trainer.eval()


if __name__ == "__main__":
    main("train")
    # main("eval")