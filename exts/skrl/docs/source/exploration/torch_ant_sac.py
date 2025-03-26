import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.exploration.sac_lds import LSDSAC, SAC_DEFAULT_CONFIG
from skrl.lbs_exploration.models import *
from skrl.agents.torch.exploration.SAC_wrapper import ExplorationSAC
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import wandb
import numpy as np
from skrl.agents.torch.exploration.skill_discovery import TrajectoryEncoder
from skrl.agents.torch.exploration.torch_ant_sac_config import sweep_config
# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

def main(mode="train", config=None):
    # define models (stochastic and deterministic models) using mixins
    class StochasticActor(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, skill_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-5, max_log_std=2, squashedNormal=True):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, squashedNormal=squashedNormal)

            self.net = nn.Sequential(nn.Linear(self.num_observations + skill_space, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 2*self.num_actions))

        def compute(self, inputs, role):
            net_output = self.net(inputs["states"])
            output = net_output[:, :self.num_actions]
            log_std = net_output[:, self.num_actions:]
            return output, log_std, {}

    class Critic(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, skill_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions + skill_space, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1))

        def compute(self, inputs, role):
            return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

    class Discriminator(DeterministicMixin,Model,nn.Module):
        def __init__(self, observation_space, action_space, skill_state_space,skill_space,p_z, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)
            self.net = nn.Sequential(nn.Linear(skill_state_space, 300),
                                    nn.ReLU(),
                                    nn.Linear(300, 300),
                                    nn.ReLU(),
                                    nn.Linear(300, skill_space))
            self._p_z = p_z.to(device)
            self.skill_space = skill_space

        def forward(self, states):
            return self.net(states)
        



    # load and wrap the Isaac Gym environment
    if mode == "train":
        num_envs = 10
        headless = True
    elif mode == "eval":
        num_envs = 10
        headless = False

    env = load_isaacgym_env_preview4(task_name="Ant", num_envs=num_envs, headless=headless)
    env = wrap_env(env)

    device = env.device
    rl_device = env.rl_device


    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=200, num_envs=env.num_envs, device=rl_device)


    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = StochasticActor(env.observation_space, env.action_space, env.skill_dim, rl_device, clip_actions=True)
    models["critic_1"] = Critic(env.observation_space, env.action_space,env.skill_dim, rl_device)
    models["critic_2"] = Critic(env.observation_space, env.action_space,env.skill_dim, rl_device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space,env.skill_dim, rl_device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space,env.skill_dim, rl_device)

    # FIX ONES BELOW TO BE SAME AS ENV DISTRIBUTION
    # skill_space = env.number_of_skills
    # skill_state_space = env.skill_state_space
    # p_z = torch.full((skill_space,), 1/skill_space)
    # models["discriminator"] = Discriminator(env.observation_space,  env.action_space, skill_state_space, skill_space, p_z, device)



    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["gradient_steps"] = 50
    cfg["batch_size"] = 2048
    cfg["rollouts"] = 1
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 1e-4
    cfg["critic_learning_rate"] = 1e-4
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    cfg["grad_norm_clip"] = 0.5
    cfg["learn_entropy"] = True
    cfg["initial_entropy_value"] = 0.01
    cfg["entropy_learning_rate"] = 1e-4
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": rl_device}
    cfg["discriminator_learning_rate"] = 3e-4
    cfg["discriminator_batch_size"] = 128
    cfg["use_nstep_return"] = False
    cfg["num_envs"] = env.num_envs
    cfg["n_step"] = 1

    cfg["use_curiosity"] = False
    cfg["curiosity_lr"] = 1e-4
    cfg["curiosity_scale"] = 0.1
    cfg["anneal_curiosity"] = False
    cfg["anneal_curiosity_warmup"] = 8000
    cfg["curiosity_observation_type"] = "debugLSD"
    cfg["curiosity_beta"] = 0.1
    cfg["curiosity_epochs"] = 5
    cfg["curiosity_mini_batches"] = 32


    if config is not None:
        if config["entropy_coeff"] == "auto":
            cfg["learn_entropy"] = True
            cfg["initial_entropy_value"] = 0.1
        else:
            cfg["learn_entropy"] = False
            cfg["initial_entropy_value"] = config["entropy_coeff"]
        cfg["discount_factor"] = config["gamma"]

    if cfg["curiosity_observation_type"] == "full_state":
        curiosity_obs_space = env.observation_space.shape[0]
        state_dim = 64
    elif cfg["curiosity_observation_type"] == "velocities":
        curiosity_obs_space = 3
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "positions":
        curiosity_obs_space = 3
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "pos_vel":
        curiosity_obs_space = 6
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "xvel_yaw":
        curiosity_obs_space = 2
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "debugLSD":
        curiosity_obs_space = env.observation_space.shape[0] # remove the skills (last two elements)
        state_dim = curiosity_obs_space

    cfg["skill_discovery"] = True
    cfg["skill_space"] = env.skill_dim
    cfg["discrete_skills"] = env.discrete_skills
    cfg["mask_xy_pos"] = False
    cfg["skill_discovery_grad_steps"] = 32
    cfg["pretrain_normalizers"] = True
    cfg["sd_specific_state_ind"] = None
    cfg["on_policy_rewards"] = False

    # dual optimisation stuff:
    dual_lambda = 3000
    cfg["dual_optimisation"] = False
    cfg["dual_lambda"] = nn.Parameter(torch.log(torch.tensor(dual_lambda)))
    cfg["dual_lambda_lr"] = 1e-4
    cfg["dual_slack"] = 1e-6
    cfg["skill_discovery_distance_metric"] = "l2"

    if cfg["skill_discovery_distance_metric"] == "l2":
        use_spectral_norm = True
    else:
        use_spectral_norm = False

    cfg["dist_predictor"] = TrajectoryEncoder(curiosity_obs_space, [512,512], curiosity_obs_space, 
                                             min_log_std=1e-6,max_log_std=1e6,use_spectral_norm=False,device=rl_device)
    cfg["dist_model_lr"] = 1e-4


    if cfg["skill_discovery"]:
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": rl_device}

    if cfg["skill_discovery"]:
        cfg["lsd_model"] = TrajectoryEncoder(curiosity_obs_space, [512,512], cfg["skill_space"], 
                                             min_log_std=-5,max_log_std=2,use_spectral_norm=use_spectral_norm,device=rl_device)
        cfg["lsd_lr"] = 3e-5


    cfg["curiosity_preprocessor"] = RunningStandardScaler
    cfg["curiosity_preprocessor_kwargs"] = {"size": curiosity_obs_space, "device": device}


    cfg["limit_curiosity_obs"] = 3.14*torch.ones(curiosity_obs_space)
    hidden_dim = [5,5,3]
    cfg["curiosity_model"]= LBS(curiosity_obs_space,env.action_space.shape[0],state_dim,hidden_dim,device=rl_device,beta=cfg["curiosity_beta"])
    # cfg["curiosity_model"] = RND(curiosity_obs_space, [5,5,3], 1, activation=nn.ReLU(), device=device)




    if mode == "train":


        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 1
        cfg["experiment"]["checkpoint_interval"] = 100000
        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)


        # agent = ExplorationSAC(models=models,
        #             memory=memory,
        #             cfg=cfg,
        #             observation_space=env.observation_space,
        #             action_space=env.action_space,
        #             skill_state_space = skill_state_space,
        #             skill_space = skill_space,
        #             device=device)
        agent = LSDSAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=rl_device)


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 25000*cfg["rollouts"], "headless": True}

        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # start training
        trainer.train()

    elif mode == "eval":
        cfg["random_timesteps"] = 0
        cfg["learning_starts"] = 0
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0
        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

        # agent = ExplorationSAC(models=models,
        #             memory=memory,
        #             cfg=cfg,
        #             observation_space=env.observation_space,
        #             action_space=env.action_space,
        #             skill_state_space = skill_state_space,
        #             skill_space = skill_space,
        #             device=device)
        agent = LSDSAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=rl_device)
        


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 500, "headless": True}

        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # path = "runs/torch/Ant/24-02-09_10-41-47-378103_LSDSAC/checkpoints/" + "agent_33000.pt"
        # path = "runs/torch/Ant/24-02-09_14-57-40-904810_LSDSAC/checkpoints/" + "agent_30500.pt"
        # path = "runs/torch/Ant/24-02-10_10-49-41-675383_LSDSAC/checkpoints/" + "agent_80000.pt"
        # path = "runs/torch/Ant/24-02-10_19-02-06-806734_LSDSAC/checkpoints/" + "agent_65000.pt"
        # path = "runs/torch/Ant/24-02-10_23-43-29-344373_LSDSAC/checkpoints/" + "agent_42000.pt"
        # path = "runs/torch/Ant/24-02-11_15-30-42-392242_LSDSAC/checkpoints/" + "agent_80000.pt"

        # Test
        # path = "runs/torch/Ant/24-02-12_05-36-23-698728_LSDSAC/checkpoints/" + "agent_54500.pt"
        # path = "runs/torch/Ant/24-02-12_15-36-43-206898_LSDSAC/checkpoints/" + "agent_100000.pt"
        # path = "runs/torch/Ant/24-02-13_05-35-43-211135_LSDSAC/checkpoints/" + "agent_20000.pt"
        # path = "runs/torch/Ant/24-02-13_13-13-52-342623_LSDSAC/checkpoints/" + "agent_16000.pt"
        # path = "runs/torch/Ant/24-02-14_15-59-51-326680_LSDSAC/checkpoints/" + "agent_600000.pt"
        path = "runs/torch/Ant/24-02-16_10-58-40-223397_LSDSAC/checkpoints/" + "agent_1300000.pt"
        agent.load(path)

        # # start evaluation
        trainer.eval()

if __name__ == "__main__":

    # main("train")
    main("eval")
# 
