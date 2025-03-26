import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo import LBSPPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo_Lipschitz import LSDPPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo_hierarchical import HierarchicalLSDPPO
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.lbs_exploration.models import *
from skrl.agents.torch.exploration.RND import RND
from skrl.agents.torch.exploration.historyEncoder import LongHistoryEncoder

from skrl.agents.torch.exploration.skill_discovery import TrajectoryEncoder
from gym import spaces
import numpy as np
def main(mode="train", path=None):
    # seed for reproducibility
    set_seed(4)  # e.g. `set_seed(42)` for fixed seed

    class LowLevelPolicy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space,skill_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction,squashedNormal=True)

            self.net = nn.Sequential(nn.Linear(self.num_observations + skill_space, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.num_actions))
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), self.log_std_parameter, {}


    # define models (stochastic and deterministic models) using mixins
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction,squashedNormal=True)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.num_actions))
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), self.log_std_parameter, {}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    if mode == "train":
        headless = False
        num_envs = 100
        task_name = "AnymalTerrainPathfindingSkills"
    else:
        headless = False
        num_envs = 100
        task_name = "AnymalTerrainPathfindingSkills"

    # load and wrap the Isaac Gym environment
    env = load_isaacgym_env_preview4(task_name=task_name,num_envs=num_envs,headless=headless)
    env = wrap_env(env)

    device = env.device
    rl_device = env.rl_device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    high_level_action_space = spaces.Box(-np.ones(env.skill_dim), np.ones(env.skill_dim), dtype=np.float32)
    models = {}
    models["policy"] = Policy(env.observation_space, high_level_action_space, rl_device)
    models["value"] = Value(env.observation_space, high_level_action_space, rl_device)

    models["low_level_policy"] = LowLevelPolicy(env.observation_space, env.action_space, env.skill_dim, rl_device)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 48  # memory_size
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 12  # 24 * 4096 / 16384
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 500
    cfg["learning_starts"] = 500
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.001
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 1
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "runs/torch/AnymalTerrainPathfindingSkills"
    cfg["entropy_reward_scale"] = 0.0
    cfg["use_nstep_return"] = False
    cfg["num_envs"] = env.num_envs



    cfg["use_curiosity"] = False
    cfg["curiosity_lr"] = 5e-4
    cfg["curiosity_scale"] = 0.1
    cfg["anneal_curiosity"] = False
    cfg["anneal_curiosity_warmup"] = 20000
    cfg["anneal_curiosity_end"] = 50000
    cfg["curiosity_observation_type"] = "debugLSD"
    cfg["curiosity_beta"] = 0.1
    cfg["curiosity_epochs"] = 5
    cfg["curiosity_mini_batches"] = 12


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

    cfg["skill_discovery"] = False
    cfg["skill_space"] = env.skill_dim
    cfg["discrete_skills"] = env.discrete_skills
    cfg["mask_xy_pos"] = False
    cfg["skill_discovery_epochs"] = 4
    cfg["pretrain_normalizers"] = False
    cfg["intrinsic_reward_scale"] = 1.0
    cfg["sd_specific_state_ind"] = [0,1,2,3,4,5,6,7,8,9,194,195,196]#[0,1,190,191]
    # [0,1,2] lin vel, [3,4,5] ang vel, [6,7,8,9] quaternion, [190,191,192,193] contacts, [194,195,196] base pos
    cfg["on_policy_rewards"] = True


    # dual optimisation stuff:
    dual_lambda = 30
    cfg["dual_optimisation"] = True
    cfg["dual_lambda"] = nn.Parameter(torch.log(torch.tensor(dual_lambda)))
    cfg["dual_lambda_lr"] = 1e-4
    cfg["dual_slack"] = 1e-6
    cfg["skill_discovery_distance_metric"] = "one"
    cfg["learn_dual_lambda"] = True
    cfg["curiosity_obs_space"] = env.observation_space.shape[0]

    if not cfg["dual_optimisation"] and cfg["skill_discovery_distance_metric"] == "l2":
        use_spectral_norm = True
    else:
        use_spectral_norm = False

    cfg["dist_predictor"] = TrajectoryEncoder(curiosity_obs_space, [512,512], curiosity_obs_space, 
                                             min_log_std=1e-6,max_log_std=1e6,use_spectral_norm=False,device=device)
    cfg["dist_model_lr"] = 1e-4


    if cfg["skill_discovery"]:
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": rl_device}

    if cfg["skill_discovery"]:
        cfg["lsd_model"] = TrajectoryEncoder(curiosity_obs_space, [512,512], cfg["skill_space"], 
                                             min_log_std=-5,max_log_std=2,use_spectral_norm=use_spectral_norm,device=rl_device)
        cfg["lsd_lr"] = 5e-3


    cfg["curiosity_preprocessor"] = RunningStandardScaler
    cfg["curiosity_preprocessor_kwargs"] = {"size": curiosity_obs_space, "device": device}


    cfg["limit_curiosity_obs"] = 3.14*torch.ones(curiosity_obs_space)
    hidden_dim = [5,5,3]
    cfg["curiosity_model"]= LBS(curiosity_obs_space,env.action_space.shape[0],state_dim,hidden_dim,device=device,beta=cfg["curiosity_beta"])
    # cfg["curiosity_model"] = RND(curiosity_obs_space, [5,5,3], 1, activation=nn.ReLU(), device=device)





    if mode == "train":
        cfg["experiment"]["write_interval"] = 5
        cfg["experiment"]["checkpoint_interval"] = 1000
        cfg["experiment"]["directory"] = "runs/torch/AnymalTerrainPathfindingSkills"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)


        # agent = LBSPPO(models=models,
        #     memory=memory,
        #     cfg=cfg,
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     device=device)
        agent = HierarchicalLSDPPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=high_level_action_space,
            device=device)
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 100000, "headless": True, "init_at_random_ep_len": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-03_11-21-02-421323_LSDPPO/checkpoints/agent_200000.pt"
        agent.load_for_training(path)
        # start training
        trainer.train()
    else:
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                                "mode":"offline"}   
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0
        cfg["random_timesteps"] = 0
        cfg["learning_starts"] = 0

        # agent = LBSPPO(models=models,
        #             memory=memory,
        #             cfg=cfg,
        #             observation_space=env.observation_space,
        #             action_space=env.action_space,
        #             device=device)
        agent = HierarchicalLSDPPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=high_level_action_space,
            device=rl_device)


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 295, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        # path = "runs/torch/AnymalTerrain/24-01-25_11-07-28-890398_LBSPPO/checkpoints/agent_24000.pt" # default ppo
        # path = "runs/torch/AnymalTerrain/24-01-25_12-17-03-947679_LBSPPO/checkpoints/agent_2000.pt" # curiosity 0.1
        # path = "runs/torch/AnymalTerrain/24-01-25_12-19-52-628674_LBSPPO/checkpoints/agent_24000.pt" # default ppo flat
        # path = "runs/torch/AnymalTerrain/24-01-25_13-14-08-453459_LBSPPO/checkpoints/agent_24000.pt" # curiosity 1e-5 flat
        # path = "runs/torch/AnymalTerrain/24-01-25_15-57-29-035792_LBSPPO/checkpoints/agent_24000.pt" # curiosity 1e-1 flat
        # path = "runs/torch/AnymalTerrain/24-01-25_16-22-58-279781_LBSPPO/checkpoints/agent_24000.pt" # curiosity 1 flat
        # path = "runs/torch/AnymalTerrain/24-01-25_17-20-57-871710_LBSPPO/checkpoints/agent_24000.pt" # curiosity 10 flat
        # path = "runs/torch/AnymalTerrain/24-01-26_11-36-37-868804_LBSPPO/checkpoints/agent_11500.pt" # /
        # path = "runs/torch/AnymalTerrain/24-01-26_11-51-52-676593_LBSPPO/checkpoints/agent_24000.pt" # / # curiosity 0.1
        # path = "runs/torch/AnymalTerrain/24-01-26_12-14-38-148587_LBSPPO/checkpoints/agent_24000.pt" # / # default ppo
        # path = "runs/torch/AnymalTerrain/24-01-26_13-05-03-295748_LBSPPO/checkpoints/agent_24000.pt" # / # no obs/act clip
        # path = "runs/torch/AnymalTerrain/24-01-26_14-20-58-503681_LBSPPO/checkpoints/agent_24000.pt" # / # 
        # path = "runs/torch/AnymalTerrain/24-01-26_15-37-41-739814_LBSPPO/checkpoints/agent_2500.pt" # / # curiosity 0.01\
        # path = "runs/torch/AnymalTerrain/24-01-28_12-58-13-136085_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf default
        # path = "runs/torch/AnymalTerrain/24-01-28_13-20-10-440236_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf curiosity 0.1
        # path = "runs/torch/AnymalTerrain/24-01-29_12-02-36-941278_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf curiosity 0.01
        #
        #
        # Trying with actuator network and only linear velocity curiosity
        # path = "runs/torch/AnymalTerrain/24-01-29_15-46-12-105554_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf curiosity 0.1
        # path = "runs/torch/AnymalTerrain/24-01-29_14-13-10-661960_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf curiosity 0.001
        # path = "runs/torch/AnymalTerrain/24-01-29_15-21-20-481893_LBSPPO/checkpoints/agent_24000.pt" # / # new urdf default ppo

        # And rough terrain

        # path = "runs/torch/AnymalTerrain/24-01-29_17-32-44-448886_LBSPPO/checkpoints/agent_24000.pt" # / # curiosity 0.1
        # path = "runs/torch/AnymalTerrain/24-01-29_18-02-50-680809_LBSPPO/checkpoints/agent_24000.pt" # / # default
        # path = "runs/torch/AnymalTerrain/24-01-30_10-28-03-728360_LBSPPO/checkpoints/agent_24000.pt" # / # curiosity 0.01
        # path = "runs/torch/AnymalTerrain/24-01-30_11-59-51-743275_LBSPPO/checkpoints/agent_24000.pt" # / # curiosity no z vel
        # path = "runs/torch/AnymalTerrain/24-01-30_16-22-16-746963_LBSPPO/checkpoints/agent_24000.pt" # / # curiosity base_pos

        # path = "runs/torch/AnymalTerrain/24-01-31_16-30-42-536428_LBSPPO/checkpoints/agent_24000.pt" # / # 


        # PATHFINDING
        # path = "runs/torch/AnymalTerrainPathfinding/24-01-31_18-06-40-222326_LBSPPO/checkpoints/agent_22500.pt" # / # 
        # path = "runs/torch/AnymalTerrainPathfinding/24-01-31_18-33-00-840957_LBSPPO/checkpoints/agent_24000.pt" # / #
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_10-44-48-059171_LBSPPO/checkpoints/agent_17000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_11-02-34-566788_LBSPPO/checkpoints/agent_16500.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_13-50-22-968721_LBSPPO/checkpoints/agent_24000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_14-17-25-476641_LBSPPO/checkpoints/agent_24000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_15-53-19-741564_LBSPPO/checkpoints/agent_40000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_18-41-04-124342_LBSPPO/checkpoints/agent_28500.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_12-12-25-924306_LBSPPO/checkpoints/agent_24000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_12-39-08-013104_LBSPPO/checkpoints/agent_24000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_14-02-15-632614_LBSPPO/checkpoints/agent_18000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_15-34-42-724064_LBSPPO/checkpoints/agent_24000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_16-26-36-861165_LBSPPO/checkpoints/agent_11500.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_16-54-44-984977_LBSPPO/checkpoints/agent_24000.pt"


        # path = "runs/torch/AnymalTerrainPathfinding/24-02-02_18-55-19-940804_LBSPPO/checkpoints/agent_48000.pt"
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-03_20-13-32-652403_LBSPPO/checkpoints/agent_64000.pt"

        # One with RND:
        # path = "runs/torch/AnymalTerrainPathfinding/24-02-01_19-09-27-400012_LBSPPO/checkpoints/agent_40000.pt"
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-15_10-18-55-406493_LSDPPO/checkpoints/agent_31000.pt"


        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-19_14-13-37-875325_LSDPPO/checkpoints/agent_10000.pt"
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-23_17-06-32-238077_LSDPPO/checkpoints/agent_100000.pt"
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-25_12-24-19-759585_LSDPPO/checkpoints/agent_100000.pt"
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-28_10-30-23-144950_LSDPPO/checkpoints/agent_100000.pt"
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-28_13-37-41-156235_LSDPPO/checkpoints/agent_100000.pt"

        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-28_15-42-35-142085_LSDPPO/checkpoints/agent_100000.pt"
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-28_17-59-56-013432_LSDPPO/checkpoints/agent_100000.pt"


        # METRA one:
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-02-29_18-45-35-407949_LSDPPO/checkpoints/agent_100000.pt"

        path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-01_23-07-31-847437_LSDPPO/checkpoints/agent_200000.pt"
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-02_16-29-19-880103_LSDPPO/checkpoints/agent_200000.pt"
        # THis one was trained with randomized initil yaw:
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-02_22-54-02-056134_LSDPPO/checkpoints/agent_200000.pt"
        # And with higher learning rate:
        path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-03_11-21-02-421323_LSDPPO/checkpoints/agent_200000.pt"
        # And with rough terrain:
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-03_17-33-25-439270_LSDPPO/checkpoints/agent_200000.pt"
        # And with rough terrain and lower lambda learning rate:
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-04_00-25-47-652004_LSDPPO/checkpoints/agent_200000.pt"
        # With higher lambda learning rate and gaps:
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-04_14-01-27-310833_LSDPPO/checkpoints/agent_200000.pt"
        # Only on gaps:
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-04_21-14-46-051215_LSDPPO/checkpoints/agent_200000.pt"
        # With lower velocity limits on flat:
        # path = "runs/torch/AnymalTerrainPathfindingSkills/24-03-05_11-10-18-803553_LSDPPO/checkpoints/agent_63000.pt"
        agent.load_for_training(path,eval=True)        # # start evaluation
        trainer.eval()
if __name__ == "__main__":
    # main("train")
    main("eval")