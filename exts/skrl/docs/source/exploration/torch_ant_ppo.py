import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo import LBSPPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.exploration.ppo_Lipschitz import LSDPPO, PPO_DEFAULT_CONFIG
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

def main(mode="train", path=None):
    # seed for reproducibility
    set_seed(42)  # e.g. `set_seed(42)` for fixed seed


    # define models (stochastic and deterministic models) using mixins
    class Policy(GaussianMixin, Model):
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

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, skill_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(nn.Linear(self.num_observations + skill_space, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1))

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    if mode == "train":
        headless = True
        num_envs = 4096
        task_name = "Ant"
    else:
        headless = False
        num_envs = 100
        task_name = "Ant"

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
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.skill_dim, rl_device)
    models["value"] = Value(env.observation_space, env.action_space, env.skill_dim, rl_device)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 48  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 12  # 24 * 4096 / 16384
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 2000
    cfg["learning_starts"] = 2000
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
    cfg["experiment"]["directory"] = "runs/torch/Ant"
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

    cfg["skill_discovery"] = True
    cfg["skill_space"] = env.skill_dim
    cfg["discrete_skills"] = env.discrete_skills
    cfg["mask_xy_pos"] = False
    cfg["skill_discovery_epochs"] = 4
    cfg["pretrain_normalizers"] = True
    cfg["intrinsic_reward_scale"] = 1.0
    cfg["sd_specific_state_ind"] = None # positions and linear velocities
    cfg["on_policy_rewards"] = True


    # dual optimisation stuff:
    dual_lambda = 30
    cfg["dual_optimisation"] = False
    cfg["dual_lambda"] = nn.Parameter(torch.log(torch.tensor(dual_lambda)))
    cfg["dual_lambda_lr"] = 1e-4
    cfg["dual_slack"] = 1e-3
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
    cfg["curiosity_model"]= LBS(curiosity_obs_space,env.action_space.shape[0],state_dim,hidden_dim,device=device,beta=cfg["curiosity_beta"])
    # cfg["curiosity_model"] = RND(curiosity_obs_space, [5,5,3], 1, activation=nn.ReLU(), device=device)





    if mode == "train":
        cfg["experiment"]["write_interval"] = 1
        cfg["experiment"]["checkpoint_interval"] = 500
        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = True           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
        cfg["experiment"]["wandb_save_files"] = ["./skrl/agents/torch/exploration/torch_ant_ppo.py",
                                                "./skrl/agents/torch/exploration/ppo_Lipschitz.py"]

        # agent = LBSPPO(models=models,
        #     memory=memory,
        #     cfg=cfg,
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     device=device)
        agent = LSDPPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 200000, "headless": True, "init_at_random_ep_len": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # start training
        trainer.train()
    else:

        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

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
        agent = LSDPPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=rl_device)


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 195, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        
        # LSD on plane.
        # path = "runs/torch/Ant/24-02-16_16-07-18-186344_LSDPPO/checkpoints/" + "agent_104000.pt"
        # LSD on trimesh.
        # path = "runs/torch/Ant/24-02-17_11-50-17-006503_LSDPPO/checkpoints/" + "agent_200000.pt"
        # METRA on plane (with modified env spacing)
        # path = "runs/torch/Ant/24-02-17_12-23-09-308294_LSDPPO/checkpoints/" + "agent_200000.pt"
        # METRA on plane (with original env spacing of 5)
        # path = "runs/torch/Ant/24-02-16_19-12-37-894346_LSDPPO/checkpoints/" + "agent_200000.pt"

        # Test:
        # path = "runs/torch/Ant/24-02-19_10-52-09-310295_LSDPPO/checkpoints/" + "agent_38500.pt"
        agent.load(path)

        # # start evaluation
        trainer.eval()
if __name__ == "__main__":
    # main("train")
    main("eval")